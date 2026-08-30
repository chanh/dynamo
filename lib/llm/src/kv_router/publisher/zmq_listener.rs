// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::cache_loss::{
    CacheEvidenceBatch, CacheEvidenceMutation, CacheEvidenceStoredBlock, CacheOwner, CacheTier,
};
use dynamo_kv_router::protocols::*;
use dynamo_kv_router::zmq_wire::*;

use crate::kv_router::metrics::kv_publisher_metrics;
use crate::utils::zmq::{connect_sub_socket, multipart_message};

pub(super) struct DecodedZmqKvBatch {
    pub(super) source_cursor: u64,
    pub(super) batch: KvEventBatch,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SourceSequenceTracker {
    last: Option<u64>,
}

// Match the supported OffloadingConnector's production tracker bound. A lower
// consumer bound would make otherwise supported CPU-cache churn permanently
// incomplete before the producer reaches its own safety limit.
const MAX_EVIDENCE_HASH_MAPPINGS: usize = 4_000_000;
const PARENT_SEQUENCE_HASH_ALGORITHM: &str = "dynamo_xxh3_u32le_seed1337_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceHashMapping {
    Known(CacheEvidenceStoredBlock),
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvidenceLookaheadMapping {
    parent_external_hash: Option<u64>,
    sequence_hash: u64,
    block: CacheEvidenceStoredBlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceLookaheadMappingState {
    Known(EvidenceLookaheadMapping),
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceMutationError {
    Invalid,
    UnresolvedHash,
    AmbiguousHash,
    MappingOverflow,
}

impl EvidenceMutationError {
    fn metric_reason(self) -> &'static str {
        match self {
            Self::Invalid => "evidence_conversion_error",
            Self::UnresolvedHash => "evidence_hash_unresolved",
            Self::AmbiguousHash => "evidence_hash_ambiguous",
            Self::MappingOverflow => "evidence_hash_mapping_overflow",
        }
    }
}

#[derive(Debug)]
struct EvidenceHashResolver {
    // The listener lifetime binds the source incarnation. DP ranks may share
    // one source, so the owner remains part of the identity fence. External
    // block hashes are scoped to a cache group and may repeat across groups.
    mappings: HashMap<(CacheOwner, u32, u64), EvidenceHashMapping>,
    lookahead_mappings: HashMap<(CacheOwner, u32, u64), EvidenceLookaheadMappingState>,
    capacity: usize,
}

impl EvidenceHashResolver {
    fn new(capacity: usize) -> Self {
        Self {
            mappings: HashMap::new(),
            lookahead_mappings: HashMap::new(),
            capacity,
        }
    }

    fn clear(&mut self) {
        self.mappings.clear();
        self.lookahead_mappings.clear();
    }

    fn clear_owner(&mut self, owner: CacheOwner) {
        self.mappings
            .retain(|(mapped_owner, _, _), _| *mapped_owner != owner);
        self.lookahead_mappings
            .retain(|(mapped_owner, _, _), _| *mapped_owner != owner);
    }

    fn learn_gpu(
        &mut self,
        owner: CacheOwner,
        group: u32,
        blocks: &[CacheEvidenceStoredBlock],
    ) -> Result<(), EvidenceMutationError> {
        let mut pending = HashMap::with_capacity(blocks.len());
        let mut ambiguous = HashSet::new();
        for &block in blocks {
            let key = (owner, group, block.external_hash);
            if let Some(previous) = pending.insert(key, block)
                && previous != block
            {
                ambiguous.insert(key);
            }
            match self.mappings.get(&key) {
                Some(EvidenceHashMapping::Known(existing)) if *existing != block => {
                    ambiguous.insert(key);
                }
                Some(EvidenceHashMapping::Ambiguous) => {
                    ambiguous.insert(key);
                }
                _ => {}
            }
        }

        if !ambiguous.is_empty() {
            for &key in &ambiguous {
                if self.mappings.contains_key(&key) {
                    self.mappings.insert(key, EvidenceHashMapping::Ambiguous);
                }
            }
            let new_ambiguous = ambiguous
                .iter()
                .filter(|key| !self.mappings.contains_key(key))
                .count();
            if self
                .mappings
                .len()
                .saturating_add(self.lookahead_mappings.len())
                .saturating_add(new_ambiguous)
                > self.capacity
            {
                return Err(EvidenceMutationError::MappingOverflow);
            }
            for key in ambiguous {
                self.mappings.insert(key, EvidenceHashMapping::Ambiguous);
            }
            return Err(EvidenceMutationError::AmbiguousHash);
        }

        let new_entries = pending
            .keys()
            .filter(|key| !self.mappings.contains_key(key))
            .count();
        if self
            .mappings
            .len()
            .saturating_add(self.lookahead_mappings.len())
            .saturating_add(new_entries)
            > self.capacity
        {
            return Err(EvidenceMutationError::MappingOverflow);
        }

        for (key, block) in pending {
            self.mappings
                .entry(key)
                .or_insert(EvidenceHashMapping::Known(block));
        }
        Ok(())
    }

    fn learn_gpu_lookahead(
        &mut self,
        owner: CacheOwner,
        group: u32,
        mapping: EvidenceLookaheadMapping,
    ) -> Result<(), EvidenceMutationError> {
        let key = (owner, group, mapping.block.external_hash);
        match self.lookahead_mappings.get(&key) {
            Some(EvidenceLookaheadMappingState::Known(known)) if *known == mapping => {
                return Ok(());
            }
            Some(_) => {
                self.lookahead_mappings
                    .insert(key, EvidenceLookaheadMappingState::Ambiguous);
                return Err(EvidenceMutationError::AmbiguousHash);
            }
            None => {}
        }
        if !self.lookahead_mappings.contains_key(&key)
            && self
                .mappings
                .len()
                .saturating_add(self.lookahead_mappings.len())
                .saturating_add(1)
                > self.capacity
        {
            return Err(EvidenceMutationError::MappingOverflow);
        }
        self.lookahead_mappings
            .insert(key, EvidenceLookaheadMappingState::Known(mapping));
        Ok(())
    }

    fn resolve_cpu_lookahead(
        &self,
        owner: CacheOwner,
        group: u32,
        external_hash: u64,
    ) -> Option<EvidenceLookaheadMapping> {
        self.lookahead_mappings
            .get(&(owner, group, external_hash))
            .and_then(|mapping| match mapping {
                EvidenceLookaheadMappingState::Known(mapping) => Some(*mapping),
                EvidenceLookaheadMappingState::Ambiguous => None,
            })
    }

    fn resolve_cpu(
        &self,
        owner: CacheOwner,
        group: u32,
        external_hashes: &[u64],
    ) -> Result<Vec<CacheEvidenceStoredBlock>, EvidenceMutationError> {
        external_hashes
            .iter()
            .map(
                |&external_hash| match self.mappings.get(&(owner, group, external_hash)) {
                    Some(EvidenceHashMapping::Known(block)) => Ok(*block),
                    Some(EvidenceHashMapping::Ambiguous) => {
                        Err(EvidenceMutationError::AmbiguousHash)
                    }
                    None => Err(EvidenceMutationError::UnresolvedHash),
                },
            )
            .collect()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SourceSequenceObservation {
    Contiguous,
    Gap { missing: u64 },
    Stale,
}

impl SourceSequenceTracker {
    fn observe(&mut self, sequence: u64) -> SourceSequenceObservation {
        let observation = match self.last {
            None if sequence == 0 => SourceSequenceObservation::Contiguous,
            None => SourceSequenceObservation::Gap { missing: sequence },
            Some(last) if sequence == last.saturating_add(1) => {
                SourceSequenceObservation::Contiguous
            }
            Some(last) if sequence > last => SourceSequenceObservation::Gap {
                missing: sequence - last - 1,
            },
            Some(_) => SourceSequenceObservation::Stale,
        };
        if !matches!(observation, SourceSequenceObservation::Stale) {
            self.last = Some(sequence);
        }
        observation
    }
}

fn filter_preserves_publisher_telemetry(reason: ZmqEventFilterReason) -> bool {
    matches!(reason, ZmqEventFilterReason::NonMainAttentionKind)
}

struct EvidenceHeartbeatState {
    owners: HashSet<CacheOwner>,
    source_cursor: Option<u64>,
}

impl EvidenceHeartbeatState {
    fn new(owner: CacheOwner) -> Self {
        Self {
            owners: HashSet::from([owner]),
            source_cursor: None,
        }
    }

    fn observe(&mut self, owner: CacheOwner, source_cursor: u64) {
        self.owners.insert(owner);
        self.source_cursor = Some(source_cursor);
    }

    fn entries(&self) -> impl Iterator<Item = (CacheOwner, Option<u64>)> + '_ {
        self.owners
            .iter()
            .copied()
            .map(|owner| (owner, self.source_cursor))
    }
}

/// Decode the transport envelope shared by legacy and residency-aware inputs.
///
/// Callers retain their own malformed-input and protocol-version policies.
pub(super) fn decode_zmq_kv_batch(
    mut frames: crate::utils::zmq::MultipartMessage,
) -> Result<DecodedZmqKvBatch> {
    if frames.len() != 3 {
        anyhow::bail!("expected three ZMQ frames, received {}", frames.len());
    }
    let payload = frames.pop().expect("frame count was validated");
    let sequence = frames.pop().expect("frame count was validated");
    let sequence: [u8; 8] = sequence.try_into().map_err(|sequence: Vec<u8>| {
        anyhow::anyhow!(
            "ZMQ sequence must contain eight bytes, received {}",
            sequence.len()
        )
    })?;
    let batch = decode_event_batch(&payload).context("failed to decode KV event batch")?;
    Ok(DecodedZmqKvBatch {
        source_cursor: u64::from_be_bytes(sequence),
        batch,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    worker_id: WorkerId,
    configured_dp_rank: DpRank,
    evidence_incarnation_id: u64,
    tx: mpsc::UnboundedSender<Vec<PlacementEvent>>,
    evidence_tx: Option<mpsc::UnboundedSender<CacheEvidenceBatch>>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
    next_event_id: Arc<AtomicU64>,
    image_token_id: Option<u32>,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let mut normalizer = ZmqEventNormalizer::new(kv_block_size).with_image_token_id(image_token_id);
    let evidence_warning_count = Arc::new(AtomicU32::new(0));
    let mut evidence_hash_resolver = EvidenceHashResolver::new(MAX_EVIDENCE_HASH_MAPPINGS);
    let socket = match connect_sub_socket(&zmq_endpoint, Some(&zmq_topic)).await {
        Ok(socket) => socket,
        Err(error) => {
            tracing::error!(endpoint = %zmq_endpoint, topic = %zmq_topic, error = %error, "ZMQ listener failed to connect");
            return;
        }
    };
    let mut socket = socket;
    let metrics = kv_publisher_metrics();
    if cancellation_token.is_cancelled() {
        return;
    }

    let mut messages_processed = 0u64;
    let mut source_sequence = SourceSequenceTracker::default();
    let mut heartbeat_state = EvidenceHeartbeatState::new(CacheOwner {
        worker_id,
        dp_rank: configured_dp_rank,
    });
    let mut heartbeat = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_secs(1),
        Duration::from_secs(1),
    );
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let exit_reason = 'main: loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                break 'main String::from("cancellation token cancelled");
            }

            msg_result = socket.next() => {
                let frames = match msg_result {
                    Some(Ok(frames)) => multipart_message(frames),
                    Some(Err(error)) => {
                        tracing::error!(endpoint = %zmq_endpoint, error = %error, "ZMQ listener recv failed");
                        break 'main format!("ZMQ recv failed: {error}");
                    }
                    None => break 'main String::from("ZMQ stream ended"),
                };
                let DecodedZmqKvBatch {
                    source_cursor: engine_seq,
                    batch,
                } = match decode_zmq_kv_batch(frames) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        tracing::warn!(%error, "Failed to decode ZMQ KV batch");
                        evidence_hash_resolver.clear();
                        if let Some(metrics) = &metrics {
                            metrics.mark_telemetry_incomplete("source_decode_error");
                        }
                        continue;
                    }
                };

                match source_sequence.observe(engine_seq) {
                    SourceSequenceObservation::Contiguous => {}
                    SourceSequenceObservation::Gap { missing } => {
                        tracing::warn!(engine_seq, missing, "ZMQ KV source sequence gap");
                        evidence_hash_resolver.clear();
                        if let Some(metrics) = &metrics {
                            metrics.record_source_gap(missing);
                        }
                    }
                    SourceSequenceObservation::Stale => {
                        tracing::warn!(engine_seq, "Discarding stale ZMQ KV source batch");
                        if let Some(metrics) = &metrics {
                            metrics.record_source_out_of_order();
                        }
                        continue;
                    }
                }

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (engine_seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    engine_seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
                let owner = CacheOwner { worker_id, dp_rank };
                heartbeat_state.observe(owner, engine_seq);
                let mut telemetry_complete = true;
                let clear_epoch_ids: std::collections::HashSet<_> = batch
                    .events
                    .iter()
                    .filter_map(RawKvEvent::epoch_id)
                    .collect();
                let clear_epoch_id = (clear_epoch_ids.len() == 1)
                    .then(|| (*clear_epoch_ids.iter().next().expect("one epoch id")).to_string());
                if !clear_epoch_ids.is_empty()
                    && (clear_epoch_ids.len() != 1
                        || batch.events.iter().any(|event| {
                            !matches!(event, RawKvEvent::AllBlocksCleared { .. })
                                || event.epoch_id() != clear_epoch_id.as_deref()
                        }))
                {
                    telemetry_complete = false;
                }
                let mutations = batch.events.iter().filter_map(|event| {
                    match cache_evidence_mutation_with_resolver(
                        event,
                        owner,
                        image_token_id,
                        &evidence_warning_count,
                        &mut evidence_hash_resolver,
                    ) {
                        Ok(mutation) => mutation,
                        Err(error) => {
                            telemetry_complete = false;
                            if let Some(metrics) = &metrics {
                                metrics.increment_zmq_conversion_issue(
                                    event.event_type_label(),
                                    error.metric_reason(),
                                );
                                metrics.mark_telemetry_incomplete(error.metric_reason());
                            }
                            None
                        }
                    }
                }).collect();
                if let Some(evidence_tx) = &evidence_tx {
                    if let Some(metrics) = &metrics {
                        metrics.increment_evidence_queue_depth();
                    }
                    if evidence_tx
                        .send(CacheEvidenceBatch {
                            owner,
                            source_cursor: engine_seq,
                            source_incarnation_id: Some(evidence_incarnation_id),
                            heartbeat: false,
                            watermark_source_cursor: None,
                            telemetry_complete,
                            mutations,
                            barrier_id: batch.barrier_id,
                            epoch_id: batch.epoch_id.or(clear_epoch_id),
                        })
                        .is_err()
                    {
                        tracing::warn!("Failed to send cache evidence - receiver dropped");
                        if let Some(metrics) = &metrics {
                            metrics.decrement_evidence_queue_depth();
                            metrics.mark_telemetry_incomplete("evidence_channel_dropped");
                        }
                        break 'main String::from("cache-evidence receiver dropped");
                    }
                }
                let mut events = Vec::with_capacity(batch.events.len());
                for raw_event in batch.events {
                    let event_type = raw_event.event_type_label();
                    if let Some(metrics) = &metrics {
                        metrics.increment_zmq_event("received", event_type);
                    }
                    let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                    let raw_event = match normalizer.preprocess_with_reason(raw_event, worker) {
                        Ok(raw_event) => raw_event,
                        Err(reason) => {
                            if let Some(metrics) = &metrics {
                                metrics.increment_zmq_filtered_event(event_type, reason.as_label());
                                if !filter_preserves_publisher_telemetry(reason) {
                                    metrics.mark_telemetry_incomplete("filtered_event");
                                }
                            }
                            continue;
                        }
                    };
                    if let Some(metrics) = &metrics {
                        metrics.increment_zmq_event("accepted", event_type);
                    }
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let Some(event) =
                        normalizer.normalize_preprocessed(raw_event, event_id, worker)
                    else {
                        if let Some(metrics) = &metrics {
                            metrics.increment_zmq_conversion_issue(event_type, "conversion_none");
                            metrics.mark_telemetry_incomplete("conversion_error");
                        }
                        continue;
                    };
                    if matches!(event.event.data, KvCacheEventData::Stored(ref data) if data.blocks.is_empty())
                        && let Some(metrics) = &metrics
                    {
                        metrics.increment_zmq_suspicious_event(event_type, "empty_store_blocks");
                        metrics.mark_telemetry_incomplete("empty_store_blocks");
                    }
                    events.push(event);
                }
                if !events.is_empty() {
                    let event_count = events.len() as u64;
                    if tx.send(events).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        break 'main String::from("channel receiver dropped");
                    }
                    messages_processed += event_count;
                }
            }

            // Keep this after socket.next() in the biased select: a watermark
            // must never overtake source input that is already ready.
            _ = heartbeat.tick(), if evidence_tx.is_some() => {
                let evidence_tx = evidence_tx.as_ref().expect("heartbeat requires evidence channel");
                for (owner, watermark_source_cursor) in heartbeat_state.entries() {
                    if let Some(metrics) = &metrics {
                        metrics.increment_evidence_queue_depth();
                    }
                    if evidence_tx.send(CacheEvidenceBatch {
                        owner,
                        source_cursor: watermark_source_cursor.unwrap_or(0),
                        source_incarnation_id: Some(evidence_incarnation_id),
                        heartbeat: true,
                        watermark_source_cursor,
                        telemetry_complete: true,
                        mutations: Vec::new(),
                        barrier_id: None,
                        epoch_id: None,
                    }).is_err() {
                        tracing::warn!("Failed to send cache-evidence heartbeat - receiver dropped");
                        if let Some(metrics) = &metrics {
                            metrics.decrement_evidence_queue_depth();
                            metrics.mark_telemetry_incomplete("evidence_channel_dropped");
                        }
                        break 'main String::from("cache-evidence receiver dropped");
                    }
                }
            }
        }
    };

    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}

#[cfg(test)]
fn cache_evidence_mutation(
    event: &RawKvEvent,
    image_token_id: Option<u32>,
    warning_count: &Arc<AtomicU32>,
) -> Result<Option<CacheEvidenceMutation>, EvidenceMutationError> {
    cache_evidence_mutation_with_resolver(
        event,
        CacheOwner {
            worker_id: 0,
            dp_rank: 0,
        },
        image_token_id,
        warning_count,
        &mut EvidenceHashResolver::new(MAX_EVIDENCE_HASH_MAPPINGS),
    )
}

fn cache_evidence_mutation_with_resolver(
    event: &RawKvEvent,
    owner: CacheOwner,
    image_token_id: Option<u32>,
    warning_count: &Arc<AtomicU32>,
    resolver: &mut EvidenceHashResolver,
) -> Result<Option<CacheEvidenceMutation>, EvidenceMutationError> {
    if !matches!(event.ownership(), Ok(KvEventOwnership::Framework)) {
        return Err(EvidenceMutationError::Invalid);
    }
    if matches!(event.locality(), Some(Locality::Remote | Locality::Unknown)) {
        return Err(EvidenceMutationError::Invalid);
    }

    let tier = |medium: Option<&str>| match medium {
        None => Ok(CacheTier::Gpu),
        Some(medium) => match StorageTier::from_kv_medium(medium) {
            Some(StorageTier::Device) => Ok(CacheTier::Gpu),
            Some(StorageTier::HostPinned) => Ok(CacheTier::Cpu),
            Some(StorageTier::Disk | StorageTier::External) => Err(EvidenceMutationError::Invalid),
            None => Err(EvidenceMutationError::Invalid),
        },
    };

    match event {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            parent_sequence_hash,
            parent_sequence_hash_algorithm,
            eagle_lookahead_sequence_hash,
            eagle_lookahead_sequence_hash_algorithm,
            eagle_lookahead_token_ids,
            token_ids,
            block_size,
            medium,
            lora_name,
            cache_namespace,
            block_mm_infos,
            is_eagle,
            group_idx,
            ..
        } => {
            let eagle_lookahead = match (
                eagle_lookahead_sequence_hash,
                eagle_lookahead_sequence_hash_algorithm.as_deref(),
                eagle_lookahead_token_ids.as_deref(),
            ) {
                (None, None, None) => None,
                (Some(sequence), Some(PARENT_SEQUENCE_HASH_ALGORITHM), Some(tokens)) => {
                    Some((sequence.into_u64(), tokens))
                }
                _ => return Err(EvidenceMutationError::Invalid),
            };
            let parent_sequence_hash = match (
                parent_sequence_hash,
                parent_sequence_hash_algorithm.as_deref(),
            ) {
                (None, None) => None,
                (Some(hash), Some(PARENT_SEQUENCE_HASH_ALGORITHM)) => Some(hash.into_u64()),
                _ => return Err(EvidenceMutationError::Invalid),
            };
            if parent_sequence_hash.is_some() && parent_block_hash.is_none() {
                return Err(EvidenceMutationError::Invalid);
            }
            let external_hashes: Vec<_> = block_hashes
                .iter()
                .copied()
                .map(BlockHashValue::into_u64)
                .collect();
            let store_tier = tier(medium.as_deref())?;
            if let Some((sequence_hash, lookahead_tokens)) = eagle_lookahead {
                if parent_sequence_hash.is_some()
                    || !token_ids.is_empty()
                    || block_hashes.len() != 1
                    || *block_size == 0
                    || lookahead_tokens.len() != *block_size
                    || *is_eagle != Some(false)
                    || block_mm_infos.is_some()
                {
                    return Err(EvidenceMutationError::Invalid);
                }
                let [stored] = create_stored_blocks(
                    (*block_size)
                        .try_into()
                        .map_err(|_| EvidenceMutationError::Invalid)?,
                    lookahead_tokens,
                    &[*block_size as u64],
                    &external_hashes,
                    lora_name.as_deref(),
                    cache_namespace.as_deref(),
                    warning_count,
                    block_mm_infos.as_deref(),
                    Some(false),
                    image_token_id,
                )
                .try_into()
                .map_err(|_| EvidenceMutationError::Invalid)?;
                let mapping = EvidenceLookaheadMapping {
                    parent_external_hash: parent_block_hash.map(BlockHashValue::into_u64),
                    sequence_hash,
                    block: CacheEvidenceStoredBlock {
                        external_hash: stored.block_hash.0,
                        tokens_hash: stored.tokens_hash.0,
                    },
                };
                if store_tier == CacheTier::Gpu {
                    resolver.learn_gpu_lookahead(
                        owner,
                        group_idx.ok_or(EvidenceMutationError::Invalid)?,
                        mapping,
                    )?;
                }
                return Ok(Some(CacheEvidenceMutation::StoreEagleLookahead {
                    tier: store_tier,
                    group_idx: group_idx.ok_or(EvidenceMutationError::Invalid)?,
                    parent_external_hash: mapping.parent_external_hash,
                    sequence_hash: mapping.sequence_hash,
                    block: mapping.block,
                }));
            }
            let stored = if store_tier == CacheTier::Cpu && token_ids.is_empty() && *block_size == 0
            {
                let group = group_idx.ok_or(EvidenceMutationError::Invalid)?;
                match resolver.resolve_cpu(owner, group, &external_hashes) {
                    Ok(stored) => stored,
                    Err(EvidenceMutationError::UnresolvedHash) if external_hashes.len() == 1 => {
                        let mapping = resolver
                            .resolve_cpu_lookahead(owner, group, external_hashes[0])
                            .ok_or(EvidenceMutationError::UnresolvedHash)?;
                        return Ok(Some(CacheEvidenceMutation::StoreEagleLookahead {
                            tier: CacheTier::Cpu,
                            group_idx: group,
                            parent_external_hash: mapping.parent_external_hash,
                            sequence_hash: mapping.sequence_hash,
                            block: mapping.block,
                        }));
                    }
                    Err(error) => return Err(error),
                }
            } else {
                let num_block_tokens = vec![*block_size as u64; external_hashes.len()];
                let stored = create_stored_blocks(
                    (*block_size)
                        .try_into()
                        .map_err(|_| EvidenceMutationError::Invalid)?,
                    token_ids,
                    &num_block_tokens,
                    &external_hashes,
                    lora_name.as_deref(),
                    cache_namespace.as_deref(),
                    warning_count,
                    block_mm_infos.as_deref(),
                    *is_eagle,
                    image_token_id,
                );
                if stored.len() != external_hashes.len() {
                    return Err(EvidenceMutationError::Invalid);
                }
                let stored: Vec<_> = stored
                    .into_iter()
                    .map(|block| CacheEvidenceStoredBlock {
                        external_hash: block.block_hash.0,
                        tokens_hash: block.tokens_hash.0,
                    })
                    .collect();
                if store_tier == CacheTier::Gpu
                    && let Some(group) = group_idx
                {
                    resolver.learn_gpu(owner, *group, &stored)?;
                }
                stored
            };
            if let Some(parent_sequence_hash) = parent_sequence_hash {
                Ok(Some(CacheEvidenceMutation::StoreWithParentAttestation {
                    tier: store_tier,
                    group_idx: group_idx.ok_or(EvidenceMutationError::Invalid)?,
                    parent_external_hash: parent_block_hash
                        .ok_or(EvidenceMutationError::Invalid)?
                        .into_u64(),
                    parent_sequence_hash,
                    blocks: stored,
                }))
            } else {
                Ok(Some(CacheEvidenceMutation::Store {
                    tier: store_tier,
                    group_idx: *group_idx,
                    parent_external_hash: parent_block_hash.map(BlockHashValue::into_u64),
                    blocks: stored,
                }))
            }
        }
        RawKvEvent::BlockRemoved {
            block_hashes,
            medium,
            group_idx,
            ..
        } => Ok(Some(CacheEvidenceMutation::Remove {
            tier: tier(medium.as_deref())?,
            group_idx: *group_idx,
            block_hashes: block_hashes
                .iter()
                .copied()
                .map(BlockHashValue::into_u64)
                .collect(),
        })),
        RawKvEvent::AllBlocksCleared {
            medium, epoch_id, ..
        } => {
            if medium.is_none() || epoch_id.is_some() {
                resolver.clear_owner(owner);
            }
            let cleared_tier = match medium.as_deref() {
                Some(medium) => Some(tier(Some(medium))?),
                None => None,
            };
            Ok(Some(CacheEvidenceMutation::Clear { tier: cleared_tier }))
        }
        RawKvEvent::Ignored => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::cache_loss::{
        CacheEvidenceLedger, CacheGroupHashSequence, CacheGroupKind, KnownBool, KnownPrefixLength,
    };

    const TEST_OWNER: CacheOwner = CacheOwner {
        worker_id: 7,
        dp_rank: 0,
    };

    fn constituent_store(
        group_idx: u32,
        block_size: usize,
        block_hashes: &[u64],
        parent_block_hash: Option<u64>,
        token_ids: &[u32],
        kind: KvCacheSpecKind,
        sliding_window: Option<u32>,
    ) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: block_hashes
                .iter()
                .copied()
                .map(BlockHashValue::Unsigned)
                .collect(),
            parent_block_hash: parent_block_hash.map(BlockHashValue::Unsigned),
            parent_sequence_hash: None,
            parent_sequence_hash_algorithm: None,
            eagle_lookahead_sequence_hash: None,
            eagle_lookahead_sequence_hash_algorithm: None,
            eagle_lookahead_token_ids: None,
            token_ids: token_ids.to_vec(),
            block_size,
            medium: Some("CPU".to_string()),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(group_idx),
            kv_cache_spec_kind: Some(kind),
            kv_cache_spec_sliding_window: sliding_window,
            locality: Some(Locality::Local),
            ownership: None,
        }
    }

    fn sequence_hashes(tokens: &[u32], block_size: u32) -> Vec<u64> {
        TokensWithHashes::new(tokens.to_vec(), block_size)
            .get_or_compute_seq_hashes()
            .to_vec()
    }

    fn resolver_store(
        medium: &str,
        group_idx: u32,
        external_hash: u64,
        token_ids: Vec<u32>,
        block_size: usize,
    ) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(external_hash)],
            parent_block_hash: None,
            parent_sequence_hash: None,
            parent_sequence_hash_algorithm: None,
            eagle_lookahead_sequence_hash: None,
            eagle_lookahead_sequence_hash_algorithm: None,
            eagle_lookahead_token_ids: None,
            token_ids,
            block_size,
            medium: Some(medium.to_string()),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(group_idx),
            kv_cache_spec_kind: Some(KvCacheSpecKind::FullAttention),
            kv_cache_spec_sliding_window: None,
            locality: Some(Locality::Local),
            ownership: None,
        }
    }

    fn resolver_store_many(
        medium: &str,
        group_idx: u32,
        external_hashes: &[u64],
        token_ids: Vec<u32>,
        block_size: usize,
    ) -> RawKvEvent {
        let mut event = resolver_store(medium, group_idx, 0, token_ids, block_size);
        let RawKvEvent::BlockStored { block_hashes, .. } = &mut event else {
            unreachable!("resolver store helper always returns a store");
        };
        *block_hashes = external_hashes
            .iter()
            .copied()
            .map(BlockHashValue::Unsigned)
            .collect();
        event
    }

    fn resolve_with(
        resolver: &mut EvidenceHashResolver,
        event: &RawKvEvent,
    ) -> Result<Option<CacheEvidenceMutation>, EvidenceMutationError> {
        cache_evidence_mutation_with_resolver(
            event,
            TEST_OWNER,
            None,
            &Arc::new(AtomicU32::new(0)),
            resolver,
        )
    }

    #[test]
    fn hash_only_cpu_store_resolves_only_after_same_source_gpu_store() {
        let gpu = resolver_store("GPU", 0, 41, vec![1; 16], 16);
        let cpu = resolver_store("CPU", 0, 41, vec![], 0);
        let mut resolver = EvidenceHashResolver::new(8);

        assert_eq!(
            resolve_with(&mut resolver, &cpu),
            Err(EvidenceMutationError::UnresolvedHash)
        );
        let gpu_mutation = resolve_with(&mut resolver, &gpu).unwrap().unwrap();
        let cpu_mutation = resolve_with(&mut resolver, &cpu).unwrap().unwrap();
        let CacheEvidenceMutation::Store {
            blocks: gpu_blocks, ..
        } = gpu_mutation
        else {
            panic!("GPU mutation must be a store");
        };
        let CacheEvidenceMutation::Store {
            tier,
            blocks: cpu_blocks,
            ..
        } = cpu_mutation
        else {
            panic!("CPU mutation must be a store");
        };
        assert_eq!(tier, CacheTier::Cpu);
        assert_eq!(cpu_blocks, gpu_blocks);

        let other_owner = CacheOwner {
            worker_id: TEST_OWNER.worker_id,
            dp_rank: 1,
        };
        assert_eq!(
            cache_evidence_mutation_with_resolver(
                &cpu,
                other_owner,
                None,
                &Arc::new(AtomicU32::new(0)),
                &mut resolver,
            ),
            Err(EvidenceMutationError::UnresolvedHash)
        );
    }

    #[test]
    fn hash_only_cpu_store_resolves_external_hashes_per_cache_group() {
        let group_zero_gpu = resolver_store("GPU", 0, 41, vec![1; 16], 16);
        let group_one_gpu = resolver_store("GPU", 1, 41, vec![2; 16], 16);
        let group_zero_cpu = resolver_store("CPU", 0, 41, vec![], 0);
        let group_one_cpu = resolver_store("CPU", 1, 41, vec![], 0);
        let mut resolver = EvidenceHashResolver::new(8);

        let group_zero = resolve_with(&mut resolver, &group_zero_gpu)
            .unwrap()
            .unwrap();
        let group_one = resolve_with(&mut resolver, &group_one_gpu)
            .unwrap()
            .unwrap();
        let group_zero_cpu = resolve_with(&mut resolver, &group_zero_cpu)
            .unwrap()
            .unwrap();
        let group_one_cpu = resolve_with(&mut resolver, &group_one_cpu)
            .unwrap()
            .unwrap();

        let blocks = |mutation: CacheEvidenceMutation| match mutation {
            CacheEvidenceMutation::Store { blocks, .. } => blocks,
            _ => panic!("resolver store must remain a store mutation"),
        };
        assert_eq!(blocks(group_zero_cpu), blocks(group_zero));
        assert_eq!(blocks(group_one_cpu), blocks(group_one));
    }

    #[test]
    fn hash_only_cpu_mapping_survives_gpu_eviction() {
        let gpu = resolver_store("GPU", 0, 51, vec![2; 16], 16);
        let cpu = resolver_store("CPU", 0, 51, vec![], 0);
        let remove = RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(51)],
            medium: Some("GPU".to_string()),
            group_idx: Some(0),
            kv_cache_spec_kind: Some(KvCacheSpecKind::FullAttention),
            kv_cache_spec_sliding_window: None,
            locality: Some(Locality::Local),
            ownership: None,
        };
        let mut resolver = EvidenceHashResolver::new(8);

        resolve_with(&mut resolver, &gpu).unwrap();
        resolve_with(&mut resolver, &remove).unwrap();
        assert!(resolve_with(&mut resolver, &cpu).unwrap().is_some());
    }

    #[test]
    fn hash_mapping_collision_is_ambiguous_until_source_restart() {
        let first = resolver_store("GPU", 0, 61, vec![3; 16], 16);
        let conflicting = resolver_store("GPU", 0, 61, vec![4; 16], 16);
        let cpu = resolver_store("CPU", 0, 61, vec![], 0);
        let mut resolver = EvidenceHashResolver::new(8);

        resolve_with(&mut resolver, &first).unwrap();
        assert_eq!(
            resolve_with(&mut resolver, &conflicting),
            Err(EvidenceMutationError::AmbiguousHash)
        );
        assert_eq!(
            resolve_with(&mut resolver, &cpu),
            Err(EvidenceMutationError::AmbiguousHash)
        );

        let mut restarted_source = EvidenceHashResolver::new(8);
        assert_eq!(
            resolve_with(&mut restarted_source, &cpu),
            Err(EvidenceMutationError::UnresolvedHash)
        );
    }

    #[test]
    fn hash_mapping_capacity_fails_closed_without_eviction() {
        let first = resolver_store("GPU", 0, 71, vec![5; 16], 16);
        let second = resolver_store("GPU", 0, 72, vec![6; 16], 16);
        let cpu_first = resolver_store("CPU", 0, 71, vec![], 0);
        let mut resolver = EvidenceHashResolver::new(1);

        resolve_with(&mut resolver, &first).unwrap();
        assert_eq!(
            resolve_with(&mut resolver, &second),
            Err(EvidenceMutationError::MappingOverflow)
        );
        assert!(resolve_with(&mut resolver, &cpu_first).unwrap().is_some());
        assert_eq!(resolver.mappings.len(), 1);
    }

    #[test]
    fn rejected_multi_block_gpu_store_cannot_partially_seed_on_overflow() {
        let gpu = resolver_store_many("GPU", 0, &[91, 92], vec![8; 32], 16);
        let cpu_first = resolver_store("CPU", 0, 91, vec![], 0);
        let mut resolver = EvidenceHashResolver::new(1);

        assert_eq!(
            resolve_with(&mut resolver, &gpu),
            Err(EvidenceMutationError::MappingOverflow)
        );
        assert!(resolver.mappings.is_empty());
        assert_eq!(
            resolve_with(&mut resolver, &cpu_first),
            Err(EvidenceMutationError::UnresolvedHash)
        );
    }

    #[test]
    fn rejected_multi_block_gpu_collision_quarantines_only_conflicting_hash() {
        let seed = resolver_store("GPU", 0, 102, vec![9; 16], 16);
        let conflicting = resolver_store_many("GPU", 0, &[101, 102], vec![10; 32], 16);
        let cpu_new = resolver_store("CPU", 0, 101, vec![], 0);
        let cpu_conflict = resolver_store("CPU", 0, 102, vec![], 0);
        let mut resolver = EvidenceHashResolver::new(8);

        resolve_with(&mut resolver, &seed).unwrap();
        assert_eq!(
            resolve_with(&mut resolver, &conflicting),
            Err(EvidenceMutationError::AmbiguousHash)
        );
        assert_eq!(
            resolve_with(&mut resolver, &cpu_new),
            Err(EvidenceMutationError::UnresolvedHash)
        );
        assert_eq!(
            resolve_with(&mut resolver, &cpu_conflict),
            Err(EvidenceMutationError::AmbiguousHash)
        );
    }

    #[test]
    fn new_ambiguous_tombstone_cannot_exceed_mapping_capacity() {
        let duplicate = resolver_store_many("GPU", 0, &[111, 111], vec![11; 32], 16);
        let mut resolver = EvidenceHashResolver::new(0);

        assert_eq!(
            resolve_with(&mut resolver, &duplicate),
            Err(EvidenceMutationError::MappingOverflow)
        );
        assert!(resolver.mappings.is_empty());
    }

    #[test]
    fn cold_epoch_clear_discards_hash_mapping_generation() {
        let gpu = resolver_store("GPU", 0, 81, vec![7; 16], 16);
        let cpu = resolver_store("CPU", 0, 81, vec![], 0);
        let clear = RawKvEvent::AllBlocksCleared {
            medium: Some("GPU".to_string()),
            ownership: None,
            epoch_id: Some("00000000000000000000000000000001".to_string()),
        };
        let mut resolver = EvidenceHashResolver::new(8);

        resolve_with(&mut resolver, &gpu).unwrap();
        resolve_with(&mut resolver, &clear).unwrap();
        assert_eq!(
            resolve_with(&mut resolver, &cpu),
            Err(EvidenceMutationError::UnresolvedHash)
        );
    }

    #[test]
    fn only_expected_non_main_attention_filters_preserve_publisher_telemetry() {
        assert!(filter_preserves_publisher_telemetry(
            ZmqEventFilterReason::NonMainAttentionKind
        ));

        for reason in [
            ZmqEventFilterReason::IgnoredEvent,
            ZmqEventFilterReason::NonLocalLocality,
            ZmqEventFilterReason::UnknownMedium,
            ZmqEventFilterReason::UnsupportedOwnership,
            ZmqEventFilterReason::UnknownOwnership,
            ZmqEventFilterReason::AmbiguousCacheNamespace,
            ZmqEventFilterReason::UnknownKind,
            ZmqEventFilterReason::NonMainAttentionGroup,
            ZmqEventFilterReason::UnlearnedGroupIdx,
        ] {
            assert!(!filter_preserves_publisher_telemetry(reason), "{reason:?}");
        }
    }

    #[test]
    fn source_sequence_tracker_detects_gap_and_drops_stale_batches() {
        let mut tracker = SourceSequenceTracker::default();

        assert_eq!(
            tracker.observe(2),
            SourceSequenceObservation::Gap { missing: 2 }
        );
        assert_eq!(tracker.observe(3), SourceSequenceObservation::Contiguous);
        assert_eq!(tracker.observe(3), SourceSequenceObservation::Stale);
        assert_eq!(tracker.observe(1), SourceSequenceObservation::Stale);
        assert_eq!(
            tracker.observe(6),
            SourceSequenceObservation::Gap { missing: 2 }
        );
        assert_eq!(tracker.last, Some(6));
    }

    #[test]
    fn interleaved_dp_owners_share_the_global_heartbeat_cursor() {
        let rank_zero = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let rank_one = CacheOwner {
            worker_id: 7,
            dp_rank: 1,
        };
        let mut heartbeat = EvidenceHeartbeatState::new(rank_zero);
        heartbeat.observe(rank_zero, 3);
        heartbeat.observe(rank_one, 4);

        let entries: HashSet<_> = heartbeat.entries().collect();
        assert_eq!(
            entries,
            HashSet::from([(rank_zero, Some(4)), (rank_one, Some(4))])
        );
    }

    #[test]
    fn evidence_preserves_filtered_group_and_defaults_missing_medium_to_gpu() {
        let event = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(42)],
            parent_block_hash: None,
            parent_sequence_hash: None,
            parent_sequence_hash_algorithm: None,
            eagle_lookahead_sequence_hash: None,
            eagle_lookahead_sequence_hash_algorithm: None,
            eagle_lookahead_token_ids: None,
            token_ids: vec![1; 8],
            block_size: 8,
            medium: None,
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(2),
            kv_cache_spec_kind: Some(KvCacheSpecKind::SlidingWindow),
            kv_cache_spec_sliding_window: Some(128),
            locality: Some(Locality::Local),
            ownership: None,
        };

        assert_eq!(
            cache_evidence_mutation(&event, None, &Arc::new(AtomicU32::new(0))),
            Ok(Some(CacheEvidenceMutation::Store {
                tier: CacheTier::Gpu,
                group_idx: Some(2),
                parent_external_hash: None,
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 42,
                    tokens_hash: compute_block_hash_for_seq(
                        &[1; 8],
                        8,
                        BlockHashOptions::default(),
                    )[0]
                    .0,
                }],
            }))
        );
    }

    #[test]
    fn evidence_accepts_only_complete_known_parent_attestation_pair() {
        let mut event = constituent_store(
            4,
            8,
            &[42],
            Some(41),
            &[1; 8],
            KvCacheSpecKind::SlidingWindow,
            Some(128),
        );
        if let RawKvEvent::BlockStored {
            parent_sequence_hash,
            parent_sequence_hash_algorithm,
            ..
        } = &mut event
        {
            *parent_sequence_hash = Some(BlockHashValue::Unsigned(u64::MAX));
            *parent_sequence_hash_algorithm = Some(PARENT_SEQUENCE_HASH_ALGORITHM.to_string());
        }
        assert!(matches!(
            cache_evidence_mutation(&event, None, &Arc::new(AtomicU32::new(0))),
            Ok(Some(CacheEvidenceMutation::StoreWithParentAttestation {
                group_idx: 4,
                parent_external_hash: 41,
                parent_sequence_hash: u64::MAX,
                ..
            }))
        ));

        if let RawKvEvent::BlockStored {
            parent_sequence_hash_algorithm,
            ..
        } = &mut event
        {
            *parent_sequence_hash_algorithm = Some("unknown".to_string());
        }
        assert_eq!(
            cache_evidence_mutation(&event, None, &Arc::new(AtomicU32::new(0))),
            Err(EvidenceMutationError::Invalid)
        );
        if let RawKvEvent::BlockStored {
            parent_sequence_hash_algorithm,
            ..
        } = &mut event
        {
            *parent_sequence_hash_algorithm = None;
        }
        assert_eq!(
            cache_evidence_mutation(&event, None, &Arc::new(AtomicU32::new(0))),
            Err(EvidenceMutationError::Invalid)
        );
    }

    #[test]
    fn producer_golden_vectors_preserve_attested_child_sequences() {
        let cases = [
            (
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [101],
                    "parent_block_hash": 100,
                    "parent_sequence_hash": 14643705804678351452_u64,
                    "parent_sequence_hash_algorithm": PARENT_SEQUENCE_HASH_ALGORITHM,
                    "token_ids": [5, 6, 7, 8],
                    "block_size": 4,
                    "medium": "CPU",
                    "group_idx": 1,
                    "kv_cache_spec_kind": "full_attention"
                }),
                1,
                vec![4945711292740353085_u64],
            ),
            (
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [201],
                    "parent_block_hash": 200,
                    "parent_sequence_hash": 9967753091837871098_u64,
                    "parent_sequence_hash_algorithm": PARENT_SEQUENCE_HASH_ALGORITHM,
                    "token_ids": [5, 6, 7, 8],
                    "block_size": 4,
                    "medium": "CPU",
                    "lora_name": "adapter-a",
                    "cache_salt": "tenant-a",
                    "group_idx": 4,
                    "kv_cache_spec_kind": "full_attention"
                }),
                4,
                vec![8253787442643436499_u64],
            ),
            (
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [301],
                    "parent_block_hash": 300,
                    "parent_sequence_hash": 1914873892905420881_u64,
                    "parent_sequence_hash_algorithm": PARENT_SEQUENCE_HASH_ALGORITHM,
                    "token_ids": [[5, 6], [6, 7], [7, 8], [8, 9]],
                    "block_size": 4,
                    "medium": "CPU",
                    "group_idx": 3,
                    "kv_cache_spec_kind": "full_attention"
                }),
                3,
                vec![13136491855314414045_u64],
            ),
            (
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [401, 402, 403],
                    "parent_block_hash": 400,
                    "parent_sequence_hash": 1921452330601040443_u64,
                    "parent_sequence_hash_algorithm": PARENT_SEQUENCE_HASH_ALGORITHM,
                    "token_ids": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                    "block_size": 4,
                    "medium": "CPU",
                    "group_idx": 1,
                    "kv_cache_spec_kind": "full_attention"
                }),
                1,
                vec![
                    16910271745392797383_u64,
                    1149017327988146166_u64,
                    4629705925025678689_u64,
                ],
            ),
        ];

        for (wire, group, expected_children) in cases {
            let encoded = rmp_serde::to_vec_named(&wire).unwrap();
            let event: RawKvEvent = rmp_serde::from_slice(&encoded).unwrap();
            let mutation = cache_evidence_mutation(&event, None, &Arc::new(AtomicU32::new(0)))
                .unwrap()
                .unwrap();
            let mut ledger = CacheEvidenceLedger::new(32);
            ledger.record_group_catalog(TEST_OWNER, CacheTier::Cpu, [group]);
            ledger.seal_physical_scope();
            let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
                owner: TEST_OWNER,
                source_cursor: 1,
                source_incarnation_id: None,
                heartbeat: false,
                watermark_source_cursor: None,
                mutations: vec![mutation],
                barrier_id: None,
                epoch_id: None,
                telemetry_complete: true,
            });

            assert!(result.is_complete());
            for child in expected_children {
                assert_eq!(ledger.resident_on(child, TEST_OWNER), KnownBool::Yes);
            }
            let RawKvEvent::BlockStored {
                parent_sequence_hash: Some(parent),
                ..
            } = event
            else {
                unreachable!();
            };
            assert_eq!(
                ledger.resident_on(parent.into_u64(), TEST_OWNER),
                KnownBool::No
            );
        }
    }

    #[test]
    fn live_n513_sparse_eagle_wire_preserves_gpu_cpu_mapping() {
        use base64::Engine as _;
        use sha2::{Digest, Sha256};

        let encoded = base64::engine::general_purpose::STANDARD
            .decode(include_str!("fixtures/live_n513_gpu_cpu.msgpack.b64").trim())
            .unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&encoded)),
            "8b39fa9e476d3f3d5089a0bc23f52215fa3cd4c7de848cce6d7d0ff602b5f9a8"
        );
        let batch: KvEventBatch = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(batch.events.len(), 3);
        assert!(matches!(
            &batch.events[0],
            RawKvEvent::BlockStored { block_hashes, group_idx: Some(2), medium: Some(medium), .. }
                if block_hashes.len() == 3 && medium == "GPU"
        ));
        assert!(matches!(
            &batch.events[1],
            RawKvEvent::BlockStored { block_hashes, group_idx: Some(2), medium: Some(medium), .. }
                if block_hashes.len() == 2 && medium == "GPU"
        ));

        let warning_count = Arc::new(AtomicU32::new(0));
        let mut resolver = EvidenceHashResolver::new(MAX_EVIDENCE_HASH_MAPPINGS);
        let mutations: Vec<_> = batch
            .events
            .iter()
            .map(|event| {
                cache_evidence_mutation_with_resolver(
                    event,
                    TEST_OWNER,
                    None,
                    &warning_count,
                    &mut resolver,
                )
                .unwrap()
                .unwrap()
            })
            .collect();
        let (external_hash, local_hash, parent_sequence_hash) = match &mutations[2] {
            CacheEvidenceMutation::StoreWithParentAttestation {
                blocks,
                parent_sequence_hash,
                ..
            } => {
                assert_eq!(blocks.len(), 1);
                (
                    blocks[0].external_hash,
                    blocks[0].tokens_hash,
                    *parent_sequence_hash,
                )
            }
            mutation => panic!("unexpected CPU mutation: {mutation:?}"),
        };
        let gpu_match = match &mutations[1] {
            CacheEvidenceMutation::StoreWithParentAttestation { blocks, .. } => blocks
                .iter()
                .find(|block| block.external_hash == external_hash)
                .copied(),
            mutation => panic!("unexpected GPU mutation: {mutation:?}"),
        };
        assert_eq!(gpu_match.unwrap().tokens_hash, local_hash);
        let canonical_hash =
            compute_next_seq_hash(parent_sequence_hash, LocalBlockHash(local_hash));

        let mut ledger = CacheEvidenceLedger::new(64);
        ledger.record_group_catalog(TEST_OWNER, CacheTier::Gpu, [2]);
        ledger.record_group_catalog(TEST_OWNER, CacheTier::Cpu, [2]);
        ledger.seal_physical_scope();
        let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
            owner: TEST_OWNER,
            source_cursor: 1,
            source_incarnation_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            mutations,
            barrier_id: None,
            epoch_id: None,
            telemetry_complete: true,
        });
        assert!(result.is_complete());
        assert_eq!(
            ledger.resident_on(canonical_hash, TEST_OWNER),
            KnownBool::Yes
        );

        for (source_cursor, medium, expected) in
            [(2, "GPU", KnownBool::Yes), (3, "CPU", KnownBool::No)]
        {
            let remove = RawKvEvent::BlockRemoved {
                block_hashes: vec![BlockHashValue::Unsigned(external_hash)],
                medium: Some(medium.to_string()),
                group_idx: Some(2),
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
            };
            let mutation = cache_evidence_mutation_with_resolver(
                &remove,
                TEST_OWNER,
                None,
                &warning_count,
                &mut resolver,
            )
            .unwrap()
            .unwrap();
            let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
                owner: TEST_OWNER,
                source_cursor,
                source_incarnation_id: None,
                heartbeat: false,
                watermark_source_cursor: None,
                mutations: vec![mutation],
                barrier_id: None,
                epoch_id: None,
                telemetry_complete: true,
            });
            assert!(result.is_complete());
            assert_eq!(ledger.resident_on(canonical_hash, TEST_OWNER), expected);
        }
    }

    #[test]
    fn terminal_eagle_wire_matches_native_hybrid_prefix() {
        use base64::Engine as _;
        use sha2::{Digest, Sha256};

        let fixtures = [
            (
                575usize,
                include_str!("fixtures/terminal_eagle_575.msgpack.b64"),
                "d2864f80d685d8962a7e4ff68614693839f811b9851157cd4c1e8fbf24e41a40",
                KnownPrefixLength::Known(0),
                [2usize, 2, 2, 2, 16],
            ),
            (
                576usize,
                include_str!("fixtures/terminal_eagle_576.msgpack.b64"),
                "1b2f5947ae3976f7a61a1fc19ade43308559c02660821b003a2703ff3c8e45dd",
                KnownPrefixLength::Known(0),
                [2usize, 3, 3, 2, 16],
            ),
            (
                577usize,
                include_str!("fixtures/terminal_eagle_577.msgpack.b64"),
                "dc6cd99a4a13d6a14c1c66c6f04b82a95f3686eb8dae45fea76ec3d97e210dfc",
                KnownPrefixLength::Known(512),
                [2usize, 3, 3, 2, 16],
            ),
        ];

        for (num_tokens, encoded, expected_digest, expected_prefix, expected_blocks) in fixtures {
            let wire = base64::engine::general_purpose::STANDARD
                .decode(encoded.trim())
                .unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(&wire)), expected_digest);
            let batch: KvEventBatch = rmp_serde::from_slice(&wire).unwrap();

            let mut blocks_by_group = [0usize; 5];
            for event in &batch.events {
                let RawKvEvent::BlockStored {
                    block_hashes,
                    group_idx: Some(group_idx),
                    ..
                } = event
                else {
                    panic!("unexpected terminal EAGLE fixture event: {event:?}");
                };
                blocks_by_group[*group_idx as usize] += block_hashes.len();
            }
            assert_eq!(blocks_by_group, expected_blocks);
            let lookahead_events: Vec<_> = batch
                .events
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        RawKvEvent::BlockStored {
                            eagle_lookahead_token_ids: Some(_),
                            ..
                        }
                    )
                })
                .collect();
            assert_eq!(lookahead_events.len(), usize::from(num_tokens == 576));
            if num_tokens == 576 {
                let RawKvEvent::BlockStored {
                    token_ids,
                    is_eagle,
                    eagle_lookahead_sequence_hash: Some(_),
                    eagle_lookahead_sequence_hash_algorithm: Some(algorithm),
                    eagle_lookahead_token_ids: Some(lookahead_tokens),
                    ..
                } = lookahead_events[0]
                else {
                    panic!("missing exact terminal EAGLE lookahead event");
                };
                assert!(token_ids.is_empty());
                assert_eq!(*is_eagle, Some(false));
                assert_eq!(algorithm, PARENT_SEQUENCE_HASH_ALGORITHM);
                assert_eq!(lookahead_tokens.len(), 64);

                let mut legacy_view = (*lookahead_events[0]).clone();
                let RawKvEvent::BlockStored {
                    eagle_lookahead_sequence_hash,
                    eagle_lookahead_sequence_hash_algorithm,
                    eagle_lookahead_token_ids,
                    ..
                } = &mut legacy_view
                else {
                    unreachable!();
                };
                *eagle_lookahead_sequence_hash = None;
                *eagle_lookahead_sequence_hash_algorithm = None;
                *eagle_lookahead_token_ids = None;
                let mut legacy_resolver = EvidenceHashResolver::new(MAX_EVIDENCE_HASH_MAPPINGS);
                let legacy_warning_count = Arc::new(AtomicU32::new(0));
                assert_eq!(
                    cache_evidence_mutation_with_resolver(
                        &legacy_view,
                        TEST_OWNER,
                        None,
                        &legacy_warning_count,
                        &mut legacy_resolver,
                    ),
                    Err(EvidenceMutationError::Invalid)
                );
            }

            let warning_count = Arc::new(AtomicU32::new(0));
            let mut resolver = EvidenceHashResolver::new(MAX_EVIDENCE_HASH_MAPPINGS);
            let mutations = batch
                .events
                .iter()
                .map(|event| {
                    cache_evidence_mutation_with_resolver(
                        event,
                        TEST_OWNER,
                        None,
                        &warning_count,
                        &mut resolver,
                    )
                    .unwrap()
                    .unwrap()
                })
                .collect();
            let mut ledger = CacheEvidenceLedger::new(4096);
            ledger.record_group_catalog(TEST_OWNER, CacheTier::Gpu, 0..5);
            ledger.seal_physical_scope();
            let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
                owner: TEST_OWNER,
                source_cursor: 1,
                source_incarnation_id: None,
                heartbeat: false,
                watermark_source_cursor: None,
                mutations,
                barrier_id: None,
                epoch_id: None,
                telemetry_complete: true,
            });
            assert!(result.is_complete());

            let tokens: Vec<u32> = (0..num_tokens)
                .map(|index| 100 + (index % 2) as u32)
                .collect();
            let specs = [
                (0, CacheGroupKind::FullAttention, 256, None, false),
                (1, CacheGroupKind::SlidingWindow, 64, Some(128), false),
                (2, CacheGroupKind::SlidingWindow, 64, Some(128), true),
                (3, CacheGroupKind::SlidingWindow, 4, Some(8), false),
                (4, CacheGroupKind::SlidingWindow, 8, Some(128), false),
            ];
            let groups: Vec<_> = specs
                .into_iter()
                .map(|(group_idx, kind, block_size, sliding_window, is_eagle)| {
                    let sequence_hashes = TokensWithHashes::new(tokens.clone(), block_size)
                        .with_lora_name("terminal-eagle-lora".to_string())
                        .with_cache_namespace("terminal-eagle-salt".to_string())
                        .with_is_eagle(is_eagle)
                        .get_or_compute_seq_hashes()
                        .to_vec();
                    let eagle_lookahead_block_hashes = if is_eagle {
                        TokensWithHashes::new(tokens.clone(), block_size)
                            .with_lora_name("terminal-eagle-lora".to_string())
                            .with_cache_namespace("terminal-eagle-salt".to_string())
                            .get_or_compute_block_hashes()
                            .iter()
                            .map(|hash| hash.0)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    CacheGroupHashSequence {
                        group_idx,
                        kind,
                        block_size,
                        sliding_window,
                        is_eagle,
                        alignment_tokens: 256,
                        sequence_hashes,
                        eagle_lookahead_block_hashes,
                    }
                })
                .collect();
            assert_eq!(
                ledger.resident_prefix_on(&groups, num_tokens as u64, TEST_OWNER),
                expected_prefix
            );
        }
    }

    #[test]
    fn terminal_eagle_n576_wire_matches_cross_request_n577_native_hit() {
        use base64::Engine as _;

        let wire = base64::engine::general_purpose::STANDARD
            .decode(include_str!("fixtures/terminal_eagle_576.msgpack.b64").trim())
            .unwrap();
        let batch: KvEventBatch = rmp_serde::from_slice(&wire).unwrap();
        let warning_count = Arc::new(AtomicU32::new(0));
        let mut resolver = EvidenceHashResolver::new(MAX_EVIDENCE_HASH_MAPPINGS);
        let mutations = batch
            .events
            .iter()
            .map(|event| {
                cache_evidence_mutation_with_resolver(
                    event,
                    TEST_OWNER,
                    None,
                    &warning_count,
                    &mut resolver,
                )
                .unwrap()
                .unwrap()
            })
            .collect();
        let mut ledger = CacheEvidenceLedger::new(4096);
        ledger.record_group_catalog(TEST_OWNER, CacheTier::Gpu, 0..5);
        ledger.seal_physical_scope();
        assert!(
            ledger
                .apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
                    owner: TEST_OWNER,
                    source_cursor: 1,
                    source_incarnation_id: None,
                    heartbeat: false,
                    watermark_source_cursor: None,
                    mutations,
                    barrier_id: None,
                    epoch_id: None,
                    telemetry_complete: true,
                })
                .is_complete()
        );

        let tokens: Vec<u32> = (0..577).map(|index| 100 + (index % 2) as u32).collect();
        let specs = [
            (0, CacheGroupKind::FullAttention, 256, None, false),
            (1, CacheGroupKind::SlidingWindow, 64, Some(128), false),
            (2, CacheGroupKind::SlidingWindow, 64, Some(128), true),
            (3, CacheGroupKind::SlidingWindow, 4, Some(8), false),
            (4, CacheGroupKind::SlidingWindow, 8, Some(128), false),
        ];
        let groups: Vec<_> = specs
            .into_iter()
            .map(|(group_idx, kind, block_size, sliding_window, is_eagle)| {
                let sequence_hashes = TokensWithHashes::new(tokens.clone(), block_size)
                    .with_lora_name("terminal-eagle-lora".to_string())
                    .with_cache_namespace("terminal-eagle-salt".to_string())
                    .with_is_eagle(is_eagle)
                    .get_or_compute_seq_hashes()
                    .to_vec();
                let eagle_lookahead_block_hashes = if is_eagle {
                    TokensWithHashes::new(tokens.clone(), block_size)
                        .with_lora_name("terminal-eagle-lora".to_string())
                        .with_cache_namespace("terminal-eagle-salt".to_string())
                        .get_or_compute_block_hashes()
                        .iter()
                        .map(|hash| hash.0)
                        .collect()
                } else {
                    Vec::new()
                };
                CacheGroupHashSequence {
                    group_idx,
                    kind,
                    block_size,
                    sliding_window,
                    is_eagle,
                    alignment_tokens: 256,
                    sequence_hashes,
                    eagle_lookahead_block_hashes,
                }
            })
            .collect();

        // Exact vLLM returns 512 for this cross-request transition. The marker
        // qualifies the preceding canonical EAGLE prefix without advertising
        // the nonexistent B+1 canonical terminal unit.
        assert_eq!(
            ledger.resident_prefix_on(&groups, 577, TEST_OWNER),
            KnownPrefixLength::Known(512)
        );
    }

    #[test]
    fn constituent_events_preserve_hybrid_prefix_and_duplicate_refcounts() {
        const HBF: usize = 3;
        const FULL_BLOCK_SIZE: usize = 256;
        const SWA_BLOCK_SIZE: usize = 64;
        const PROMPT_TOKENS: usize = HBF * FULL_BLOCK_SIZE;

        let original_tokens: Vec<u32> = (0..PROMPT_TOKENS as u32).collect();
        let warning_count = Arc::new(AtomicU32::new(0));
        let full_event = constituent_store(
            0,
            FULL_BLOCK_SIZE,
            &[101, 102, 103],
            None,
            &original_tokens,
            KvCacheSpecKind::FullAttention,
            None,
        );
        let mut store_mutations = vec![
            cache_evidence_mutation(&full_event, None, &warning_count)
                .unwrap()
                .unwrap(),
        ];

        let mut swa_external_hashes = Vec::new();
        for chunk_idx in 0..(PROMPT_TOKENS / (HBF * SWA_BLOCK_SIZE)) {
            let first_hash = 201 + chunk_idx as u64 * HBF as u64;
            let hashes: Vec<_> = (first_hash..first_hash + HBF as u64).collect();
            let token_start = chunk_idx * HBF * SWA_BLOCK_SIZE;
            let token_end = token_start + HBF * SWA_BLOCK_SIZE;
            let event = constituent_store(
                1,
                SWA_BLOCK_SIZE,
                &hashes,
                swa_external_hashes.last().copied(),
                &original_tokens[token_start..token_end],
                KvCacheSpecKind::SlidingWindow,
                Some(128),
            );
            store_mutations.push(
                cache_evidence_mutation(&event, None, &warning_count)
                    .unwrap()
                    .unwrap(),
            );
            swa_external_hashes.extend(hashes);
        }

        let CacheEvidenceMutation::Store { blocks, .. } = &store_mutations[0] else {
            panic!("constituent store must remain a store mutation");
        };
        assert_eq!(blocks.len(), HBF);

        let mut ledger = CacheEvidenceLedger::new(64);
        ledger.record_group_catalog(TEST_OWNER, CacheTier::Cpu, [0, 1]);
        ledger.seal_physical_scope();
        let stores = CacheEvidenceBatch {
            owner: TEST_OWNER,
            source_cursor: 1,
            source_incarnation_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: store_mutations.clone(),
            barrier_id: None,
            epoch_id: None,
        };
        assert!(ledger.apply_evidence_batch(&stores));
        assert!(ledger.apply_evidence_batch(&CacheEvidenceBatch {
            source_cursor: 2,
            ..stores.clone()
        }));
        assert_eq!(ledger.stats().cpu_physical_blocks, 15);

        let mut changed_tail = original_tokens.clone();
        for token in &mut changed_tail[2 * FULL_BLOCK_SIZE..] {
            *token += 10_000;
        }
        let original_full_hashes = sequence_hashes(&original_tokens, FULL_BLOCK_SIZE as u32);
        let changed_full_hashes = sequence_hashes(&changed_tail, FULL_BLOCK_SIZE as u32);
        assert_eq!(original_full_hashes[..2], changed_full_hashes[..2]);
        assert_ne!(original_full_hashes[2], changed_full_hashes[2]);
        let full_group = CacheGroupHashSequence {
            group_idx: 0,
            kind: CacheGroupKind::FullAttention,
            block_size: FULL_BLOCK_SIZE as u32,
            sliding_window: None,
            is_eagle: false,
            alignment_tokens: FULL_BLOCK_SIZE as u32,
            sequence_hashes: changed_full_hashes,
            eagle_lookahead_block_hashes: Vec::new(),
        };
        let original_swa_hashes = sequence_hashes(&original_tokens, SWA_BLOCK_SIZE as u32);
        let changed_swa_hashes = sequence_hashes(&changed_tail, SWA_BLOCK_SIZE as u32);
        assert_eq!(original_swa_hashes[..8], changed_swa_hashes[..8]);
        assert_ne!(original_swa_hashes[8], changed_swa_hashes[8]);
        let swa_group = CacheGroupHashSequence {
            group_idx: 1,
            kind: CacheGroupKind::SlidingWindow,
            block_size: SWA_BLOCK_SIZE as u32,
            sliding_window: Some(128),
            is_eagle: false,
            alignment_tokens: FULL_BLOCK_SIZE as u32,
            sequence_hashes: changed_swa_hashes,
            eagle_lookahead_block_hashes: Vec::new(),
        };

        assert_eq!(
            ledger.resident_prefix_on(
                std::slice::from_ref(&full_group),
                PROMPT_TOKENS as u64,
                TEST_OWNER,
            ),
            KnownPrefixLength::Known(512)
        );
        assert_eq!(
            ledger.resident_prefix_on(
                std::slice::from_ref(&swa_group),
                PROMPT_TOKENS as u64,
                TEST_OWNER,
            ),
            KnownPrefixLength::Known(512)
        );
        assert_eq!(
            ledger.resident_prefix_on(
                &[full_group.clone(), swa_group.clone()],
                PROMPT_TOKENS as u64,
                TEST_OWNER,
            ),
            KnownPrefixLength::Known(512)
        );

        let mut remove_mutations = vec![CacheEvidenceMutation::Remove {
            tier: CacheTier::Cpu,
            group_idx: Some(0),
            block_hashes: vec![101, 102, 103],
        }];
        remove_mutations.extend(swa_external_hashes.chunks(HBF).map(|hashes| {
            CacheEvidenceMutation::Remove {
                tier: CacheTier::Cpu,
                group_idx: Some(1),
                block_hashes: hashes.to_vec(),
            }
        }));
        let removes = CacheEvidenceBatch {
            source_cursor: 3,
            mutations: remove_mutations,
            ..stores
        };
        assert!(ledger.apply_evidence_batch(&removes));
        assert_eq!(ledger.stats().cpu_physical_blocks, 15);
        assert_eq!(
            ledger.resident_prefix_on(
                std::slice::from_ref(&full_group),
                PROMPT_TOKENS as u64,
                TEST_OWNER,
            ),
            KnownPrefixLength::Known(512)
        );

        assert!(ledger.apply_evidence_batch(&CacheEvidenceBatch {
            source_cursor: 4,
            ..removes
        }));
        assert_eq!(ledger.stats().cpu_physical_blocks, 0);
        assert_eq!(
            ledger.resident_prefix_on(&[full_group, swa_group], PROMPT_TOKENS as u64, TEST_OWNER,),
            KnownPrefixLength::Known(0)
        );
    }

    #[test]
    fn evidence_marks_nonlocal_and_unknown_media_incomplete() {
        let mut event = RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(42)],
            medium: Some("CPU".to_string()),
            group_idx: Some(0),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: Some(Locality::Remote),
            ownership: None,
        };
        let warning_count = Arc::new(AtomicU32::new(0));
        assert_eq!(
            cache_evidence_mutation(&event, None, &warning_count),
            Err(EvidenceMutationError::Invalid)
        );
        if let RawKvEvent::BlockRemoved {
            medium, locality, ..
        } = &mut event
        {
            *locality = Some(Locality::Local);
            *medium = Some("FUTURE_TIER".to_string());
        }
        assert_eq!(
            cache_evidence_mutation(&event, None, &warning_count),
            Err(EvidenceMutationError::Invalid)
        );
    }
}
