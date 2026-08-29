// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
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
                let mutations = batch
                    .events
                    .iter()
                    .filter_map(|event| match cache_evidence_mutation(
                        event,
                        image_token_id,
                        &evidence_warning_count,
                    ) {
                        Ok(mutation) => mutation,
                        Err(()) => {
                            telemetry_complete = false;
                            None
                        }
                    })
                    .collect();
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

fn cache_evidence_mutation(
    event: &RawKvEvent,
    image_token_id: Option<u32>,
    warning_count: &Arc<AtomicU32>,
) -> Result<Option<CacheEvidenceMutation>, ()> {
    if !matches!(event.ownership(), Ok(KvEventOwnership::Framework)) {
        return Err(());
    }
    if matches!(event.locality(), Some(Locality::Remote | Locality::Unknown)) {
        return Err(());
    }

    let tier = |medium: Option<&str>| match medium {
        None => Ok(CacheTier::Gpu),
        Some(medium) => match StorageTier::from_kv_medium(medium) {
            Some(StorageTier::Device) => Ok(CacheTier::Gpu),
            Some(StorageTier::HostPinned) => Ok(CacheTier::Cpu),
            Some(StorageTier::Disk | StorageTier::External) => Err(()),
            None => Err(()),
        },
    };

    match event {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
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
            let external_hashes: Vec<_> = block_hashes
                .iter()
                .copied()
                .map(BlockHashValue::into_u64)
                .collect();
            let num_block_tokens = vec![*block_size as u64; external_hashes.len()];
            let stored = create_stored_blocks(
                (*block_size).try_into().map_err(|_| ())?,
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
                return Err(());
            }
            Ok(Some(CacheEvidenceMutation::Store {
                tier: tier(medium.as_deref())?,
                group_idx: *group_idx,
                parent_external_hash: parent_block_hash.map(BlockHashValue::into_u64),
                blocks: stored
                    .into_iter()
                    .map(|block| CacheEvidenceStoredBlock {
                        external_hash: block.block_hash.0,
                        tokens_hash: block.tokens_hash.0,
                    })
                    .collect(),
            }))
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
        RawKvEvent::AllBlocksCleared { medium, .. } => {
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
        CacheEvidenceLedger, CacheGroupHashSequence, CacheGroupKind, KnownPrefixLength,
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
            Err(())
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
            Err(())
        );
    }
}
