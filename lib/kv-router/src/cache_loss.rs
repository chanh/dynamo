// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-level attribution for KV-cache reuse and loss.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::protocols::{LocalBlockHash, compute_next_seq_hash};

pub const KV_CACHE_EVIDENCE_SUBJECT: &str = "kv-cache-evidence-events";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct CacheOwner {
    pub worker_id: u64,
    pub dp_rank: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTier {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnownBool {
    Yes,
    No,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheEvidenceApplyIntegrityFailure {
    IncompleteBatch,
    MissingGroup,
    MissingParent,
    ConflictingMapping,
}

impl CacheEvidenceApplyIntegrityFailure {
    pub const fn metric_label(self) -> &'static str {
        match self {
            Self::IncompleteBatch => "incomplete_batch",
            Self::MissingGroup => "missing_group",
            Self::MissingParent => "missing_parent",
            Self::ConflictingMapping => "conflicting_mapping",
        }
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
pub struct CacheEvidenceApplyResult {
    failures: HashSet<CacheEvidenceApplyIntegrityFailure>,
}

impl CacheEvidenceApplyResult {
    pub fn is_complete(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn failures(&self) -> impl Iterator<Item = CacheEvidenceApplyIntegrityFailure> + '_ {
        self.failures.iter().copied()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum CacheEvidenceMutation {
    Store {
        tier: CacheTier,
        group_idx: Option<u32>,
        parent_external_hash: Option<u64>,
        blocks: Vec<CacheEvidenceStoredBlock>,
    },
    Remove {
        tier: CacheTier,
        group_idx: Option<u32>,
        block_hashes: Vec<u64>,
    },
    Clear {
        /// Missing on legacy publishers and therefore conservatively clears all tiers.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tier: Option<CacheTier>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CacheEvidenceStoredBlock {
    /// Engine sequence hash used by later removal events.
    pub external_hash: u64,
    /// Token-derived hash computed with this cache group's real block size.
    pub tokens_hash: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheGroupBlockQuery {
    pub group_idx: u32,
    pub tokens_hash: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheGroupKind {
    FullAttention,
    SlidingWindow,
}

/// Request hashes for one physical KV-cache group. Hashes are rolling
/// sequence hashes at this group's actual block size, not raw token-block
/// hashes, so identical chunks under different parents remain distinct.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheGroupHashSequence {
    pub group_idx: u32,
    pub kind: CacheGroupKind,
    pub block_size: u32,
    pub sliding_window: Option<u32>,
    pub is_eagle: bool,
    pub alignment_tokens: u32,
    pub sequence_hashes: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnownPrefixLength {
    Known(u64),
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheEvidenceStats {
    pub history_blocks: usize,
    pub history_saturated: bool,
    pub history_complete: bool,
    pub gpu_physical_blocks: usize,
    pub cpu_physical_blocks: usize,
    pub expected_owners: usize,
    pub cataloged_owners: usize,
    pub physical_scope_complete: bool,
    pub physical_telemetry_complete: bool,
}

/// One engine publisher batch. Empty mutation batches are retained so the
/// frontend can detect a missing source cursor even when a batch contained no
/// routing-indexable cache event.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CacheEvidenceBatch {
    pub owner: CacheOwner,
    pub source_cursor: u64,
    /// Incarnation advertised by the source membership record. Missing on
    /// legacy publishers, whose evidence cannot be freshness-fenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_incarnation_id: Option<u64>,
    /// An ordered heartbeat proves that every source batch through this cursor
    /// was admitted to the evidence queue before this batch.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub heartbeat: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watermark_source_cursor: Option<u64>,
    /// False when one or more events in this source batch could not be
    /// represented as local GPU/CPU residency evidence.
    pub telemetry_complete: bool,
    pub mutations: Vec<CacheEvidenceMutation>,
    /// Ordered route-time cut emitted by a barrier-capable engine. Missing on
    /// legacy batches and ordinary mutations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barrier_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PhysicalCopy {
    owner: CacheOwner,
    tier: CacheTier,
    group: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StoreMutationResult {
    Applied,
    MissingParent,
    Invalid,
    RejectedMapping,
}

/// Bounded evidence ledger used to build the first two funnel gates.
///
/// History never evicts silently. Once its capacity is exhausted, absent
/// hashes become unknown while hashes already retained remain positive facts.
/// Physical copies are reference-counted because chunked offload events can
/// announce the same logical block more than once.
#[derive(Debug)]
pub struct CacheEvidenceLedger {
    history_capacity: usize,
    seen_blocks: HashSet<u64>,
    history_saturated: bool,
    history_complete: bool,
    history_epoch: u64,
    copies: HashMap<(u64, PhysicalCopy), u32>,
    external_copies: HashMap<(u64, PhysicalCopy), (u64, u32)>,
    required_groups: HashMap<(CacheOwner, CacheTier), BTreeSet<u32>>,
    expected_owners: Option<HashSet<CacheOwner>>,
    physical_scope_complete: bool,
    physical_telemetry_complete: bool,
}

impl CacheEvidenceLedger {
    pub fn new(history_capacity: usize) -> Self {
        assert!(history_capacity > 0, "history capacity must be positive");
        Self {
            history_capacity,
            seen_blocks: HashSet::with_capacity(history_capacity),
            history_saturated: false,
            // A new frontend can reconstruct current residency, but not blocks
            // that warm workers saw and evicted before it subscribed. Absence
            // therefore remains unknown until durable history or a proven
            // all-owner cold epoch is implemented.
            history_complete: false,
            history_epoch: 0,
            copies: HashMap::new(),
            external_copies: HashMap::new(),
            required_groups: HashMap::new(),
            expected_owners: None,
            physical_scope_complete: false,
            physical_telemetry_complete: true,
        }
    }

    pub fn record_seen_blocks(&mut self, hashes: impl IntoIterator<Item = u64>) {
        for hash in hashes {
            if self.seen_blocks.contains(&hash) {
                continue;
            }
            if self.seen_blocks.len() == self.history_capacity {
                self.history_saturated = true;
                continue;
            }
            self.seen_blocks.insert(hash);
        }
    }

    pub fn record_seen_blocks_for_epoch(
        &mut self,
        epoch: u64,
        hashes: impl IntoIterator<Item = u64>,
    ) -> bool {
        if epoch != self.history_epoch {
            return false;
        }
        self.record_seen_blocks(hashes);
        true
    }

    pub fn history_epoch(&self) -> u64 {
        self.history_epoch
    }

    pub fn advance_history_epoch(&mut self) -> u64 {
        self.history_epoch = self.history_epoch.wrapping_add(1).max(1);
        self.seen_blocks.clear();
        self.history_saturated = false;
        self.history_complete = true;
        self.physical_telemetry_complete = true;
        self.history_epoch
    }

    pub fn commit_cold_history_epoch(&mut self, owners: &HashSet<CacheOwner>) -> Option<u64> {
        if !self.expected_owners_match(owners)
            || !self.physical_scope_complete
            || self
                .copies
                .keys()
                .any(|(_, copy)| owners.contains(&copy.owner))
            || self
                .external_copies
                .keys()
                .any(|(_, copy)| owners.contains(&copy.owner))
        {
            self.mark_history_incomplete();
            return None;
        }
        Some(self.advance_history_epoch())
    }

    pub fn reusable(&self, hash: u64) -> KnownBool {
        if self.seen_blocks.contains(&hash) {
            KnownBool::Yes
        } else if self.history_saturated || !self.history_complete {
            KnownBool::Unknown
        } else {
            KnownBool::No
        }
    }

    pub fn mark_history_incomplete(&mut self) {
        self.history_complete = false;
    }

    pub fn mark_history_incomplete_for_epoch(&mut self, epoch: u64) -> bool {
        if epoch != self.history_epoch {
            return false;
        }
        self.history_complete = false;
        true
    }

    pub fn record_group_catalog(
        &mut self,
        owner: CacheOwner,
        tier: CacheTier,
        groups: impl IntoIterator<Item = u32>,
    ) {
        self.required_groups
            .insert((owner, tier), groups.into_iter().collect());
        self.refresh_physical_scope();
    }

    pub fn set_expected_owners(&mut self, owners: impl IntoIterator<Item = CacheOwner>) {
        let owners: HashSet<_> = owners.into_iter().collect();
        let retired: Vec<_> = self
            .required_groups
            .keys()
            .map(|(owner, _)| *owner)
            .filter(|owner| !owners.contains(owner))
            .collect();
        for owner in retired {
            self.retire_owner(owner);
        }
        self.expected_owners = Some(owners);
        self.refresh_physical_scope();
    }

    pub fn expected_owners_match(&self, owners: &HashSet<CacheOwner>) -> bool {
        self.expected_owners.as_ref() == Some(owners)
    }

    /// Declare that every cache owner/tier in the measured routing domain has
    /// supplied its group catalog. Before this point, negative residency is
    /// unknown rather than false.
    pub fn seal_physical_scope(&mut self) {
        self.physical_scope_complete = true;
    }

    pub fn mark_physical_telemetry_incomplete(&mut self) {
        self.physical_telemetry_complete = false;
        // The same ordered stream carries authoritative Store events. Once an
        // event may be missing, absence from the bounded reuse history is no
        // longer evidence that a prefix was never computed in this epoch.
        self.history_complete = false;
    }

    pub fn stats(&self) -> CacheEvidenceStats {
        let mut gpu_physical_blocks = 0;
        let mut cpu_physical_blocks = 0;
        for (_, copy) in self.external_copies.keys() {
            match copy.tier {
                CacheTier::Gpu => gpu_physical_blocks += 1,
                CacheTier::Cpu => cpu_physical_blocks += 1,
            }
        }
        CacheEvidenceStats {
            history_blocks: self.seen_blocks.len(),
            history_saturated: self.history_saturated,
            history_complete: self.history_complete,
            gpu_physical_blocks,
            cpu_physical_blocks,
            expected_owners: self.expected_owners.as_ref().map_or(0, HashSet::len),
            cataloged_owners: self
                .required_groups
                .keys()
                .map(|(owner, _)| *owner)
                .collect::<HashSet<_>>()
                .len(),
            physical_scope_complete: self.physical_scope_complete,
            physical_telemetry_complete: self.physical_telemetry_complete,
        }
    }

    /// Apply one transport batch after its publisher/source ordering has been
    /// validated by the subscriber.
    pub fn apply_evidence_batch(&mut self, batch: &CacheEvidenceBatch) -> bool {
        self.apply_evidence_batch_with_diagnostics(batch)
            .is_complete()
    }

    pub fn apply_evidence_batch_with_diagnostics(
        &mut self,
        batch: &CacheEvidenceBatch,
    ) -> CacheEvidenceApplyResult {
        let mut result = CacheEvidenceApplyResult::default();
        // Report this batch's integrity independently from cumulative stream
        // completeness. A verified cold epoch must be able to apply its clear
        // batches after an earlier gap, while the cumulative poison remains
        // set until the coordinator atomically commits the epoch.
        let prior_telemetry_complete = self.physical_telemetry_complete;
        self.physical_telemetry_complete = true;
        if !batch.telemetry_complete {
            self.mark_physical_telemetry_incomplete();
            result
                .failures
                .insert(CacheEvidenceApplyIntegrityFailure::IncompleteBatch);
        }

        let mut store_run_start = None;
        for (index, mutation) in batch.mutations.iter().enumerate() {
            if matches!(mutation, CacheEvidenceMutation::Store { .. }) {
                store_run_start.get_or_insert(index);
                continue;
            }
            if let Some(start) = store_run_start.take() {
                self.apply_store_run(batch.owner, &batch.mutations[start..index], &mut result);
            }
            match mutation {
                CacheEvidenceMutation::Store { .. } => unreachable!("stores were handled above"),
                CacheEvidenceMutation::Remove {
                    tier,
                    group_idx,
                    block_hashes,
                } => {
                    let Some(group) = *group_idx else {
                        self.mark_physical_telemetry_incomplete();
                        result
                            .failures
                            .insert(CacheEvidenceApplyIntegrityFailure::MissingGroup);
                        continue;
                    };
                    for &hash in block_hashes {
                        self.remove_external(batch.owner, *tier, group, hash);
                    }
                }
                CacheEvidenceMutation::Clear { tier } => match tier {
                    Some(tier) => self.clear_owner_tier(batch.owner, *tier),
                    None => self.clear_owner(batch.owner),
                },
            }
        }
        if let Some(start) = store_run_start {
            self.apply_store_run(batch.owner, &batch.mutations[start..], &mut result);
        }
        let batch_complete = self.physical_telemetry_complete;
        self.physical_telemetry_complete = prior_telemetry_complete && batch_complete;
        debug_assert_eq!(batch_complete, result.is_complete());
        result
    }

    /// Resolve the canonical sequence hashes changed by one mutation before it
    /// is applied. Callers use this to fence route snapshots without scanning
    /// unrelated residency.
    pub fn affected_sequence_hashes(
        &self,
        owner: CacheOwner,
        mutation: &CacheEvidenceMutation,
    ) -> HashSet<u64> {
        match mutation {
            CacheEvidenceMutation::Store {
                group_idx: Some(group),
                parent_external_hash,
                blocks,
                ..
            } => {
                let mut parent = match parent_external_hash {
                    Some(external) => match self.parent_sequence_hash(owner, *group, *external) {
                        Ok(parent) => Some(parent),
                        Err(_) => return HashSet::new(),
                    },
                    None => None,
                };
                blocks
                    .iter()
                    .map(|block| {
                        let hash = parent.map_or(block.tokens_hash, |parent| {
                            compute_next_seq_hash(parent, LocalBlockHash(block.tokens_hash))
                        });
                        parent = Some(hash);
                        hash
                    })
                    .collect()
            }
            CacheEvidenceMutation::Remove {
                tier,
                group_idx: Some(group),
                block_hashes,
            } => {
                let copy = PhysicalCopy {
                    owner,
                    tier: *tier,
                    group: *group,
                };
                block_hashes
                    .iter()
                    .filter_map(|external| {
                        self.external_copies
                            .get(&(*external, copy))
                            .map(|(sequence, _)| *sequence)
                    })
                    .collect()
            }
            CacheEvidenceMutation::Clear { tier } => self
                .external_copies
                .iter()
                .filter_map(|((_, copy), (sequence, _))| {
                    (copy.owner == owner && tier.is_none_or(|tier| copy.tier == tier))
                        .then_some(*sequence)
                })
                .collect(),
            CacheEvidenceMutation::Store { .. } | CacheEvidenceMutation::Remove { .. } => {
                HashSet::new()
            }
        }
    }

    fn apply_store_run(
        &mut self,
        owner: CacheOwner,
        mutations: &[CacheEvidenceMutation],
        result: &mut CacheEvidenceApplyResult,
    ) {
        let mut ready = VecDeque::new();
        let mut waiting: HashMap<(u32, u64), Vec<&CacheEvidenceMutation>> = HashMap::new();
        for mutation in mutations {
            let CacheEvidenceMutation::Store {
                group_idx,
                parent_external_hash,
                ..
            } = mutation
            else {
                unreachable!("store-only batch was checked by the caller");
            };
            let Some(group) = *group_idx else {
                self.mark_physical_telemetry_incomplete();
                result
                    .failures
                    .insert(CacheEvidenceApplyIntegrityFailure::MissingGroup);
                continue;
            };
            match parent_external_hash {
                None => ready.push_back(mutation),
                Some(parent) if self.parent_sequence_hash(owner, group, *parent).is_ok() => {
                    ready.push_back(mutation);
                }
                Some(parent) => waiting.entry((group, *parent)).or_default().push(mutation),
            }
        }

        while let Some(mutation) = ready.pop_front() {
            let CacheEvidenceMutation::Store {
                tier,
                group_idx,
                parent_external_hash,
                blocks,
            } = mutation
            else {
                unreachable!("store-only batch was checked by the caller");
            };
            let Some(group) = *group_idx else {
                unreachable!("invalid groups were rejected before queueing");
            };
            match self.apply_store_mutation(
                owner,
                *tier,
                Some(group),
                *parent_external_hash,
                blocks,
            ) {
                StoreMutationResult::Applied => {}
                StoreMutationResult::MissingParent => {
                    self.mark_physical_telemetry_incomplete();
                    result
                        .failures
                        .insert(CacheEvidenceApplyIntegrityFailure::MissingParent);
                    continue;
                }
                StoreMutationResult::RejectedMapping => {
                    self.mark_physical_telemetry_incomplete();
                    result
                        .failures
                        .insert(CacheEvidenceApplyIntegrityFailure::ConflictingMapping);
                    continue;
                }
                StoreMutationResult::Invalid => {
                    self.mark_physical_telemetry_incomplete();
                    result
                        .failures
                        .insert(CacheEvidenceApplyIntegrityFailure::MissingGroup);
                    continue;
                }
            }
            for block in blocks {
                if let Some(dependents) = waiting.remove(&(group, block.external_hash)) {
                    ready.extend(dependents);
                }
            }
        }
        if !waiting.is_empty() {
            self.mark_physical_telemetry_incomplete();
            result
                .failures
                .insert(CacheEvidenceApplyIntegrityFailure::MissingParent);
        }
    }

    fn apply_store_mutation(
        &mut self,
        owner: CacheOwner,
        tier: CacheTier,
        group: Option<u32>,
        parent_external_hash: Option<u64>,
        blocks: &[CacheEvidenceStoredBlock],
    ) -> StoreMutationResult {
        let Some(group) = group else {
            return StoreMutationResult::Invalid;
        };
        let mut parent_sequence_hash = match parent_external_hash {
            Some(parent) => match self.parent_sequence_hash(owner, group, parent) {
                Ok(sequence_hash) => Some(sequence_hash),
                Err(()) => return StoreMutationResult::MissingParent,
            },
            None => None,
        };
        let copy = PhysicalCopy { owner, tier, group };
        let mut mappings = Vec::with_capacity(blocks.len());
        let mut proposed = HashMap::new();
        for block in blocks {
            let sequence_hash = parent_sequence_hash.map_or(block.tokens_hash, |parent| {
                compute_next_seq_hash(parent, LocalBlockHash(block.tokens_hash))
            });
            if self
                .external_copies
                .get(&(block.external_hash, copy))
                .is_some_and(|(mapped_hash, _)| *mapped_hash != sequence_hash)
                || proposed
                    .insert(block.external_hash, sequence_hash)
                    .is_some_and(|mapped_hash| mapped_hash != sequence_hash)
            {
                return StoreMutationResult::RejectedMapping;
            }
            mappings.push((block.external_hash, sequence_hash));
            parent_sequence_hash = Some(sequence_hash);
        }
        self.record_seen_blocks(mappings.iter().map(|(_, sequence_hash)| *sequence_hash));
        for (external_hash, sequence_hash) in mappings {
            let applied = self.try_store_mapped(owner, tier, group, external_hash, sequence_hash);
            debug_assert!(applied, "store mapping was validated before mutation");
        }
        StoreMutationResult::Applied
    }

    fn parent_sequence_hash(
        &self,
        owner: CacheOwner,
        group: u32,
        external_hash: u64,
    ) -> Result<u64, ()> {
        let mut resolved = None;
        for tier in [CacheTier::Gpu, CacheTier::Cpu] {
            let copy = PhysicalCopy { owner, tier, group };
            let Some((sequence_hash, _)) = self.external_copies.get(&(external_hash, copy)) else {
                continue;
            };
            match resolved {
                Some(existing) if existing != *sequence_hash => return Err(()),
                Some(_) => {}
                None => resolved = Some(*sequence_hash),
            }
        }
        resolved.ok_or(())
    }

    pub fn store(&mut self, owner: CacheOwner, tier: CacheTier, group: u32, hash: u64) {
        self.store_mapped(owner, tier, group, hash, hash);
    }

    pub fn store_mapped(
        &mut self,
        owner: CacheOwner,
        tier: CacheTier,
        group: u32,
        external_hash: u64,
        sequence_hash: u64,
    ) {
        self.try_store_mapped(owner, tier, group, external_hash, sequence_hash);
    }

    fn try_store_mapped(
        &mut self,
        owner: CacheOwner,
        tier: CacheTier,
        group: u32,
        external_hash: u64,
        sequence_hash: u64,
    ) -> bool {
        let copy = PhysicalCopy { owner, tier, group };
        if self
            .external_copies
            .get(&(external_hash, copy))
            .is_some_and(|(mapped_hash, _)| *mapped_hash != sequence_hash)
        {
            self.mark_physical_telemetry_incomplete();
            return false;
        }
        let external = self
            .external_copies
            .entry((external_hash, copy))
            .or_insert((sequence_hash, 0));
        external.1 = external.1.saturating_add(1);
        let count = self.copies.entry((sequence_hash, copy)).or_default();
        *count = count.saturating_add(1);
        // Store events include generated blocks as they become complete. They
        // therefore extend bounded reuse history even when no later request
        // has yet carried those generated tokens back in its prompt.
        self.record_seen_blocks([sequence_hash]);
        true
    }

    pub fn remove(&mut self, owner: CacheOwner, tier: CacheTier, group: u32, hash: u64) {
        self.remove_external(owner, tier, group, hash);
    }

    pub fn remove_external(
        &mut self,
        owner: CacheOwner,
        tier: CacheTier,
        group: u32,
        external_hash: u64,
    ) {
        let copy = PhysicalCopy { owner, tier, group };
        let external_key = (external_hash, copy);
        let Some((tokens_hash, external_count)) = self.external_copies.get_mut(&external_key)
        else {
            self.mark_physical_telemetry_incomplete();
            return;
        };
        let tokens_hash = *tokens_hash;
        if *external_count == 1 {
            self.external_copies.remove(&external_key);
        } else {
            *external_count -= 1;
        }
        let key = (tokens_hash, copy);
        let Some(count) = self.copies.get_mut(&key) else {
            self.mark_physical_telemetry_incomplete();
            return;
        };
        if *count == 1 {
            self.copies.remove(&key);
        } else {
            *count -= 1;
        }
    }

    pub fn clear_owner(&mut self, owner: CacheOwner) {
        self.copies.retain(|(_, copy), _| copy.owner != owner);
        self.external_copies
            .retain(|(_, copy), _| copy.owner != owner);
    }

    pub fn clear_owner_tier(&mut self, owner: CacheOwner, tier: CacheTier) {
        self.copies
            .retain(|(_, copy), _| copy.owner != owner || copy.tier != tier);
        self.external_copies
            .retain(|(_, copy), _| copy.owner != owner || copy.tier != tier);
    }

    /// Remove all physical state and cache-shape declarations for a worker
    /// that has left the measured routing domain.
    pub fn retire_owner(&mut self, owner: CacheOwner) {
        self.clear_owner(owner);
        self.required_groups
            .retain(|(declared_owner, _), _| *declared_owner != owner);
        self.refresh_physical_scope();
    }

    pub fn resident_anywhere(&self, hash: u64) -> KnownBool {
        if self
            .required_groups
            .iter()
            .any(|(&(owner, tier), _)| self.resident_in_declared_cache(hash, owner, tier))
        {
            KnownBool::Yes
        } else {
            self.negative_physical_fact()
        }
    }

    pub fn resident_on(&self, hash: u64, owner: CacheOwner) -> KnownBool {
        let mut has_declared_tier = false;
        for tier in [CacheTier::Gpu, CacheTier::Cpu] {
            if self.required_groups.contains_key(&(owner, tier)) {
                has_declared_tier = true;
                if self.resident_in_declared_cache(hash, owner, tier) {
                    return KnownBool::Yes;
                }
            }
        }
        if has_declared_tier {
            self.negative_physical_fact()
        } else {
            KnownBool::Unknown
        }
    }

    /// Test one logical prompt span against every cache group declared for a
    /// worker. Different groups may use different block sizes and therefore
    /// different token hashes for the same span.
    pub fn resident_groups_on(
        &self,
        queries: &[CacheGroupBlockQuery],
        owner: CacheOwner,
    ) -> KnownBool {
        let required: BTreeSet<_> = [CacheTier::Gpu, CacheTier::Cpu]
            .into_iter()
            .filter_map(|tier| self.required_groups.get(&(owner, tier)))
            .flatten()
            .copied()
            .collect();
        if required.is_empty() {
            return KnownBool::Unknown;
        }
        let mut query_by_group: HashMap<u32, Vec<u64>> = HashMap::new();
        for query in queries {
            query_by_group
                .entry(query.group_idx)
                .or_default()
                .push(query.tokens_hash);
        }
        if !required
            .iter()
            .all(|group| query_by_group.contains_key(group))
        {
            return KnownBool::Unknown;
        }
        let resident = required.iter().all(|&group| {
            query_by_group[&group].iter().all(|&hash| {
                [CacheTier::Gpu, CacheTier::Cpu].into_iter().any(|tier| {
                    self.copies
                        .get(&(hash, PhysicalCopy { owner, tier, group }))
                        .is_some_and(|&count| count > 0)
                })
            })
        });
        if resident {
            KnownBool::Yes
        } else {
            self.negative_physical_fact()
        }
    }

    pub fn resident_groups_anywhere(&self, queries: &[CacheGroupBlockQuery]) -> KnownBool {
        let owners: HashSet<_> = self
            .required_groups
            .keys()
            .map(|(owner, _)| *owner)
            .collect();
        let mut saw_unknown = owners.is_empty();
        for owner in owners {
            match self.resident_groups_on(queries, owner) {
                KnownBool::Yes => return KnownBool::Yes,
                KnownBool::Unknown => saw_unknown = true,
                KnownBool::No => {}
            }
        }
        if saw_unknown {
            KnownBool::Unknown
        } else {
            self.negative_physical_fact()
        }
    }

    /// Reproduce vLLM's hybrid cache-hit boundary over prior request history.
    pub fn reusable_prefix(
        &self,
        groups: &[CacheGroupHashSequence],
        max_tokens: u64,
    ) -> KnownPrefixLength {
        hybrid_prefix(groups, max_tokens, |_, hash| self.reusable(hash))
    }

    /// Longest prefix physically reusable on one worker across its GPU and CPU
    /// copies. A group may be split across the two tiers.
    pub fn resident_prefix_on(
        &self,
        groups: &[CacheGroupHashSequence],
        max_tokens: u64,
        owner: CacheOwner,
    ) -> KnownPrefixLength {
        hybrid_prefix(groups, max_tokens, |group, hash| {
            let mut declared = false;
            for tier in [CacheTier::Gpu, CacheTier::Cpu] {
                if self.required_groups.contains_key(&(owner, tier)) {
                    declared = true;
                }
                if self
                    .copies
                    .get(&(hash, PhysicalCopy { owner, tier, group }))
                    .is_some_and(|&count| count > 0)
                {
                    return KnownBool::Yes;
                }
            }
            if declared {
                self.negative_physical_fact()
            } else {
                KnownBool::Unknown
            }
        })
    }

    /// Maximum physically reusable prefix on any worker. If an incompletely
    /// observed worker could exceed the best known result, the answer remains
    /// unknown rather than becoming a false zero or lower bound.
    pub fn resident_prefix_anywhere(
        &self,
        groups: &[CacheGroupHashSequence],
        max_tokens: u64,
    ) -> KnownPrefixLength {
        let owners: HashSet<_> = self
            .required_groups
            .keys()
            .map(|(owner, _)| *owner)
            .collect();
        let mut best = 0;
        let mut unknown = owners.is_empty();
        for owner in owners {
            match self.resident_prefix_on(groups, max_tokens, owner) {
                KnownPrefixLength::Known(tokens) => best = best.max(tokens),
                KnownPrefixLength::Unknown => unknown = true,
            }
        }
        if best == max_tokens || !unknown {
            KnownPrefixLength::Known(best)
        } else {
            KnownPrefixLength::Unknown
        }
    }

    fn resident_in_declared_cache(&self, hash: u64, owner: CacheOwner, tier: CacheTier) -> bool {
        self.required_groups
            .get(&(owner, tier))
            .is_some_and(|groups| {
                !groups.is_empty()
                    && groups.iter().all(|&group| {
                        self.copies
                            .get(&(hash, PhysicalCopy { owner, tier, group }))
                            .is_some_and(|&count| count > 0)
                    })
            })
    }

    fn negative_physical_fact(&self) -> KnownBool {
        if self.physical_scope_complete && self.physical_telemetry_complete {
            KnownBool::No
        } else {
            KnownBool::Unknown
        }
    }

    fn refresh_physical_scope(&mut self) {
        let Some(expected) = &self.expected_owners else {
            return;
        };
        self.physical_scope_complete = !expected.is_empty()
            && expected.iter().all(|owner| {
                self.required_groups
                    .iter()
                    .any(|((declared_owner, _), groups)| {
                        declared_owner == owner && !groups.is_empty()
                    })
            });
    }
}

fn hybrid_prefix(
    groups: &[CacheGroupHashSequence],
    max_tokens: u64,
    mut present: impl FnMut(u32, u64) -> KnownBool,
) -> KnownPrefixLength {
    if groups.is_empty() {
        return KnownPrefixLength::Unknown;
    }
    if groups.iter().any(|group| {
        group.block_size == 0
            || group.alignment_tokens == 0
            || group.alignment_tokens % group.block_size != 0
            || matches!(group.kind, CacheGroupKind::SlidingWindow)
                && group.sliding_window.is_none_or(|window| window == 0)
    }) {
        return KnownPrefixLength::Unknown;
    }

    let mut hit = max_tokens;
    let mut eagle_verified = HashSet::new();
    loop {
        let mut current = hit;
        for group in groups {
            let drop_eagle = group.is_eagle && !eagle_verified.contains(&group.group_idx);
            let lookup_limit = if drop_eagle {
                current
                    .saturating_add(u64::from(group.block_size))
                    .min(max_tokens)
            } else {
                current
            };
            let next = match group_prefix(group, lookup_limit, drop_eagle, &mut present) {
                KnownPrefixLength::Known(value) => value,
                KnownPrefixLength::Unknown => return KnownPrefixLength::Unknown,
            };
            if drop_eagle {
                eagle_verified.insert(group.group_idx);
            } else if next < current {
                eagle_verified.clear();
            }
            current = next;
        }
        if current >= hit {
            return KnownPrefixLength::Known(hit);
        }
        hit = current;
    }
}

fn group_prefix(
    group: &CacheGroupHashSequence,
    max_tokens: u64,
    drop_eagle: bool,
    present: &mut impl FnMut(u32, u64) -> KnownBool,
) -> KnownPrefixLength {
    match group.kind {
        CacheGroupKind::FullAttention => {
            let max_blocks = (max_tokens / u64::from(group.block_size)) as usize;
            let mut hit_blocks = 0usize;
            for &hash in group.sequence_hashes.iter().take(max_blocks) {
                match present(group.group_idx, hash) {
                    KnownBool::Yes => hit_blocks += 1,
                    KnownBool::No => break,
                    KnownBool::Unknown => return KnownPrefixLength::Unknown,
                }
            }
            let mut hit = hit_blocks as u64 * u64::from(group.block_size);
            if drop_eagle && hit > 0 {
                hit = hit.saturating_sub(u64::from(group.alignment_tokens.min(group.block_size)));
            }
            hit -= hit % u64::from(group.alignment_tokens);
            KnownPrefixLength::Known(hit)
        }
        CacheGroupKind::SlidingWindow => {
            sliding_window_prefix(group, max_tokens, drop_eagle, present)
        }
    }
}

fn sliding_window_prefix(
    group: &CacheGroupHashSequence,
    max_tokens: u64,
    drop_eagle: bool,
    present: &mut impl FnMut(u32, u64) -> KnownBool,
) -> KnownPrefixLength {
    let block_size = u64::from(group.block_size);
    let alignment = u64::from(group.alignment_tokens);
    let max_blocks = ((max_tokens / block_size) as usize).min(group.sequence_hashes.len());
    let window = u64::from(group.sliding_window.expect("validated above"));
    let mut needed = window.saturating_sub(1).div_ceil(block_size) as usize;
    if drop_eagle {
        needed += 1;
    }

    for end in (needed..=max_blocks).rev() {
        let post_drop = end.saturating_sub(usize::from(drop_eagle));
        if !(post_drop as u64 * block_size).is_multiple_of(alignment) {
            continue;
        }
        let mut all_present = true;
        for &hash in &group.sequence_hashes[end - needed..end] {
            match present(group.group_idx, hash) {
                KnownBool::Yes => {}
                KnownBool::No => {
                    all_present = false;
                    break;
                }
                KnownBool::Unknown => return KnownPrefixLength::Unknown,
            }
        }
        if all_present {
            return KnownPrefixLength::Known(post_drop as u64 * block_size);
        }
    }

    // vLLM falls back to the leading contiguous prefix when the request is
    // shorter than a complete window or no aligned tail window is present.
    let mut leading = 0usize;
    for &hash in group.sequence_hashes.iter().take(max_blocks) {
        match present(group.group_idx, hash) {
            KnownBool::Yes => leading += 1,
            KnownBool::No => break,
            KnownBool::Unknown => return KnownPrefixLength::Unknown,
        }
    }
    if drop_eagle && leading > 0 {
        leading -= 1;
    }
    let mut hit = leading as u64 * block_size;
    hit -= hit % alignment;
    KnownPrefixLength::Known(hit)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenSource {
    Recomputed,
    Gpu,
    Cpu,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenObservation {
    pub complete: bool,
    pub reusable: bool,
    pub physically_resident: bool,
    pub router_visible: bool,
    pub selected_resident_at_routing: bool,
    pub selected_resident_at_lookup: bool,
    pub source: TokenSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenAttribution {
    UnavoidableRecomputation,
    RetentionLoss,
    RouterVisibilityLoss,
    RoutingChoiceLoss,
    EvictionRaceLoss,
    RetrievalFailure,
    GpuHit,
    CpuHit,
    Incomplete,
}

impl TokenObservation {
    /// Prefer the measured outcome, then assign recomputation to the first
    /// failed stage in the causal chain.
    pub fn classify(self) -> TokenAttribution {
        if !self.complete || self.source == TokenSource::Unknown {
            return TokenAttribution::Incomplete;
        }
        match self.source {
            TokenSource::Gpu => return TokenAttribution::GpuHit,
            TokenSource::Cpu => return TokenAttribution::CpuHit,
            TokenSource::Recomputed => {}
            TokenSource::Unknown => unreachable!("handled above"),
        }
        if !self.reusable {
            return TokenAttribution::UnavoidableRecomputation;
        }
        if !self.physically_resident {
            return TokenAttribution::RetentionLoss;
        }
        if !self.router_visible {
            return TokenAttribution::RouterVisibilityLoss;
        }
        if !self.selected_resident_at_routing {
            return TokenAttribution::RoutingChoiceLoss;
        }
        if !self.selected_resident_at_lookup {
            return TokenAttribution::EvictionRaceLoss;
        }
        TokenAttribution::RetrievalFailure
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CacheLossFunnel {
    pub f0_backend_prompt_tokens: u64,
    pub f1_reusable_tokens: u64,
    pub f2_physically_resident_tokens: u64,
    pub f3_router_visible_tokens: u64,
    pub f4_selected_resident_tokens: u64,
    pub f5_lookup_resident_tokens: u64,
    pub f6_reused_tokens: u64,
    pub unavoidable_recomputation: u64,
    pub retention_loss: u64,
    pub router_visibility_loss: u64,
    pub routing_choice_loss: u64,
    pub eviction_race_loss: u64,
    pub retrieval_failure: u64,
    pub gpu_hits: u64,
    pub cpu_hits: u64,
    pub incomplete_tokens: u64,
    pub router_false_positive_tokens: u64,
    pub router_false_negative_tokens: u64,
}

impl CacheLossFunnel {
    pub fn observe(&mut self, observation: TokenObservation) -> TokenAttribution {
        self.observe_n(observation, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe_prefix_counts(
        &mut self,
        prompt_tokens: u64,
        reusable_prefix_tokens: u64,
        physical_prefix_tokens: u64,
        router_visible_prefix_tokens: u64,
        selected_resident_prefix_tokens: u64,
        lookup_resident_prefix_tokens: u64,
        gpu_hit_tokens: u64,
        cpu_hit_tokens: u64,
    ) {
        let mut boundaries = vec![
            0,
            prompt_tokens,
            reusable_prefix_tokens.min(prompt_tokens),
            physical_prefix_tokens.min(prompt_tokens),
            router_visible_prefix_tokens.min(prompt_tokens),
            selected_resident_prefix_tokens.min(prompt_tokens),
            lookup_resident_prefix_tokens.min(prompt_tokens),
            gpu_hit_tokens.min(prompt_tokens),
            gpu_hit_tokens
                .saturating_add(cpu_hit_tokens)
                .min(prompt_tokens),
        ];
        boundaries.sort_unstable();
        boundaries.dedup();
        for window in boundaries.windows(2) {
            let start = window[0];
            let count = window[1] - start;
            if count == 0 {
                continue;
            }
            let source = if start < gpu_hit_tokens {
                TokenSource::Gpu
            } else if start < gpu_hit_tokens.saturating_add(cpu_hit_tokens) {
                TokenSource::Cpu
            } else {
                TokenSource::Recomputed
            };
            self.observe_n(
                TokenObservation {
                    complete: true,
                    reusable: start < reusable_prefix_tokens,
                    physically_resident: start < physical_prefix_tokens,
                    router_visible: start < router_visible_prefix_tokens,
                    selected_resident_at_routing: start < selected_resident_prefix_tokens,
                    selected_resident_at_lookup: start < lookup_resident_prefix_tokens,
                    source,
                },
                count,
            );
        }
    }

    pub fn observe_incomplete(&mut self, prompt_tokens: u64) {
        self.observe_n(
            TokenObservation {
                complete: false,
                reusable: false,
                physically_resident: false,
                router_visible: false,
                selected_resident_at_routing: false,
                selected_resident_at_lookup: false,
                source: TokenSource::Unknown,
            },
            prompt_tokens,
        );
    }

    fn observe_n(&mut self, observation: TokenObservation, count: u64) -> TokenAttribution {
        self.f0_backend_prompt_tokens = self.f0_backend_prompt_tokens.saturating_add(count);
        if observation.complete {
            if observation.router_visible && !observation.physically_resident {
                self.router_false_positive_tokens =
                    self.router_false_positive_tokens.saturating_add(count);
            }
            if observation.physically_resident && !observation.router_visible {
                self.router_false_negative_tokens =
                    self.router_false_negative_tokens.saturating_add(count);
            }
        }

        let attribution = observation.classify();
        match attribution {
            TokenAttribution::Incomplete => {
                self.incomplete_tokens = self.incomplete_tokens.saturating_add(count);
                return attribution;
            }
            TokenAttribution::UnavoidableRecomputation => {
                self.unavoidable_recomputation =
                    self.unavoidable_recomputation.saturating_add(count);
            }
            TokenAttribution::RetentionLoss => {
                self.retention_loss = self.retention_loss.saturating_add(count);
                self.f1_reusable_tokens = self.f1_reusable_tokens.saturating_add(count);
            }
            TokenAttribution::RouterVisibilityLoss => {
                self.router_visibility_loss = self.router_visibility_loss.saturating_add(count);
                self.f1_reusable_tokens = self.f1_reusable_tokens.saturating_add(count);
                self.f2_physically_resident_tokens =
                    self.f2_physically_resident_tokens.saturating_add(count);
            }
            TokenAttribution::RoutingChoiceLoss => {
                self.routing_choice_loss = self.routing_choice_loss.saturating_add(count);
                self.f1_reusable_tokens = self.f1_reusable_tokens.saturating_add(count);
                self.f2_physically_resident_tokens =
                    self.f2_physically_resident_tokens.saturating_add(count);
                self.f3_router_visible_tokens = self.f3_router_visible_tokens.saturating_add(count);
            }
            TokenAttribution::EvictionRaceLoss => {
                self.eviction_race_loss = self.eviction_race_loss.saturating_add(count);
                self.increment_through_f4(count);
            }
            TokenAttribution::RetrievalFailure => {
                self.retrieval_failure = self.retrieval_failure.saturating_add(count);
                self.increment_through_f5(count);
            }
            TokenAttribution::GpuHit => {
                self.gpu_hits = self.gpu_hits.saturating_add(count);
                self.increment_through_f6(count);
            }
            TokenAttribution::CpuHit => {
                self.cpu_hits = self.cpu_hits.saturating_add(count);
                self.increment_through_f6(count);
            }
        }
        attribution
    }

    pub fn complete_tokens(&self) -> u64 {
        self.f0_backend_prompt_tokens - self.incomplete_tokens
    }

    pub fn attributed_complete_tokens(&self) -> u64 {
        self.unavoidable_recomputation
            + self.retention_loss
            + self.router_visibility_loss
            + self.routing_choice_loss
            + self.eviction_race_loss
            + self.retrieval_failure
            + self.gpu_hits
            + self.cpu_hits
    }

    pub fn conservation_error(&self) -> i128 {
        i128::from(self.complete_tokens()) - i128::from(self.attributed_complete_tokens())
    }

    fn increment_through_f4(&mut self, count: u64) {
        self.f1_reusable_tokens = self.f1_reusable_tokens.saturating_add(count);
        self.f2_physically_resident_tokens =
            self.f2_physically_resident_tokens.saturating_add(count);
        self.f3_router_visible_tokens = self.f3_router_visible_tokens.saturating_add(count);
        self.f4_selected_resident_tokens = self.f4_selected_resident_tokens.saturating_add(count);
    }

    fn increment_through_f5(&mut self, count: u64) {
        self.increment_through_f4(count);
        self.f5_lookup_resident_tokens = self.f5_lookup_resident_tokens.saturating_add(count);
    }

    fn increment_through_f6(&mut self, count: u64) {
        self.increment_through_f5(count);
        self.f6_reused_tokens = self.f6_reused_tokens.saturating_add(count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKER_A: CacheOwner = CacheOwner {
        worker_id: 1,
        dp_rank: 0,
    };
    const WORKER_B: CacheOwner = CacheOwner {
        worker_id: 2,
        dp_rank: 0,
    };

    fn recomputed() -> TokenObservation {
        TokenObservation {
            complete: true,
            reusable: true,
            physically_resident: true,
            router_visible: true,
            selected_resident_at_routing: true,
            selected_resident_at_lookup: true,
            source: TokenSource::Recomputed,
        }
    }

    #[test]
    fn controlled_losses_are_mutually_exclusive_and_conserve_tokens() {
        let cases = [
            (
                TokenObservation {
                    reusable: false,
                    ..recomputed()
                },
                TokenAttribution::UnavoidableRecomputation,
            ),
            (
                TokenObservation {
                    physically_resident: false,
                    router_visible: false,
                    selected_resident_at_routing: false,
                    selected_resident_at_lookup: false,
                    ..recomputed()
                },
                TokenAttribution::RetentionLoss,
            ),
            (
                TokenObservation {
                    router_visible: false,
                    selected_resident_at_routing: false,
                    selected_resident_at_lookup: false,
                    ..recomputed()
                },
                TokenAttribution::RouterVisibilityLoss,
            ),
            (
                TokenObservation {
                    selected_resident_at_routing: false,
                    selected_resident_at_lookup: false,
                    ..recomputed()
                },
                TokenAttribution::RoutingChoiceLoss,
            ),
            (
                TokenObservation {
                    selected_resident_at_lookup: false,
                    ..recomputed()
                },
                TokenAttribution::EvictionRaceLoss,
            ),
            (recomputed(), TokenAttribution::RetrievalFailure),
            (
                TokenObservation {
                    source: TokenSource::Gpu,
                    ..recomputed()
                },
                TokenAttribution::GpuHit,
            ),
            (
                TokenObservation {
                    source: TokenSource::Cpu,
                    ..recomputed()
                },
                TokenAttribution::CpuHit,
            ),
        ];
        let mut funnel = CacheLossFunnel::default();
        for (observation, expected) in cases {
            assert_eq!(funnel.observe(observation), expected);
        }

        assert_eq!(funnel.f0_backend_prompt_tokens, 8);
        assert_eq!(funnel.f6_reused_tokens, 2);
        assert_eq!(funnel.gpu_hits, 1);
        assert_eq!(funnel.cpu_hits, 1);
        assert_eq!(funnel.conservation_error(), 0);
    }

    #[test]
    fn incomplete_tokens_are_explicit_and_excluded_from_conservation() {
        let mut funnel = CacheLossFunnel::default();
        let attribution = funnel.observe(TokenObservation {
            complete: false,
            source: TokenSource::Unknown,
            ..recomputed()
        });

        assert_eq!(attribution, TokenAttribution::Incomplete);
        assert_eq!(funnel.f0_backend_prompt_tokens, 1);
        assert_eq!(funnel.incomplete_tokens, 1);
        assert_eq!(funnel.complete_tokens(), 0);
        assert_eq!(funnel.conservation_error(), 0);
    }

    #[test]
    fn aggregate_prefix_observation_matches_token_level_attribution() {
        let mut funnel = CacheLossFunnel::default();
        funnel.observe_prefix_counts(100, 80, 70, 60, 50, 40, 20, 10);

        assert_eq!(funnel.f0_backend_prompt_tokens, 100);
        assert_eq!(funnel.unavoidable_recomputation, 20);
        assert_eq!(funnel.retention_loss, 10);
        assert_eq!(funnel.router_visibility_loss, 10);
        assert_eq!(funnel.routing_choice_loss, 10);
        assert_eq!(funnel.eviction_race_loss, 10);
        assert_eq!(funnel.retrieval_failure, 10);
        assert_eq!(funnel.gpu_hits, 20);
        assert_eq!(funnel.cpu_hits, 10);
        assert_eq!(funnel.conservation_error(), 0);
    }

    #[test]
    fn actual_hit_wins_over_stale_router_evidence() {
        let mut funnel = CacheLossFunnel::default();
        let attribution = funnel.observe(TokenObservation {
            router_visible: false,
            selected_resident_at_routing: false,
            source: TokenSource::Cpu,
            ..recomputed()
        });

        assert_eq!(attribution, TokenAttribution::CpuHit);
        assert_eq!(funnel.f6_reused_tokens, 1);
        assert_eq!(funnel.router_false_negative_tokens, 1);
        assert_eq!(funnel.conservation_error(), 0);
    }

    #[test]
    fn bounded_history_never_turns_overflow_into_a_cold_miss() {
        let mut ledger = CacheEvidenceLedger::new(2);
        ledger.record_seen_blocks([10, 20, 30]);

        assert_eq!(ledger.reusable(10), KnownBool::Yes);
        assert_eq!(ledger.reusable(20), KnownBool::Yes);
        assert_eq!(ledger.reusable(30), KnownBool::Unknown);
        assert_eq!(ledger.reusable(40), KnownBool::Unknown);
    }

    #[test]
    fn incomplete_history_never_turns_missing_observation_into_a_cold_miss() {
        let mut ledger = CacheEvidenceLedger::new(2);
        ledger.record_seen_blocks([10]);
        ledger.mark_history_incomplete();

        assert_eq!(ledger.reusable(10), KnownBool::Yes);
        assert_eq!(ledger.reusable(20), KnownBool::Unknown);
        assert!(!ledger.stats().history_complete);
    }

    #[test]
    fn frontend_restart_against_warm_workers_keeps_prior_evictions_unknown() {
        let mut restarted = CacheEvidenceLedger::new(8);
        restarted.set_expected_owners([WORKER_A]);
        restarted.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        restarted.seal_physical_scope();

        assert_eq!(restarted.reusable(41), KnownBool::Unknown);
        assert_eq!(restarted.resident_anywhere(41), KnownBool::No);
        assert!(!restarted.stats().history_complete);
    }

    #[test]
    fn multi_group_residency_requires_every_declared_group() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0, 1, 2]);
        ledger.seal_physical_scope();

        ledger.store(WORKER_A, CacheTier::Cpu, 0, 99);
        ledger.store(WORKER_A, CacheTier::Cpu, 1, 99);
        assert_eq!(ledger.resident_on(99, WORKER_A), KnownBool::No);

        ledger.store(WORKER_A, CacheTier::Cpu, 2, 99);
        assert_eq!(ledger.resident_on(99, WORKER_A), KnownBool::Yes);
    }

    #[test]
    fn tier_qualified_clear_preserves_other_tier_residency() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store(WORKER_A, CacheTier::Gpu, 0, 99);
        ledger.store(WORKER_A, CacheTier::Cpu, 0, 99);

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 0,
            source_incarnation_id: Some(1),
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Clear {
                tier: Some(CacheTier::Gpu),
            }],
            barrier_id: None,
            epoch_id: None,
        });

        assert_eq!(ledger.resident_on(99, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.stats().gpu_physical_blocks, 0);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);
    }

    #[test]
    fn clear_evidence_decodes_legacy_and_tier_qualified_shapes() {
        let legacy: CacheEvidenceMutation =
            serde_json::from_value(serde_json::json!({ "operation": "clear" })).unwrap();
        let qualified: CacheEvidenceMutation = serde_json::from_value(serde_json::json!({
            "operation": "clear",
            "tier": "gpu"
        }))
        .unwrap();

        assert_eq!(legacy, CacheEvidenceMutation::Clear { tier: None });
        assert_eq!(
            qualified,
            CacheEvidenceMutation::Clear {
                tier: Some(CacheTier::Gpu)
            }
        );
    }

    #[test]
    fn duplicate_announcements_are_reference_counted() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();

        ledger.store(WORKER_A, CacheTier::Cpu, 0, 7);
        ledger.store(WORKER_A, CacheTier::Cpu, 0, 7);
        ledger.remove(WORKER_A, CacheTier::Cpu, 0, 7);
        assert_eq!(ledger.resident_on(7, WORKER_A), KnownBool::Yes);

        ledger.remove(WORKER_A, CacheTier::Cpu, 0, 7);
        assert_eq!(ledger.resident_on(7, WORKER_A), KnownBool::No);
    }

    #[test]
    fn physical_block_metrics_do_not_double_count_duplicate_announcements() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.store(WORKER_A, CacheTier::Cpu, 0, 7);
        ledger.store(WORKER_A, CacheTier::Cpu, 0, 7);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);
        ledger.remove(WORKER_A, CacheTier::Cpu, 0, 7);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);
        ledger.remove(WORKER_A, CacheTier::Cpu, 0, 7);
        assert_eq!(ledger.stats().cpu_physical_blocks, 0);
    }

    #[test]
    fn duplicate_copies_across_workers_are_independent() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_B, CacheTier::Gpu, [0]);
        ledger.seal_physical_scope();
        ledger.store(WORKER_A, CacheTier::Gpu, 0, 5);
        ledger.store(WORKER_B, CacheTier::Gpu, 0, 5);

        ledger.remove(WORKER_A, CacheTier::Gpu, 0, 5);
        assert_eq!(ledger.resident_on(5, WORKER_A), KnownBool::No);
        assert_eq!(ledger.resident_on(5, WORKER_B), KnownBool::Yes);
        assert_eq!(ledger.resident_anywhere(5), KnownBool::Yes);
    }

    #[test]
    fn gaps_make_negative_residency_unknown_but_preserve_positive_facts() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store(WORKER_A, CacheTier::Cpu, 0, 1);
        ledger.mark_physical_telemetry_incomplete();

        assert_eq!(ledger.resident_on(1, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(2, WORKER_A), KnownBool::Unknown);
    }

    #[test]
    fn evidence_batches_apply_all_groups_and_fail_closed_on_missing_group() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0, 1]);
        ledger.seal_physical_scope();
        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 4,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: None,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 90,
                        tokens_hash: 9,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(1),
                    parent_external_hash: None,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 91,
                        tokens_hash: 9,
                    }],
                },
            ],
        });
        assert_eq!(ledger.resident_on(9, WORKER_A), KnownBool::Yes);

        let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 5,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Remove {
                tier: CacheTier::Cpu,
                group_idx: None,
                block_hashes: vec![9],
            }],
        });
        assert_eq!(
            result.failures().collect::<HashSet<_>>(),
            HashSet::from([CacheEvidenceApplyIntegrityFailure::MissingGroup])
        );
        assert_eq!(ledger.resident_on(10, WORKER_A), KnownBool::Unknown);
    }

    #[test]
    fn incomplete_batch_reports_bounded_integrity_reason() {
        let mut ledger = CacheEvidenceLedger::new(16);
        let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: false,
            mutations: Vec::new(),
        });

        assert_eq!(
            result.failures().collect::<HashSet<_>>(),
            HashSet::from([CacheEvidenceApplyIntegrityFailure::IncompleteBatch])
        );
    }

    #[test]
    fn legacy_evidence_batch_defaults_freshness_fields() {
        let batch: CacheEvidenceBatch = serde_json::from_value(serde_json::json!({
            "owner": { "worker_id": 7, "dp_rank": 0 },
            "source_cursor": 4,
            "telemetry_complete": true,
            "mutations": []
        }))
        .unwrap();
        assert_eq!(batch.source_incarnation_id, None);
        assert!(!batch.heartbeat);
        assert_eq!(batch.watermark_source_cursor, None);
    }

    #[test]
    fn group_aware_residency_accepts_distinct_hashes_and_mixed_tiers() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0, 1]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0, 1]);
        ledger.seal_physical_scope();
        ledger.store_mapped(WORKER_A, CacheTier::Gpu, 0, 100, 10);
        ledger.store_mapped(WORKER_A, CacheTier::Cpu, 1, 200, 20);
        ledger.store_mapped(WORKER_A, CacheTier::Cpu, 1, 201, 21);

        let queries = [
            CacheGroupBlockQuery {
                group_idx: 0,
                tokens_hash: 10,
            },
            CacheGroupBlockQuery {
                group_idx: 1,
                tokens_hash: 20,
            },
            CacheGroupBlockQuery {
                group_idx: 1,
                tokens_hash: 21,
            },
        ];
        assert_eq!(
            ledger.resident_groups_on(&queries, WORKER_A),
            KnownBool::Yes
        );
        assert_eq!(ledger.resident_groups_anywhere(&queries), KnownBool::Yes);

        ledger.remove_external(WORKER_A, CacheTier::Cpu, 1, 200);
        assert_eq!(ledger.resident_groups_on(&queries, WORKER_A), KnownBool::No);
    }

    #[test]
    fn negative_residency_waits_for_every_active_worker_catalog() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.set_expected_owners([WORKER_A, WORKER_B]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        assert_eq!(ledger.resident_anywhere(99), KnownBool::Unknown);

        ledger.record_group_catalog(WORKER_B, CacheTier::Gpu, [0]);
        assert_eq!(ledger.resident_anywhere(99), KnownBool::No);
    }

    #[test]
    fn physical_chain_distinguishes_equal_token_blocks_under_different_parents() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.seal_physical_scope();
        for (cursor, parent, external, local) in [
            (0, None, 100, 10),
            (1, Some(100), 101, 20),
            (2, None, 200, 11),
            (3, Some(200), 201, 20),
        ] {
            ledger.apply_evidence_batch(&CacheEvidenceBatch {
                owner: WORKER_A,
                source_cursor: cursor,
                source_incarnation_id: None,
                barrier_id: None,
                epoch_id: None,
                heartbeat: false,
                watermark_source_cursor: None,
                telemetry_complete: true,
                mutations: vec![CacheEvidenceMutation::Store {
                    tier: CacheTier::Gpu,
                    group_idx: Some(0),
                    parent_external_hash: parent,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: external,
                        tokens_hash: local,
                    }],
                }],
            });
        }

        let first_child = compute_next_seq_hash(10, LocalBlockHash(20));
        let second_child = compute_next_seq_hash(11, LocalBlockHash(20));
        assert_ne!(first_child, second_child);
        assert_eq!(ledger.resident_on(first_child, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(second_child, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(20, WORKER_A), KnownBool::No);
    }

    #[test]
    fn cpu_suffix_store_chains_from_gpu_parent() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store_mapped(WORKER_A, CacheTier::Gpu, 0, 100, 10);

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Store {
                tier: CacheTier::Cpu,
                group_idx: Some(0),
                parent_external_hash: Some(100),
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 101,
                    tokens_hash: 20,
                }],
            }],
        });

        let child = compute_next_seq_hash(10, LocalBlockHash(20));
        assert_eq!(ledger.resident_on(child, WORKER_A), KnownBool::Yes);
        assert!(ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.stats().gpu_physical_blocks, 1);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);

        ledger.remove_external(WORKER_A, CacheTier::Cpu, 0, 101);
        assert_eq!(ledger.resident_on(10, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(child, WORKER_A), KnownBool::No);
    }

    #[test]
    fn store_only_batch_resolves_parent_emitted_later() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(100),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 101,
                        tokens_hash: 20,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: None,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 100,
                        tokens_hash: 10,
                    }],
                },
            ],
        });

        let child = compute_next_seq_hash(10, LocalBlockHash(20));
        assert_eq!(ledger.resident_on(10, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(child, WORKER_A), KnownBool::Yes);
        assert!(ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.stats().cpu_physical_blocks, 2);
    }

    #[test]
    fn store_only_batch_fails_closed_when_parent_never_arrives() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();

        let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Store {
                tier: CacheTier::Cpu,
                group_idx: Some(0),
                parent_external_hash: Some(100),
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 101,
                    tokens_hash: 20,
                }],
            }],
        });

        assert_eq!(
            result.failures().collect::<HashSet<_>>(),
            HashSet::from([CacheEvidenceApplyIntegrityFailure::MissingParent])
        );
        assert!(!ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.stats().cpu_physical_blocks, 0);
    }

    #[test]
    fn remove_is_a_barrier_before_an_out_of_order_store_run() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store_mapped(WORKER_A, CacheTier::Cpu, 0, 100, 10);

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![
                CacheEvidenceMutation::Remove {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    block_hashes: vec![100],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(200),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 201,
                        tokens_hash: 21,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: None,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 200,
                        tokens_hash: 20,
                    }],
                },
            ],
        });

        let child = compute_next_seq_hash(20, LocalBlockHash(21));
        assert_eq!(ledger.resident_on(10, WORKER_A), KnownBool::No);
        assert_eq!(ledger.resident_on(20, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(child, WORKER_A), KnownBool::Yes);
        assert!(ledger.stats().physical_telemetry_complete);
    }

    #[test]
    fn clear_separates_store_dependency_runs() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store_mapped(WORKER_A, CacheTier::Cpu, 0, 100, 10);

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(100),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 101,
                        tokens_hash: 11,
                    }],
                },
                CacheEvidenceMutation::Clear { tier: None },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(200),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 201,
                        tokens_hash: 21,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: None,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 200,
                        tokens_hash: 20,
                    }],
                },
            ],
        });

        let old_child = compute_next_seq_hash(10, LocalBlockHash(11));
        let new_child = compute_next_seq_hash(20, LocalBlockHash(21));
        assert_eq!(ledger.resident_on(10, WORKER_A), KnownBool::No);
        assert_eq!(ledger.resident_on(old_child, WORKER_A), KnownBool::No);
        assert_eq!(ledger.resident_on(20, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(new_child, WORKER_A), KnownBool::Yes);
        assert!(ledger.stats().physical_telemetry_complete);
    }

    #[test]
    fn store_run_resolves_transitive_dependencies_across_tiers() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Gpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(101),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 102,
                        tokens_hash: 30,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(100),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 101,
                        tokens_hash: 20,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Gpu,
                    group_idx: Some(0),
                    parent_external_hash: None,
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 100,
                        tokens_hash: 10,
                    }],
                },
            ],
        });

        let child = compute_next_seq_hash(10, LocalBlockHash(20));
        let grandchild = compute_next_seq_hash(child, LocalBlockHash(30));
        assert_eq!(ledger.resident_on(10, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(child, WORKER_A), KnownBool::Yes);
        assert_eq!(ledger.resident_on(grandchild, WORKER_A), KnownBool::Yes);
        assert!(ledger.stats().physical_telemetry_complete);
    }

    #[test]
    fn rejected_parent_store_does_not_unlock_dependents() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store_mapped(WORKER_A, CacheTier::Cpu, 0, 101, 999);

        let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: Some(100),
                    blocks: vec![CacheEvidenceStoredBlock {
                        external_hash: 102,
                        tokens_hash: 30,
                    }],
                },
                CacheEvidenceMutation::Store {
                    tier: CacheTier::Cpu,
                    group_idx: Some(0),
                    parent_external_hash: None,
                    blocks: vec![
                        CacheEvidenceStoredBlock {
                            external_hash: 100,
                            tokens_hash: 10,
                        },
                        CacheEvidenceStoredBlock {
                            external_hash: 101,
                            tokens_hash: 20,
                        },
                    ],
                },
            ],
        });

        assert_eq!(
            result.failures().collect::<HashSet<_>>(),
            HashSet::from([
                CacheEvidenceApplyIntegrityFailure::ConflictingMapping,
                CacheEvidenceApplyIntegrityFailure::MissingParent,
            ])
        );
        assert!(!ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);
        assert!(!ledger.external_copies.keys().any(|(hash, _)| *hash == 100));
        assert!(!ledger.external_copies.keys().any(|(hash, _)| *hash == 102));
    }

    #[test]
    fn cross_tier_parent_resolution_rejects_inconsistent_mappings() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.seal_physical_scope();
        ledger.store_mapped(WORKER_A, CacheTier::Gpu, 0, 100, 10);
        ledger.store_mapped(WORKER_A, CacheTier::Cpu, 0, 100, 11);

        let result = ledger.apply_evidence_batch_with_diagnostics(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Store {
                tier: CacheTier::Cpu,
                group_idx: Some(0),
                parent_external_hash: Some(100),
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 101,
                    tokens_hash: 20,
                }],
            }],
        });

        assert_eq!(
            result.failures().collect::<HashSet<_>>(),
            HashSet::from([CacheEvidenceApplyIntegrityFailure::MissingParent])
        );
        assert!(!ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.stats().gpu_physical_blocks, 1);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);
        assert!(!ledger.external_copies.keys().any(|(hash, _)| *hash == 101));
    }

    #[test]
    fn cross_tier_parent_resolution_preserves_group_isolation() {
        let mut ledger = CacheEvidenceLedger::new(16);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [1]);
        ledger.store_mapped(WORKER_A, CacheTier::Gpu, 0, 100, 10);

        ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: None,
            barrier_id: None,
            epoch_id: None,
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Store {
                tier: CacheTier::Cpu,
                group_idx: Some(1),
                parent_external_hash: Some(100),
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 101,
                    tokens_hash: 20,
                }],
            }],
        });

        assert!(!ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.stats().cpu_physical_blocks, 0);
    }

    #[test]
    fn sliding_window_hit_uses_the_latest_aligned_contiguous_window() {
        let group = CacheGroupHashSequence {
            group_idx: 1,
            kind: CacheGroupKind::SlidingWindow,
            block_size: 8,
            sliding_window: Some(16),
            is_eagle: false,
            alignment_tokens: 32,
            sequence_hashes: (1..=12).collect(),
        };
        let resident: HashSet<_> = [7, 8].into_iter().collect();
        assert_eq!(
            hybrid_prefix(&[group], 96, |_, hash| if resident.contains(&hash) {
                KnownBool::Yes
            } else {
                KnownBool::No
            }),
            KnownPrefixLength::Known(64)
        );
    }

    #[test]
    fn hybrid_hit_reconciles_full_and_sliding_groups() {
        let groups = [
            CacheGroupHashSequence {
                group_idx: 0,
                kind: CacheGroupKind::FullAttention,
                block_size: 32,
                sliding_window: None,
                is_eagle: false,
                alignment_tokens: 32,
                sequence_hashes: vec![10, 20, 30],
            },
            CacheGroupHashSequence {
                group_idx: 1,
                kind: CacheGroupKind::SlidingWindow,
                block_size: 8,
                sliding_window: Some(16),
                is_eagle: false,
                alignment_tokens: 32,
                sequence_hashes: (101..=112).collect(),
            },
        ];
        let resident: HashSet<_> = [10, 20, 107, 108, 111, 112].into_iter().collect();
        assert_eq!(
            hybrid_prefix(&groups, 96, |_, hash| if resident.contains(&hash) {
                KnownBool::Yes
            } else {
                KnownBool::No
            }),
            KnownPrefixLength::Known(64)
        );
    }

    #[test]
    fn unknown_hybrid_evidence_never_becomes_a_zero_hit() {
        let group = CacheGroupHashSequence {
            group_idx: 0,
            kind: CacheGroupKind::FullAttention,
            block_size: 32,
            sliding_window: None,
            is_eagle: false,
            alignment_tokens: 32,
            sequence_hashes: vec![1, 2],
        };
        assert_eq!(
            hybrid_prefix(&[group], 64, |_, hash| if hash == 1 {
                KnownBool::Yes
            } else {
                KnownBool::Unknown
            }),
            KnownPrefixLength::Unknown
        );
    }

    #[test]
    fn stream_integrity_loss_invalidates_history_until_atomic_cold_epoch_recovery() {
        let mut ledger = CacheEvidenceLedger::new(8);
        ledger.set_expected_owners([WORKER_A]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.advance_history_epoch();
        assert!(ledger.stats().history_complete);
        assert!(ledger.stats().physical_telemetry_complete);

        ledger.mark_physical_telemetry_incomplete();
        assert!(!ledger.stats().history_complete);
        assert!(!ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.reusable(99), KnownBool::Unknown);

        assert!(ledger.apply_evidence_batch(&CacheEvidenceBatch {
            owner: WORKER_A,
            source_cursor: 1,
            source_incarnation_id: Some(7),
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Clear {
                tier: Some(CacheTier::Gpu),
            }],
            barrier_id: None,
            epoch_id: Some("0123456789abcdef0123456789abcdef".to_string()),
        }));
        assert!(!ledger.stats().physical_telemetry_complete);

        assert!(
            ledger
                .commit_cold_history_epoch(&HashSet::from([WORKER_A]))
                .is_some()
        );
        assert!(ledger.stats().history_complete);
        assert!(ledger.stats().physical_telemetry_complete);
        assert_eq!(ledger.reusable(99), KnownBool::No);
    }

    #[test]
    fn stale_epoch_failures_cannot_poison_new_history_epoch() {
        let mut ledger = CacheEvidenceLedger::new(8);
        let old_epoch = ledger.history_epoch();
        let new_epoch = ledger.advance_history_epoch();

        assert_ne!(old_epoch, new_epoch);
        assert!(!ledger.mark_history_incomplete_for_epoch(old_epoch));
        assert!(ledger.stats().history_complete);
        assert!(ledger.mark_history_incomplete_for_epoch(new_epoch));
        assert!(!ledger.stats().history_complete);
    }

    #[test]
    fn partial_cold_epoch_cannot_clear_cumulative_poison_or_residual_tier() {
        let mut ledger = CacheEvidenceLedger::new(8);
        ledger.set_expected_owners([WORKER_A]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(WORKER_A, CacheTier::Cpu, [0]);
        ledger.store(WORKER_A, CacheTier::Cpu, 0, 99);
        ledger.mark_physical_telemetry_incomplete();
        ledger.clear_owner_tier(WORKER_A, CacheTier::Gpu);

        assert_eq!(
            ledger.commit_cold_history_epoch(&HashSet::from([WORKER_A])),
            None
        );
        assert!(!ledger.stats().physical_telemetry_complete);
        assert!(!ledger.stats().history_complete);
        assert_eq!(ledger.stats().cpu_physical_blocks, 1);
    }
}
