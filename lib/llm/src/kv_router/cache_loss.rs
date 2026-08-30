// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridges raw engine residency events into the cache-loss evidence ledger.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dynamo_kv_router::{
    cache_loss::{
        CacheEvidenceBatch, CacheEvidenceLedger, CacheEvidenceMutation, CacheGroupHashSequence,
        CacheGroupKind, CacheOwner, CacheTier, KV_CACHE_EVIDENCE_SUBJECT, KnownPrefixLength,
    },
    protocols::{BlockExtraInfo, StorageTier, TokensWithHashes, WorkerWithDpRank},
    zmq_wire::{KvEventOwnership, Locality, RawKvEvent, RawKvEventObserver},
};
use dynamo_runtime::{
    component::{Component, Instance},
    pipeline::PushRouter,
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
    transports::event_plane::EventSubscriber,
};
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::discovery::{KvSourceMembershipView, KvSourceMembershipWatch};
use crate::kv_router::metrics::CacheEvidenceMetrics;
use crate::protocols::common::timing::CacheGroupObservation;

pub const CACHE_LOSS_FUNNEL_ENABLED_ENV: &str = "DYN_CACHE_LOSS_FUNNEL_ENABLED";
const CACHE_LOSS_HISTORY_BLOCKS_ENV: &str = "DYN_CACHE_LOSS_HISTORY_BLOCKS";
const DEFAULT_CACHE_LOSS_HISTORY_BLOCKS: usize = 1_000_000;
const CACHE_LOSS_EVIDENCE_FRESHNESS_MS_ENV: &str = "DYN_CACHE_LOSS_EVIDENCE_FRESHNESS_MS";
const DEFAULT_CACHE_LOSS_EVIDENCE_FRESHNESS_MS: u64 = 5_000;
const CACHE_LOSS_BARRIER_TIMEOUT_MS_ENV: &str = "DYN_CACHE_LOSS_BARRIER_TIMEOUT_MS";
const DEFAULT_CACHE_LOSS_BARRIER_TIMEOUT_MS: u64 = 250;
const CACHE_LOSS_BARRIER_PENDING_ENV: &str = "DYN_CACHE_LOSS_BARRIER_PENDING";
const DEFAULT_CACHE_LOSS_BARRIER_PENDING: usize = 1_024;
const CACHE_LOSS_COLD_EPOCH_SINGLE_FRONTEND_ENV: &str = "DYN_CACHE_LOSS_COLD_EPOCH_SINGLE_FRONTEND";
const CACHE_LOSS_COLD_EPOCH_READINESS_MS_ENV: &str = "DYN_CACHE_LOSS_COLD_EPOCH_READINESS_MS";
const DEFAULT_COLD_EPOCH_READINESS_MS: u64 = 120_000;
const COLD_EPOCH_LEASE_MS: u64 = 30_000;
const COLD_EPOCH_BEGIN_TIMEOUT: Duration = Duration::from_secs(12);
const COLD_EPOCH_RELEASE_TIMEOUT: Duration = Duration::from_secs(1);

struct DispatchGateRelease {
    sender: watch::Sender<bool>,
    ledger: Arc<Mutex<CacheEvidenceLedger>>,
    committed: bool,
}

impl Drop for DispatchGateRelease {
    fn drop(&mut self) {
        if !self.committed {
            self.ledger.lock().mark_history_incomplete();
        }
        self.sender.send_replace(true);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BarrierOutcome {
    Pending,
    Exact,
    Incomplete(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BarrierControlDisposition {
    Success,
    PermanentlyUnavailable,
}

#[derive(Clone, Debug)]
pub struct CacheEvidenceBarrierTicket {
    outcome: watch::Receiver<BarrierOutcome>,
}

impl CacheEvidenceBarrierTicket {
    fn outcome(&self) -> BarrierOutcome {
        *self.outcome.borrow()
    }

    async fn wait(&mut self) -> BarrierOutcome {
        while self.outcome() == BarrierOutcome::Pending {
            if self.outcome.changed().await.is_err() {
                return BarrierOutcome::Incomplete("barrier_coordinator_closed");
            }
        }
        self.outcome()
    }
}

struct PendingCut {
    relevant_hashes: HashSet<u64>,
    unmarked_owners: HashSet<CacheOwner>,
    outcome: watch::Sender<BarrierOutcome>,
}

struct BarrierRound {
    owners: HashMap<CacheOwner, u64>,
    controls_pending: HashSet<CacheOwner>,
    cuts: Vec<PendingCut>,
    started_at: Instant,
}

#[derive(Default)]
struct BarrierCoordinator {
    owners: HashMap<CacheOwner, u64>,
    serving_incarnations: HashMap<CacheOwner, u64>,
    permanently_unavailable: HashMap<CacheOwner, u64>,
    rounds: HashMap<u64, BarrierRound>,
    open_round: Option<u64>,
    next_id: u64,
    pending: usize,
    max_pending: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColdEpochOutcome {
    Pending,
    EvidenceReady,
    Incomplete(&'static str),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ColdEpochOwnerFence {
    evidence_incarnation: u64,
    serving_incarnation: u64,
    expected_tiers: HashSet<CacheTier>,
}

struct ColdEpochOwnerProgress {
    fence: ColdEpochOwnerFence,
    clear_cursor: Option<u64>,
    marker_seen: bool,
}

struct ColdEpochRound {
    epoch_id: String,
    barrier_id: u64,
    owners: HashMap<CacheOwner, ColdEpochOwnerProgress>,
    controls_succeeded: bool,
    outcome: watch::Sender<ColdEpochOutcome>,
}

#[derive(Default)]
struct ColdEpochCoordinator {
    active: Option<ColdEpochRound>,
}

impl ColdEpochCoordinator {
    fn begin(
        &mut self,
        epoch_id: String,
        barrier_id: u64,
        owners: HashMap<CacheOwner, ColdEpochOwnerFence>,
    ) -> anyhow::Result<watch::Receiver<ColdEpochOutcome>> {
        anyhow::ensure!(!owners.is_empty(), "cold epoch has no eligible owners");
        anyhow::ensure!(self.active.is_none(), "cold epoch is already active");
        let (outcome, receiver) = watch::channel(ColdEpochOutcome::Pending);
        self.active = Some(ColdEpochRound {
            epoch_id,
            barrier_id,
            owners: owners
                .into_iter()
                .map(|(owner, fence)| {
                    (
                        owner,
                        ColdEpochOwnerProgress {
                            fence,
                            clear_cursor: None,
                            marker_seen: false,
                        },
                    )
                })
                .collect(),
            controls_succeeded: false,
            outcome,
        });
        Ok(receiver)
    }

    fn fail(&mut self, reason: &'static str) {
        if let Some(round) = self.active.take() {
            round
                .outcome
                .send_replace(ColdEpochOutcome::Incomplete(reason));
        }
    }

    fn controls_succeeded(&mut self, epoch_id: &str) {
        let Some(round) = self
            .active
            .as_mut()
            .filter(|round| round.epoch_id == epoch_id)
        else {
            return;
        };
        round.controls_succeeded = true;
        Self::finish_if_ready(round);
    }

    fn accepts_clear_baseline(&self, batch: &CacheEvidenceBatch) -> bool {
        let Some(epoch_id) = batch.epoch_id.as_deref() else {
            return false;
        };
        let Some(round) = self
            .active
            .as_ref()
            .filter(|round| round.epoch_id == epoch_id)
        else {
            return false;
        };
        let Some(progress) = round.owners.get(&batch.owner) else {
            return false;
        };
        if batch.barrier_id.is_some()
            || !batch.telemetry_complete
            || batch.source_incarnation_id != Some(progress.fence.evidence_incarnation)
            || progress.clear_cursor.is_some()
            || batch.mutations.len() != progress.fence.expected_tiers.len()
        {
            return false;
        }
        let tiers: HashSet<_> = batch
            .mutations
            .iter()
            .filter_map(|mutation| match mutation {
                CacheEvidenceMutation::Clear { tier: Some(tier) } => Some(*tier),
                _ => None,
            })
            .collect();
        tiers == progress.fence.expected_tiers && tiers.len() == batch.mutations.len()
    }

    fn observe_clear_batch(&mut self, batch: &CacheEvidenceBatch) -> bool {
        let Some(epoch_id) = batch.epoch_id.as_deref() else {
            return false;
        };
        let Some(round) = self.active.as_mut() else {
            return true;
        };
        if round.epoch_id != epoch_id || batch.barrier_id.is_some() || !batch.telemetry_complete {
            self.fail("cold_epoch_invalid_clear");
            return true;
        }
        let Some(progress) = round.owners.get_mut(&batch.owner) else {
            self.fail("cold_epoch_membership_changed");
            return true;
        };
        if batch.source_incarnation_id != Some(progress.fence.evidence_incarnation)
            || progress.clear_cursor.is_some()
            || batch.mutations.len() != progress.fence.expected_tiers.len()
        {
            self.fail("cold_epoch_invalid_clear");
            return true;
        }
        let tiers: HashSet<_> = batch
            .mutations
            .iter()
            .filter_map(|mutation| match mutation {
                CacheEvidenceMutation::Clear { tier: Some(tier) } => Some(*tier),
                _ => None,
            })
            .collect();
        if tiers != progress.fence.expected_tiers || tiers.len() != batch.mutations.len() {
            self.fail("cold_epoch_invalid_clear");
            return true;
        }
        progress.clear_cursor = Some(batch.source_cursor);
        true
    }

    fn observe_marker(&mut self, batch: &CacheEvidenceBatch) -> bool {
        let Some(epoch_id) = batch.epoch_id.as_deref() else {
            return false;
        };
        let Some(round) = self.active.as_mut() else {
            return true;
        };
        if round.epoch_id != epoch_id
            || batch.barrier_id != Some(round.barrier_id)
            || !batch.mutations.is_empty()
            || !batch.telemetry_complete
        {
            self.fail("cold_epoch_invalid_marker");
            return true;
        }
        let Some(progress) = round.owners.get_mut(&batch.owner) else {
            self.fail("cold_epoch_membership_changed");
            return true;
        };
        if batch.source_incarnation_id != Some(progress.fence.evidence_incarnation)
            || progress
                .clear_cursor
                .and_then(|cursor| cursor.checked_add(1))
                != Some(batch.source_cursor)
            || progress.marker_seen
        {
            self.fail("cold_epoch_invalid_marker");
            return true;
        }
        progress.marker_seen = true;
        Self::finish_if_ready(round);
        true
    }

    fn unexpected_mutation(&mut self, owner: CacheOwner) {
        if self.active.as_ref().is_some_and(|round| {
            round
                .owners
                .get(&owner)
                .is_some_and(|progress| progress.clear_cursor.is_some() && !progress.marker_seen)
        }) {
            self.fail("cold_epoch_mutation_after_clear");
        }
    }

    fn finish_if_ready(round: &mut ColdEpochRound) {
        if round.controls_succeeded
            && round
                .owners
                .values()
                .all(|progress| progress.clear_cursor.is_some() && progress.marker_seen)
        {
            round.outcome.send_replace(ColdEpochOutcome::EvidenceReady);
        }
    }

    fn complete(&mut self, epoch_id: &str) {
        if self
            .active
            .as_ref()
            .is_some_and(|round| round.epoch_id == epoch_id)
        {
            self.active = None;
        }
    }
}

impl BarrierCoordinator {
    fn new(max_pending: usize) -> Self {
        Self {
            // Barrier ids share the workers' event stream across frontend replicas.
            // A random starting point prevents one replica's marker satisfying
            // another replica's round after simultaneous startup.
            next_id: rand::random::<u64>().max(1),
            max_pending,
            ..Default::default()
        }
    }

    #[cfg(test)]
    fn set_owners(&mut self, owners: HashMap<CacheOwner, u64>) {
        let serving_incarnations = owners.clone();
        self.set_owner_incarnations(owners, serving_incarnations);
    }

    fn set_owner_incarnations(
        &mut self,
        owners: HashMap<CacheOwner, u64>,
        serving_incarnations: HashMap<CacheOwner, u64>,
    ) {
        if self.owners != owners || self.serving_incarnations != serving_incarnations {
            self.fail_all("barrier_membership_changed");
            self.owners = owners;
            self.serving_incarnations = serving_incarnations;
            self.permanently_unavailable.retain(|owner, incarnation| {
                self.owners
                    .get(owner)
                    .is_none_or(|active| active == incarnation)
            });
        }
    }

    fn begin(
        &mut self,
        relevant_hashes: HashSet<u64>,
    ) -> (CacheEvidenceBarrierTicket, Option<u64>) {
        let (tx, rx) = watch::channel(BarrierOutcome::Pending);
        let ticket = CacheEvidenceBarrierTicket { outcome: rx };
        if self.owners.is_empty()
            || self
                .permanently_unavailable
                .iter()
                .any(|(owner, incarnation)| self.owners.get(owner) == Some(incarnation))
        {
            tx.send_replace(BarrierOutcome::Incomplete("barrier_missing_capability"));
            return (ticket, None);
        }
        if self.pending >= self.max_pending {
            tx.send_replace(BarrierOutcome::Incomplete("barrier_journal_overflow"));
            return (ticket, None);
        }

        let mut created = None;
        let id = self.open_round.unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1).max(1);
            self.rounds.insert(
                id,
                BarrierRound {
                    owners: self.owners.clone(),
                    controls_pending: self.owners.keys().copied().collect(),
                    cuts: Vec::new(),
                    started_at: Instant::now(),
                },
            );
            self.open_round = Some(id);
            created = Some(id);
            id
        });
        let round = self.rounds.get_mut(&id).expect("open barrier round exists");
        round.cuts.push(PendingCut {
            relevant_hashes,
            unmarked_owners: round.owners.keys().copied().collect(),
            outcome: tx,
        });
        self.pending += 1;
        (ticket, created)
    }

    fn matches_selected_incarnation(
        &self,
        owner: CacheOwner,
        serving_incarnation: Option<u64>,
    ) -> bool {
        serving_incarnation.is_some()
            && self.serving_incarnations.get(&owner).copied() == serving_incarnation
    }

    fn mark_permanently_unavailable(&mut self, owner: CacheOwner, incarnation: u64) -> usize {
        if self.owners.get(&owner) != Some(&incarnation) {
            return 0;
        }
        self.permanently_unavailable.insert(owner, incarnation);
        self.fail_owner(owner, "barrier_permanently_unavailable")
    }

    fn dispatch(&mut self, id: u64) -> Option<(HashMap<CacheOwner, u64>, usize)> {
        let round = self.rounds.get_mut(&id)?;
        if self.open_round == Some(id) {
            self.open_round = None;
        }
        Some((round.owners.clone(), round.cuts.len()))
    }

    fn marker(&mut self, owner: CacheOwner, barrier_id: u64) -> Option<Duration> {
        let mut remove_round = false;
        let mut rtt = None;
        if let Some(round) = self.rounds.get_mut(&barrier_id) {
            if !round.owners.contains_key(&owner) {
                return None;
            }
            for cut in &mut round.cuts {
                if cut.outcome.borrow().eq(&BarrierOutcome::Pending) {
                    cut.unmarked_owners.remove(&owner);
                    if cut.unmarked_owners.is_empty() && round.controls_pending.is_empty() {
                        cut.outcome.send_replace(BarrierOutcome::Exact);
                        self.pending = self.pending.saturating_sub(1);
                    }
                }
            }
            remove_round = round
                .cuts
                .iter()
                .all(|cut| *cut.outcome.borrow() != BarrierOutcome::Pending);
            if remove_round {
                rtt = Some(round.started_at.elapsed());
            }
        }
        if remove_round {
            self.rounds.remove(&barrier_id);
        }
        rtt
    }

    fn controls_succeeded(&mut self, barrier_id: u64) -> Option<Duration> {
        let round = self.rounds.get_mut(&barrier_id)?;
        round.controls_pending.clear();
        for cut in &mut round.cuts {
            if *cut.outcome.borrow() == BarrierOutcome::Pending && cut.unmarked_owners.is_empty() {
                cut.outcome.send_replace(BarrierOutcome::Exact);
                self.pending = self.pending.saturating_sub(1);
            }
        }
        if round
            .cuts
            .iter()
            .all(|cut| *cut.outcome.borrow() != BarrierOutcome::Pending)
        {
            let rtt = round.started_at.elapsed();
            self.rounds.remove(&barrier_id);
            Some(rtt)
        } else {
            None
        }
    }

    fn mutation(&mut self, owner: CacheOwner, hashes: &HashSet<u64>, clear: bool) -> usize {
        let mut completed = Vec::new();
        let mut failed = 0;
        for (&id, round) in &mut self.rounds {
            for cut in &mut round.cuts {
                if *cut.outcome.borrow() != BarrierOutcome::Pending
                    || !cut.unmarked_owners.contains(&owner)
                {
                    continue;
                }
                if clear || !cut.relevant_hashes.is_disjoint(hashes) {
                    cut.outcome
                        .send_replace(BarrierOutcome::Incomplete(if clear {
                            "barrier_clear"
                        } else {
                            "barrier_relevant_mutation"
                        }));
                    self.pending = self.pending.saturating_sub(1);
                    failed += 1;
                }
            }
            if round
                .cuts
                .iter()
                .all(|cut| *cut.outcome.borrow() != BarrierOutcome::Pending)
            {
                completed.push(id);
            }
        }
        for id in completed {
            self.rounds.remove(&id);
            if self.open_round == Some(id) {
                self.open_round = None;
            }
        }
        failed
    }

    fn fail_owner(&mut self, owner: CacheOwner, reason: &'static str) -> usize {
        let mut completed = Vec::new();
        let mut failed = 0;
        for (&id, round) in &mut self.rounds {
            for cut in &mut round.cuts {
                if *cut.outcome.borrow() == BarrierOutcome::Pending
                    && cut.unmarked_owners.contains(&owner)
                {
                    cut.outcome.send_replace(BarrierOutcome::Incomplete(reason));
                    self.pending = self.pending.saturating_sub(1);
                    failed += 1;
                }
            }
            if round
                .cuts
                .iter()
                .all(|cut| *cut.outcome.borrow() != BarrierOutcome::Pending)
            {
                completed.push(id);
            }
        }
        for id in completed {
            self.rounds.remove(&id);
            if self.open_round == Some(id) {
                self.open_round = None;
            }
        }
        failed
    }

    fn fail_round(&mut self, id: u64, reason: &'static str) -> usize {
        let Some(round) = self.rounds.remove(&id) else {
            return 0;
        };
        let mut failed = 0;
        for cut in round.cuts {
            if *cut.outcome.borrow() == BarrierOutcome::Pending {
                cut.outcome.send_replace(BarrierOutcome::Incomplete(reason));
                self.pending = self.pending.saturating_sub(1);
                failed += 1;
            }
        }
        if self.open_round == Some(id) {
            self.open_round = None;
        }
        failed
    }

    fn fail_all(&mut self, reason: &'static str) -> usize {
        let ids: Vec<_> = self.rounds.keys().copied().collect();
        let mut failed = 0;
        for id in ids {
            failed += self.fail_round(id, reason);
        }
        failed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceSequenceObservation {
    Contiguous,
    Gap,
    Stale,
    PublisherRestart,
}

#[derive(Default)]
struct EvidenceSequenceTracker {
    publishers: HashMap<u64, u64>,
    owner_publishers: HashMap<CacheOwner, u64>,
    retired_publishers: HashMap<CacheOwner, HashSet<u64>>,
}

impl EvidenceSequenceTracker {
    #[cfg(test)]
    fn observe(
        &mut self,
        owner: CacheOwner,
        publisher_id: u64,
        source_cursor: u64,
    ) -> EvidenceSequenceObservation {
        self.observe_with_initial_baseline(owner, publisher_id, source_cursor, false)
    }

    fn observe_with_initial_baseline(
        &mut self,
        owner: CacheOwner,
        publisher_id: u64,
        source_cursor: u64,
        allow_initial_nonzero: bool,
    ) -> EvidenceSequenceObservation {
        let previous_publisher = self.owner_publishers.get(&owner).copied();
        let publisher_restarted =
            previous_publisher.is_some_and(|previous| previous != publisher_id);
        if self
            .retired_publishers
            .get(&owner)
            .is_some_and(|retired| retired.contains(&publisher_id))
        {
            return EvidenceSequenceObservation::Stale;
        }
        let sequence_observation = match self.publishers.get(&publisher_id).copied() {
            Some(last) => {
                if source_cursor <= last {
                    EvidenceSequenceObservation::Stale
                } else if source_cursor == last.saturating_add(1) {
                    EvidenceSequenceObservation::Contiguous
                } else {
                    EvidenceSequenceObservation::Gap
                }
            }
            None if source_cursor != 0 && !allow_initial_nonzero => {
                EvidenceSequenceObservation::Gap
            }
            None => EvidenceSequenceObservation::Contiguous,
        };
        if sequence_observation == EvidenceSequenceObservation::Stale {
            return sequence_observation;
        }
        self.publishers.insert(publisher_id, source_cursor);
        self.owner_publishers.insert(owner, publisher_id);
        if publisher_restarted {
            self.retired_publishers.entry(owner).or_default().insert(
                previous_publisher.expect("publisher restart requires a previous publisher"),
            );
            EvidenceSequenceObservation::PublisherRestart
        } else {
            sequence_observation
        }
    }

    fn owners(&self) -> impl Iterator<Item = CacheOwner> + '_ {
        self.owner_publishers.keys().copied()
    }

    fn last_cursor(&self, publisher_id: u64) -> Option<u64> {
        self.publishers.get(&publisher_id).copied()
    }

    fn retire(&mut self, owner: CacheOwner) {
        if let Some(publisher_id) = self.owner_publishers.remove(&owner) {
            self.retired_publishers
                .entry(owner)
                .or_default()
                .insert(publisher_id);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceFreshnessStatus {
    BoundedFresh,
    Missing,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceWatermarkObservation {
    Current,
    Stale,
    Ahead,
}

fn observe_watermark(
    applied_cursor: Option<u64>,
    watermark_cursor: Option<u64>,
) -> EvidenceWatermarkObservation {
    match (applied_cursor, watermark_cursor) {
        (None, None) => EvidenceWatermarkObservation::Current,
        (Some(applied), Some(watermark)) if applied == watermark => {
            EvidenceWatermarkObservation::Current
        }
        (Some(applied), Some(watermark)) if watermark < applied => {
            EvidenceWatermarkObservation::Stale
        }
        _ => EvidenceWatermarkObservation::Ahead,
    }
}

#[derive(Clone, Copy)]
struct EvidenceWatermark {
    incarnation_id: u64,
    received_at: Instant,
}

struct EvidenceFreshness {
    expected: HashMap<CacheOwner, Option<u64>>,
    watermarks: HashMap<CacheOwner, EvidenceWatermark>,
    max_age: Duration,
}

impl EvidenceFreshness {
    fn new(max_age: Duration) -> Self {
        Self {
            expected: HashMap::new(),
            watermarks: HashMap::new(),
            max_age,
        }
    }

    fn set_expected(&mut self, expected: HashMap<CacheOwner, Option<u64>>) -> Vec<CacheOwner> {
        let changed = self
            .expected
            .iter()
            .filter_map(|(&owner, incarnation)| {
                (expected.get(&owner) != Some(incarnation)).then_some(owner)
            })
            .collect();
        self.watermarks.retain(|owner, watermark| {
            expected.get(owner).copied().flatten() == Some(watermark.incarnation_id)
        });
        self.expected = expected;
        changed
    }

    fn accepts(&self, owner: CacheOwner, incarnation_id: u64) -> bool {
        self.expected.get(&owner).copied().flatten() == Some(incarnation_id)
    }

    fn observe(&mut self, owner: CacheOwner, incarnation_id: u64, now: Instant) -> bool {
        if !self.accepts(owner, incarnation_id) {
            return false;
        }
        self.watermarks.insert(
            owner,
            EvidenceWatermark {
                incarnation_id,
                received_at: now,
            },
        );
        true
    }

    fn status(&self, now: Instant) -> (EvidenceFreshnessStatus, usize, Option<Duration>) {
        if self.expected.is_empty() || self.expected.values().any(Option::is_none) {
            return (EvidenceFreshnessStatus::Missing, 0, None);
        }
        let mut fresh = 0;
        let mut max_age = Duration::ZERO;
        let mut missing = false;
        for (&owner, &incarnation_id) in &self.expected {
            let Some(watermark) = self.watermarks.get(&owner) else {
                missing = true;
                continue;
            };
            if Some(watermark.incarnation_id) != incarnation_id {
                missing = true;
                continue;
            }
            let age = now.saturating_duration_since(watermark.received_at);
            max_age = max_age.max(age);
            if age <= self.max_age {
                fresh += 1;
            }
        }
        if missing {
            (EvidenceFreshnessStatus::Missing, fresh, None)
        } else if fresh == self.expected.len() {
            (EvidenceFreshnessStatus::BoundedFresh, fresh, Some(max_age))
        } else {
            (EvidenceFreshnessStatus::Stale, fresh, Some(max_age))
        }
    }
}

pub fn enabled() -> bool {
    std::env::var(CACHE_LOSS_FUNNEL_ENABLED_ENV)
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

fn history_capacity() -> usize {
    std::env::var(CACHE_LOSS_HISTORY_BLOCKS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_CACHE_LOSS_HISTORY_BLOCKS)
}

fn evidence_freshness() -> Duration {
    Duration::from_millis(
        std::env::var(CACHE_LOSS_EVIDENCE_FRESHNESS_MS_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_CACHE_LOSS_EVIDENCE_FRESHNESS_MS),
    )
}

fn barrier_timeout() -> Duration {
    Duration::from_millis(
        std::env::var(CACHE_LOSS_BARRIER_TIMEOUT_MS_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_CACHE_LOSS_BARRIER_TIMEOUT_MS),
    )
}

fn barrier_pending_capacity() -> usize {
    std::env::var(CACHE_LOSS_BARRIER_PENDING_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_CACHE_LOSS_BARRIER_PENDING)
}

fn cold_epoch_readiness_timeout() -> Duration {
    Duration::from_millis(
        std::env::var(CACHE_LOSS_COLD_EPOCH_READINESS_MS_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_COLD_EPOCH_READINESS_MS),
    )
}

fn update_freshness_metrics(
    freshness: &Mutex<EvidenceFreshness>,
    metrics: &CacheEvidenceMetrics,
) -> EvidenceFreshnessStatus {
    let freshness = freshness.lock();
    let (status, fresh_owners, max_age) = freshness.status(Instant::now());
    metrics.update_freshness(freshness.expected.len(), fresh_owners, max_age);
    status
}

fn barrier_owners(
    view: &crate::discovery::KvSourceMembershipView,
) -> (HashMap<CacheOwner, u64>, HashMap<CacheOwner, u64>) {
    let mut owners = HashMap::new();
    let mut serving_incarnations = HashMap::new();
    for (worker, status) in &view.sources {
        let Some(source) = status.active_source() else {
            return (HashMap::new(), HashMap::new());
        };
        if view.cache_evidence_barrier_enabled(worker.worker_id) != Some(true) {
            return (HashMap::new(), HashMap::new());
        }
        let Some(incarnation) = source.evidence_incarnation_id else {
            return (HashMap::new(), HashMap::new());
        };
        let Some(serving_incarnation) = view
            .cache_evidence_serving_incarnations
            .get(worker)
            .copied()
            .flatten()
        else {
            return (HashMap::new(), HashMap::new());
        };
        let owner = CacheOwner {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
        };
        owners.insert(owner, incarnation);
        serving_incarnations.insert(owner, serving_incarnation);
    }
    (owners, serving_incarnations)
}

fn cold_epoch_owners(
    view: &crate::discovery::KvSourceMembershipView,
) -> Option<HashMap<CacheOwner, ColdEpochOwnerFence>> {
    let mut owners = HashMap::new();
    for (worker, status) in &view.sources {
        if view.cache_evidence_epoch_enabled.get(&worker.worker_id) != Some(&Some(true)) {
            return None;
        }
        let evidence_incarnation = status.active_source()?.evidence_incarnation_id?;
        let serving_incarnation = view
            .cache_evidence_serving_incarnations
            .get(worker)
            .copied()
            .flatten()?;
        let expected_tiers = view
            .cache_evidence_epoch_media
            .get(worker)
            .and_then(Option::as_ref)?
            .iter()
            .map(|medium| match medium.as_str() {
                "GPU" => Some(CacheTier::Gpu),
                "CPU" => Some(CacheTier::Cpu),
                _ => None,
            })
            .collect::<Option<HashSet<_>>>()?;
        if expected_tiers.is_empty() || !expected_tiers.contains(&CacheTier::Gpu) {
            return None;
        }
        owners.insert(
            CacheOwner {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
            },
            ColdEpochOwnerFence {
                evidence_incarnation,
                serving_incarnation,
                expected_tiers,
            },
        );
    }
    (!owners.is_empty()).then_some(owners)
}

fn reconcile_history_membership(
    previous: &mut Option<HashMap<CacheOwner, ColdEpochOwnerFence>>,
    current: Option<HashMap<CacheOwner, ColdEpochOwnerFence>>,
    ledger: &mut CacheEvidenceLedger,
) -> bool {
    if *previous == current {
        return false;
    }
    *previous = current;
    ledger.mark_history_incomplete();
    true
}

pub struct CacheEvidenceSubscription {
    ledger: Arc<Mutex<CacheEvidenceLedger>>,
    group_shapes: Arc<Mutex<HashMap<CacheOwner, Vec<CacheGroupObservation>>>>,
    freshness: Arc<Mutex<EvidenceFreshness>>,
    metrics: Arc<CacheEvidenceMetrics>,
    barriers: Arc<Mutex<BarrierCoordinator>>,
    cold_epoch: Arc<Mutex<ColdEpochCoordinator>>,
    dispatch_gate: watch::Receiver<bool>,
    dispatch_fence: Arc<tokio::sync::RwLock<()>>,
    barrier_tx: mpsc::UnboundedSender<u64>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
    barrier_task: JoinHandle<()>,
    cold_epoch_task: Option<JoinHandle<()>>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum CacheEvidenceQuality {
    /// No complete physical-cache attribution is available.
    #[default]
    Incomplete,
    /// Physical state is bounded by recent ordered watermarks, without a route-time barrier.
    BoundedPhysical,
    /// Physical state is proven through an ordered route-time barrier.
    Exact,
}

impl CacheEvidenceQuality {
    pub(crate) fn metric_label(&self) -> &'static str {
        match self {
            Self::Incomplete => "incomplete",
            Self::BoundedPhysical => "bounded_physical",
            Self::Exact => "exact",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CacheLossRouteObservation {
    pub prompt_tokens: u64,
    pub reusable_prefix_tokens: Option<u64>,
    pub physical_prefix_tokens: Option<u64>,
    pub router_visible_prefix_tokens: Option<u64>,
    pub selected_physical_prefix_tokens: Option<u64>,
    pub group_hashes: Option<Vec<CacheGroupHashSequence>>,
    pub quality: CacheEvidenceQuality,
    pub complete: bool,
    pub incomplete_reason: Option<&'static str>,
    barrier: Option<CacheEvidenceBarrierTicket>,
}

impl CacheLossRouteObservation {
    /// Apply an already resolved barrier without awaiting. Returns false only
    /// while a barrier is still pending, so streaming metrics cannot consume a
    /// provisional bounded observation after its exactness result is ready.
    pub(crate) fn finalize_barrier_if_ready(&mut self) -> bool {
        let Some(ticket) = self.barrier.as_ref() else {
            return true;
        };
        let outcome = ticket.outcome();
        if outcome == BarrierOutcome::Pending {
            return false;
        }
        self.barrier.take();
        self.apply_barrier_outcome(outcome);
        true
    }

    pub(crate) async fn finalize_barrier(&mut self) {
        if self.finalize_barrier_if_ready() {
            return;
        }
        let Some(mut ticket) = self.barrier.take() else {
            return;
        };
        let outcome = ticket.wait().await;
        self.apply_barrier_outcome(outcome);
    }

    fn apply_barrier_outcome(&mut self, outcome: BarrierOutcome) {
        match outcome {
            BarrierOutcome::Exact => self.quality = CacheEvidenceQuality::Exact,
            BarrierOutcome::Incomplete(reason) => {
                self.complete = false;
                self.quality = CacheEvidenceQuality::Incomplete;
                self.incomplete_reason = Some(reason);
            }
            BarrierOutcome::Pending => unreachable!("barrier wait returned pending"),
        }
    }
}

impl CacheEvidenceSubscription {
    pub fn ledger(&self) -> Arc<Mutex<CacheEvidenceLedger>> {
        Arc::clone(&self.ledger)
    }

    pub async fn wait_for_dispatch_gate(&self) -> anyhow::Result<()> {
        let mut gate = self.dispatch_gate.clone();
        while !*gate.borrow() {
            gate.changed()
                .await
                .map_err(|_| anyhow::anyhow!("cache history epoch dispatch gate closed"))?;
        }
        Ok(())
    }

    pub async fn acquire_dispatch_fence(&self) -> tokio::sync::OwnedRwLockReadGuard<()> {
        Arc::clone(&self.dispatch_fence).read_owned().await
    }

    pub fn record_group_catalog(
        &self,
        owner: CacheOwner,
        groups: &[CacheGroupObservation],
    ) -> bool {
        record_group_catalog(
            owner,
            groups,
            &self.group_shapes,
            &self.ledger,
            &self.barriers,
            &self.cold_epoch,
            Some(&self.metrics),
        )
    }

    pub fn group_catalog(&self, owner: CacheOwner) -> Option<Vec<CacheGroupObservation>> {
        self.group_shapes.lock().get(&owner).cloned()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn snapshot_route(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
        selected: WorkerWithDpRank,
        selected_serving_incarnation: Option<u64>,
        router_visible_prefix_tokens: u64,
    ) -> CacheLossRouteObservation {
        let prompt_tokens = tokens.len() as u64;
        let max_hit_tokens = prompt_tokens.saturating_sub(1);
        let owner = CacheOwner {
            worker_id: selected.worker_id,
            dp_rank: selected.dp_rank,
        };
        // Serialize the catalog, ledger snapshot, and cut registration with evidence
        // application so a concurrent catalog change cannot escape invalidation.
        let mut barriers = self.barriers.lock();
        let expected_owners: HashSet<_> = barriers.owners.keys().copied().collect();
        if !barriers.matches_selected_incarnation(owner, selected_serving_incarnation) {
            return CacheLossRouteObservation {
                prompt_tokens,
                incomplete_reason: Some("barrier_membership_mismatch"),
                ..Default::default()
            };
        }
        let catalogs = self.group_shapes.lock();
        if catalogs.keys().copied().collect::<HashSet<_>>() != expected_owners {
            return CacheLossRouteObservation {
                prompt_tokens,
                incomplete_reason: Some("barrier_catalog_membership_mismatch"),
                ..Default::default()
            };
        }
        let Some(catalog) = catalogs.get(&owner).cloned() else {
            return CacheLossRouteObservation {
                prompt_tokens,
                incomplete_reason: Some("missing_cache_group_catalog"),
                ..Default::default()
            };
        };
        drop(catalogs);
        if block_mm_infos.is_some() {
            return CacheLossRouteObservation {
                prompt_tokens,
                incomplete_reason: Some("hybrid_multimodal_hashing_not_supported"),
                ..Default::default()
            };
        }
        let Some(group_hashes) = cache_group_hashes(tokens, &catalog, lora_name, cache_namespace)
        else {
            return CacheLossRouteObservation {
                prompt_tokens,
                incomplete_reason: Some("unsupported_cache_group_shape"),
                ..Default::default()
            };
        };

        let freshness = update_freshness_metrics(&self.freshness, &self.metrics);
        if freshness != EvidenceFreshnessStatus::BoundedFresh {
            return CacheLossRouteObservation {
                prompt_tokens,
                router_visible_prefix_tokens: Some(
                    router_visible_prefix_tokens.min(max_hit_tokens),
                ),
                group_hashes: Some(group_hashes),
                incomplete_reason: Some(match freshness {
                    EvidenceFreshnessStatus::Missing => "missing_cache_evidence_watermark",
                    EvidenceFreshnessStatus::Stale => "stale_cache_evidence_watermark",
                    EvidenceFreshnessStatus::BoundedFresh => unreachable!(),
                }),
                ..Default::default()
            };
        }

        let ledger = self.ledger.lock();
        if !ledger.expected_owners_match(&expected_owners) {
            return CacheLossRouteObservation {
                prompt_tokens,
                incomplete_reason: Some("barrier_ledger_membership_mismatch"),
                ..Default::default()
            };
        }
        let reusable = ledger.reusable_prefix(&group_hashes, max_hit_tokens);
        let physical = ledger.resident_prefix_anywhere(&group_hashes, max_hit_tokens);
        let selected_physical = ledger.resident_prefix_on(&group_hashes, max_hit_tokens, owner);
        self.metrics.update_state(ledger.stats());
        let (
            KnownPrefixLength::Known(reusable),
            KnownPrefixLength::Known(physical),
            KnownPrefixLength::Known(selected_physical),
        ) = (reusable, physical, selected_physical)
        else {
            return CacheLossRouteObservation {
                prompt_tokens,
                router_visible_prefix_tokens: Some(
                    router_visible_prefix_tokens.min(max_hit_tokens),
                ),
                group_hashes: Some(group_hashes),
                incomplete_reason: Some("incomplete_history_or_residency_evidence"),
                ..Default::default()
            };
        };
        // A recent ordered watermark bounds subscriber lag, but it is not a route-time
        // barrier. A mutation may occur after the watermark and remain queued while this
        // snapshot is taken. Emit a separately labeled operational funnel; only a future
        // ordered route-time cut may use CacheEvidenceQuality::Exact.
        let relevant_hashes = group_hashes
            .iter()
            .flat_map(|group| group.sequence_hashes.iter().copied())
            .collect();
        let (barrier, command) = barriers.begin(relevant_hashes);
        match barrier.outcome() {
            BarrierOutcome::Incomplete("barrier_missing_capability") => {
                self.metrics
                    .observe_barrier_incomplete("missing_capability");
            }
            BarrierOutcome::Incomplete("barrier_journal_overflow") => {
                self.metrics.observe_barrier_incomplete("journal_overflow");
            }
            _ => {}
        }
        self.metrics.update_barrier_pending(barriers.pending);
        if let Some(barrier_id) = command
            && self.barrier_tx.send(barrier_id).is_err()
        {
            barriers.fail_round(barrier_id, "barrier_coordinator_closed");
            self.metrics.update_barrier_pending(barriers.pending);
        }
        CacheLossRouteObservation {
            prompt_tokens,
            reusable_prefix_tokens: Some(reusable),
            physical_prefix_tokens: Some(physical),
            router_visible_prefix_tokens: Some(router_visible_prefix_tokens.min(max_hit_tokens)),
            selected_physical_prefix_tokens: Some(selected_physical),
            group_hashes: Some(group_hashes),
            quality: CacheEvidenceQuality::BoundedPhysical,
            complete: true,
            incomplete_reason: None,
            barrier: Some(barrier),
        }
    }

    pub fn history_epoch(&self) -> u64 {
        self.ledger.lock().history_epoch()
    }

    pub fn record_completed_token_history(
        &self,
        owner: CacheOwner,
        history_epoch: u64,
        tokens: &[u32],
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
    ) -> bool {
        if self.ledger.lock().history_epoch() != history_epoch {
            return false;
        }
        let Some(catalog) = self.group_catalog(owner) else {
            self.ledger
                .lock()
                .mark_history_incomplete_for_epoch(history_epoch);
            self.metrics.update_state(self.ledger.lock().stats());
            return false;
        };
        let Some(groups) = cache_group_hashes(tokens, &catalog, lora_name, cache_namespace) else {
            self.ledger
                .lock()
                .mark_history_incomplete_for_epoch(history_epoch);
            self.metrics.update_state(self.ledger.lock().stats());
            return false;
        };
        let mut ledger = self.ledger.lock();
        let recorded = ledger.record_seen_blocks_for_epoch(
            history_epoch,
            groups
                .iter()
                .flat_map(|group| group.sequence_hashes.iter().copied()),
        );
        self.metrics.update_state(ledger.stats());
        recorded
    }

    pub fn mark_history_incomplete(&self) {
        self.ledger.lock().mark_history_incomplete();
        self.metrics.update_state(self.ledger.lock().stats());
    }

    pub fn mark_history_incomplete_for_epoch(&self, epoch: u64) -> bool {
        let mut ledger = self.ledger.lock();
        let marked = ledger.mark_history_incomplete_for_epoch(epoch);
        self.metrics.update_state(ledger.stats());
        marked
    }

    pub async fn shutdown(self) {
        self.cancel.cancel();
        if let Err(error) = self.task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "Cache-evidence subscriber failed during shutdown");
        }
        if let Err(error) = self.barrier_task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "Cache-evidence barrier dispatcher failed during shutdown");
        }
        if let Some(task) = self.cold_epoch_task
            && let Err(error) = task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "Cold cache-history epoch task failed during shutdown");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_group_catalog(
    owner: CacheOwner,
    groups: &[CacheGroupObservation],
    group_shapes: &Mutex<HashMap<CacheOwner, Vec<CacheGroupObservation>>>,
    ledger: &Mutex<CacheEvidenceLedger>,
    barriers: &Mutex<BarrierCoordinator>,
    cold_epoch: &Mutex<ColdEpochCoordinator>,
    metrics: Option<&CacheEvidenceMetrics>,
) -> bool {
    let mut barriers = barriers.lock();
    let valid = !groups.is_empty()
        && groups.iter().all(|group| {
            matches!(
                group.kind.as_str(),
                "full_attention"
                    | "mla_attention"
                    | "sink_full_attention"
                    | "sliding_window"
                    | "sliding_window_mla"
            ) && group.block_size > 0
                && group
                    .alignment_tokens
                    .is_some_and(|alignment| alignment > 0 && alignment % group.block_size == 0)
        })
        && groups
            .iter()
            .map(|group| group.group_idx)
            .collect::<HashSet<_>>()
            .len()
            == groups.len();
    if !valid {
        barriers.fail_all("barrier_catalog_changed");
        cold_epoch.lock().fail("cold_epoch_catalog_changed");
        let mut state = ledger.lock();
        state.mark_physical_telemetry_incomplete();
        state.mark_history_incomplete();
        if let Some(metrics) = metrics {
            metrics.update_state(state.stats());
        }
        return false;
    }
    let mut normalized = groups.to_vec();
    normalized.sort_unstable_by_key(|group| group.group_idx);
    let (catalog_consistent, catalog_changed) = {
        let mut catalogs = group_shapes.lock();
        let consistent = catalogs.values().all(|existing| existing == &normalized);
        let changed = catalogs.get(&owner) != Some(&normalized);
        if consistent {
            catalogs.insert(owner, normalized.clone());
        }
        (consistent, changed)
    };
    if !catalog_consistent {
        barriers.fail_all("barrier_catalog_changed");
        cold_epoch.lock().fail("cold_epoch_catalog_changed");
        let mut state = ledger.lock();
        state.mark_physical_telemetry_incomplete();
        state.mark_history_incomplete();
        if let Some(metrics) = metrics {
            metrics.update_state(state.stats());
        }
        return false;
    }
    if catalog_changed {
        barriers.fail_all("barrier_catalog_changed");
        cold_epoch.lock().fail("cold_epoch_catalog_changed");
        if let Some(metrics) = metrics {
            metrics.observe_barrier_incomplete("catalog_changed");
        }
    }
    let mut state = ledger.lock();
    for tier in [CacheTier::Gpu, CacheTier::Cpu] {
        state.record_group_catalog(owner, tier, normalized.iter().map(|group| group.group_idx));
    }
    if let Some(metrics) = metrics {
        metrics.update_state(state.stats());
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn sync_attested_group_catalogs(
    view: &KvSourceMembershipView,
    group_shapes: &Mutex<HashMap<CacheOwner, Vec<CacheGroupObservation>>>,
    ledger: &Mutex<CacheEvidenceLedger>,
    barriers: &Mutex<BarrierCoordinator>,
    cold_epoch: &Mutex<ColdEpochCoordinator>,
    metrics: Option<&CacheEvidenceMetrics>,
) {
    let active: HashSet<_> = view
        .sources
        .iter()
        .filter_map(|(worker, status)| {
            status.active_source().map(|_| {
                (
                    *worker,
                    CacheOwner {
                        worker_id: worker.worker_id,
                        dp_rank: worker.dp_rank,
                    },
                )
            })
        })
        .collect();
    let active_owners: HashSet<_> = active.iter().map(|(_, owner)| *owner).collect();
    for (worker, owner) in active {
        let accepted = view
            .cache_evidence_cache_group_catalogs
            .get(&worker)
            .and_then(Option::as_deref)
            .is_some_and(|groups| {
                record_group_catalog(
                    owner,
                    groups,
                    group_shapes,
                    ledger,
                    barriers,
                    cold_epoch,
                    metrics,
                )
            });
        if !accepted && group_shapes.lock().remove(&owner).is_some() {
            barriers.lock().fail_all("barrier_catalog_changed");
            cold_epoch.lock().fail("cold_epoch_catalog_changed");
            let mut state = ledger.lock();
            state.mark_physical_telemetry_incomplete();
            state.mark_history_incomplete();
            if let Some(metrics) = metrics {
                metrics.update_state(state.stats());
            }
        }
    }
    group_shapes
        .lock()
        .retain(|owner, _| active_owners.contains(owner));
}

fn cache_group_hashes(
    tokens: &[u32],
    catalog: &[CacheGroupObservation],
    lora_name: Option<&str>,
    cache_namespace: Option<&str>,
) -> Option<Vec<CacheGroupHashSequence>> {
    let mut groups = Vec::with_capacity(catalog.len());
    for group in catalog {
        let kind = match group.kind.as_str() {
            "full_attention" | "mla_attention" | "sink_full_attention" => {
                CacheGroupKind::FullAttention
            }
            "sliding_window" | "sliding_window_mla" => CacheGroupKind::SlidingWindow,
            _ => return None,
        };
        let alignment_tokens = group.alignment_tokens?;
        if group.block_size == 0
            || alignment_tokens == 0
            || alignment_tokens % group.block_size != 0
        {
            return None;
        }
        let mut hashed =
            TokensWithHashes::new(tokens.to_vec(), group.block_size).with_is_eagle(group.is_eagle);
        if let Some(lora_name) = lora_name {
            hashed = hashed.with_lora_name(lora_name.to_string());
        }
        if let Some(cache_namespace) = cache_namespace {
            hashed = hashed.with_cache_namespace(cache_namespace.to_string());
        }
        let sequence_hashes = hashed.get_or_compute_seq_hashes().to_vec();
        groups.push(CacheGroupHashSequence {
            group_idx: group.group_idx,
            kind,
            block_size: group.block_size,
            sliding_window: group.sliding_window,
            is_eagle: group.is_eagle,
            alignment_tokens,
            sequence_hashes,
        });
    }
    Some(groups)
}

pub async fn start_cache_evidence_subscriber(
    component: Component,
    membership_watch: KvSourceMembershipWatch,
    parent_cancel: CancellationToken,
) -> CacheEvidenceSubscription {
    let ledger = Arc::new(Mutex::new(CacheEvidenceLedger::new(history_capacity())));
    let group_shapes = Arc::new(Mutex::new(HashMap::new()));
    let freshness = Arc::new(Mutex::new(EvidenceFreshness::new(evidence_freshness())));
    let metrics = CacheEvidenceMetrics::from_component(&component);
    let barriers = Arc::new(Mutex::new(BarrierCoordinator::new(
        barrier_pending_capacity(),
    )));
    let cold_epoch = Arc::new(Mutex::new(ColdEpochCoordinator::default()));
    let cold_epoch_enabled = std::env::var(CACHE_LOSS_COLD_EPOCH_SINGLE_FRONTEND_ENV)
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"));
    if cold_epoch_enabled {
        tracing::warn!(
            "Cold cache-history epoch enabled with the explicit single-frontend safety precondition; multi-frontend coordination is unsupported"
        );
    }
    // Catalogs are learned from completed requests. Keep dispatch open during
    // bounded readiness warmup, then the epoch task closes it immediately
    // before taking its fenced owner/catalog snapshot.
    let (dispatch_gate_tx, dispatch_gate) = watch::channel(true);
    let dispatch_fence = Arc::new(tokio::sync::RwLock::new(()));
    let (barrier_tx, barrier_rx) = mpsc::unbounded_channel();
    let cancel = parent_cancel.child_token();
    let serving_endpoint = membership_watch.borrow().serving_endpoint.clone();
    let cold_epoch_membership = membership_watch.fork_receiver();
    let task = tokio::spawn(run_cache_evidence_subscriber(
        component.clone(),
        membership_watch,
        Arc::clone(&ledger),
        Arc::clone(&group_shapes),
        Arc::clone(&freshness),
        Arc::clone(&metrics),
        Arc::clone(&barriers),
        Arc::clone(&cold_epoch),
        cancel.clone(),
    ));
    let barrier_task = tokio::spawn(run_barrier_dispatcher(
        component.clone(),
        serving_endpoint.clone(),
        barrier_rx,
        Arc::clone(&barriers),
        Arc::clone(&metrics),
        cancel.clone(),
    ));
    let cold_epoch_task = cold_epoch_enabled.then(|| {
        tokio::spawn(run_cold_epoch_once(
            component,
            serving_endpoint,
            cold_epoch_membership,
            Arc::clone(&ledger),
            Arc::clone(&group_shapes),
            Arc::clone(&cold_epoch),
            Arc::clone(&metrics),
            Arc::clone(&dispatch_fence),
            dispatch_gate_tx,
            cancel.clone(),
        ))
    });
    CacheEvidenceSubscription {
        ledger,
        group_shapes,
        freshness,
        metrics,
        barriers,
        cold_epoch,
        dispatch_gate,
        dispatch_fence,
        barrier_tx,
        cancel,
        task,
        barrier_task,
        cold_epoch_task,
    }
}

async fn run_barrier_dispatcher(
    component: Component,
    serving_endpoint: dynamo_runtime::protocols::EndpointId,
    mut commands: mpsc::UnboundedReceiver<u64>,
    barriers: Arc<Mutex<BarrierCoordinator>>,
    metrics: Arc<CacheEvidenceMetrics>,
    cancel: CancellationToken,
) {
    let control_component = match component
        .drt()
        .namespace(serving_endpoint.namespace.clone())
        .and_then(|namespace| namespace.component(serving_endpoint.component.clone()))
    {
        Ok(component) => component,
        Err(error) => {
            tracing::warn!(%error, "Failed to resolve cache-evidence barrier component");
            barriers.lock().fail_all("barrier_control_unavailable");
            return;
        }
    };
    let client = match control_component
        .endpoint("cache_evidence_barrier")
        .client()
        .await
    {
        Ok(client) => client,
        Err(error) => {
            tracing::warn!(%error, "Failed to create cache-evidence barrier client");
            barriers.lock().fail_all("barrier_control_unavailable");
            return;
        }
    };
    let client = match PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
        client,
        Default::default(),
    )
    .await
    {
        Ok(client) => Arc::new(client),
        Err(error) => {
            tracing::warn!(%error, "Failed to initialize cache-evidence barrier client");
            barriers.lock().fail_all("barrier_control_unavailable");
            return;
        }
    };

    while let Some(barrier_id) = tokio::select! {
        _ = cancel.cancelled() => None,
        command = commands.recv() => command,
    } {
        tokio::time::sleep(Duration::from_millis(1)).await;
        let Some((owners, cuts)) = barriers.lock().dispatch(barrier_id) else {
            continue;
        };
        metrics.observe_barrier_coalesced(cuts);
        let client = Arc::clone(&client);
        let barriers = Arc::clone(&barriers);
        let metrics = Arc::clone(&metrics);
        let cancel = cancel.clone();
        tokio::spawn(async move {
            let started = Instant::now();
            let requests = owners.into_iter().map(|(owner, incarnation)| {
                let client = Arc::clone(&client);
                async move {
                    let request = serde_json::json!({
                        "barrier_id": barrier_id,
                        "data_parallel_rank": owner.dp_rank,
                    });
                    let mut responses = client.direct(request.into(), owner.worker_id).await?;
                    let disposition =
                        validate_barrier_responses(&mut responses, barrier_id).await?;
                    anyhow::Ok((owner, incarnation, disposition))
                }
            });
            let timeout = barrier_timeout();
            let responses = tokio::select! {
                _ = cancel.cancelled() => {
                    let mut barriers = barriers.lock();
                    barriers.fail_round(barrier_id, "barrier_coordinator_closed");
                    metrics.update_barrier_pending(barriers.pending);
                    return;
                },
                responses = tokio::time::timeout(timeout, futures::future::join_all(requests)) => {
                    match responses {
                        Ok(responses) => responses,
                        Err(_) => {
                            let mut barrier_state = barriers.lock();
                            barrier_state.fail_round(barrier_id, "barrier_timeout");
                            metrics.update_barrier_pending(barrier_state.pending);
                            metrics.observe_barrier_incomplete("timeout");
                            return;
                        }
                    }
                }
            };
            for result in responses {
                match result {
                    Ok((_, _, BarrierControlDisposition::Success)) => {}
                    Ok((owner, incarnation, BarrierControlDisposition::PermanentlyUnavailable)) => {
                        let mut barrier_state = barriers.lock();
                        barrier_state.mark_permanently_unavailable(owner, incarnation);
                        metrics.update_barrier_pending(barrier_state.pending);
                        metrics.observe_barrier_incomplete("permanently_unavailable");
                        return;
                    }
                    Err(error) => {
                        tracing::warn!(barrier_id, %error, "Cache-evidence barrier dispatch failed");
                        let mut barrier_state = barriers.lock();
                        barrier_state.fail_round(barrier_id, "barrier_control_failed");
                        metrics.update_barrier_pending(barrier_state.pending);
                        metrics.observe_barrier_incomplete("control_failed");
                        return;
                    }
                }
            }
            let control_rtt = {
                let mut barrier_state = barriers.lock();
                let rtt = barrier_state.controls_succeeded(barrier_id);
                metrics.update_barrier_pending(barrier_state.pending);
                rtt
            };
            if let Some(rtt) = control_rtt {
                metrics.observe_barrier_rtt(rtt);
                return;
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            tokio::select! {
                _ = cancel.cancelled() => {
                    let mut barriers = barriers.lock();
                    barriers.fail_round(barrier_id, "barrier_coordinator_closed");
                    metrics.update_barrier_pending(barriers.pending);
                },
                _ = tokio::time::sleep(remaining) => {
                    let mut barrier_state = barriers.lock();
                    if barrier_state.rounds.contains_key(&barrier_id) {
                        barrier_state.fail_round(barrier_id, "barrier_timeout");
                        metrics.update_barrier_pending(barrier_state.pending);
                        metrics.observe_barrier_incomplete("timeout");
                    }
                }
            }
        });
    }
    barriers.lock().fail_all("barrier_coordinator_closed");
}

async fn validate_barrier_responses<S>(
    responses: &mut S,
    barrier_id: u64,
) -> anyhow::Result<BarrierControlDisposition>
where
    S: futures::Stream<Item = Annotated<serde_json::Value>> + Unpin,
{
    let response = responses
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("barrier endpoint returned no response"))?;
    let payload = response
        .into_result()?
        .ok_or_else(|| anyhow::anyhow!("barrier endpoint returned no payload"))?;
    anyhow::ensure!(
        payload
            .get("barrier_id")
            .and_then(serde_json::Value::as_u64)
            == Some(barrier_id),
        "barrier endpoint returned a mismatched barrier id"
    );
    anyhow::ensure!(
        responses.next().await.is_none(),
        "barrier endpoint returned more than one response"
    );
    match (
        payload.get("status").and_then(serde_json::Value::as_str),
        payload.get("code").and_then(serde_json::Value::as_str),
    ) {
        (Some("success"), _) => Ok(BarrierControlDisposition::Success),
        (Some("error"), Some("barrier_permanently_unavailable")) => {
            Ok(BarrierControlDisposition::PermanentlyUnavailable)
        }
        _ => anyhow::bail!("barrier endpoint returned an error payload"),
    }
}

async fn cold_epoch_client(
    component: &Component,
    serving_endpoint: &dynamo_runtime::protocols::EndpointId,
    endpoint: &str,
    expected_worker_ids: &HashSet<u64>,
) -> anyhow::Result<Arc<PushRouter<serde_json::Value, Annotated<serde_json::Value>>>> {
    let control_component = component
        .drt()
        .namespace(serving_endpoint.namespace.clone())?
        .component(serving_endpoint.component.clone())?;
    let client = control_component.endpoint(endpoint).client().await?;
    wait_for_cold_epoch_workers(client.instance_source.as_ref().clone(), expected_worker_ids)
        .await?;
    Ok(Arc::new(
        PushRouter::from_client(client, Default::default()).await?,
    ))
}

type ColdEpochClient = Arc<PushRouter<serde_json::Value, Annotated<serde_json::Value>>>;

struct ColdEpochClients {
    begin: ColdEpochClient,
    commit: ColdEpochClient,
    abort: ColdEpochClient,
}

async fn cold_epoch_clients(
    component: &Component,
    serving_endpoint: &dynamo_runtime::protocols::EndpointId,
    expected_worker_ids: &HashSet<u64>,
) -> anyhow::Result<ColdEpochClients> {
    let (begin, commit, abort) = tokio::try_join!(
        cold_epoch_client(
            component,
            serving_endpoint,
            "begin_cache_evidence_epoch",
            expected_worker_ids,
        ),
        cold_epoch_client(
            component,
            serving_endpoint,
            "commit_cache_evidence_epoch",
            expected_worker_ids,
        ),
        cold_epoch_client(
            component,
            serving_endpoint,
            "abort_cache_evidence_epoch",
            expected_worker_ids,
        ),
    )?;
    Ok(ColdEpochClients {
        begin,
        commit,
        abort,
    })
}

async fn wait_for_cold_epoch_workers(
    mut instances: watch::Receiver<Vec<Instance>>,
    expected_worker_ids: &HashSet<u64>,
) -> anyhow::Result<()> {
    loop {
        let complete = {
            let discovered = instances.borrow_and_update();
            expected_worker_ids
                .iter()
                .all(|expected| discovered.iter().any(|instance| instance.id() == *expected))
        };
        if complete {
            return Ok(());
        }
        instances.changed().await.map_err(|_| {
            anyhow::anyhow!(
                "cold epoch control endpoint discovery closed before all owners appeared"
            )
        })?;
    }
}

async fn validate_cold_epoch_response<S>(
    responses: &mut S,
    epoch_id: &str,
    barrier_id: Option<u64>,
    serving_incarnation: Option<u64>,
    expected_tiers: Option<&HashSet<CacheTier>>,
) -> anyhow::Result<()>
where
    S: futures::Stream<Item = Annotated<serde_json::Value>> + Unpin,
{
    let response = responses
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("cold epoch endpoint returned no response"))?;
    let payload = response
        .into_result()?
        .ok_or_else(|| anyhow::anyhow!("cold epoch endpoint returned no payload"))?;
    anyhow::ensure!(
        payload.get("status").and_then(serde_json::Value::as_str) == Some("success"),
        "cold epoch endpoint returned an error payload"
    );
    anyhow::ensure!(
        payload.get("epoch_id").and_then(serde_json::Value::as_str) == Some(epoch_id),
        "cold epoch endpoint returned a mismatched token"
    );
    if let Some(barrier_id) = barrier_id {
        anyhow::ensure!(
            payload
                .get("barrier_id")
                .and_then(serde_json::Value::as_u64)
                == Some(barrier_id),
            "cold epoch endpoint returned a mismatched barrier id"
        );
    }
    if let Some(serving_incarnation) = serving_incarnation {
        anyhow::ensure!(
            payload
                .get("serving_incarnation")
                .and_then(serde_json::Value::as_u64)
                == Some(serving_incarnation),
            "cold epoch endpoint returned a mismatched serving incarnation"
        );
    }
    if let Some(expected_tiers) = expected_tiers {
        let returned: HashSet<_> = payload
            .get("cleared_media")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("cold epoch response omitted cleared media"))?
            .iter()
            .map(|value| match value.as_str() {
                Some("GPU") => Ok(CacheTier::Gpu),
                Some("CPU") => Ok(CacheTier::Cpu),
                _ => anyhow::bail!("cold epoch response returned invalid cleared media"),
            })
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            &returned == expected_tiers,
            "cold epoch response returned mismatched cleared media"
        );
    }
    anyhow::ensure!(
        responses.next().await.is_none(),
        "cold epoch endpoint returned more than one response"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_cold_epoch_once(
    component: Component,
    serving_endpoint: dynamo_runtime::protocols::EndpointId,
    mut membership_watch: KvSourceMembershipWatch,
    ledger: Arc<Mutex<CacheEvidenceLedger>>,
    group_shapes: Arc<Mutex<HashMap<CacheOwner, Vec<CacheGroupObservation>>>>,
    cold_epoch: Arc<Mutex<ColdEpochCoordinator>>,
    metrics: Arc<CacheEvidenceMetrics>,
    dispatch_fence: Arc<tokio::sync::RwLock<()>>,
    dispatch_gate: watch::Sender<bool>,
    cancel: CancellationToken,
) {
    loop {
        let readiness = tokio::time::timeout(cold_epoch_readiness_timeout(), async {
            loop {
                let view = membership_watch.borrow_and_update().clone();
                if let Some(owners) = cold_epoch_owners(&view) {
                    let expected: HashSet<_> = owners.keys().copied().collect();
                    if group_shapes.lock().keys().copied().collect::<HashSet<_>>() == expected
                        && ledger.lock().expected_owners_match(&expected)
                    {
                        break Some(());
                    }
                }
                tokio::select! {
                    _ = cancel.cancelled() => break None,
                    changed = membership_watch.changed() => if changed.is_err() { break None; },
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {},
                }
            }
        })
        .await;
        match readiness {
            Ok(Some(())) => break,
            Ok(None) => {
                metrics.observe_cold_epoch("cancelled");
                ledger.lock().mark_history_incomplete();
                return;
            }
            Err(_) => {
                tracing::warn!("Cold cache-history epoch readiness attempt timed out; retrying");
                metrics.observe_cold_epoch("readiness_timeout");
                ledger.lock().mark_history_incomplete();
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                }
            }
        }
    }
    let Some(prepared_owners) = cold_epoch_owners(&membership_watch.borrow()) else {
        tracing::warn!("Cold cache-history epoch membership changed before control discovery");
        metrics.observe_cold_epoch("membership");
        ledger.lock().mark_history_incomplete();
        return;
    };
    let expected_worker_ids = prepared_owners
        .keys()
        .map(|owner| owner.worker_id)
        .collect::<HashSet<_>>();
    let preparation_started = Instant::now();
    let controls = tokio::select! {
        _ = cancel.cancelled() => None,
        result = tokio::time::timeout(
            COLD_EPOCH_BEGIN_TIMEOUT,
            cold_epoch_clients(&component, &serving_endpoint, &expected_worker_ids),
        ) => match result {
            Ok(Ok(controls)) => Some(controls),
            Ok(Err(error)) => {
                tracing::warn!(%error, "Failed to initialize cold cache-history epoch controls");
                None
            }
            Err(_) => {
                tracing::warn!("Cold cache-history epoch control discovery timed out");
                None
            }
        },
    };
    let control_preparation_elapsed = preparation_started.elapsed();
    let Some(controls) = controls else {
        metrics.observe_cold_epoch(if cancel.is_cancelled() {
            "cancelled"
        } else {
            "control"
        });
        ledger.lock().mark_history_incomplete();
        return;
    };
    let dispatch_fence = tokio::select! {
        _ = cancel.cancelled() => None,
        result = tokio::time::timeout(
            COLD_EPOCH_BEGIN_TIMEOUT,
            dispatch_fence.write_owned(),
        ) => result.ok(),
    };
    let Some(_dispatch_fence) = dispatch_fence else {
        tracing::warn!("Cold cache-history epoch timed out draining selected requests");
        metrics.observe_cold_epoch("dispatch_drain");
        ledger.lock().mark_history_incomplete();
        return;
    };
    dispatch_gate.send_replace(false);
    let mut gate_release = DispatchGateRelease {
        sender: dispatch_gate,
        ledger: Arc::clone(&ledger),
        committed: false,
    };
    let Some(owners) = cold_epoch_owners(&membership_watch.borrow()) else {
        tracing::warn!("Cold cache-history epoch membership changed before fencing");
        metrics.observe_cold_epoch("membership");
        return;
    };
    if owners != prepared_owners {
        tracing::warn!("Cold cache-history epoch membership changed during control discovery");
        metrics.observe_cold_epoch("membership");
        return;
    }
    let expected: HashSet<_> = owners.keys().copied().collect();
    if group_shapes.lock().keys().copied().collect::<HashSet<_>>() != expected
        || !ledger.lock().expected_owners_match(&expected)
    {
        tracing::warn!("Cold cache-history epoch catalog changed before fencing");
        metrics.observe_cold_epoch("catalog");
        return;
    }
    let initial_catalogs = group_shapes.lock().clone();
    let epoch_id = format!("{:032x}", rand::random::<u128>());
    let barrier_id = rand::random::<u64>().max(1);
    let issued_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let deadline_unix_ms = issued_at_unix_ms.saturating_add(COLD_EPOCH_LEASE_MS);
    let mut outcome = match cold_epoch
        .lock()
        .begin(epoch_id.clone(), barrier_id, owners.clone())
    {
        Ok(outcome) => outcome,
        Err(error) => {
            tracing::warn!(%error, "Failed to initialize cold cache-history epoch");
            metrics.observe_cold_epoch("init");
            return;
        }
    };

    let begin_phase = async {
        let requests = owners.iter().map(|(&owner, fence)| {
            let client = Arc::clone(&controls.begin);
            let epoch_id = epoch_id.clone();
            let fence = fence.clone();
            async move {
                let request = serde_json::json!({
                    "epoch_id": epoch_id,
                    "barrier_id": barrier_id,
                    "lease_ms": COLD_EPOCH_LEASE_MS,
                    "serving_incarnation": fence.serving_incarnation,
                    "issued_at_unix_ms": issued_at_unix_ms,
                    "deadline_unix_ms": deadline_unix_ms,
                    "data_parallel_rank": owner.dp_rank,
                });
                let mut responses = client.direct(request.into(), owner.worker_id).await?;
                validate_cold_epoch_response(
                    &mut responses,
                    &epoch_id,
                    Some(barrier_id),
                    Some(fence.serving_incarnation),
                    Some(&fence.expected_tiers),
                )
                .await
            }
        });
        let results = futures::future::join_all(requests).await;
        anyhow::ensure!(
            results.into_iter().all(|result| result.is_ok()),
            "one or more cold epoch begin controls failed"
        );
        anyhow::Ok(())
    };
    let begin_timeout = COLD_EPOCH_BEGIN_TIMEOUT.saturating_sub(control_preparation_elapsed);
    let began = tokio::select! {
        _ = cancel.cancelled() => false,
        result = tokio::time::timeout(begin_timeout, begin_phase) => {
            result.is_ok_and(|result| result.is_ok())
        }
    };
    if began {
        cold_epoch.lock().controls_succeeded(&epoch_id);
    } else {
        cold_epoch.lock().fail("cold_epoch_begin_failed");
        metrics.observe_cold_epoch(if cancel.is_cancelled() {
            "cancelled"
        } else {
            "control"
        });
    }

    let evidence_outcome = if began {
        let evidence = async {
            loop {
                let current = *outcome.borrow();
                match current {
                    ColdEpochOutcome::Pending => outcome.changed().await?,
                    result => return Ok::<_, watch::error::RecvError>(result),
                }
            }
        };
        tokio::select! {
            _ = cancel.cancelled() => Some(ColdEpochOutcome::Incomplete("cold_epoch_cancelled")),
            result = tokio::time::timeout(COLD_EPOCH_BEGIN_TIMEOUT, evidence) => {
                result.ok().and_then(Result::ok)
            }
        }
    } else {
        None
    };
    if let Some(ColdEpochOutcome::Incomplete(reason)) = evidence_outcome {
        metrics.observe_cold_epoch(match reason {
            "cold_epoch_membership_changed" => "membership",
            "cold_epoch_catalog_changed" => "catalog",
            "cold_epoch_cancelled" => "cancelled",
            _ => "evidence",
        });
    }
    let mut evidence_ready = evidence_outcome == Some(ColdEpochOutcome::EvidenceReady);

    if evidence_ready {
        let mut epoch_state = cold_epoch.lock();
        let membership_matches =
            cold_epoch_owners(&membership_watch.borrow()).as_ref() == Some(&owners);
        let catalogs_match = *group_shapes.lock() == initial_catalogs;
        let expected: HashSet<_> = owners.keys().copied().collect();
        let mut state = ledger.lock();
        if membership_matches && catalogs_match && state.expected_owners_match(&expected) {
            if state.commit_cold_history_epoch(&expected).is_none() {
                epoch_state.fail("cold_epoch_residual_physical_state");
                metrics.observe_cold_epoch("residual_physical_state");
                evidence_ready = false;
            }
        } else {
            epoch_state.fail("cold_epoch_snapshot_changed");
            metrics.observe_cold_epoch("snapshot");
            state.mark_history_incomplete();
            evidence_ready = false;
        }
        metrics.update_state(state.stats());
    } else {
        let mut state = ledger.lock();
        state.mark_history_incomplete();
        metrics.update_state(state.stats());
    }

    let release_client = if evidence_ready {
        Arc::clone(&controls.commit)
    } else {
        Arc::clone(&controls.abort)
    };
    let release_phase = async {
        let requests = owners.keys().copied().map(|owner| {
            let client = Arc::clone(&release_client);
            let epoch_id = epoch_id.clone();
            async move {
                let request = serde_json::json!({
                    "epoch_id": epoch_id,
                    "data_parallel_rank": owner.dp_rank,
                });
                let mut responses = client.direct(request.into(), owner.worker_id).await?;
                validate_cold_epoch_response(&mut responses, &epoch_id, None, None, None).await
            }
        });
        let results = futures::future::join_all(requests).await;
        anyhow::ensure!(
            results.into_iter().all(|result| result.is_ok()),
            "one or more cold epoch release controls failed"
        );
        anyhow::Ok(())
    };
    let released = tokio::select! {
        _ = cancel.cancelled() => false,
        result = tokio::time::timeout(COLD_EPOCH_RELEASE_TIMEOUT, release_phase) => {
            result.is_ok_and(|result| result.is_ok())
        }
    };
    if !released {
        let mut state = ledger.lock();
        state.mark_history_incomplete();
        metrics.update_state(state.stats());
        metrics.observe_cold_epoch("release");
    } else if evidence_ready {
        metrics.observe_cold_epoch("success");
    } else if began && evidence_outcome.is_none() {
        metrics.observe_cold_epoch("evidence");
    }
    gate_release.committed = evidence_ready && released;
    cold_epoch.lock().complete(&epoch_id);
    drop(gate_release);
}

#[allow(clippy::too_many_arguments)]
async fn run_cache_evidence_subscriber(
    component: Component,
    mut membership_watch: KvSourceMembershipWatch,
    ledger: Arc<Mutex<CacheEvidenceLedger>>,
    group_shapes: Arc<Mutex<HashMap<CacheOwner, Vec<CacheGroupObservation>>>>,
    freshness: Arc<Mutex<EvidenceFreshness>>,
    metrics: Arc<CacheEvidenceMetrics>,
    barriers: Arc<Mutex<BarrierCoordinator>>,
    cold_epoch: Arc<Mutex<ColdEpochCoordinator>>,
    cancel: CancellationToken,
) {
    let mut source_sequences = EvidenceSequenceTracker::default();
    loop {
        let initial_view = membership_watch.borrow_and_update().clone();
        {
            let mut barrier_state = barriers.lock();
            let (owners, serving_incarnations) = barrier_owners(&initial_view);
            barrier_state.set_owner_incarnations(owners, serving_incarnations);
            metrics.update_barrier_pending(barrier_state.pending);
        }
        let mut active: std::collections::HashSet<_> = initial_view
            .sources
            .keys()
            .map(|worker| CacheOwner {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
            })
            .collect();
        let mut history_membership = cold_epoch_owners(&initial_view);
        sync_attested_group_catalogs(
            &initial_view,
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            Some(&metrics),
        );
        let _ = freshness.lock().set_expected(
            initial_view
                .sources
                .iter()
                .map(|(worker, status)| {
                    (
                        CacheOwner {
                            worker_id: worker.worker_id,
                            dp_rank: worker.dp_rank,
                        },
                        status
                            .active_source()
                            .and_then(|source| source.evidence_incarnation_id),
                    )
                })
                .collect(),
        );
        update_freshness_metrics(&freshness, &metrics);
        ledger.lock().set_expected_owners(active.iter().copied());
        metrics.update_state(ledger.lock().stats());
        let endpoint = initial_view.resolved_kv_state_endpoint().cloned();
        let Some(endpoint) = endpoint else {
            ledger.lock().mark_physical_telemetry_incomplete();
            metrics.update_state(ledger.lock().stats());
            tokio::select! {
                _ = cancel.cancelled() => return,
                changed = membership_watch.changed() => if changed.is_err() { return; },
            }
            continue;
        };

        let subscriber =
            EventSubscriber::for_endpoint_id(component.drt(), &endpoint, KV_CACHE_EVIDENCE_SUBJECT)
                .await;
        let mut subscriber = match subscriber {
            Ok(subscriber) => subscriber.typed::<CacheEvidenceBatch>(),
            Err(error) => {
                tracing::warn!(%error, %endpoint, "Failed to subscribe to cache evidence");
                ledger.lock().mark_physical_telemetry_incomplete();
                metrics.update_state(ledger.lock().stats());
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    changed = membership_watch.changed() => if changed.is_err() { return; },
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                }
                continue;
            }
        };
        let mut freshness_tick = tokio::time::interval(Duration::from_secs(1));
        freshness_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                changed = membership_watch.changed() => {
                    if changed.is_err() { return; }
                    let current = membership_watch.borrow_and_update();
                    let next_history_membership = cold_epoch_owners(&current);
                    let history_membership_changed = history_membership != next_history_membership;
                    if history_membership_changed {
                        cold_epoch.lock().fail("cold_epoch_membership_changed");
                    }
                    {
                        let mut barrier_state = barriers.lock();
                        let (owners, serving_incarnations) = barrier_owners(&current);
                        barrier_state.set_owner_incarnations(owners, serving_incarnations);
                        metrics.update_barrier_pending(barrier_state.pending);
                    }
                    if current.resolved_kv_state_endpoint() != Some(&endpoint) {
                        cold_epoch.lock().fail("cold_epoch_membership_changed");
                        barriers
                            .lock()
                            .fail_all("barrier_membership_changed");
                        ledger.lock().mark_history_incomplete();
                        break;
                    }
                    let next_active: std::collections::HashSet<_> = current.sources.keys().map(|worker| CacheOwner {
                        worker_id: worker.worker_id,
                        dp_rank: worker.dp_rank,
                    }).collect();
                    let restarted = freshness.lock().set_expected(current.sources.iter().map(|(worker, status)| {
                        (
                            CacheOwner {
                                worker_id: worker.worker_id,
                                dp_rank: worker.dp_rank,
                            },
                            status.active_source().and_then(|source| source.evidence_incarnation_id),
                        )
                    }).collect());
                    update_freshness_metrics(&freshness, &metrics);
                    let retired: Vec<_> = source_sequences.owners().filter(|owner| !next_active.contains(owner)).collect();
                    let reset: HashSet<_> = retired.iter().chain(&restarted).copied().collect();
                    for &owner in &reset {
                        source_sequences.retire(owner);
                        group_shapes.lock().remove(&owner);
                    }
                    sync_attested_group_catalogs(
                        &current,
                        &group_shapes,
                        &ledger,
                        &barriers,
                        &cold_epoch,
                        Some(&metrics),
                    );
                    let mut state = ledger.lock();
                    reconcile_history_membership(
                        &mut history_membership,
                        next_history_membership,
                        &mut state,
                    );
                    state.set_expected_owners(next_active.iter().copied());
                    for owner in reset {
                        state.retire_owner(owner);
                    }
                    metrics.update_state(state.stats());
                    active = next_active;
                }
                event = subscriber.next() => {
                    let Some(event) = event else {
                        barriers
                            .lock()
                            .fail_all("barrier_subscriber_closed");
                        ledger.lock().mark_physical_telemetry_incomplete();
                        cold_epoch.lock().fail("cold_epoch_subscriber_closed");
                        metrics.update_state(ledger.lock().stats());
                        break;
                    };
                    let (_envelope, batch) = match event {
                        Ok(value) => value,
                        Err(error) => {
                            tracing::warn!(%error, %endpoint, "Failed to decode cache evidence");
                            ledger.lock().mark_physical_telemetry_incomplete();
                            barriers.lock().fail_all("barrier_decode_error");
                            cold_epoch.lock().fail("cold_epoch_decode_error");
                            metrics.observe_batch("decode_error");
                            metrics.update_state(ledger.lock().stats());
                            continue;
                        }
                    };
                    if !active.contains(&batch.owner) {
                        metrics.observe_batch("inactive_owner");
                        continue;
                    }
                    let Some(source_incarnation_id) = batch.source_incarnation_id else {
                        ledger.lock().mark_physical_telemetry_incomplete();
                        cold_epoch.lock().fail("cold_epoch_legacy_batch");
                        metrics.observe_batch("legacy");
                        metrics.update_state(ledger.lock().stats());
                        continue;
                    };
                    if !freshness.lock().accepts(batch.owner, source_incarnation_id) {
                        metrics.observe_batch("wrong_incarnation");
                        continue;
                    }
                    if batch.heartbeat {
                        if !batch.mutations.is_empty() || !batch.telemetry_complete {
                            ledger.lock().mark_physical_telemetry_incomplete();
                            barriers
                                .lock()
                                .fail_owner(batch.owner, "barrier_invalid_watermark");
                            metrics.observe_batch("invalid_watermark");
                            metrics.update_state(ledger.lock().stats());
                            continue;
                        }
                        match observe_watermark(
                            source_sequences.last_cursor(source_incarnation_id),
                            batch.watermark_source_cursor,
                        ) {
                            EvidenceWatermarkObservation::Current => {}
                            EvidenceWatermarkObservation::Stale => {
                                metrics.observe_batch("stale_watermark");
                                continue;
                            }
                            EvidenceWatermarkObservation::Ahead => {
                                ledger.lock().mark_physical_telemetry_incomplete();
                                barriers
                                    .lock()
                                    .fail_owner(batch.owner, "barrier_watermark_ahead");
                                metrics.observe_batch("watermark_ahead");
                                metrics.update_state(ledger.lock().stats());
                                continue;
                            }
                        }
                        freshness.lock().observe(
                            batch.owner,
                            source_incarnation_id,
                            Instant::now(),
                        );
                        metrics.observe_batch("watermark");
                        update_freshness_metrics(&freshness, &metrics);
                        continue;
                    }
                    let accepts_clear_baseline = cold_epoch.lock().accepts_clear_baseline(&batch);
                    let sequence_observation = source_sequences.observe_with_initial_baseline(
                        batch.owner,
                        source_incarnation_id,
                        batch.source_cursor,
                        accepts_clear_baseline,
                    );
                    if sequence_observation == EvidenceSequenceObservation::PublisherRestart {
                        group_shapes.lock().remove(&batch.owner);
                    }
                    let mut barrier_state = barriers.lock();
                    let mut state = ledger.lock();
                    match sequence_observation {
                        EvidenceSequenceObservation::Contiguous => {}
                        EvidenceSequenceObservation::Gap => {
                            state.mark_physical_telemetry_incomplete();
                            cold_epoch.lock().fail("cold_epoch_gap");
                            if barrier_state.fail_owner(batch.owner, "barrier_gap") > 0 {
                                metrics.observe_barrier_incomplete("gap");
                            }
                            metrics.observe_batch("gap");
                        }
                        EvidenceSequenceObservation::Stale => {
                            metrics.observe_batch("stale");
                            continue;
                        }
                        EvidenceSequenceObservation::PublisherRestart => {
                            cold_epoch.lock().fail("cold_epoch_restart");
                            state.mark_history_incomplete();
                            if barrier_state.fail_owner(batch.owner, "barrier_restart") > 0 {
                                metrics.observe_barrier_incomplete("restart");
                            }
                            metrics.observe_batch("publisher_restart");
                            state.retire_owner(batch.owner);
                            if batch.source_cursor != 0 {
                                state.mark_physical_telemetry_incomplete();
                            }
                        }
                    }
                    if batch.epoch_id.is_some() {
                        if batch.mutations.is_empty() {
                            cold_epoch.lock().observe_marker(&batch);
                            metrics.observe_batch("cold_epoch_marker");
                            continue;
                        }
                        cold_epoch.lock().observe_clear_batch(&batch);
                        barrier_state.fail_owner(batch.owner, "barrier_clear");
                        let applied = state.apply_evidence_batch(&batch);
                        if !applied {
                            cold_epoch.lock().fail("cold_epoch_apply_integrity_failure");
                        }
                        metrics.update_state(state.stats());
                        metrics.observe_batch("cold_epoch_clear");
                        continue;
                    }
                    if let Some(barrier_id) = batch.barrier_id {
                        if !batch.mutations.is_empty() || !batch.telemetry_complete {
                            if barrier_state
                                .fail_owner(batch.owner, "barrier_invalid_marker")
                                > 0
                            {
                                metrics.observe_barrier_incomplete("invalid_marker");
                            }
                            metrics.observe_batch("invalid_barrier");
                            continue;
                        }
                        drop(state);
                        if let Some(rtt) = barrier_state.marker(batch.owner, barrier_id) {
                            metrics.observe_barrier_rtt(rtt);
                        }
                        metrics.update_barrier_pending(barrier_state.pending);
                        metrics.observe_batch("barrier");
                        continue;
                    }
                    if !batch.telemetry_complete
                        && barrier_state
                            .fail_owner(batch.owner, "barrier_incomplete_batch")
                            > 0
                    {
                        cold_epoch.lock().fail("cold_epoch_incomplete_batch");
                        metrics.observe_barrier_incomplete("incomplete_batch");
                    }
                    for mutation in &batch.mutations {
                        cold_epoch.lock().unexpected_mutation(batch.owner);
                        let hashes = state.affected_sequence_hashes(batch.owner, mutation);
                        let clear = matches!(mutation, CacheEvidenceMutation::Clear { tier: None });
                        let unresolved = hashes.is_empty()
                            && match mutation {
                                CacheEvidenceMutation::Store { blocks, .. }
                                | CacheEvidenceMutation::StoreWithParentAttestation {
                                    blocks,
                                    ..
                                } => !blocks.is_empty(),
                                CacheEvidenceMutation::Remove { block_hashes, .. } => {
                                    !block_hashes.is_empty()
                                }
                                CacheEvidenceMutation::Clear { .. } => false,
                            };
                        if unresolved {
                            state.mark_history_incomplete();
                            cold_epoch.lock().fail("cold_epoch_unresolved_mutation");
                            if barrier_state
                                .fail_owner(batch.owner, "barrier_unresolved_mutation")
                                > 0
                            {
                                metrics.observe_barrier_incomplete("unresolved_mutation");
                            }
                            continue;
                        }
                        if barrier_state.mutation(
                            batch.owner,
                            &hashes,
                            clear,
                        ) > 0
                        {
                            metrics.observe_barrier_incomplete(if clear {
                                "clear"
                            } else {
                                "relevant_mutation"
                            });
                        }
                    }
                    metrics.observe_batch("applied");
                    for mutation in &batch.mutations {
                        match mutation {
                            CacheEvidenceMutation::Store { tier, .. }
                            | CacheEvidenceMutation::StoreWithParentAttestation { tier, .. } => {
                                metrics.observe_mutation("store", cache_tier_label(*tier));
                            }
                            CacheEvidenceMutation::Remove { tier, .. } => {
                                metrics.observe_mutation("remove", cache_tier_label(*tier));
                            }
                            CacheEvidenceMutation::Clear { .. } => {
                                metrics.observe_mutation("clear", "all");
                            }
                        }
                    }
                    let apply_result = state.apply_evidence_batch_with_diagnostics(&batch);
                    if !apply_result.is_complete() {
                        for failure in apply_result.failures() {
                            metrics.observe_apply_integrity_failure(failure.metric_label());
                        }
                        if barrier_state
                            .fail_owner(batch.owner, "barrier_apply_integrity_failure")
                            > 0
                        {
                            cold_epoch.lock().fail("cold_epoch_apply_integrity_failure");
                            metrics.observe_barrier_incomplete("apply_integrity_failure");
                        }
                    }
                    metrics.update_state(state.stats());
                    metrics.update_barrier_pending(barrier_state.pending);
                }
                _ = freshness_tick.tick() => {
                    update_freshness_metrics(&freshness, &metrics);
                }
            }
        }
    }
}

fn cache_tier_label(tier: CacheTier) -> &'static str {
    match tier {
        CacheTier::Gpu => "gpu",
        CacheTier::Cpu => "cpu",
    }
}

#[derive(Clone, Debug)]
pub struct CacheEvidenceObserver {
    ledger: Arc<Mutex<CacheEvidenceLedger>>,
}

impl CacheEvidenceObserver {
    pub fn new(history_capacity: usize) -> Self {
        Self {
            ledger: Arc::new(Mutex::new(CacheEvidenceLedger::new(history_capacity))),
        }
    }

    pub fn ledger(&self) -> Arc<Mutex<CacheEvidenceLedger>> {
        Arc::clone(&self.ledger)
    }
}

impl RawKvEventObserver for CacheEvidenceObserver {
    fn observe(&self, event: &RawKvEvent, worker: WorkerWithDpRank) {
        let owner = CacheOwner {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
        };
        let mut ledger = self.ledger.lock();
        if !matches!(event.ownership(), Ok(KvEventOwnership::Framework))
            || matches!(event.locality(), Some(Locality::Remote | Locality::Unknown))
        {
            ledger.mark_physical_telemetry_incomplete();
            return;
        }
        match event {
            RawKvEvent::BlockStored {
                block_hashes,
                medium,
                group_idx,
                ..
            } => {
                let Some(tier) = cache_tier(medium.as_deref()) else {
                    ledger.mark_physical_telemetry_incomplete();
                    return;
                };
                let group = group_idx.unwrap_or_else(|| {
                    ledger.mark_physical_telemetry_incomplete();
                    0
                });
                for hash in block_hashes {
                    ledger.store(owner, tier, group, hash.into_u64());
                }
            }
            RawKvEvent::BlockRemoved {
                block_hashes,
                medium,
                group_idx,
                ..
            } => {
                let Some(tier) = cache_tier(medium.as_deref()) else {
                    ledger.mark_physical_telemetry_incomplete();
                    return;
                };
                let group = group_idx.unwrap_or_else(|| {
                    ledger.mark_physical_telemetry_incomplete();
                    0
                });
                for hash in block_hashes {
                    ledger.remove(owner, tier, group, hash.into_u64());
                }
            }
            RawKvEvent::AllBlocksCleared { medium, .. } => match medium.as_deref() {
                None => ledger.clear_owner(owner),
                Some(medium) => match cache_tier(Some(medium)) {
                    Some(tier) => ledger.clear_owner_tier(owner, tier),
                    None => ledger.mark_physical_telemetry_incomplete(),
                },
            },
            RawKvEvent::Ignored => {}
        }
    }
}

fn cache_tier(medium: Option<&str>) -> Option<CacheTier> {
    match medium.map_or(Some(StorageTier::Device), StorageTier::from_kv_medium) {
        Some(StorageTier::Device) => Some(CacheTier::Gpu),
        Some(StorageTier::HostPinned) => Some(CacheTier::Cpu),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{KvEventSource, KvSourceStatus, KvStateEndpointResolution};
    use dynamo_kv_router::{
        cache_loss::{CacheEvidenceStoredBlock, KnownBool},
        zmq_wire::{BlockHashValue, KvCacheSpecKind, Locality},
    };

    fn catalog_group(group_idx: u32, kind: &str, block_size: u32) -> CacheGroupObservation {
        CacheGroupObservation {
            group_idx,
            kind: kind.to_string(),
            block_size,
            sliding_window: (kind == "sliding_window").then_some(128),
            is_eagle: false,
            alignment_tokens: Some(256),
        }
    }

    fn catalog_membership_view(
        catalog: Option<Vec<CacheGroupObservation>>,
    ) -> KvSourceMembershipView {
        let worker = WorkerWithDpRank::new(7, 4);
        let endpoint = dynamo_runtime::protocols::EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: "generate".to_string(),
        };
        KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint.clone()),
            sources: HashMap::from([(
                worker,
                KvSourceStatus::ActiveLiveOnly(KvEventSource {
                    kv_state_endpoint: endpoint,
                    worker,
                    publisher_id: 11,
                    evidence_incarnation_id: Some(12),
                    recovery_target: None,
                }),
            )]),
            kv_event_publishing_enabled: HashMap::from([(7, Some(true))]),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::from([(worker, false)]),
            cache_evidence_barrier_enabled: HashMap::from([(7, Some(true))]),
            serving_incarnations: HashMap::from([(7, Some(1))]),
            cache_evidence_serving_incarnations: HashMap::from([(worker, Some(99))]),
            cache_evidence_cache_group_catalogs: HashMap::from([(worker, catalog)]),
            cache_evidence_epoch_enabled: HashMap::from([(7, Some(true))]),
            cache_evidence_epoch_media: HashMap::from([(
                worker,
                Some(HashSet::from(["GPU".to_string(), "CPU".to_string()])),
            )]),
        }
    }

    #[test]
    fn attested_multigroup_catalog_seeds_g0_and_change_fails_closed() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 4,
        };
        let group_shapes = Mutex::new(HashMap::new());
        let ledger = Mutex::new(CacheEvidenceLedger::new(16));
        ledger.lock().set_expected_owners([owner]);
        let barriers = Mutex::new(BarrierCoordinator::new(8));
        let cold_epoch = Mutex::new(ColdEpochCoordinator::default());
        let catalog = vec![
            catalog_group(1, "sliding_window", 8),
            catalog_group(0, "full_attention", 256),
        ];

        assert!(record_group_catalog(
            owner,
            &catalog,
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            None,
        ));
        assert_eq!(group_shapes.lock()[&owner][0].group_idx, 0);
        assert_eq!(ledger.lock().stats().expected_owners, 1);
        assert_eq!(ledger.lock().stats().cataloged_owners, 1);
        assert!(record_group_catalog(
            owner,
            &catalog,
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            None,
        ));

        let changed = vec![catalog_group(0, "full_attention", 128)];
        assert!(!record_group_catalog(
            owner,
            &changed,
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            None,
        ));
        let stats = ledger.lock().stats();
        assert!(!stats.physical_telemetry_complete);
        assert!(!stats.history_complete);
    }

    #[test]
    fn missing_or_changed_attestation_removes_prior_catalog() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 4,
        };
        let group_shapes = Mutex::new(HashMap::new());
        let ledger = Mutex::new(CacheEvidenceLedger::new(16));
        ledger.lock().set_expected_owners([owner]);
        let barriers = Mutex::new(BarrierCoordinator::new(8));
        let cold_epoch = Mutex::new(ColdEpochCoordinator::default());
        let catalog = vec![catalog_group(0, "full_attention", 256)];

        sync_attested_group_catalogs(
            &catalog_membership_view(Some(catalog)),
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            None,
        );
        assert!(group_shapes.lock().contains_key(&owner));

        sync_attested_group_catalogs(
            &catalog_membership_view(Some(vec![catalog_group(0, "full_attention", 128)])),
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            None,
        );
        assert!(!group_shapes.lock().contains_key(&owner));
        assert!(!ledger.lock().stats().physical_telemetry_complete);

        sync_attested_group_catalogs(
            &catalog_membership_view(None),
            &group_shapes,
            &ledger,
            &barriers,
            &cold_epoch,
            None,
        );
        assert!(!group_shapes.lock().contains_key(&owner));
    }

    #[test]
    fn unsupported_attested_group_kind_never_counts_as_cataloged() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 4,
        };
        let group_shapes = Mutex::new(HashMap::new());
        let ledger = Mutex::new(CacheEvidenceLedger::new(16));
        ledger.lock().set_expected_owners([owner]);

        assert!(!record_group_catalog(
            owner,
            &[catalog_group(0, "unknown_attention", 256)],
            &group_shapes,
            &ledger,
            &Mutex::new(BarrierCoordinator::new(8)),
            &Mutex::new(ColdEpochCoordinator::default()),
            None,
        ));
        assert_eq!(ledger.lock().stats().cataloged_owners, 0);
    }

    fn store(group: u32, medium: &str) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(42)],
            parent_block_hash: None,
            parent_sequence_hash: None,
            parent_sequence_hash_algorithm: None,
            token_ids: vec![1, 2],
            block_size: 2,
            medium: Some(medium.to_string()),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(group),
            kv_cache_spec_kind: Some(KvCacheSpecKind::MlaAttention),
            kv_cache_spec_sliding_window: None,
            locality: Some(Locality::Local),
            ownership: None,
        }
    }

    #[test]
    fn observer_tracks_all_required_groups_before_declaring_residency() {
        let observer = CacheEvidenceObserver::new(16);
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        {
            let ledger = observer.ledger();
            let mut ledger = ledger.lock();
            ledger.record_group_catalog(owner, CacheTier::Cpu, [0, 1]);
            ledger.seal_physical_scope();
        }

        observer.observe(&store(0, "CPU"), worker);
        assert_eq!(
            observer.ledger().lock().resident_on(42, owner),
            KnownBool::No
        );
        observer.observe(&store(1, "CPU"), worker);
        assert_eq!(
            observer.ledger().lock().resident_on(42, owner),
            KnownBool::Yes
        );
    }

    #[test]
    fn evidence_sequence_tracker_detects_gaps_stale_batches_and_restarts() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let mut tracker = EvidenceSequenceTracker::default();
        assert_eq!(
            tracker.observe(owner, 11, 0),
            EvidenceSequenceObservation::Contiguous
        );
        assert_eq!(
            tracker.observe(owner, 11, 2),
            EvidenceSequenceObservation::Gap
        );
        assert_eq!(
            tracker.observe(owner, 11, 1),
            EvidenceSequenceObservation::Stale
        );
        assert_eq!(
            tracker.observe(owner, 12, 0),
            EvidenceSequenceObservation::PublisherRestart
        );
        assert_eq!(
            tracker.observe(owner, 12, 1),
            EvidenceSequenceObservation::Contiguous
        );
    }

    #[test]
    fn first_late_evidence_batch_is_a_gap() {
        let mut tracker = EvidenceSequenceTracker::default();
        assert_eq!(
            tracker.observe(
                CacheOwner {
                    worker_id: 7,
                    dp_rank: 0,
                },
                11,
                3,
            ),
            EvidenceSequenceObservation::Gap
        );
    }

    #[test]
    fn validated_cold_epoch_clear_can_baseline_an_unseen_publisher() {
        let owner = barrier_owner(7);
        let mut coordinator = ColdEpochCoordinator::default();
        let outcome = coordinator
            .begin(
                "0123456789abcdef0123456789abcdef".to_string(),
                41,
                HashMap::from([(owner, epoch_fence(70))]),
            )
            .unwrap();
        let mut clear = epoch_batch(
            owner,
            70,
            None,
            vec![
                CacheEvidenceMutation::Clear {
                    tier: Some(CacheTier::Gpu),
                },
                CacheEvidenceMutation::Clear {
                    tier: Some(CacheTier::Cpu),
                },
            ],
        );
        clear.source_cursor = 50;
        assert!(coordinator.accepts_clear_baseline(&clear));

        let mut tracker = EvidenceSequenceTracker::default();
        assert_eq!(
            tracker.observe_with_initial_baseline(owner, 70, 50, true),
            EvidenceSequenceObservation::Contiguous
        );
        assert!(coordinator.observe_clear_batch(&clear));

        let mut marker = epoch_batch(owner, 70, Some(41), Vec::new());
        marker.source_cursor = 51;
        assert!(!coordinator.accepts_clear_baseline(&marker));
        assert_eq!(
            tracker.observe(owner, 70, 51),
            EvidenceSequenceObservation::Contiguous
        );
        assert!(coordinator.observe_marker(&marker));
        coordinator.controls_succeeded("0123456789abcdef0123456789abcdef");
        assert_eq!(*outcome.borrow(), ColdEpochOutcome::EvidenceReady);
    }

    #[test]
    fn invalid_epoch_clear_cannot_baseline_an_unseen_publisher() {
        let owner = barrier_owner(7);
        let mut coordinator = ColdEpochCoordinator::default();
        coordinator
            .begin(
                "0123456789abcdef0123456789abcdef".to_string(),
                41,
                HashMap::from([(owner, epoch_fence(70))]),
            )
            .unwrap();
        let mut missing_cpu = epoch_batch(
            owner,
            70,
            None,
            vec![CacheEvidenceMutation::Clear {
                tier: Some(CacheTier::Gpu),
            }],
        );
        missing_cpu.source_cursor = 50;

        assert!(!coordinator.accepts_clear_baseline(&missing_cpu));
        assert_eq!(
            EvidenceSequenceTracker::default().observe_with_initial_baseline(owner, 70, 50, false,),
            EvidenceSequenceObservation::Gap
        );
    }

    #[test]
    fn interleaved_dp_ranks_share_the_publisher_sequence() {
        let mut tracker = EvidenceSequenceTracker::default();
        let rank_zero = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let rank_one = CacheOwner {
            worker_id: 7,
            dp_rank: 1,
        };
        assert_eq!(
            tracker.observe(rank_zero, 11, 0),
            EvidenceSequenceObservation::Contiguous
        );
        assert_eq!(
            tracker.observe(rank_one, 11, 1),
            EvidenceSequenceObservation::Contiguous
        );
        assert_eq!(
            tracker.observe(rank_zero, 11, 2),
            EvidenceSequenceObservation::Contiguous
        );
    }

    #[test]
    fn delayed_retired_publisher_cannot_reclaim_an_owner() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let mut tracker = EvidenceSequenceTracker::default();
        assert_eq!(
            tracker.observe(owner, 11, 0),
            EvidenceSequenceObservation::Contiguous
        );
        assert_eq!(
            tracker.observe(owner, 12, 0),
            EvidenceSequenceObservation::PublisherRestart
        );
        assert_eq!(
            tracker.observe(owner, 11, 1),
            EvidenceSequenceObservation::Stale
        );
        assert_eq!(
            tracker.observe(owner, 12, 1),
            EvidenceSequenceObservation::Contiguous
        );
    }

    #[test]
    fn retiring_an_owner_fences_its_previous_publisher() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let mut tracker = EvidenceSequenceTracker::default();
        assert_eq!(
            tracker.observe(owner, 11, 0),
            EvidenceSequenceObservation::Contiguous
        );
        tracker.retire(owner);
        assert_eq!(
            tracker.observe(owner, 11, 1),
            EvidenceSequenceObservation::Stale
        );
        assert_eq!(
            tracker.observe(owner, 12, 0),
            EvidenceSequenceObservation::Contiguous
        );
    }

    #[test]
    fn watermark_must_follow_every_applied_source_batch() {
        assert_eq!(
            observe_watermark(Some(3), Some(4)),
            EvidenceWatermarkObservation::Ahead
        );
        assert_eq!(
            observe_watermark(Some(4), Some(3)),
            EvidenceWatermarkObservation::Stale
        );
        assert_eq!(
            observe_watermark(Some(4), Some(4)),
            EvidenceWatermarkObservation::Current
        );
    }

    #[test]
    fn idle_heartbeat_establishes_bounded_freshness() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let now = Instant::now();
        let mut freshness = EvidenceFreshness::new(Duration::from_secs(5));
        let _ = freshness.set_expected(HashMap::from([(owner, Some(11))]));
        assert_eq!(
            observe_watermark(None, None),
            EvidenceWatermarkObservation::Current
        );
        assert!(freshness.observe(owner, 11, now));
        assert_eq!(
            freshness.status(now).0,
            EvidenceFreshnessStatus::BoundedFresh
        );
    }

    #[test]
    fn bounded_and_exact_attribution_have_distinct_metric_buckets() {
        assert_eq!(CacheEvidenceQuality::default().metric_label(), "incomplete");
        assert_eq!(
            CacheEvidenceQuality::BoundedPhysical.metric_label(),
            "bounded_physical"
        );
        assert_eq!(CacheEvidenceQuality::Exact.metric_label(), "exact");
    }

    #[test]
    fn old_watermark_becomes_stale() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let now = Instant::now();
        let mut freshness = EvidenceFreshness::new(Duration::from_secs(5));
        let _ = freshness.set_expected(HashMap::from([(owner, Some(11))]));
        assert!(freshness.observe(owner, 11, now));
        assert_eq!(
            freshness.status(now + Duration::from_secs(6)).0,
            EvidenceFreshnessStatus::Stale
        );
    }

    #[test]
    fn membership_restart_requires_the_new_incarnation_watermark() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let now = Instant::now();
        let mut freshness = EvidenceFreshness::new(Duration::from_secs(5));
        let _ = freshness.set_expected(HashMap::from([(owner, Some(11))]));
        assert!(freshness.observe(owner, 11, now));
        let restarted = freshness.set_expected(HashMap::from([(owner, Some(12))]));
        assert_eq!(restarted, vec![owner]);
        assert!(!freshness.accepts(owner, 11));
        assert_eq!(freshness.status(now).0, EvidenceFreshnessStatus::Missing);
    }

    #[test]
    fn inactive_owner_cannot_refresh_evidence() {
        let owner = CacheOwner {
            worker_id: 7,
            dp_rank: 0,
        };
        let mut freshness = EvidenceFreshness::new(Duration::from_secs(5));
        let _ = freshness.set_expected(HashMap::new());
        assert!(!freshness.observe(owner, 11, Instant::now()));
    }

    #[test]
    fn request_hashes_follow_each_cache_groups_real_block_shape() {
        let groups = cache_group_hashes(
            &(0..64).collect::<Vec<_>>(),
            &[
                CacheGroupObservation {
                    group_idx: 0,
                    kind: "full_attention".to_string(),
                    block_size: 32,
                    sliding_window: None,
                    is_eagle: false,
                    alignment_tokens: Some(32),
                },
                CacheGroupObservation {
                    group_idx: 1,
                    kind: "sliding_window".to_string(),
                    block_size: 8,
                    sliding_window: Some(16),
                    is_eagle: false,
                    alignment_tokens: Some(32),
                },
            ],
            None,
            None,
        )
        .unwrap();

        assert_eq!(groups[0].sequence_hashes.len(), 2);
        assert_eq!(groups[1].sequence_hashes.len(), 8);
        assert_eq!(groups[0].kind, CacheGroupKind::FullAttention);
        assert_eq!(groups[1].kind, CacheGroupKind::SlidingWindow);
    }

    fn barrier_owner(worker_id: u64) -> CacheOwner {
        CacheOwner {
            worker_id,
            dp_rank: 0,
        }
    }

    fn barrier_coordinator(max_pending: usize) -> BarrierCoordinator {
        let mut coordinator = BarrierCoordinator::new(max_pending);
        coordinator.set_owners(HashMap::from([
            (barrier_owner(7), 70),
            (barrier_owner(8), 80),
        ]));
        coordinator
    }

    fn provisional_route(ticket: CacheEvidenceBarrierTicket) -> CacheLossRouteObservation {
        CacheLossRouteObservation {
            quality: CacheEvidenceQuality::BoundedPhysical,
            complete: true,
            barrier: Some(ticket),
            ..Default::default()
        }
    }

    #[test]
    fn exact_choice_loss_waits_for_every_contiguous_owner_marker() {
        let mut coordinator = barrier_coordinator(8);
        let (first, command) = coordinator.begin(HashSet::from([41]));
        let (second, coalesced) = coordinator.begin(HashSet::from([42]));
        let barrier_id = command.unwrap();
        assert_eq!(coalesced, None);
        assert_eq!(coordinator.dispatch(barrier_id).unwrap().1, 2);
        coordinator.controls_succeeded(barrier_id);

        coordinator.marker(barrier_owner(7), barrier_id);
        assert_eq!(first.outcome(), BarrierOutcome::Pending);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(first.outcome(), BarrierOutcome::Exact);
        assert_eq!(second.outcome(), BarrierOutcome::Exact);
    }

    #[test]
    fn resolved_exact_is_applied_before_first_streamed_observation() {
        let mut coordinator = barrier_coordinator(8);
        let (ticket, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.controls_succeeded(barrier_id);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);
        let mut route = provisional_route(ticket);

        assert!(route.finalize_barrier_if_ready());
        assert_eq!(route.quality, CacheEvidenceQuality::Exact);
        assert!(route.complete);
        assert!(route.barrier.is_none());
    }

    #[test]
    fn resolved_incomplete_is_applied_before_first_streamed_observation() {
        let mut missing = BarrierCoordinator::new(8);
        let (missing_ticket, _) = missing.begin(HashSet::from([41]));

        let mut control = barrier_coordinator(8);
        let (control_ticket, command) = control.begin(HashSet::from([41]));
        control.fail_round(command.unwrap(), "barrier_control_failed");

        let mut gap = barrier_coordinator(8);
        let (gap_ticket, _) = gap.begin(HashSet::from([41]));
        gap.fail_owner(barrier_owner(7), "barrier_gap");

        let mut mutation = barrier_coordinator(8);
        let (mutation_ticket, _) = mutation.begin(HashSet::from([41]));
        mutation.mutation(barrier_owner(7), &HashSet::from([41]), false);

        for (ticket, reason) in [
            (missing_ticket, "barrier_missing_capability"),
            (control_ticket, "barrier_control_failed"),
            (gap_ticket, "barrier_gap"),
            (mutation_ticket, "barrier_relevant_mutation"),
        ] {
            let mut route = provisional_route(ticket);
            assert!(route.finalize_barrier_if_ready());
            assert_eq!(route.quality, CacheEvidenceQuality::Incomplete);
            assert!(!route.complete);
            assert_eq!(route.incomplete_reason, Some(reason));
            assert!(route.barrier.is_none());
        }
    }

    #[tokio::test]
    async fn barrier_control_response_requires_one_matching_success() {
        let success = || {
            Annotated::from_data(serde_json::json!({
                "status": "success",
                "barrier_id": 41
            }))
        };

        let mut valid = futures::stream::iter([success()]);
        assert_eq!(
            validate_barrier_responses(&mut valid, 41).await.unwrap(),
            BarrierControlDisposition::Success
        );

        let mut permanent = futures::stream::iter([Annotated::from_data(serde_json::json!({
            "status": "error",
            "code": "barrier_permanently_unavailable",
            "barrier_id": 41
        }))]);
        assert_eq!(
            validate_barrier_responses(&mut permanent, 41)
                .await
                .unwrap(),
            BarrierControlDisposition::PermanentlyUnavailable
        );

        let invalid_cases = [
            vec![Annotated::from_data(
                serde_json::json!({ "status": "error", "message": "engine failed" }),
            )],
            vec![Annotated::from_data(
                serde_json::json!({ "status": "success", "barrier_id": 42 }),
            )],
            vec![Annotated::from_data(serde_json::json!({
                "status": "error",
                "code": "barrier_timeout",
                "barrier_id": 41
            }))],
            vec![success(), success()],
            vec![
                success(),
                Annotated::from_data(serde_json::json!({ "status": "progress" })),
            ],
            Vec::new(),
        ];
        for responses in invalid_cases {
            let mut responses = futures::stream::iter(responses);
            assert!(
                validate_barrier_responses(&mut responses, 41)
                    .await
                    .is_err()
            );
        }
    }

    #[test]
    fn eviction_race_fails_only_the_relevant_snapshot_before_its_marker() {
        let mut coordinator = barrier_coordinator(8);
        let (relevant, command) = coordinator.begin(HashSet::from([41]));
        let (unrelated, _) = coordinator.begin(HashSet::from([42]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.controls_succeeded(barrier_id);

        coordinator.mutation(barrier_owner(7), &HashSet::from([41]), false);
        assert_eq!(
            relevant.outcome(),
            BarrierOutcome::Incomplete("barrier_relevant_mutation")
        );
        assert_eq!(unrelated.outcome(), BarrierOutcome::Pending);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(unrelated.outcome(), BarrierOutcome::Exact);
    }

    #[test]
    fn mutation_after_owner_marker_does_not_retroactively_invalidate_cut() {
        let mut coordinator = barrier_coordinator(8);
        let (ticket, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.controls_succeeded(barrier_id);

        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.mutation(barrier_owner(7), &HashSet::from([41]), false);
        assert_eq!(ticket.outcome(), BarrierOutcome::Pending);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(ticket.outcome(), BarrierOutcome::Exact);
    }

    #[test]
    fn clear_before_marker_cannot_become_exact() {
        let mut coordinator = barrier_coordinator(8);
        let (ticket, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.controls_succeeded(barrier_id);

        coordinator.mutation(barrier_owner(7), &HashSet::new(), true);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);

        assert_eq!(
            ticket.outcome(),
            BarrierOutcome::Incomplete("barrier_clear")
        );
    }

    #[test]
    fn marker_before_control_error_cannot_complete_the_cut() {
        let mut coordinator = barrier_coordinator(8);
        let (ticket, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(ticket.outcome(), BarrierOutcome::Pending);

        coordinator.fail_round(barrier_id, "barrier_control_failed");
        assert_eq!(
            ticket.outcome(),
            BarrierOutcome::Incomplete("barrier_control_failed")
        );
    }

    #[test]
    fn apply_integrity_failure_invalidates_pending_cut_before_marker() {
        let owner = barrier_owner(7);
        let mut ledger = CacheEvidenceLedger::new(8);
        ledger.record_group_catalog(owner, CacheTier::Gpu, [0]);
        let initial = CacheEvidenceBatch {
            owner,
            source_cursor: 1,
            source_incarnation_id: Some(70),
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations: vec![CacheEvidenceMutation::Store {
                tier: CacheTier::Gpu,
                group_idx: Some(0),
                parent_external_hash: None,
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 100,
                    tokens_hash: 41,
                }],
            }],
            barrier_id: None,
            epoch_id: None,
        };
        assert!(ledger.apply_evidence_batch(&initial));

        let mut coordinator = barrier_coordinator(8);
        let (ticket, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        let conflicting = CacheEvidenceBatch {
            source_cursor: 2,
            mutations: vec![CacheEvidenceMutation::Store {
                tier: CacheTier::Gpu,
                group_idx: Some(0),
                parent_external_hash: None,
                blocks: vec![CacheEvidenceStoredBlock {
                    external_hash: 100,
                    tokens_hash: 42,
                }],
            }],
            ..initial
        };
        assert!(!ledger.apply_evidence_batch(&conflicting));
        coordinator.fail_owner(owner, "barrier_apply_integrity_failure");
        coordinator.controls_succeeded(barrier_id);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(
            ticket.outcome(),
            BarrierOutcome::Incomplete("barrier_apply_integrity_failure")
        );
    }

    #[test]
    fn barrier_fails_closed_for_clear_gap_restart_membership_catalog_and_timeout() {
        for (reason, fail) in [
            ("barrier_clear", 0_u8),
            ("barrier_gap", 1),
            ("barrier_restart", 2),
            ("barrier_membership_changed", 3),
            ("barrier_catalog_changed", 4),
            ("barrier_timeout", 5),
        ] {
            let mut coordinator = barrier_coordinator(8);
            let (ticket, command) = coordinator.begin(HashSet::from([41]));
            let barrier_id = command.unwrap();
            coordinator.dispatch(barrier_id);
            match fail {
                0 => {
                    coordinator.mutation(barrier_owner(7), &HashSet::new(), true);
                }
                1 => {
                    coordinator.fail_owner(barrier_owner(7), "barrier_gap");
                }
                2 => {
                    coordinator.fail_owner(barrier_owner(7), "barrier_restart");
                }
                3 => coordinator.set_owners(HashMap::from([(barrier_owner(7), 71)])),
                4 => {
                    coordinator.fail_all("barrier_catalog_changed");
                }
                5 => {
                    coordinator.fail_round(barrier_id, "barrier_timeout");
                }
                _ => unreachable!(),
            }
            assert_eq!(ticket.outcome(), BarrierOutcome::Incomplete(reason));
        }
    }

    #[test]
    fn barrier_fails_closed_for_missing_capability_and_journal_overflow() {
        let mut missing = BarrierCoordinator::new(1);
        let (ticket, command) = missing.begin(HashSet::from([41]));
        assert_eq!(command, None);
        assert_eq!(
            ticket.outcome(),
            BarrierOutcome::Incomplete("barrier_missing_capability")
        );

        let mut full = barrier_coordinator(1);
        let (_first, command) = full.begin(HashSet::from([41]));
        assert!(command.is_some());
        let (overflow, command) = full.begin(HashSet::from([42]));
        assert_eq!(command, None);
        assert_eq!(
            overflow.outcome(),
            BarrierOutcome::Incomplete("barrier_journal_overflow")
        );
    }

    #[test]
    fn barrier_timeout_is_cut_local_and_next_success_can_be_exact() {
        let owner = barrier_owner(7);
        let mut ledger = CacheEvidenceLedger::new(8);
        ledger.set_expected_owners([owner]);
        ledger.record_group_catalog(owner, CacheTier::Gpu, [0]);
        assert!(ledger.stats().physical_telemetry_complete);

        let mut coordinator = barrier_coordinator(8);
        let (timed_out, command) = coordinator.begin(HashSet::from([41]));
        coordinator.fail_round(command.unwrap(), "barrier_timeout");
        assert_eq!(
            timed_out.outcome(),
            BarrierOutcome::Incomplete("barrier_timeout")
        );
        assert!(ledger.stats().physical_telemetry_complete);

        let (next, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.controls_succeeded(barrier_id);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(next.outcome(), BarrierOutcome::Exact);
        assert!(ledger.stats().physical_telemetry_complete);
    }

    #[test]
    fn permanent_unavailability_circuit_breaks_until_new_incarnation() {
        let mut coordinator = barrier_coordinator(8);
        let (failed, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.mark_permanently_unavailable(barrier_owner(7), 70);
        assert_eq!(
            failed.outcome(),
            BarrierOutcome::Incomplete("barrier_permanently_unavailable")
        );

        let (same_incarnation, command) = coordinator.begin(HashSet::from([41]));
        assert_eq!(command, None);
        assert_eq!(
            same_incarnation.outcome(),
            BarrierOutcome::Incomplete("barrier_missing_capability")
        );
        coordinator.set_owners(HashMap::from([
            (barrier_owner(7), 70),
            (barrier_owner(8), 80),
        ]));
        let (_, command) = coordinator.begin(HashSet::from([41]));
        assert_eq!(command, None);

        coordinator.set_owners(HashMap::from([
            (barrier_owner(7), 71),
            (barrier_owner(8), 80),
        ]));
        let (recovered, command) = coordinator.begin(HashSet::from([41]));
        let barrier_id = command.unwrap();
        coordinator.dispatch(barrier_id);
        coordinator.controls_succeeded(barrier_id);
        coordinator.marker(barrier_owner(7), barrier_id);
        coordinator.marker(barrier_owner(8), barrier_id);
        assert_eq!(recovered.outcome(), BarrierOutcome::Exact);
    }

    #[test]
    fn selected_owner_requires_the_exact_fenced_serving_incarnation() {
        let mut coordinator = barrier_coordinator(8);
        assert!(coordinator.matches_selected_incarnation(barrier_owner(7), Some(70)));
        assert!(!coordinator.matches_selected_incarnation(barrier_owner(7), Some(71)));
        assert!(!coordinator.matches_selected_incarnation(barrier_owner(9), Some(90)));
        assert!(!coordinator.matches_selected_incarnation(barrier_owner(7), None));

        coordinator.set_owner_incarnations(
            HashMap::from([(barrier_owner(7), 700)]),
            HashMap::from([(barrier_owner(7), 71)]),
        );
        assert!(!coordinator.matches_selected_incarnation(barrier_owner(7), Some(70)));
        assert!(coordinator.matches_selected_incarnation(barrier_owner(7), Some(71)));
        assert!(!coordinator.matches_selected_incarnation(barrier_owner(8), Some(80)));
    }

    fn epoch_fence(evidence_incarnation: u64) -> ColdEpochOwnerFence {
        ColdEpochOwnerFence {
            evidence_incarnation,
            serving_incarnation: evidence_incarnation + 1_000,
            expected_tiers: HashSet::from([CacheTier::Gpu, CacheTier::Cpu]),
        }
    }

    #[test]
    fn identical_membership_refresh_preserves_completed_cold_history_epoch() {
        let owner = barrier_owner(7);
        let owners = HashSet::from([owner]);
        let membership = Some(HashMap::from([(owner, epoch_fence(70))]));
        let mut previous = membership.clone();
        let mut ledger = CacheEvidenceLedger::new(8);
        ledger.set_expected_owners(owners.iter().copied());
        ledger.record_group_catalog(owner, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(owner, CacheTier::Cpu, [0]);
        assert!(ledger.commit_cold_history_epoch(&owners).is_some());

        assert!(!reconcile_history_membership(
            &mut previous,
            membership,
            &mut ledger,
        ));
        assert!(ledger.stats().history_complete);
    }

    #[test]
    fn cold_epoch_commit_exports_completed_empty_history_state() {
        let owner = barrier_owner(7);
        let owners = HashSet::from([owner]);
        let mut ledger = CacheEvidenceLedger::new(8);
        ledger.set_expected_owners(owners.iter().copied());
        ledger.record_group_catalog(owner, CacheTier::Gpu, [0]);
        ledger.record_group_catalog(owner, CacheTier::Cpu, [0]);
        ledger.record_seen_blocks([11, 12]);

        assert!(ledger.commit_cold_history_epoch(&owners).is_some());
        let exported = ledger.stats();
        assert!(exported.history_complete);
        assert_eq!(exported.history_blocks, 0);
    }

    fn cold_epoch_control_instance(instance_id: u64) -> Instance {
        Instance {
            component: "backend".to_string(),
            endpoint: "begin_cache_evidence_epoch".to_string(),
            namespace: "test".to_string(),
            instance_id,
            transport: dynamo_runtime::component::TransportType::Nats("test".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    #[tokio::test]
    async fn cold_epoch_control_waits_for_delayed_initial_discovery() {
        let (instances_tx, instances_rx) = watch::channel(Vec::new());
        let mut waiter = tokio::spawn(async move {
            wait_for_cold_epoch_workers(instances_rx, &HashSet::from([7])).await
        });

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut waiter)
                .await
                .is_err()
        );
        instances_tx.send_replace(vec![cold_epoch_control_instance(7)]);

        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cold_epoch_control_requires_every_owner_in_discovery() {
        let (instances_tx, instances_rx) = watch::channel(vec![cold_epoch_control_instance(7)]);
        let mut waiter = tokio::spawn(async move {
            wait_for_cold_epoch_workers(instances_rx, &HashSet::from([7, 8])).await
        });

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut waiter)
                .await
                .is_err()
        );
        instances_tx.send_replace(vec![
            cold_epoch_control_instance(7),
            cold_epoch_control_instance(8),
        ]);

        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cold_epoch_control_response_is_strictly_correlated() {
        let success = || {
            Annotated::from_data(serde_json::json!({
                "status": "success",
                "epoch_id": "0123456789abcdef0123456789abcdef",
                "barrier_id": 41,
                "serving_incarnation": 1070,
                "cleared_media": ["CPU", "GPU"]
            }))
        };
        let mut valid = futures::stream::iter(vec![success()]);
        validate_cold_epoch_response(
            &mut valid,
            "0123456789abcdef0123456789abcdef",
            Some(41),
            Some(1070),
            Some(&HashSet::from([CacheTier::Gpu, CacheTier::Cpu])),
        )
        .await
        .unwrap();

        for responses in [
            Vec::new(),
            vec![success(), success()],
            vec![Annotated::from_data(serde_json::json!({
                "status": "error",
                "epoch_id": "0123456789abcdef0123456789abcdef",
                "barrier_id": 41,
            }))],
            vec![Annotated::from_data(serde_json::json!({
                "status": "success",
                "epoch_id": "1123456789abcdef0123456789abcdef",
                "barrier_id": 41,
                "serving_incarnation": 1070,
                "cleared_media": ["CPU", "GPU"]
            }))],
        ] {
            let mut responses = futures::stream::iter(responses);
            assert!(
                validate_cold_epoch_response(
                    &mut responses,
                    "0123456789abcdef0123456789abcdef",
                    Some(41),
                    Some(1070),
                    Some(&HashSet::from([CacheTier::Gpu, CacheTier::Cpu])),
                )
                .await
                .is_err()
            );
        }
    }

    fn epoch_batch(
        owner: CacheOwner,
        source_incarnation_id: u64,
        barrier_id: Option<u64>,
        mutations: Vec<CacheEvidenceMutation>,
    ) -> CacheEvidenceBatch {
        CacheEvidenceBatch {
            owner,
            source_cursor: if barrier_id.is_some() { 2 } else { 1 },
            source_incarnation_id: Some(source_incarnation_id),
            heartbeat: false,
            watermark_source_cursor: None,
            telemetry_complete: true,
            mutations,
            barrier_id,
            epoch_id: Some("0123456789abcdef0123456789abcdef".to_string()),
        }
    }

    #[test]
    fn cold_epoch_requires_exact_tier_clears_markers_and_controls_for_every_owner() {
        let owners = HashMap::from([
            (barrier_owner(7), epoch_fence(70)),
            (barrier_owner(8), epoch_fence(80)),
        ]);
        let mut coordinator = ColdEpochCoordinator::default();
        let outcome = coordinator
            .begin("0123456789abcdef0123456789abcdef".to_string(), 41, owners)
            .unwrap();
        for (owner, incarnation) in [(barrier_owner(7), 70), (barrier_owner(8), 80)] {
            assert!(coordinator.observe_clear_batch(&epoch_batch(
                owner,
                incarnation,
                None,
                vec![
                    CacheEvidenceMutation::Clear {
                        tier: Some(CacheTier::Gpu),
                    },
                    CacheEvidenceMutation::Clear {
                        tier: Some(CacheTier::Cpu),
                    },
                ],
            )));
            assert!(coordinator.observe_marker(&epoch_batch(
                owner,
                incarnation,
                Some(41),
                Vec::new(),
            )));
        }
        assert_eq!(*outcome.borrow(), ColdEpochOutcome::Pending);
        coordinator.controls_succeeded("0123456789abcdef0123456789abcdef");
        assert_eq!(*outcome.borrow(), ColdEpochOutcome::EvidenceReady);
    }

    #[test]
    fn cold_epoch_fails_closed_on_missing_cpu_duplicate_clear_and_mutation_race() {
        for mutations in [
            vec![CacheEvidenceMutation::Clear {
                tier: Some(CacheTier::Gpu),
            }],
            vec![
                CacheEvidenceMutation::Clear {
                    tier: Some(CacheTier::Gpu),
                },
                CacheEvidenceMutation::Clear {
                    tier: Some(CacheTier::Gpu),
                },
            ],
        ] {
            let mut coordinator = ColdEpochCoordinator::default();
            let outcome = coordinator
                .begin(
                    "0123456789abcdef0123456789abcdef".to_string(),
                    41,
                    HashMap::from([(barrier_owner(7), epoch_fence(70))]),
                )
                .unwrap();
            coordinator.observe_clear_batch(&epoch_batch(barrier_owner(7), 70, None, mutations));
            assert!(matches!(
                *outcome.borrow(),
                ColdEpochOutcome::Incomplete("cold_epoch_invalid_clear")
            ));
        }

        let mut coordinator = ColdEpochCoordinator::default();
        let outcome = coordinator
            .begin(
                "0123456789abcdef0123456789abcdef".to_string(),
                41,
                HashMap::from([(barrier_owner(7), epoch_fence(70))]),
            )
            .unwrap();
        coordinator.observe_clear_batch(&epoch_batch(
            barrier_owner(7),
            70,
            None,
            vec![
                CacheEvidenceMutation::Clear {
                    tier: Some(CacheTier::Gpu),
                },
                CacheEvidenceMutation::Clear {
                    tier: Some(CacheTier::Cpu),
                },
            ],
        ));
        coordinator.unexpected_mutation(barrier_owner(7));
        assert_eq!(
            *outcome.borrow(),
            ColdEpochOutcome::Incomplete("cold_epoch_mutation_after_clear")
        );
    }

    #[tokio::test]
    async fn cold_epoch_fence_drains_selected_requests_before_closing_dispatch() {
        let fence = Arc::new(tokio::sync::RwLock::new(()));
        let selected_before_epoch = Arc::clone(&fence).read_owned().await;
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let writer = {
            let fence = Arc::clone(&fence);
            tokio::spawn(async move {
                let guard = fence.write_owned().await;
                entered_tx.send(()).unwrap();
                guard
            })
        };

        assert!(
            tokio::time::timeout(Duration::from_millis(10), entered_rx.recv())
                .await
                .is_err(),
            "epoch must wait for a selection that has not dispatched"
        );
        drop(selected_before_epoch);
        tokio::time::timeout(Duration::from_secs(1), entered_rx.recv())
            .await
            .expect("epoch acquired dispatch fence")
            .expect("epoch fence observer remained live");
        let epoch_guard = writer.await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), Arc::clone(&fence).read_owned())
                .await
                .is_err(),
            "new selections must wait while the cold epoch owns the fence"
        );
        drop(epoch_guard);
        tokio::time::timeout(Duration::from_secs(1), fence.read_owned())
            .await
            .expect("selection resumes after epoch release");
    }

    #[tokio::test]
    async fn cancelled_cold_epoch_writer_does_not_block_new_selections() {
        let fence = Arc::new(tokio::sync::RwLock::new(()));
        let held_selection = Arc::clone(&fence).read_owned().await;

        assert!(
            tokio::time::timeout(Duration::from_millis(10), Arc::clone(&fence).write_owned())
                .await
                .is_err()
        );
        tokio::time::timeout(Duration::from_secs(1), Arc::clone(&fence).read_owned())
            .await
            .expect("timed-out epoch writer must leave the read queue usable");
        drop(held_selection);
    }

    #[test]
    fn cold_epoch_scope_guard_reopens_dispatch_and_keeps_history_incomplete() {
        let ledger = Arc::new(Mutex::new(CacheEvidenceLedger::new(8)));
        let (sender, receiver) = watch::channel(false);
        {
            let _release = DispatchGateRelease {
                sender,
                ledger: Arc::clone(&ledger),
                committed: false,
            };
        }

        assert!(*receiver.borrow());
        assert!(!ledger.lock().stats().history_complete);
    }

    #[tokio::test]
    async fn readiness_timeout_keeps_dispatch_open_and_later_catalog_retry_succeeds() {
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (_sender, receiver) = watch::channel(true);
        let attempt = |ready: Arc<std::sync::atomic::AtomicBool>| async move {
            loop {
                if ready.load(std::sync::atomic::Ordering::Acquire) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        };

        assert!(
            tokio::time::timeout(Duration::from_millis(5), attempt(Arc::clone(&ready)))
                .await
                .is_err()
        );
        assert!(*receiver.borrow(), "readiness must not close dispatch");

        ready.store(true, std::sync::atomic::Ordering::Release);
        tokio::time::timeout(Duration::from_secs(1), attempt(ready))
            .await
            .expect("a later readiness attempt observes newly learned catalogs");
        assert!(*receiver.borrow());
    }
}
