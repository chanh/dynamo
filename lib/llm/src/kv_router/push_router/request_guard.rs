// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc, time::Duration};

use dynamo_kv_router::{protocols::WorkerWithDpRank, selector::WorkerSelector};
use dynamo_runtime::{
    error::DynamoError,
    metrics::frontend_perf::{STAGE_DISPATCH, StageGuard},
    protocols::annotated::Annotated,
};

use crate::{
    kv_router::{
        KvRouter,
        metrics::{CacheLossTerminalKind, RouterRequestMetrics},
        prompt_source_outcome::{RegistrationError, TerminalPromptSourceRegistration, WaitError},
        scheduler::DefaultWorkerSelector,
    },
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        preprocessor::MigrationState,
        timing::{CacheSourceObservation, RequestPhase, RequestTracker},
    },
};
use dynamo_kv_router::cache_loss::CacheLossFunnel;

/// Owns scheduler cleanup after a worker is selected.
///
/// `worker` is captured at construction so cleanup targets the booking this
/// guard acquired, even if cleanup is delayed.
struct RequestCleanup<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    chooser: Arc<KvRouter<Sel>>,
    context_id: String,
    worker: WorkerWithDpRank,
    scheduler_tracked: bool,
    freed: bool,
}

impl<Sel> RequestCleanup<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn new(
        chooser: Arc<KvRouter<Sel>>,
        context_id: String,
        worker: WorkerWithDpRank,
        scheduler_tracked: bool,
    ) -> Self {
        Self {
            chooser,
            context_id,
            worker,
            scheduler_tracked,
            freed: false,
        }
    }

    async fn finish(&mut self) {
        if self.freed {
            return;
        }
        if self.scheduler_tracked
            && let Err(error) = self
                .chooser
                .free_if_worker(&self.context_id, self.worker)
                .await
        {
            tracing::warn!(
                request_id = %self.context_id,
                worker = ?self.worker,
                %error,
                "Failed to free request"
            );
        }
        self.freed = true;
    }
}

impl<Sel> Drop for RequestCleanup<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        if self.freed || !self.scheduler_tracked {
            return;
        }

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                request_id = %self.context_id,
                "No tokio runtime for request cleanup"
            );
            return;
        };

        let chooser = self.chooser.clone();
        let context_id = self.context_id.clone();
        let worker = self.worker;
        handle.spawn(async move {
            let result = chooser.free_if_worker(&context_id, worker).await;
            if let Err(error) = result {
                tracing::warn!(
                    request_id = %context_id,
                    ?worker,
                    %error,
                    "Failed to free request from drop guard"
                );
            }
        });
    }
}

/// Owns request-scoped timing and metrics state.
struct RequestObservability {
    tracker: Option<Arc<RequestTracker>>,
    request_metrics: Arc<RouterRequestMetrics>,
    cumulative_osl: usize,
    metrics_recorded: bool,
    first_token_recorded: bool,
    dispatch_guard: Option<StageGuard>,
    dispatched: bool,
}

impl RequestObservability {
    fn new(
        tracker: Option<Arc<RequestTracker>>,
        request_metrics: Arc<RouterRequestMetrics>,
    ) -> Self {
        Self {
            tracker,
            request_metrics,
            cumulative_osl: 0,
            metrics_recorded: false,
            first_token_recorded: false,
            dispatch_guard: None,
            dispatched: false,
        }
    }

    fn request_metrics(&self) -> &RouterRequestMetrics {
        &self.request_metrics
    }

    fn start_dispatch(&mut self, phase_label: &str) {
        self.dispatch_guard = Some(StageGuard::new(STAGE_DISPATCH, phase_label));
    }

    fn record_prefill_start(&self) {
        if let Some(tracker) = &self.tracker {
            tracker.record_prefill_start();
        }
    }

    fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }

    fn dispatched(&self) -> bool {
        self.dispatched
    }

    fn observe_response(&mut self) {
        // Taking the guard ends dispatch latency exactly once; later responses see None.
        self.dispatch_guard.take();
    }

    fn observe_tokens(&mut self, new_tokens: usize) {
        if !self.first_token_recorded && new_tokens > 0 {
            if let Some(tracker) = &self.tracker {
                tracker.record_first_token();
                if tracker.phase() == RequestPhase::Decode {
                    tracker.record_decode_first_token();
                }
                if let Some(ttft) = tracker.ttft_ms() {
                    self.request_metrics
                        .time_to_first_token_seconds
                        .observe(ttft / 1000.0);
                }
            }
            self.first_token_recorded = true;
        }

        self.cumulative_osl += new_tokens;
    }

    fn cumulative_osl(&self) -> usize {
        self.cumulative_osl
    }

    fn cache_source_observation(&self) -> Option<CacheSourceObservation> {
        self.tracker.as_ref()?.cache_source_observation().cloned()
    }

    fn observe_output_block_boundary(&self) {
        let Some(tracker) = &self.tracker else {
            return;
        };

        // Refresh finish time at block boundaries so the streaming ITL sample stays current.
        tracker.record_osl(self.cumulative_osl);
        tracker.record_finish();
        if let Some(avg_itl) = tracker.avg_itl_ms() {
            self.request_metrics
                .inter_token_latency_seconds
                .observe(avg_itl / 1000.0);
        }
    }

    fn record_metrics(&mut self) {
        // A failed dispatch never reached the backend and must not count as a request.
        if self.metrics_recorded || !self.dispatched {
            return;
        }
        self.metrics_recorded = true;

        if let Some(tracker) = &self.tracker {
            tracker.record_finish();
            tracker.record_osl(self.cumulative_osl);
            if let Some(latency) = tracker.kv_transfer_estimated_latency_secs() {
                self.request_metrics
                    .kv_transfer_estimated_latency_seconds
                    .observe(latency);
            }
        }
        if self.cumulative_osl > 0 {
            self.request_metrics
                .output_sequence_tokens
                .observe(self.cumulative_osl as f64);
        }
        self.request_metrics.requests_total.inc();
    }
}

struct OutputBlockUpdate {
    decay_fraction: Option<f64>,
}

struct CacheHistoryTracker {
    epoch: u64,
    prompt: Vec<u32>,
    branches: HashMap<u32, Vec<u32>>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    prompt_recorded: bool,
    finalized: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestTerminalLifecycle {
    Open,
    Aborting,
    Finished,
    Aborted,
}

impl CacheHistoryTracker {
    fn new(request: &PreprocessedRequest, epoch: u64) -> Self {
        Self {
            epoch,
            prompt: request.token_ids.clone(),
            branches: HashMap::new(),
            lora_name: request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.clone()),
            cache_namespace: request
                .routing
                .as_ref()
                .and_then(|routing| routing.cache_namespace.clone()),
            prompt_recorded: false,
            finalized: false,
        }
    }

    fn observe(&mut self, index: u32, token_ids: &[u32]) {
        if token_ids.is_empty() {
            return;
        }
        self.branches
            .entry(index)
            .or_default()
            .extend_from_slice(token_ids);
    }

    fn completed_sequences(&self) -> impl Iterator<Item = Vec<u32>> + '_ {
        self.branches.values().filter_map(|output| {
            let computed_output = output.get(..output.len().saturating_sub(1))?;
            let mut sequence = Vec::with_capacity(self.prompt.len() + computed_output.len());
            sequence.extend_from_slice(&self.prompt);
            sequence.extend_from_slice(computed_output);
            Some(sequence)
        })
    }

    fn aborted_sequence(
        &self,
        computed_output_tokens: usize,
        prompt_complete: bool,
    ) -> Result<Vec<u32>, &'static str> {
        if !prompt_complete {
            return Err("prompt_incomplete");
        }
        if self.branches.len() > 1 {
            return Err("multiple_output_branches");
        }
        let output = self
            .branches
            .values()
            .next()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if computed_output_tokens > output.len() {
            return Err("computed_output_exceeds_delivered_output");
        }
        let mut sequence = Vec::with_capacity(self.prompt.len() + computed_output_tokens);
        sequence.extend_from_slice(&self.prompt);
        sequence.extend_from_slice(&output[..computed_output_tokens]);
        Ok(sequence)
    }
}

/// Tracks when streamed output grows into a new scheduler accounting block.
struct OutputBlockTracker {
    track_output_blocks: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
}

impl OutputBlockTracker {
    fn new(
        track_output_blocks: bool,
        isl_tokens: usize,
        block_size: usize,
        expected_output_tokens: Option<u32>,
    ) -> Self {
        Self {
            track_output_blocks,
            current_total_blocks: isl_tokens.div_ceil(block_size),
            isl_tokens,
            block_size,
            expected_output_tokens,
        }
    }

    fn observe(&mut self, cumulative_osl: usize) -> Option<OutputBlockUpdate> {
        if !self.track_output_blocks {
            return None;
        }

        let new_total_blocks = (self.isl_tokens + cumulative_osl).div_ceil(self.block_size);
        if new_total_blocks <= self.current_total_blocks {
            return None;
        }

        // Advance before returning so a failed scheduler update preserves existing no-retry behavior.
        self.current_total_blocks = new_total_blocks;
        let decay_fraction = self
            .expected_output_tokens
            .map(|expected| (1.0 - cumulative_osl as f64 / expected.max(1) as f64).max(0.0));
        Some(OutputBlockUpdate { decay_fraction })
    }
}

/// Coordinates scheduler cleanup, observability, and streamed load tracking.
///
/// Session-affinity lifetime is separate: `AffinityAcquire` and
/// `AffinityLease` own binding commit, release, and invalidation.
pub(super) struct RequestGuard<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    cleanup: RequestCleanup<Sel>,
    observability: RequestObservability,
    output_blocks: OutputBlockTracker,
    prefill_marked: bool,
    migration_state: Option<MigrationState>,
    cache_loss_route: Option<crate::kv_router::cache_loss::CacheLossRouteObservation>,
    cache_loss_recorded: bool,
    cache_history: Option<CacheHistoryTracker>,
    terminal_prompt_source: Option<TerminalPromptSourceRegistration>,
    terminal_prompt_source_registration_error: Option<RegistrationError>,
    terminal_lifecycle: RequestTerminalLifecycle,
}

impl<Sel> RequestGuard<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub(super) fn new(
        chooser: Arc<KvRouter<Sel>>,
        request_metrics: Arc<RouterRequestMetrics>,
        context_id: String,
        worker: WorkerWithDpRank,
        request: &PreprocessedRequest,
        scheduler_tracked: bool,
    ) -> Self {
        // Snapshot request-scoped inputs now so the guard can outlive the
        // PreprocessedRequest after it is moved into backend dispatch.
        let block_size = chooser.block_size() as usize;
        let isl_tokens = request.token_ids.len();
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        let track_output_blocks =
            scheduler_tracked && chooser.kv_router_config().router_track_output_blocks;
        if scheduler_tracked {
            request_metrics.requests_started_total().inc();
        }
        let cache_history = if chooser.cache_loss_history_enabled() {
            if request.block_mm_routing_info().1.is_none() {
                Some(CacheHistoryTracker::new(
                    request,
                    chooser.cache_loss_history_epoch().unwrap_or(0),
                ))
            } else {
                chooser.mark_cache_loss_history_incomplete();
                None
            }
        } else {
            None
        };

        Self {
            cleanup: RequestCleanup::new(chooser, context_id, worker, scheduler_tracked),
            observability: RequestObservability::new(request.tracker.clone(), request_metrics),
            output_blocks: OutputBlockTracker::new(
                track_output_blocks,
                isl_tokens,
                block_size,
                expected_output_tokens,
            ),
            prefill_marked: false,
            migration_state: request.migration_state.clone(),
            cache_loss_route: None,
            cache_loss_recorded: false,
            cache_history,
            terminal_prompt_source: None,
            terminal_prompt_source_registration_error: None,
            terminal_lifecycle: RequestTerminalLifecycle::Open,
        }
    }

    pub(super) fn set_cache_loss_route(
        &mut self,
        observation: Option<crate::kv_router::cache_loss::CacheLossRouteObservation>,
    ) {
        self.cache_loss_route = observation;
        if self.cache_loss_route.is_some() && self.cleanup.scheduler_tracked {
            match self.cleanup.chooser.prompt_source_registry() {
                Some(registry) => match registry.register(self.cleanup.context_id.clone()) {
                    Ok(registration) => self.terminal_prompt_source = Some(registration),
                    Err(error) => self.terminal_prompt_source_registration_error = Some(error),
                },
                None => {
                    self.terminal_prompt_source_registration_error =
                        Some(RegistrationError::Unavailable)
                }
            }
        }
    }

    pub(super) async fn finalize_cache_loss_route(&mut self) {
        if let Some(route) = self.cache_loss_route.as_mut() {
            route.finalize_barrier().await;
        }
    }

    pub(super) fn terminal_prompt_source_nonce(&self) -> Option<u64> {
        self.terminal_prompt_source
            .as_ref()
            .map(TerminalPromptSourceRegistration::nonce)
    }

    pub(super) fn record_migration_failure(&self, error: Option<DynamoError>) {
        if let Some(state) = self.migration_state.as_ref() {
            state.record_failure(self.cleanup.worker.worker_id, error);
        }
    }

    pub(super) fn request_metrics(&self) -> &RouterRequestMetrics {
        self.observability.request_metrics()
    }

    pub(super) fn start_dispatch(&mut self, phase_label: &str) {
        self.observability.start_dispatch(phase_label);
    }

    pub(super) fn record_prefill_start(&self) {
        self.observability.record_prefill_start();
    }

    pub(super) fn mark_dispatched(&mut self) {
        self.observability.mark_dispatched();
    }

    pub(super) async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        self.observability.observe_response();

        if let (Some(data), Some(history)) = (item.data.as_ref(), self.cache_history.as_mut()) {
            history.observe(data.index.unwrap_or(0), &data.token_ids);
        }

        if !self.prefill_marked {
            let has_tokens = item
                .data
                .as_ref()
                .is_some_and(|data| !data.token_ids.is_empty());
            if has_tokens {
                if self.cleanup.scheduler_tracked
                    && let Err(error) = self
                        .cleanup
                        .chooser
                        .mark_prefill_completed(&self.cleanup.context_id)
                        .await
                {
                    tracing::warn!(
                        request_id = %self.cleanup.context_id,
                        %error,
                        "Failed to mark prefill completed"
                    );
                }
                self.prefill_marked = true;
            }
        }

        let new_tokens = item.data.as_ref().map_or(0, |data| data.token_ids.len());
        self.observability.observe_tokens(new_tokens);
        let cumulative_osl = self.observability.cumulative_osl();
        let Some(update) = self.output_blocks.observe(cumulative_osl) else {
            return;
        };

        if let Err(error) = self
            .cleanup
            .chooser
            .add_output_block(&self.cleanup.context_id, update.decay_fraction)
        {
            tracing::warn!(
                request_id = %self.cleanup.context_id,
                %error,
                "Failed to add output block"
            );
        }

        self.observability.observe_output_block_boundary();
    }

    fn observe_cache_loss_if_ready(&mut self, terminal_kind: CacheLossTerminalKind) {
        if self.cache_loss_recorded || self.cache_loss_route.is_none() {
            return;
        }
        if !self
            .cache_loss_route
            .as_mut()
            .expect("presence checked above")
            .finalize_barrier_if_ready()
        {
            return;
        }
        let Some(source) = self.observability.cache_source_observation() else {
            return;
        };
        self.observe_cache_loss_source(source, terminal_kind);
    }

    fn observe_cache_loss_source(
        &mut self,
        source: CacheSourceObservation,
        terminal_kind: CacheLossTerminalKind,
    ) {
        if self.cache_loss_recorded || self.cache_loss_route.is_none() {
            return;
        }
        self.cache_loss_recorded = true;
        self.terminal_prompt_source.take();
        let route = self
            .cache_loss_route
            .take()
            .expect("presence checked above");

        if let Some(groups) = source.cache_groups.as_deref() {
            self.cleanup
                .chooser
                .record_cache_group_catalog(self.cleanup.worker, groups);
        }

        if source.complete && source.prompt_tokens == Some(route.prompt_tokens) {
            self.record_prompt_cache_history();
        }

        let mut funnel = CacheLossFunnel::default();
        let complete =
            route.complete && source.complete && source.prompt_tokens == Some(route.prompt_tokens);
        if !complete {
            funnel.observe_incomplete(route.prompt_tokens);
            let result = if !route.complete {
                "incomplete_route"
            } else {
                "incomplete_worker"
            };
            self.observability.request_metrics().observe_cache_loss(
                &funnel,
                terminal_kind,
                "incomplete",
                result,
            );
            return;
        }

        let Some((gpu_hits, cpu_hits, cpu_lookup, reusable, physical, router_visible, selected)) =
            source
                .gpu_hit_tokens
                .zip(source.cpu_hit_tokens)
                .zip(source.cpu_lookup_tokens)
                .zip(route.reusable_prefix_tokens)
                .zip(route.physical_prefix_tokens)
                .zip(route.router_visible_prefix_tokens)
                .zip(route.selected_physical_prefix_tokens)
                .map(
                    |((((((gpu, cpu), lookup), reusable), physical), visible), selected)| {
                        (gpu, cpu, lookup, reusable, physical, visible, selected)
                    },
                )
        else {
            funnel.observe_incomplete(route.prompt_tokens);
            self.observability.request_metrics().observe_cache_loss(
                &funnel,
                terminal_kind,
                "incomplete",
                "missing_join",
            );
            return;
        };
        funnel.observe_prefix_counts(
            route.prompt_tokens,
            reusable,
            physical,
            router_visible,
            selected,
            gpu_hits.saturating_add(cpu_lookup),
            gpu_hits,
            cpu_hits,
        );
        self.observability.request_metrics().observe_cache_loss(
            &funnel,
            terminal_kind,
            route.quality.metric_label(),
            "complete",
        );
    }

    fn finalize_cache_loss(&mut self, terminal_kind: CacheLossTerminalKind) {
        self.observe_cache_loss_if_ready(terminal_kind);
        if self.cache_loss_recorded || !self.observability.dispatched() {
            return;
        }
        let Some(route) = self.cache_loss_route.take() else {
            return;
        };
        self.cache_loss_recorded = true;
        let mut funnel = CacheLossFunnel::default();
        funnel.observe_incomplete(route.prompt_tokens);
        self.observability.request_metrics().observe_cache_loss(
            &funnel,
            terminal_kind,
            "incomplete",
            "missing_join",
        );
    }

    fn record_prompt_cache_history(&mut self) {
        let Some(history) = self.cache_history.as_mut() else {
            return;
        };
        if history.prompt_recorded {
            return;
        }
        history.prompt_recorded = self.cleanup.chooser.record_completed_token_history(
            self.cleanup.worker,
            history.epoch,
            &history.prompt,
            history.lora_name.as_deref(),
            history.cache_namespace.as_deref(),
        );
    }

    fn finalize_cache_history(&mut self) {
        let Some(history) = self.cache_history.as_mut() else {
            return;
        };
        if history.finalized {
            return;
        }
        history.finalized = true;
        for sequence in history.completed_sequences() {
            self.cleanup.chooser.record_completed_token_history(
                self.cleanup.worker,
                history.epoch,
                &sequence,
                history.lora_name.as_deref(),
                history.cache_namespace.as_deref(),
            );
        }
    }

    fn abandon_cache_history(&mut self) {
        if !self.observability.dispatched() {
            return;
        }
        let Some(history) = self.cache_history.as_mut() else {
            return;
        };
        if history.finalized {
            return;
        }
        history.finalized = true;
        self.cleanup
            .chooser
            .mark_cache_loss_history_incomplete_for_epoch(history.epoch);
    }

    fn finalize_aborted_cache_history(
        &mut self,
        computed_output_tokens: u64,
        unobserved_computed_output_tokens: u64,
        prompt_complete: bool,
        incomplete_reason: Option<&str>,
    ) {
        let Some(history) = self.cache_history.as_mut() else {
            return;
        };
        if history.finalized {
            return;
        }
        history.finalized = true;
        if unobserved_computed_output_tokens > 0 || incomplete_reason.is_some() {
            self.cleanup
                .chooser
                .mark_cache_loss_history_incomplete_for_epoch(history.epoch);
            return;
        }
        let Ok(computed_output_tokens) = usize::try_from(computed_output_tokens) else {
            self.cleanup
                .chooser
                .mark_cache_loss_history_incomplete_for_epoch(history.epoch);
            return;
        };
        let Ok(sequence) = history.aborted_sequence(computed_output_tokens, prompt_complete) else {
            self.cleanup
                .chooser
                .mark_cache_loss_history_incomplete_for_epoch(history.epoch);
            return;
        };
        self.cleanup.chooser.record_completed_token_history(
            self.cleanup.worker,
            history.epoch,
            &sequence,
            history.lora_name.as_deref(),
            history.cache_namespace.as_deref(),
        );
    }

    async fn await_terminal_cache_source(&mut self) {
        if self.cache_loss_recorded || !self.observability.dispatched() {
            return;
        }
        if let Some(error) = self.terminal_prompt_source_registration_error.take() {
            let result = match error {
                RegistrationError::Full => "terminal_registry_full",
                RegistrationError::Duplicate => "terminal_registry_duplicate",
                RegistrationError::Unavailable => "terminal_registry_unavailable",
            };
            self.record_terminal_incomplete(result);
            return;
        }
        let Some(registration) = self.terminal_prompt_source.take() else {
            return;
        };
        match registration.wait(Duration::from_secs(5)).await {
            Ok(outcome) => {
                if let Some(route) = self.cache_loss_route.as_mut() {
                    route.finalize_barrier().await;
                }
                self.finalize_aborted_cache_history(
                    outcome.num_computed_output_tokens,
                    outcome.num_unobserved_computed_output_tokens,
                    outcome.cache_source.complete,
                    outcome.generated_history_incomplete_reason.as_deref(),
                );
                self.observe_cache_loss_source(
                    outcome.cache_source,
                    CacheLossTerminalKind::Cancelled,
                );
            }
            Err(WaitError::Timeout) => self.record_terminal_incomplete("terminal_timeout"),
            Err(WaitError::PublisherClosed) => {
                self.record_terminal_incomplete("terminal_publisher_closed")
            }
        }
    }

    fn record_terminal_incomplete(&mut self, result: &'static str) {
        if self.cache_loss_recorded || !self.observability.dispatched() {
            return;
        }
        let Some(route) = self.cache_loss_route.take() else {
            return;
        };
        if let Some(history) = self.cache_history.as_mut() {
            self.cleanup
                .chooser
                .mark_cache_loss_history_incomplete_for_epoch(history.epoch);
            history.finalized = true;
        }
        self.cache_loss_recorded = true;
        let mut funnel = CacheLossFunnel::default();
        funnel.observe_incomplete(route.prompt_tokens);
        self.observability.request_metrics().observe_cache_loss(
            &funnel,
            CacheLossTerminalKind::Cancelled,
            "incomplete",
            result,
        );
    }

    pub(super) async fn finish(&mut self) {
        // Metrics must observe the completed request before cleanup releases its state.
        self.finalize_cache_loss_route().await;
        self.finalize_cache_loss(CacheLossTerminalKind::Completed);
        self.finalize_cache_history();
        self.terminal_lifecycle = RequestTerminalLifecycle::Finished;
        self.observability.record_metrics();
        self.cleanup.finish().await;
    }

    pub(super) async fn abort(&mut self) {
        self.terminal_lifecycle = RequestTerminalLifecycle::Aborting;
        self.cleanup.finish().await;
        self.await_terminal_cache_source().await;
        self.finalize_cache_loss(CacheLossTerminalKind::Cancelled);
        self.finalize_cache_history();
        self.terminal_lifecycle = RequestTerminalLifecycle::Aborted;
    }
}

impl<Sel> Drop for RequestGuard<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        // RequestCleanup drops immediately afterward and performs resource cleanup.
        match self.terminal_lifecycle {
            RequestTerminalLifecycle::Aborting => {
                self.record_terminal_incomplete("terminal_wait_cancelled");
            }
            RequestTerminalLifecycle::Open => {
                self.finalize_cache_loss(CacheLossTerminalKind::Dropped);
                self.abandon_cache_history();
            }
            RequestTerminalLifecycle::Finished | RequestTerminalLifecycle::Aborted => {}
        }
        self.observability.record_metrics();
    }
}

#[cfg(test)]
mod cache_history_tests {
    use super::CacheHistoryTracker;
    use std::collections::HashMap;

    #[test]
    fn generated_history_keeps_choices_separate_and_excludes_sampled_tail() {
        let tracker = CacheHistoryTracker {
            epoch: 0,
            prompt: vec![1, 2, 3],
            branches: HashMap::from([(0, vec![4, 5, 6]), (1, vec![7, 8])]),
            lora_name: None,
            cache_namespace: None,
            prompt_recorded: true,
            finalized: false,
        };

        let mut sequences: Vec<Vec<u32>> = tracker.completed_sequences().collect();
        sequences.sort();
        assert_eq!(sequences, vec![vec![1, 2, 3, 4, 5], vec![1, 2, 3, 7]]);
    }

    #[test]
    fn one_sampled_output_adds_no_generated_kv_history() {
        let tracker = CacheHistoryTracker {
            epoch: 0,
            prompt: vec![1, 2, 3],
            branches: HashMap::from([(0, vec![4])]),
            lora_name: None,
            cache_namespace: None,
            prompt_recorded: true,
            finalized: false,
        };

        assert_eq!(
            tracker.completed_sequences().collect::<Vec<_>>(),
            vec![vec![1, 2, 3]]
        );
    }
}
