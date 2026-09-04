// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict longest-cache-reuse worker selection.

mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy};
use picker::LongestCacheReusePicker;
use scorer::ActiveLoadScorer;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let _: Parameters = parameters.deserialize()?;
    Ok(Arc::new(move |config, worker_type, _partition| {
        create_policy(config, worker_type.as_str())
    }))
}

fn create_policy(config: &KvRouterConfig, worker_label: &'static str) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config.clone(),
        worker_label,
        vec![Box::new(ActiveLoadScorer {
            prefill_load_scale: config.prefill_load_scale,
            active_request_weight: config.decode_active_request_weight,
        })],
        Box::new(LongestCacheReusePicker),
    )
}

/// Register the `longest-cache-reuse` policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("longest-cache-reuse", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{SchedulingRequest, WorkerSelectionInput, WorkerSelector};

    use super::*;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(4096)
        }
    }

    fn request(overlaps: [usize; 2], loads: [usize; 2]) -> SchedulingRequest {
        let workers = [
            WorkerWithDpRank::from_worker_id(0),
            WorkerWithDpRank::from_worker_id(1),
        ];
        let mut overlap = OverlapSignals::default();
        for (worker, blocks) in workers.into_iter().zip(overlaps) {
            overlap.tier_overlap_blocks.device.insert(worker, blocks);
            overlap
                .effective_overlap_blocks
                .insert(worker, blocks as f64);
        }
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 1600,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap,
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (worker, active_decode_blocks) in workers.into_iter().zip(loads) {
            request.worker_loads.insert(
                worker,
                dynamo_kv_router::sequences::WorkerLoadProjection {
                    active_decode_blocks,
                    ..Default::default()
                },
            );
        }
        request
    }

    fn select(overlaps: [usize; 2], loads: [usize; 2]) -> WorkerWithDpRank {
        let policy = create_policy(&KvRouterConfig::default(), "test");
        let workers = HashMap::from([(0, TestWorker), (1, TestWorker)]);
        let request = request(overlaps, loads);
        policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap()
            .worker
    }

    #[test]
    fn greater_cache_reuse_wins_regardless_of_load() {
        assert_eq!(
            select([80, 79], [1_000_000, 0]),
            WorkerWithDpRank::from_worker_id(0)
        );
    }

    #[test]
    fn lower_load_wins_equal_cache_reuse() {
        assert_eq!(
            select([80, 80], [40, 0]),
            WorkerWithDpRank::from_worker_id(1)
        );
    }

    #[test]
    fn lower_worker_id_wins_an_exact_tie() {
        assert_eq!(
            select([80, 80], [0, 0]),
            WorkerWithDpRank::from_worker_id(0)
        );
    }
}
