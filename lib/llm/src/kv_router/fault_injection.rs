// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded cache-loss fault injection for an owned Flash A shadow arm.
//!
//! The code is absent from default builds. A feature-enabled build remains inert
//! unless every runtime identity/acknowledgement gate and the strict one-stage
//! plan match. Sampling consumes only existing mirrored Store events or router
//! context IDs; it never generates requests or changes funnel metric values.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use dynamo_kv_router::protocols::{StorageTier, WorkerWithDpRank};
use dynamo_kv_router::zmq_wire::{BlockHashValue, RawKvEvent};
use serde::Deserialize;
use xxhash_rust::xxh3::Xxh3;

const ENABLE_VALUE: &str = "shadow-only-v1";
const ACK_VALUE: &str = "I_ACKNOWLEDGE_SHADOW_ONLY_CACHE_LOSS_V1";
const TRAFFIC_CLASS_VALUE: &str = "mirrored-shadow";
const SHADOW_ARM_VALUE: &str = "flash-a";
const MODEL_VALUE: &str = "deepseek-ai/deepseek-v4-flash-0731";
const SAMPLE_DENOMINATOR: u16 = 10_000;
const MAX_INJECTIONS: u32 = 32;
const MAX_RUN_ID_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum FaultStage {
    PhysicalRetention,
    RouterVisibility,
    RoutingChoice,
}

impl FaultStage {
    pub(super) const ALL: [Self; 3] = [
        Self::PhysicalRetention,
        Self::RouterVisibility,
        Self::RoutingChoice,
    ];

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::PhysicalRetention => "physical_retention",
            Self::RouterVisibility => "router_visibility",
            Self::RoutingChoice => "routing_choice",
        }
    }

    const fn discriminator(self) -> u8 {
        match self {
            Self::PhysicalRetention => 1,
            Self::RouterVisibility => 2,
            Self::RoutingChoice => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RouteTarget {
    worker_id: u64,
    dp_rank: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FaultPlan {
    run_id: String,
    seed: u64,
    sample_permyriad: u16,
    max_injections: u32,
    stage: FaultStage,
    #[serde(default)]
    route_target: Option<RouteTarget>,
}

#[derive(Default)]
struct GateValues<'a> {
    enable: Option<&'a str>,
    acknowledgment: Option<&'a str>,
    traffic_class: Option<&'a str>,
    shadow_arm: Option<&'a str>,
    expected_shadow_arm: Option<&'a str>,
    namespace: Option<&'a str>,
    expected_namespace: Option<&'a str>,
    deployment: Option<&'a str>,
    expected_deployment: Option<&'a str>,
    model: Option<&'a str>,
    expected_model: Option<&'a str>,
    plan: Option<&'a str>,
}

pub(super) struct CacheLossFaultInjector {
    plan: Option<FaultPlan>,
    injections: AtomicU32,
}

impl CacheLossFaultInjector {
    fn disabled() -> Self {
        Self {
            plan: None,
            injections: AtomicU32::new(0),
        }
    }

    fn from_env() -> Self {
        let values = [
            std::env::var("DYN_CACHE_LOSS_FAULT_ENABLE").ok(),
            std::env::var("DYN_CACHE_LOSS_FAULT_ACK").ok(),
            std::env::var("DYN_TRAFFIC_CLASS").ok(),
            std::env::var("DYN_SHADOW_ARM").ok(),
            std::env::var("DYN_CACHE_LOSS_FAULT_EXPECTED_ARM").ok(),
            std::env::var("POD_NAMESPACE").ok(),
            std::env::var("DYN_CACHE_LOSS_FAULT_EXPECTED_NAMESPACE").ok(),
            std::env::var("DYN_DEPLOYMENT_NAME").ok(),
            std::env::var("DYN_CACHE_LOSS_FAULT_EXPECTED_DEPLOYMENT").ok(),
            std::env::var("DYN_MODEL_NAME").ok(),
            std::env::var("DYN_CACHE_LOSS_FAULT_EXPECTED_MODEL").ok(),
            std::env::var("DYN_CACHE_LOSS_FAULT_PLAN").ok(),
        ];
        Self::from_values(GateValues {
            enable: values[0].as_deref(),
            acknowledgment: values[1].as_deref(),
            traffic_class: values[2].as_deref(),
            shadow_arm: values[3].as_deref(),
            expected_shadow_arm: values[4].as_deref(),
            namespace: values[5].as_deref(),
            expected_namespace: values[6].as_deref(),
            deployment: values[7].as_deref(),
            expected_deployment: values[8].as_deref(),
            model: values[9].as_deref(),
            expected_model: values[10].as_deref(),
            plan: values[11].as_deref(),
        })
    }

    fn from_values(values: GateValues<'_>) -> Self {
        let gate_valid = values.enable == Some(ENABLE_VALUE)
            && values.acknowledgment == Some(ACK_VALUE)
            && values.traffic_class == Some(TRAFFIC_CLASS_VALUE)
            && values.shadow_arm == Some(SHADOW_ARM_VALUE)
            && values.shadow_arm == values.expected_shadow_arm
            && values.namespace.is_some()
            && values.namespace == values.expected_namespace
            && values.deployment.is_some()
            && values.deployment == values.expected_deployment
            && values.deployment.is_some_and(is_shadow_name)
            && values.model == Some(MODEL_VALUE)
            && values.model == values.expected_model;
        if !gate_valid {
            if values.plan.is_some() || values.enable.is_some() {
                tracing::warn!(
                    "Cache-loss fault injection remains disabled because the Flash A shadow gate is incomplete"
                );
            }
            return Self::disabled();
        }

        let Some(plan) = values
            .plan
            .and_then(|raw| serde_json::from_str::<FaultPlan>(raw).ok())
            .filter(FaultPlan::is_valid)
        else {
            tracing::warn!(
                "Cache-loss fault injection remains disabled because the bounded plan is invalid"
            );
            return Self::disabled();
        };

        tracing::warn!(
            run_id = %plan.run_id,
            stage = plan.stage.label(),
            sample_permyriad = plan.sample_permyriad,
            max_injections = plan.max_injections,
            "Armed bounded cache-loss fault injection for existing Flash A mirrored traffic"
        );
        Self {
            plan: Some(plan),
            injections: AtomicU32::new(0),
        }
    }

    pub(super) fn armed_stage(&self) -> Option<FaultStage> {
        self.plan.as_ref().map(|plan| plan.stage)
    }

    pub(super) fn inject_physical_retention(&self, event: &RawKvEvent) -> bool {
        self.sample_store(FaultStage::PhysicalRetention, event)
    }

    pub(super) fn suppress_router_visibility(&self, event: &RawKvEvent) -> bool {
        self.sample_store(FaultStage::RouterVisibility, event)
    }

    fn sample_store(&self, stage: FaultStage, event: &RawKvEvent) -> bool {
        let Some(plan) = self.plan.as_ref().filter(|plan| plan.stage == stage) else {
            return false;
        };
        let Some(hashes) = gpu_store_hashes(event) else {
            return false;
        };
        if self.injections.load(Ordering::Relaxed) >= plan.max_injections {
            return false;
        }
        let mut hasher = sample_hasher(plan, stage);
        let mut hash_count = 0usize;
        for hash in hashes {
            hasher.update(&hash.to_le_bytes());
            hash_count += 1;
        }
        hash_count > 0 && self.reserve_digest(plan, hasher.digest())
    }

    pub(super) fn route_target(&self, context_id: &str) -> Option<WorkerWithDpRank> {
        let plan = self
            .plan
            .as_ref()
            .filter(|plan| plan.stage == FaultStage::RoutingChoice)?;
        if self.injections.load(Ordering::Relaxed) >= plan.max_injections {
            return None;
        }
        let target = plan.route_target?;
        let mut hasher = sample_hasher(plan, FaultStage::RoutingChoice);
        hasher.update(context_id.as_bytes());
        self.reserve_digest(plan, hasher.digest())
            .then(|| WorkerWithDpRank::new(target.worker_id, target.dp_rank))
    }

    fn reserve_digest(&self, plan: &FaultPlan, digest: u64) -> bool {
        if digest % u64::from(SAMPLE_DENOMINATOR) >= u64::from(plan.sample_permyriad) {
            return false;
        }

        self.injections
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current < plan.max_injections).then_some(current + 1)
            })
            .is_ok()
    }

    #[cfg(test)]
    fn injection_count(&self) -> u32 {
        self.injections.load(Ordering::Relaxed)
    }
}

fn sample_hasher(plan: &FaultPlan, stage: FaultStage) -> Xxh3 {
    let mut hasher = Xxh3::with_seed(plan.seed);
    hasher.update(plan.run_id.as_bytes());
    hasher.update(&[stage.discriminator()]);
    hasher
}

impl FaultPlan {
    fn is_valid(&self) -> bool {
        let valid_run_id = !self.run_id.is_empty()
            && self.run_id.len() <= MAX_RUN_ID_BYTES
            && self
                .run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
        let route_shape = match self.stage {
            FaultStage::RoutingChoice => self.route_target.is_some(),
            FaultStage::PhysicalRetention | FaultStage::RouterVisibility => {
                self.route_target.is_none()
            }
        };
        valid_run_id
            && (1..=SAMPLE_DENOMINATOR).contains(&self.sample_permyriad)
            && (1..=MAX_INJECTIONS).contains(&self.max_injections)
            && route_shape
    }
}

fn is_shadow_name(name: &str) -> bool {
    name.split(|character: char| !character.is_ascii_alphanumeric())
        .any(|segment| segment.eq_ignore_ascii_case("shadow"))
}

fn gpu_store_hashes(event: &RawKvEvent) -> Option<impl Iterator<Item = u64> + '_> {
    let RawKvEvent::BlockStored {
        block_hashes,
        medium,
        ..
    } = event
    else {
        return None;
    };
    let is_gpu = medium
        .as_deref()
        .is_none_or(|medium| StorageTier::from_kv_medium(medium) == Some(StorageTier::Device));
    is_gpu.then(|| block_hashes.iter().copied().map(BlockHashValue::into_u64))
}

static CACHE_LOSS_FAULT_INJECTOR: OnceLock<CacheLossFaultInjector> = OnceLock::new();

pub(super) fn cache_loss_fault_injector() -> &'static CacheLossFaultInjector {
    CACHE_LOSS_FAULT_INJECTOR.get_or_init(CacheLossFaultInjector::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::zmq_wire::Locality;

    fn plan(stage: &str, route: &str, cap: u32) -> String {
        format!(
            r#"{{"run_id":"run-7","seed":17,"sample_permyriad":10000,"max_injections":{cap},"stage":"{stage}"{route}}}"#
        )
    }

    fn valid_gate<'a>(plan: &'a str) -> GateValues<'a> {
        GateValues {
            enable: Some(ENABLE_VALUE),
            acknowledgment: Some(ACK_VALUE),
            traffic_class: Some(TRAFFIC_CLASS_VALUE),
            shadow_arm: Some(SHADOW_ARM_VALUE),
            expected_shadow_arm: Some(SHADOW_ARM_VALUE),
            namespace: Some("prod-inference"),
            expected_namespace: Some("prod-inference"),
            deployment: Some("deepseek-shadow-flash-a"),
            expected_deployment: Some("deepseek-shadow-flash-a"),
            model: Some(MODEL_VALUE),
            expected_model: Some(MODEL_VALUE),
            plan: Some(plan),
        }
    }

    fn store(hashes: &[u64], medium: Option<&str>) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: hashes
                .iter()
                .copied()
                .map(BlockHashValue::Unsigned)
                .collect(),
            parent_block_hash: None,
            parent_sequence_hash: None,
            parent_sequence_hash_algorithm: None,
            eagle_lookahead_sequence_hash: None,
            eagle_lookahead_sequence_hash_algorithm: None,
            eagle_lookahead_token_ids: None,
            token_ids: vec![1; hashes.len()],
            block_size: 1,
            medium: medium.map(str::to_string),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(0),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: Some(Locality::Local),
            ownership: None,
        }
    }

    #[test]
    fn every_flash_a_shadow_gate_factor_is_required_exactly() {
        let plan = plan(
            "routing_choice",
            r#","route_target":{"worker_id":9,"dp_rank":2}"#,
            1,
        );
        assert_eq!(
            CacheLossFaultInjector::from_values(valid_gate(&plan)).armed_stage(),
            Some(FaultStage::RoutingChoice)
        );

        let invalid = [
            GateValues {
                enable: None,
                ..valid_gate(&plan)
            },
            GateValues {
                acknowledgment: Some("ack"),
                ..valid_gate(&plan)
            },
            GateValues {
                traffic_class: Some("shadow"),
                ..valid_gate(&plan)
            },
            GateValues {
                shadow_arm: Some("flash-b"),
                expected_shadow_arm: Some("flash-b"),
                ..valid_gate(&plan)
            },
            GateValues {
                expected_shadow_arm: Some("flash-b"),
                ..valid_gate(&plan)
            },
            GateValues {
                namespace: None,
                expected_namespace: None,
                ..valid_gate(&plan)
            },
            GateValues {
                expected_namespace: Some("other"),
                ..valid_gate(&plan)
            },
            GateValues {
                deployment: None,
                expected_deployment: None,
                ..valid_gate(&plan)
            },
            GateValues {
                deployment: Some("deepseek-production-flash-a"),
                expected_deployment: Some("deepseek-production-flash-a"),
                ..valid_gate(&plan)
            },
            GateValues {
                expected_deployment: Some("other-shadow"),
                ..valid_gate(&plan)
            },
            GateValues {
                model: Some("other/model"),
                expected_model: Some("other/model"),
                ..valid_gate(&plan)
            },
            GateValues {
                expected_model: Some("other/model"),
                ..valid_gate(&plan)
            },
        ];
        for gate in invalid {
            assert_eq!(
                CacheLossFaultInjector::from_values(gate).armed_stage(),
                None
            );
        }
    }

    #[test]
    fn existing_gpu_store_traffic_is_sampled_deterministically_and_capped() {
        let plan = plan("physical_retention", "", 1);
        let first = CacheLossFaultInjector::from_values(valid_gate(&plan));
        let second = CacheLossFaultInjector::from_values(valid_gate(&plan));

        assert!(first.inject_physical_retention(&store(&[11, 12], None)));
        assert!(second.inject_physical_retention(&store(&[11, 12], None)));
        assert!(!first.inject_physical_retention(&store(&[13], Some("CPU"))));
        assert!(!first.suppress_router_visibility(&store(&[13], None)));
        assert!(!first.inject_physical_retention(&store(&[13], None)));
        assert_eq!(first.injection_count(), 1);
    }

    #[test]
    fn route_sampling_uses_existing_request_ids_and_is_hard_capped() {
        let plan = plan(
            "routing_choice",
            r#","route_target":{"worker_id":9,"dp_rank":2}"#,
            1,
        );
        let first = CacheLossFaultInjector::from_values(valid_gate(&plan));
        let second = CacheLossFaultInjector::from_values(valid_gate(&plan));

        assert_eq!(
            first.route_target("mirrored-request-a"),
            Some(WorkerWithDpRank::new(9, 2))
        );
        assert_eq!(
            second.route_target("mirrored-request-a"),
            Some(WorkerWithDpRank::new(9, 2))
        );
        assert_eq!(first.route_target("mirrored-request-b"), None);
        assert_eq!(first.injection_count(), 1);
    }

    #[test]
    fn partial_sampling_is_reproducible_for_existing_mirrored_requests() {
        let plan = r#"{"run_id":"run-7","seed":991,"sample_permyriad":5000,"max_injections":32,"stage":"routing_choice","route_target":{"worker_id":9,"dp_rank":2}}"#;
        let first = CacheLossFaultInjector::from_values(valid_gate(plan));
        let second = CacheLossFaultInjector::from_values(valid_gate(plan));

        let first_decisions: Vec<_> = (0..32)
            .map(|index| first.route_target(&format!("mirrored-request-{index}")))
            .collect();
        let second_decisions: Vec<_> = (0..32)
            .map(|index| second.route_target(&format!("mirrored-request-{index}")))
            .collect();
        assert_eq!(first_decisions, second_decisions);
        assert!(first_decisions.iter().any(Option::is_some));
        assert!(first_decisions.iter().any(Option::is_none));
    }

    #[test]
    fn concurrent_sampling_cannot_exceed_the_hard_cap() {
        let plan = plan(
            "routing_choice",
            r#","route_target":{"worker_id":9,"dp_rank":2}"#,
            4,
        );
        let injector = std::sync::Arc::new(CacheLossFaultInjector::from_values(valid_gate(&plan)));
        let threads: Vec<_> = (0..16)
            .map(|index| {
                let injector = std::sync::Arc::clone(&injector);
                std::thread::spawn(move || {
                    injector.route_target(&format!("mirrored-request-{index}"))
                })
            })
            .collect();
        let injected = threads
            .into_iter()
            .map(|thread| thread.join().unwrap().is_some())
            .filter(|injected| *injected)
            .count();
        assert_eq!(injected, 4);
        assert_eq!(injector.injection_count(), 4);
    }

    #[test]
    fn invalid_or_multi_stage_plans_remain_inert() {
        assert_eq!(
            CacheLossFaultInjector::from_values(GateValues::default()).armed_stage(),
            None
        );
        for plan in [
            r#"{"run_id":"run","seed":1,"sample_permyriad":0,"max_injections":1,"stage":"router_visibility"}"#,
            r#"{"run_id":"run","seed":1,"sample_permyriad":10000,"max_injections":33,"stage":"router_visibility"}"#,
            r#"{"run_id":"run","seed":1,"sample_permyriad":10000,"max_injections":1,"stage":"routing_choice"}"#,
            r#"{"run_id":"run","seed":1,"sample_permyriad":10000,"max_injections":1,"stage":"physical_retention","route_target":{"worker_id":1,"dp_rank":0}}"#,
            r#"{"run_id":"run","seed":1,"sample_permyriad":10000,"max_injections":1,"stage":"router_visibility","unknown":true}"#,
        ] {
            assert_eq!(
                CacheLossFaultInjector::from_values(valid_gate(plan)).armed_stage(),
                None
            );
        }
    }
}
