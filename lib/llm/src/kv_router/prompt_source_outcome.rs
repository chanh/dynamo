// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use dynamo_runtime::{
    component::Endpoint,
    metrics::MetricsHierarchy,
    transports::event_plane::{EventPublisher, EventSubscriber},
};

use crate::protocols::common::timing::CacheSourceObservation;

const TOPIC: &str = "prompt-source-terminal-v1";
const QUEUE_CAPACITY: usize = 4096;
const REGISTRY_CAPACITY: usize = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminalPromptSourceOutcome {
    pub origin_router_id: u64,
    pub request_id: String,
    /// Correlates one registration lifecycle when clients reuse request IDs.
    /// Legacy outcomes decode as zero and cannot satisfy a current registration.
    #[serde(default)]
    pub registration_nonce: u64,
    pub cache_source: CacheSourceObservation,
    #[serde(default)]
    pub num_computed_output_tokens: u64,
    #[serde(default)]
    pub num_unobserved_computed_output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_history_incomplete_reason: Option<String>,
}

#[derive(Clone)]
pub struct TerminalPromptSourcePublisher {
    tx: mpsc::Sender<TerminalPromptSourceOutcome>,
    events_total: prometheus::IntCounterVec,
}

impl TerminalPromptSourcePublisher {
    pub async fn for_endpoint(endpoint: &Endpoint) -> anyhow::Result<Self> {
        let events_total = endpoint.metrics().create_intcountervec(
            "worker_prompt_source_terminal_publish_total",
            "Terminal prompt-source outcome publication events by result",
            &["result"],
            &[],
        )?;
        let publisher = EventPublisher::for_endpoint(endpoint, TOPIC).await?;
        let (tx, mut rx) = mpsc::channel(QUEUE_CAPACITY);
        let publish_events = events_total.clone();
        tokio::spawn(async move {
            while let Some(outcome) = rx.recv().await {
                if let Err(error) = publisher.publish(&outcome).await {
                    publish_events.with_label_values(&["publish_error"]).inc();
                    tracing::warn!(%error, "Failed to publish terminal prompt-source outcome");
                } else {
                    publish_events.with_label_values(&["published"]).inc();
                }
            }
        });
        Ok(Self { tx, events_total })
    }

    pub fn try_publish(&self, outcome: TerminalPromptSourceOutcome) -> Result<(), &'static str> {
        match self.tx.try_send(outcome) {
            Ok(()) => {
                self.events_total.with_label_values(&["enqueued"]).inc();
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.events_total.with_label_values(&["queue_full"]).inc();
                Err("full")
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.events_total.with_label_values(&["queue_closed"]).inc();
                Err("closed")
            }
        }
    }

    pub fn observe_control_result(&self, result: &str) {
        let result = match result {
            "origin_active_evicted" => "origin_active_evicted",
            "origin_pending_expired" => "origin_pending_expired",
            "origin_pending_full" => "origin_pending_full",
            "origin_missing" => "origin_missing",
            _ => "invalid_control_result",
        };
        self.events_total.with_label_values(&[result]).inc();
    }
}

struct PendingOutcome {
    generation: u64,
    tx: oneshot::Sender<TerminalPromptSourceOutcome>,
}

struct RegistryState {
    next_generation: u64,
    pending: HashMap<String, PendingOutcome>,
}

#[derive(Clone)]
pub struct TerminalPromptSourceRegistry {
    inner: Arc<TerminalPromptSourceRegistryInner>,
}

struct TerminalPromptSourceRegistryInner {
    router_id: u64,
    capacity: usize,
    state: Mutex<RegistryState>,
    events_total: prometheus::IntCounterVec,
}

pub struct TerminalPromptSourceRegistration {
    registry: TerminalPromptSourceRegistry,
    request_id: String,
    generation: u64,
    rx: oneshot::Receiver<TerminalPromptSourceOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationError {
    Full,
    Duplicate,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitError {
    Timeout,
    PublisherClosed,
}

impl TerminalPromptSourceRegistry {
    pub async fn for_endpoint(endpoint: &Endpoint, router_id: u64) -> anyhow::Result<Self> {
        let events_total = endpoint.metrics().create_intcountervec(
            "router_prompt_source_terminal_events_total",
            "Terminal prompt-source outcome handoff events by result",
            &["result"],
            &[],
        )?;
        let registry = Self {
            inner: Arc::new(TerminalPromptSourceRegistryInner {
                router_id,
                capacity: REGISTRY_CAPACITY,
                state: Mutex::new(RegistryState {
                    next_generation: 0,
                    pending: HashMap::new(),
                }),
                events_total,
            }),
        };
        let mut subscriber = EventSubscriber::for_endpoint(endpoint, TOPIC)
            .await?
            .typed::<TerminalPromptSourceOutcome>();
        let receiver = registry.clone();
        tokio::spawn(async move {
            while let Some(result) = subscriber.next().await {
                match result {
                    Ok((_envelope, outcome)) => receiver.deliver(outcome),
                    Err(error) => {
                        receiver.observe("decode_error");
                        tracing::warn!(%error, "Failed to decode terminal prompt-source outcome");
                    }
                }
            }
        });
        Ok(registry)
    }

    pub fn router_id(&self) -> u64 {
        self.inner.router_id
    }

    pub fn register(
        &self,
        request_id: String,
    ) -> Result<TerminalPromptSourceRegistration, RegistrationError> {
        let (tx, rx) = oneshot::channel();
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        // A request may legitimately run for longer than any fixed registry TTL.
        // The registration drop path removes live entries generation-safely; only
        // discard entries whose receiver is already gone.
        state.pending.retain(|_, pending| !pending.tx.is_closed());
        if state.pending.contains_key(&request_id) {
            drop(state);
            self.observe("registry_duplicate");
            return Err(RegistrationError::Duplicate);
        }
        if state.pending.len() >= self.inner.capacity {
            drop(state);
            self.observe("registry_full");
            return Err(RegistrationError::Full);
        }
        state.next_generation = state.next_generation.wrapping_add(1).max(1);
        let generation = state.next_generation;
        state
            .pending
            .insert(request_id.clone(), PendingOutcome { generation, tx });
        Ok(TerminalPromptSourceRegistration {
            registry: self.clone(),
            request_id,
            generation,
            rx,
        })
    }

    fn deliver(&self, mut outcome: TerminalPromptSourceOutcome) {
        if outcome.origin_router_id != self.inner.router_id {
            return;
        }
        outcome.cache_source = outcome.cache_source.validate();
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        let matches = state
            .pending
            .get(&outcome.request_id)
            .is_some_and(|pending| pending.generation == outcome.registration_nonce);
        let pending = matches
            .then(|| state.pending.remove(&outcome.request_id))
            .flatten();
        drop(state);
        match pending {
            Some(pending) => {
                if pending.tx.send(outcome).is_ok() {
                    self.observe("delivered");
                } else {
                    self.observe("receiver_closed");
                }
            }
            None => self.observe("orphan"),
        }
    }

    fn remove(&self, request_id: &str, generation: u64) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        if state
            .pending
            .get(request_id)
            .is_some_and(|pending| pending.generation == generation)
        {
            state.pending.remove(request_id);
        }
    }

    fn observe(&self, result: &str) {
        self.inner.events_total.with_label_values(&[result]).inc();
    }
}

impl TerminalPromptSourceRegistration {
    pub fn nonce(&self) -> u64 {
        self.generation
    }

    pub async fn wait(
        mut self,
        timeout: Duration,
    ) -> Result<TerminalPromptSourceOutcome, WaitError> {
        match tokio::time::timeout(timeout, &mut self.rx).await {
            Ok(Ok(outcome)) => Ok(outcome),
            Ok(Err(_)) => Err(WaitError::PublisherClosed),
            Err(_) => {
                self.registry.observe("timeout");
                Err(WaitError::Timeout)
            }
        }
    }
}

impl Drop for TerminalPromptSourceRegistration {
    fn drop(&mut self) {
        self.registry.remove(&self.request_id, self.generation);
    }
}

#[cfg(test)]
mod tests {
    use prometheus::{IntCounterVec, Opts};

    use super::*;

    fn registry(router_id: u64, capacity: usize) -> TerminalPromptSourceRegistry {
        TerminalPromptSourceRegistry {
            inner: Arc::new(TerminalPromptSourceRegistryInner {
                router_id,
                capacity,
                state: Mutex::new(RegistryState {
                    next_generation: 0,
                    pending: HashMap::new(),
                }),
                events_total: IntCounterVec::new(Opts::new("test_events", "test"), &["result"])
                    .unwrap(),
            }),
        }
    }

    fn outcome(origin_router_id: u64, request_id: &str) -> TerminalPromptSourceOutcome {
        TerminalPromptSourceOutcome {
            origin_router_id,
            request_id: request_id.to_string(),
            registration_nonce: 1,
            cache_source: CacheSourceObservation {
                schema_version: 1,
                complete: false,
                prompt_tokens: Some(10),
                gpu_hit_tokens: None,
                cpu_hit_tokens: None,
                cpu_lookup_tokens: None,
                retrieval_failure_tokens: None,
                recomputed_tokens: None,
                incomplete_reason: Some("cancelled_before_prompt_complete".to_string()),
                cache_groups: None,
            },
            num_computed_output_tokens: 0,
            num_unobserved_computed_output_tokens: 0,
            generated_history_incomplete_reason: None,
        }
    }

    fn publisher(
        capacity: usize,
    ) -> (
        TerminalPromptSourcePublisher,
        mpsc::Receiver<TerminalPromptSourceOutcome>,
    ) {
        let (tx, rx) = mpsc::channel(capacity);
        let events_total =
            IntCounterVec::new(Opts::new("test_publish_events", "test"), &["result"]).unwrap();
        (TerminalPromptSourcePublisher { tx, events_total }, rx)
    }

    #[tokio::test]
    async fn only_origin_router_delivers_once() {
        let registry = registry(7, 4);
        let registration = registry.register("request-1".to_string()).unwrap();

        registry.deliver(outcome(8, "request-1"));
        assert!(
            registry
                .inner
                .state
                .lock()
                .unwrap()
                .pending
                .contains_key("request-1")
        );

        registry.deliver(outcome(7, "request-1"));
        let delivered = registration.wait(Duration::from_millis(10)).await.unwrap();
        assert_eq!(delivered.request_id, "request-1");
        registry.deliver(outcome(7, "request-1"));
        assert_eq!(
            registry
                .inner
                .events_total
                .with_label_values(&["orphan"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn delayed_outcome_cannot_satisfy_reused_request_id() {
        let registry = registry(7, 4);
        let old = registry.register("reused".to_string()).unwrap();
        let old_nonce = old.nonce();
        assert_eq!(
            old.wait(Duration::from_millis(1)).await,
            Err(WaitError::Timeout)
        );

        let current = registry.register("reused".to_string()).unwrap();
        let current_nonce = current.nonce();
        assert_ne!(old_nonce, current_nonce);
        let mut legacy = outcome(7, "reused");
        legacy.registration_nonce = 0;
        registry.deliver(legacy);
        let mut delayed = outcome(7, "reused");
        delayed.registration_nonce = old_nonce;
        registry.deliver(delayed);
        assert!(
            registry
                .inner
                .state
                .lock()
                .unwrap()
                .pending
                .contains_key("reused")
        );

        let mut matching = outcome(7, "reused");
        matching.registration_nonce = current_nonce;
        registry.deliver(matching);
        let delivered = current.wait(Duration::from_millis(10)).await.unwrap();
        assert_eq!(delivered.registration_nonce, current_nonce);
    }

    #[tokio::test]
    async fn invalid_terminal_cache_source_is_explicitly_incomplete() {
        let registry = registry(7, 4);
        let registration = registry.register("request-1".to_string()).unwrap();
        let mut invalid = outcome(7, "request-1");
        invalid.cache_source = CacheSourceObservation {
            schema_version: 1,
            complete: true,
            prompt_tokens: Some(10),
            gpu_hit_tokens: Some(4),
            cpu_hit_tokens: Some(4),
            cpu_lookup_tokens: Some(4),
            retrieval_failure_tokens: Some(0),
            recomputed_tokens: Some(4),
            incomplete_reason: None,
            cache_groups: None,
        };

        registry.deliver(invalid);
        let delivered = registration.wait(Duration::from_millis(10)).await.unwrap();

        assert!(!delivered.cache_source.complete);
        assert_eq!(
            delivered.cache_source.incomplete_reason.as_deref(),
            Some("invalid_cache_source_observation")
        );
    }

    #[test]
    fn registry_rejects_overflow_without_replacing_live_request() {
        let registry = registry(7, 1);
        let _first = registry.register("first".to_string()).unwrap();
        assert_eq!(
            registry.register("second".to_string()).err(),
            Some(RegistrationError::Full)
        );
        assert!(
            registry
                .inner
                .state
                .lock()
                .unwrap()
                .pending
                .contains_key("first")
        );
    }

    #[tokio::test]
    async fn registry_rejects_duplicate_without_replacing_live_request() {
        let registry = registry(7, 2);
        let first = registry.register("same".to_string()).unwrap();
        assert_eq!(
            registry.register("same".to_string()).err(),
            Some(RegistrationError::Duplicate)
        );
        registry.deliver(outcome(7, "same"));
        assert!(first.wait(Duration::from_millis(10)).await.is_ok());
    }

    #[tokio::test]
    async fn concurrent_registration_does_not_prune_a_live_request() {
        let registry = registry(7, 2);
        let first = registry.register("long-running".to_string()).unwrap();

        let second = registry.register("concurrent".to_string()).unwrap();
        registry.deliver(outcome(7, "long-running"));

        assert!(first.wait(Duration::from_millis(10)).await.is_ok());
        drop(second);
        assert!(registry.inner.state.lock().unwrap().pending.is_empty());
    }

    #[test]
    fn publisher_counts_full_and_closed_queue_failures() {
        let (publisher, receiver) = publisher(1);
        publisher.try_publish(outcome(7, "first")).unwrap();
        assert_eq!(publisher.try_publish(outcome(7, "second")), Err("full"));
        assert_eq!(
            publisher
                .events_total
                .with_label_values(&["queue_full"])
                .get(),
            1
        );

        drop(receiver);
        assert_eq!(publisher.try_publish(outcome(7, "third")), Err("closed"));
        assert_eq!(
            publisher
                .events_total
                .with_label_values(&["queue_closed"])
                .get(),
            1
        );
    }
}
