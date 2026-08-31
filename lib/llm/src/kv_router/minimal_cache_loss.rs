// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal request-level cache-loss accounting.
//!
//! The router already keeps the per-worker cache index used for routing. This
//! module deliberately adds no second cache ledger or cache-event stream.

#[derive(Clone, Copy, Debug)]
pub struct RouteObservation {
    pub prompt_tokens: u64,
    pub best_router_tokens: u64,
    pub selected_router_tokens: u64,
}

impl RouteObservation {
    pub fn bounded(self) -> Self {
        let prompt_tokens = self.prompt_tokens;
        Self {
            prompt_tokens,
            best_router_tokens: self.best_router_tokens.min(prompt_tokens),
            selected_router_tokens: self.selected_router_tokens.min(prompt_tokens),
        }
    }
}
