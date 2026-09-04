// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict request-specific effective-cache affinity with load as a tie-breaker.

use dynamo_kv_router::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

pub(crate) struct LongestCacheReusePicker;

impl WorkerPicker for LongestCacheReusePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .zip(
                input
                    .cache()
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?,
            )
            .enumerate()
            .max_by(|(_, (left, left_cache)), (_, (right, right_cache))| {
                left_cache
                    .effective_overlap_blocks()
                    .total_cmp(&right_cache.effective_overlap_blocks())
                    .then_with(|| right.cost().total_cmp(&left.cost()))
                    .then_with(|| right.worker().cmp(&left.worker()))
            })
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}
