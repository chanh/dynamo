<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Longest Cache Reuse Policy

Use `longest-cache-reuse` to select the eligible worker with the greatest request-specific effective cache overlap. Effective overlap is the same tier-weighted value used by Dynamo's cache-loss funnel, including device and configured lower-tier cache credit. The policy does not trade cached blocks for lower load. It uses modeled active load only when multiple workers expose the same effective overlap, then uses worker ID to make an exact tie deterministic.

The policy reads:

- Effective overlap blocks from `WorkerInputs::CACHE`.
- Active prefill, decode, and request load from `WorkerInputs::LOAD` for equal-overlap ties.

Configure the policy for an aggregated worker pool:

```yaml
worker_selection:
  aggregated: longest-cache-reuse
  instances:
    - name: longest-cache-reuse
      type: longest-cache-reuse
      parameters: {}
```

Start the frontend with the policy catalog and configuration:

```bash
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config /path/to/worker-selection.yaml
```

The default Dynamo policy remains active unless the configuration selects `longest-cache-reuse`.
