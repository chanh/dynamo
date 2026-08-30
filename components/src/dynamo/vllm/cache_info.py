# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

from .dp_topology import get_dp_range_for_worker

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
    "sink_full_attention",
}
CACHE_EVIDENCE_BARRIER_CAPABILITY = "cache_evidence_barrier_v1"
CACHE_EVIDENCE_CACHE_GROUP_CATALOGS = "cache_evidence_cache_group_catalogs"
CACHE_EVIDENCE_SERVING_INCARNATIONS = "cache_evidence_serving_incarnations"
CACHE_EVIDENCE_EPOCH_CAPABILITY = "cache_evidence_epoch_v1"
CACHE_EVIDENCE_EPOCH_MEDIA = "cache_evidence_epoch_media"
DYNAMO_KV_CACHE_GROUP_METADATA_KEY = "dynamo_kv_cache_group_metadata"


def normalize_cache_group_metadata(
    group_metadata: Any,
) -> list[dict[str, Any]] | None:
    if not isinstance(group_metadata, list) or not group_metadata:
        return None
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group in group_metadata:
        if not isinstance(group, dict):
            return None
        group_idx = group.get("group_idx")
        block_size = group.get("block_size")
        kind = group.get("kind")
        is_eagle = group.get("is_eagle", False)
        if (
            isinstance(group_idx, bool)
            or not isinstance(group_idx, int)
            or not 0 <= group_idx <= 2**32 - 1
            or group_idx in seen
            or isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or not 0 < block_size <= 2**32 - 1
            or not isinstance(kind, str)
            or not kind
            or not isinstance(is_eagle, bool)
        ):
            return None
        result: dict[str, Any] = {
            "group_idx": group_idx,
            "kind": kind,
            "block_size": block_size,
            "is_eagle": is_eagle,
        }
        for key in ("sliding_window", "alignment_tokens"):
            value = group.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= 2**32 - 1
            ):
                return None
            result[key] = value
        alignment = result.get("alignment_tokens")
        if alignment is not None and alignment % block_size != 0:
            return None
        seen.add(group_idx)
        normalized.append(result)
    return sorted(normalized, key=lambda group: group["group_idx"])


def publish_cache_group_catalogs(
    runtime_config: Any,
    vllm_config: VllmConfig,
    incarnations: dict[str, str],
) -> bool:
    additional_config = vllm_config.additional_config or {}
    retained = additional_config.get(DYNAMO_KV_CACHE_GROUP_METADATA_KEY)
    if not isinstance(retained, dict) or set(retained) != set(incarnations):
        return False
    catalogs = {
        rank: {
            "serving_incarnation": incarnations[rank],
            "cache_groups": retained[rank],
        }
        for rank in sorted(incarnations, key=int)
        if retained[rank]
    }
    if set(catalogs) != set(incarnations):
        return False
    runtime_config.set_engine_specific(
        CACHE_EVIDENCE_CACHE_GROUP_CATALOGS, json.dumps(catalogs)
    )
    return True


def get_common_cache_group_metadata(
    vllm_config: VllmConfig,
) -> list[dict[str, Any]] | None:
    additional_config = vllm_config.additional_config or {}
    retained = additional_config.get(DYNAMO_KV_CACHE_GROUP_METADATA_KEY)
    if not isinstance(retained, dict) or not retained:
        return None
    catalogs = list(retained.values())
    if not catalogs[0] or any(catalog != catalogs[0] for catalog in catalogs[1:]):
        return None
    return catalogs[0]


def cache_evidence_barrier_supported(
    engine: AsyncLLM, engine_args: Any, kv_events_enabled: bool
) -> bool:
    enabled = os.environ.get("DYN_CACHE_LOSS_FUNNEL_ENABLED", "").lower()
    events = getattr(engine_args, "kv_events_config", None)
    return (
        kv_events_enabled
        and events is not None
        and bool(getattr(events, "enable_kv_cache_events", False))
        and getattr(events, "publisher", None) not in {None, "null"}
        and enabled in {"1", "true", "yes"}
        and callable(getattr(engine, "cache_evidence_barrier", None))
        and callable(getattr(engine, "cache_evidence_barrier_supported", None))
    )


async def publish_cache_evidence_barrier_capability(
    runtime_config: Any,
    engine: AsyncLLM,
    engine_args: Any,
    kv_events_enabled: bool,
    dp_range: tuple[int, int],
    vllm_config: VllmConfig | None = None,
) -> bool:
    supported = cache_evidence_barrier_supported(engine, engine_args, kv_events_enabled)
    if supported:
        support_utility = engine.cache_evidence_barrier_supported
        try:
            for dp_rank in range(dp_range[0], dp_range[0] + dp_range[1]):
                if await support_utility(data_parallel_rank=dp_rank) is not True:
                    return False
        except Exception as error:
            logger.warning(
                "Failed to verify vLLM cache-evidence barrier support: %s", error
            )
            return False
        incarnation_utility = getattr(
            engine, "cache_evidence_serving_incarnation", None
        )
        if not callable(incarnation_utility):
            return False
        incarnations: dict[str, str] = {}
        for dp_rank in range(dp_range[0], dp_range[0] + dp_range[1]):
            incarnation = await incarnation_utility(data_parallel_rank=dp_rank)
            if (
                isinstance(incarnation, bool)
                or not isinstance(incarnation, int)
                or incarnation <= 0
            ):
                return False
            incarnations[str(dp_rank)] = str(incarnation)
        runtime_config.set_engine_specific(
            CACHE_EVIDENCE_BARRIER_CAPABILITY, json.dumps(True)
        )
        runtime_config.set_engine_specific(
            CACHE_EVIDENCE_SERVING_INCARNATIONS, json.dumps(incarnations)
        )
        if vllm_config is not None:
            publish_cache_group_catalogs(runtime_config, vllm_config, incarnations)
        epoch_media_utility = getattr(engine, "cache_evidence_epoch_media", None)
        epoch_supported = callable(epoch_media_utility) and all(
            callable(getattr(engine, name, None))
            for name in (
                "begin_cache_evidence_epoch",
                "commit_cache_evidence_epoch",
                "abort_cache_evidence_epoch",
            )
        )
        if epoch_supported:
            epoch_media: dict[str, list[str]] = {}
            for dp_rank in range(dp_range[0], dp_range[0] + dp_range[1]):
                media = await epoch_media_utility(data_parallel_rank=dp_rank)
                if not isinstance(media, (list, tuple)) or set(media) not in (
                    {"GPU"},
                    {"GPU", "CPU"},
                ):
                    return supported
                epoch_media[str(dp_rank)] = sorted(set(media))
            runtime_config.set_engine_specific(
                CACHE_EVIDENCE_EPOCH_CAPABILITY, json.dumps(True)
            )
            runtime_config.set_engine_specific(
                CACHE_EVIDENCE_EPOCH_MEDIA, json.dumps(epoch_media)
            )
    return supported


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    return additional_config.get(
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        vllm_config.cache_config.block_size,
    )


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    if not group_metadata:
        return fallback_block_size

    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size", fallback_block_size)

    return fallback_block_size


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    fallback_block_size = vllm_config.cache_config.block_size
    dp_start, dp_size = get_dp_range_for_worker(vllm_config)
    retained: dict[str, list[dict[str, Any]]] = {}
    try:
        for dp_rank in range(dp_start, dp_start + dp_size):
            engine_core = engine.engine_core
            select_engine = getattr(engine_core, "_cache_evidence_engine", None)
            ranked_call = getattr(engine_core, "_call_utility_async", None)
            if callable(select_engine) and callable(ranked_call):
                group_metadata = await ranked_call(
                    "get_kv_cache_group_metadata",
                    engine=select_engine(dp_rank),
                )
            elif dp_size == 1:
                group_metadata = await engine_core.call_utility_async(
                    "get_kv_cache_group_metadata"
                )
            else:
                raise RuntimeError("rank-targeted utility calls are unavailable")
            normalized = normalize_cache_group_metadata(group_metadata)
            if normalized is None:
                raise ValueError(f"invalid cache-group metadata for DP rank {dp_rank}")
            retained[str(dp_rank)] = normalized
    except Exception as e:
        logger.warning(
            "Failed to fetch KV cache group metadata; falling back to "
            "vLLM cache_config.block_size: %s",
            e,
        )
        kv_event_block_size = fallback_block_size
        retained = {}
    else:
        kv_event_block_size = select_main_attention_block_size(
            next(iter(retained.values())),
            fallback_block_size,
        )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    if retained:
        vllm_config.additional_config[DYNAMO_KV_CACHE_GROUP_METADATA_KEY] = retained
    else:
        vllm_config.additional_config.pop(DYNAMO_KV_CACHE_GROUP_METADATA_KEY, None)
    return kv_event_block_size
