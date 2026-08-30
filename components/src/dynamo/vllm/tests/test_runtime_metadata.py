# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm.cache_info import (
    CACHE_EVIDENCE_BARRIER_CAPABILITY,
    CACHE_EVIDENCE_CACHE_GROUP_CATALOGS,
    CACHE_EVIDENCE_EPOCH_CAPABILITY,
    CACHE_EVIDENCE_EPOCH_MEDIA,
    CACHE_EVIDENCE_SERVING_INCARNATIONS,
    DYNAMO_KV_CACHE_GROUP_METADATA_KEY,
    configure_kv_event_block_size,
    get_common_cache_group_metadata,
    publish_cache_evidence_barrier_capability,
)
from dynamo.vllm.capacity import get_metrics_model_name, get_spec_decode_runtime_data
from dynamo.vllm.engine_generate import (
    VLLM_GENERATE_CAPABILITY,
    publish_engine_generate_capability,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_spec_decode_runtime_data_uses_vllm_speculative_config():
    config = SimpleNamespace(
        engine_args=SimpleNamespace(
            speculative_config={"num_speculative_tokens": 99, "method": "ignored"}
        )
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3, method="eagle")
    )

    assert get_spec_decode_runtime_data(config, vllm_config) == {
        "nextn": 3,
        "method": "eagle",
        "source": "backend_config",
    }


def test_metrics_model_name_prefers_served_model_name():
    config = SimpleNamespace(model="meta-llama/Llama-3.1-8B", served_model_name="llama")

    assert get_metrics_model_name(config) == "llama"


def test_metrics_model_name_falls_back_to_model():
    config = SimpleNamespace(model="meta-llama/Llama-3.1-8B", served_model_name=None)

    assert get_metrics_model_name(config) == "meta-llama/Llama-3.1-8B"


@pytest.mark.asyncio
async def test_cache_group_metadata_is_retained_and_attested_per_global_dp_rank(
    monkeypatch,
):
    monkeypatch.setenv("DYN_CACHE_LOSS_FUNNEL_ENABLED", "true")
    groups = [
        {
            "group_idx": 1,
            "kind": "sliding_window",
            "block_size": 256,
            "sliding_window": 4096,
            "is_eagle": False,
            "alignment_tokens": 512,
        },
        {
            "group_idx": 0,
            "kind": "mla_attention",
            "block_size": 256,
            "sliding_window": None,
            "is_eagle": False,
            "alignment_tokens": 512,
        },
    ]
    engine_core = SimpleNamespace(
        _cache_evidence_engine=Mock(side_effect=lambda rank: f"engine-{rank}"),
        _call_utility_async=AsyncMock(side_effect=[groups, groups]),
    )
    engine = SimpleNamespace(
        engine_core=engine_core,
        cache_evidence_barrier=Mock(),
        cache_evidence_barrier_supported=AsyncMock(side_effect=[True, True]),
        cache_evidence_serving_incarnation=AsyncMock(side_effect=[101, 2**63 + 123]),
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=True,
            data_parallel_rank=4,
            data_parallel_size_local=2,
        ),
    )

    assert await configure_kv_event_block_size(engine, vllm_config) == 256
    normalized = [
        {
            "group_idx": 0,
            "kind": "mla_attention",
            "block_size": 256,
            "is_eagle": False,
            "alignment_tokens": 512,
        },
        {
            "group_idx": 1,
            "kind": "sliding_window",
            "block_size": 256,
            "is_eagle": False,
            "sliding_window": 4096,
            "alignment_tokens": 512,
        },
    ]
    assert vllm_config.additional_config[DYNAMO_KV_CACHE_GROUP_METADATA_KEY] == {
        "4": normalized,
        "5": normalized,
    }
    assert get_common_cache_group_metadata(vllm_config) == normalized
    assert engine_core._call_utility_async.call_args_list == [
        call("get_kv_cache_group_metadata", engine="engine-4"),
        call("get_kv_cache_group_metadata", engine="engine-5"),
    ]

    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    engine_args = SimpleNamespace(
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True, publisher="zmq")
    )
    assert await publish_cache_evidence_barrier_capability(
        runtime_config, engine, engine_args, True, (4, 2), vllm_config
    )
    published = {
        key: json.loads(value)
        for key, value in (
            entry.args for entry in runtime_config.set_engine_specific.call_args_list
        )
    }
    assert published[CACHE_EVIDENCE_CACHE_GROUP_CATALOGS] == {
        "4": {"serving_incarnation": "101", "cache_groups": normalized},
        "5": {
            "serving_incarnation": str(2**63 + 123),
            "cache_groups": normalized,
        },
    }


@pytest.mark.asyncio
async def test_empty_cache_group_metadata_omits_attestation():
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=AsyncMock(return_value=[]))
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            data_parallel_external_lb=True,
            data_parallel_hybrid_lb=False,
            data_parallel_rank=0,
        ),
    )

    assert await configure_kv_event_block_size(engine, vllm_config) == 16
    assert DYNAMO_KV_CACHE_GROUP_METADATA_KEY not in vllm_config.additional_config


def test_vllm_token_budget_matches_rejection_policy():
    from dynamo.vllm.capacity import publish_vllm_token_budget

    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    publish_vllm_token_budget(runtime_config, 4096)

    runtime_config.set_engine_specific.assert_called_once()
    key, value = runtime_config.set_engine_specific.call_args.args
    assert key == TOKEN_BUDGET_RUNTIME_KEY
    assert json.loads(value) == {
        "combined_limit": 4096,
        "reject_prompt_overflow": True,
        "reject_total_overflow": True,
    }


@pytest.mark.asyncio
async def test_cache_evidence_barrier_capability_requires_events_method_and_opt_in(
    monkeypatch,
):
    monkeypatch.setenv("DYN_CACHE_LOSS_FUNNEL_ENABLED", "true")
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    large_incarnation = 2**63 + 123
    engine = SimpleNamespace(
        cache_evidence_barrier=Mock(),
        cache_evidence_barrier_supported=AsyncMock(side_effect=[True, True]),
        cache_evidence_serving_incarnation=AsyncMock(
            side_effect=[101, large_incarnation]
        ),
    )
    engine_args = SimpleNamespace(
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True, publisher="zmq"),
        kv_transfer_config=None,
    )

    assert await publish_cache_evidence_barrier_capability(
        runtime_config, engine, engine_args, True, (4, 2)
    )
    assert runtime_config.set_engine_specific.call_args_list == [
        call(CACHE_EVIDENCE_BARRIER_CAPABILITY, json.dumps(True)),
        call(
            CACHE_EVIDENCE_SERVING_INCARNATIONS,
            json.dumps({"4": "101", "5": str(large_incarnation)}),
        ),
    ]

    runtime_config.set_engine_specific.reset_mock()
    assert not await publish_cache_evidence_barrier_capability(
        runtime_config, SimpleNamespace(), engine_args, True, (4, 2)
    )
    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.asyncio
async def test_cache_evidence_epoch_capability_advertises_media_per_global_dp_rank(
    monkeypatch,
):
    monkeypatch.setenv("DYN_CACHE_LOSS_FUNNEL_ENABLED", "true")
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    engine = SimpleNamespace(
        cache_evidence_barrier=Mock(),
        cache_evidence_barrier_supported=AsyncMock(side_effect=[True, True]),
        cache_evidence_serving_incarnation=AsyncMock(side_effect=[101, 202]),
        cache_evidence_epoch_media=AsyncMock(
            side_effect=[("GPU",), ("GPU", "CPU")]
        ),
        begin_cache_evidence_epoch=Mock(),
        commit_cache_evidence_epoch=Mock(),
        abort_cache_evidence_epoch=Mock(),
    )
    engine_args = SimpleNamespace(
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True, publisher="zmq"),
        kv_transfer_config=None,
    )

    assert await publish_cache_evidence_barrier_capability(
        runtime_config, engine, engine_args, True, (4, 2)
    )
    values = {
        key: json.loads(value)
        for key, value in (call.args for call in runtime_config.set_engine_specific.call_args_list)
    }
    assert values[CACHE_EVIDENCE_EPOCH_CAPABILITY] is True
    assert values[CACHE_EVIDENCE_EPOCH_MEDIA] == {
        "4": ["GPU"],
        "5": ["CPU", "GPU"],
    }
    assert values[CACHE_EVIDENCE_SERVING_INCARNATIONS] == {
        "4": "101",
        "5": "202",
    }

    runtime_config.set_engine_specific.reset_mock()
    engine_args.kv_events_config.publisher = "null"
    assert not await publish_cache_evidence_barrier_capability(
        runtime_config, engine, engine_args, True, (4, 2)
    )
    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transfer_config", "actual_support", "expected"),
    [
        (None, True, True),
        (
            SimpleNamespace(
                kv_connector="OffloadingConnector",
                kv_connector_extra_config={"self_describing_kv_events": True},
            ),
            True,
            True,
        ),
        (
            SimpleNamespace(
                kv_connector="OffloadingConnector",
                kv_connector_extra_config={"self_describing_kv_events": False},
            ),
            False,
            False,
        ),
    ],
    ids=("connector-absent", "self-describing", "not-self-describing"),
)
async def test_cache_evidence_capability_uses_actual_engine_support(
    monkeypatch, transfer_config, actual_support, expected
):
    monkeypatch.setenv("DYN_CACHE_LOSS_FUNNEL_ENABLED", "true")
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    engine = SimpleNamespace(
        cache_evidence_barrier=Mock(),
        cache_evidence_barrier_supported=AsyncMock(return_value=actual_support),
        cache_evidence_serving_incarnation=AsyncMock(return_value=101),
    )
    engine_args = SimpleNamespace(
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True, publisher="zmq"),
        kv_transfer_config=transfer_config,
    )

    assert (
        await publish_cache_evidence_barrier_capability(
            runtime_config, engine, engine_args, True, (0, 1)
        )
        is expected
    )
    engine.cache_evidence_barrier_supported.assert_awaited_once_with(
        data_parallel_rank=0
    )
    if expected:
        assert runtime_config.set_engine_specific.call_args_list == [
            call(CACHE_EVIDENCE_BARRIER_CAPABILITY, json.dumps(True)),
            call(
                CACHE_EVIDENCE_SERVING_INCARNATIONS,
                json.dumps({"0": "101"}),
            ),
        ]
    else:
        runtime_config.set_engine_specific.assert_not_called()
        engine.cache_evidence_serving_incarnation.assert_not_awaited()


@pytest.mark.parametrize(
    ("model_input", "model_type", "worker_type", "expected"),
    [
        (ModelInput.Tokens, ModelType.Prefill, WorkerType.Prefill, True),
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Decode, True),
        (ModelInput.Tokens, ModelType.Completions, WorkerType.Aggregated, True),
        (ModelInput.Tokens, ModelType.Empty, WorkerType.Prefill, False),
        (ModelInput.Tokens, ModelType.Empty, WorkerType.Decode, False),
        (ModelInput.Text, ModelType.Chat, WorkerType.Aggregated, False),
        (ModelInput.Tokens, ModelType.Embedding, WorkerType.Aggregated, False),
    ],
)
def test_vllm_generate_capability_publication(
    model_input, model_type, worker_type, expected
):
    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    published = publish_engine_generate_capability(
        runtime_config, model_input, model_type, worker_type
    )

    assert published is expected
    if expected:
        runtime_config.set_engine_specific.assert_called_once_with(
            VLLM_GENERATE_CAPABILITY, json.dumps(True)
        )
    else:
        runtime_config.set_engine_specific.assert_not_called()


def test_spec_decode_runtime_data_falls_back_to_engine_args_json():
    config = SimpleNamespace(
        engine_args=SimpleNamespace(
            speculative_config='{"num_speculative_tokens": "4", "method": "ngram"}'
        )
    )
    vllm_config = SimpleNamespace(speculative_config=None)

    assert get_spec_decode_runtime_data(config, vllm_config) == {
        "nextn": 4,
        "method": "ngram",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_config",
    [None, {}, {"num_speculative_tokens": 0}, {"num_speculative_tokens": "bad"}],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_config):
    config = SimpleNamespace(
        engine_args=SimpleNamespace(speculative_config=speculative_config)
    )
    vllm_config = SimpleNamespace(speculative_config=None)

    assert get_spec_decode_runtime_data(config, vllm_config) is None
