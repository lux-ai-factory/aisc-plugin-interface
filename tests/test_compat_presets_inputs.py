"""Faces, the compatibility rule, presets and the input decorator."""
from __future__ import annotations

from dataclasses import dataclass

from aisc_plugin_interface import InputRole
from aisc_plugin_interface.adapters import (
    BUILTIN_DATA_ADAPTERS,
    BUILTIN_ENDPOINT_ADAPTERS,
)
from aisc_plugin_interface.adapters import compat, presets
from aisc_plugin_interface.adapters.base import Answer
from aisc_plugin_interface.adapters.faces import (
    parse_agent_request,
    parse_chat_request,
    render_agent_response,
    render_chat_response,
)
from aisc_plugin_interface.decorators.evaluation_input import evaluation_input
from aisc_plugin_interface.models.evaluation_input import InputType


@dataclass
class Slot:
    accepts: list[str]
    adapter: str | None = None


# --------------------------------------------------------------------------- #
# Faces
# --------------------------------------------------------------------------- #
def test_chat_face_splits_history_from_the_last_turn():
    turn = parse_chat_request(
        {
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": [{"type": "text", "text": "why?"}]},
            ],
            "temperature": 0.5,
        }
    )
    assert turn.input == "why?"
    assert turn.history == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]
    assert turn.params["temperature"] == 0.5


def test_chat_face_renders_a_completion():
    body = render_chat_response(Answer(ok=True, answer="answer"), model="gpt", request_id="r1")
    assert body["id"] == "chatcmpl-r1"
    assert body["choices"][0]["message"]["content"] == "answer"


def test_chat_face_renders_a_refusal_as_content():
    body = render_chat_response(Answer(ok=True, refused=True, refusal_reason="no"), model="gpt")
    assert body["choices"][0]["message"]["content"] == "no"


def test_agent_face_parses_and_renders():
    turn = parse_agent_request({"params": {"message": {"parts": [{"kind": "text", "text": "hello"}]}}})
    assert turn.input == "hello"
    body = render_agent_response(Answer(ok=True, answer="42"), request_id="r1")
    assert body["id"] == "r1"
    assert body["result"]["artifacts"][0]["parts"][0]["text"] == "42"
    error = render_agent_response(Answer(ok=False, error="x", detail="boom"))
    assert error["error"]["message"] == "boom"


# --------------------------------------------------------------------------- #
# Compatibility rule
# --------------------------------------------------------------------------- #
def test_tsv_matches_a_records_slot():
    candidates = compat.data_slot_candidates("tsv", Slot(["records"]), BUILTIN_DATA_ADAPTERS)
    assert [c.id for c in candidates] == ["tsv"]


def test_parquet_matches_a_table_slot():
    candidates = compat.data_slot_candidates("parquet", Slot(["table"]), BUILTIN_DATA_ADAPTERS)
    assert "parquet" in [c.id for c in candidates]


def test_the_slot_default_is_ordered_first():
    candidates = compat.data_slot_candidates("parquet", Slot(["records"], adapter="parquet"), BUILTIN_DATA_ADAPTERS)
    assert candidates[0].id == "parquet"


def test_agent_target_matches_only_the_a2a_face():
    candidates = compat.endpoint_slot_candidates("agent", Slot(["openai-chat"]), BUILTIN_ENDPOINT_ADAPTERS)
    assert [c.id for c in candidates] == ["a2a"]


def test_openai_passthrough_matches_an_openai_slot():
    candidates = compat.endpoint_slot_candidates("openai-chat", Slot(["openai-chat"]), BUILTIN_ENDPOINT_ADAPTERS)
    assert [c.id for c in candidates] == ["openai"]


def test_a_model_slot_does_not_accept_records():
    assert compat.data_candidates("onnx", {"records"}, BUILTIN_DATA_ADAPTERS) == []


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def test_the_five_presets_are_defined():
    assert {p.id for p in presets.all_presets()} == {"llm", "agent", "model-api", "dataset", "model-file"}
    llm = presets.preset("llm")
    assert llm.kind == "endpoint" and llm.protocol == "openai-chat" and llm.adapter == "openai"


# --------------------------------------------------------------------------- #
# The input decorator
# --------------------------------------------------------------------------- #
def test_llm_shorthand_sets_accepts_and_role():
    @evaluation_input(name="llm", label="LLM", input_type=InputType.LLM)
    class Plugin:
        pass

    definition = Plugin._input_definitions[0]
    assert definition.accepts == ["openai-chat"]
    assert definition.role == InputRole.SUBJECT.value


def test_dataset_shorthand_is_records():
    @evaluation_input(name="data", label="Data", input_type=InputType.DATASET)
    class Plugin:
        pass

    assert Plugin._input_definitions[0].accepts == ["records"]


def test_dataset_can_ask_for_a_table():
    @evaluation_input(name="data", label="Data", input_type=InputType.DATASET, accepts=["table"])
    class Plugin:
        pass

    assert Plugin._input_definitions[0].accepts == ["table"]


def test_a_declared_default_and_required_adapter():
    @evaluation_input(
        name="target",
        label="Target",
        input_type=InputType.LLM,
        adapter="my_pkg:BankPortalAdapter",
        adapter_required=True,
        env={"OPENAI_API_KEY": "{api_key}"},
    )
    class Plugin:
        pass

    definition = Plugin._input_definitions[0]
    assert definition.adapter == "my_pkg:BankPortalAdapter"
    assert definition.adapter_required is True
    assert definition.env == {"OPENAI_API_KEY": "{api_key}"}


def test_an_old_definition_still_parses():
    from aisc_plugin_interface.models.evaluation_input import InputDefinition

    old = {"name": "data", "label": "Data", "input_type": "dataset", "required": True}
    definition = InputDefinition.model_validate(old)
    assert definition.accepts == []
    assert definition.role == InputRole.INPUT.value
    assert definition.adapter is None
