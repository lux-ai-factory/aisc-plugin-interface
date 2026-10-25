"""Declare what a plugin assesses: a target of the AI card, a local model file, or both.

    @assesses(card=system_under_test_access(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"}))
    class ChatPlugin(BaseEvaluationPlugin): ...

    @assesses(card=dataset_through_target_access(datasets=("reference-dataset",)), local_model=onnx(input="model"))
    @evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider, input_type=InputType.MODEL)
    class ScoringPlugin(BaseEvaluationPlugin): ...

``card=`` says how the plugin reaches a target defined on the AI card (the system or one of its
components), with one of the builders below; each applies the matching decorator of
``system_under_test`` (unchanged behaviour). ``local_model=`` names the plugin's model input
(InputType.MODEL) that a model file uploaded to the project fills; declare that input below @assesses.
With both, the card access is optional at run time: a run assesses one or the other.

``required=False`` on ``system_under_test_access`` or ``dataset_through_target_access`` makes the card
access optional on its own: a run with no card target, or with a card target that has no endpoint
(training data, say), runs the plugin on its uploaded inputs unchanged, as the older decorators do
with ``required=False``. Any other failure to reach the platform still stops the run.

The engine reads the declaration with ``target_kinds_of``, ``card_protocols_of`` and
``card_needs_connection`` and builds the evaluation form from it, so a kind the plugin does not declare
cannot be picked. A subclass inherits its base's declaration and may not restate it. A plugin
declared with @system_under_test, @dataset_through_target or @assesses_inputs alone reads as ``card=``
with no local model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aisc_plugin_interface.decorators.evaluation_input import evaluation_input
from aisc_plugin_interface.model_session import check_outputs
from aisc_plugin_interface.models.evaluation_input import InputType
from aisc_plugin_interface.system_under_test import (
    SYSTEM_INPUT, _one_declaration, assesses_inputs, dataset_through_target, declaring_base,
    inherited_declaration, system_under_test, target_access_of,
)

CARD, LOCAL_MODEL = "card", "local_model"
#: card accesses that call the target, so a card target picked for them needs a connection
_CALLS_THE_TARGET = ("endpoint", "dataset_through_target", "model_session")


@dataclass(frozen=True)
class CardAccess:
    """How a plugin reaches a card target: ``access`` names it, ``apply(cls, required)`` declares it;
    ``required`` False lets a run have no card target, or one without an endpoint."""
    access: str
    apply: Callable = field(repr=False)
    required: bool = True


@dataclass(frozen=True)
class LocalModel:
    """A local model file, read through the plugin's model input named ``input``."""
    input: str


def system_under_test_access(protocols, fields=None, env=None, needs=(), required: bool = True) -> CardAccess:
    """The tool calls the target through its endpoint, as @system_under_test."""
    return CardAccess("endpoint", lambda cls, required: system_under_test(
        protocols, fields=fields, env=env, required=required, needs=needs)(cls), required)


def dataset_through_target_access(datasets, required: bool = True) -> CardAccess:
    """Each row of the named datasets is sent through the target's endpoint first, as @dataset_through_target."""
    return CardAccess("dataset_through_target",
                      lambda cls, required: dataset_through_target(datasets, required=required)(cls), required)


def inputs_access() -> CardAccess:
    """The plugin only reads its inputs and calls no system, as @assesses_inputs."""
    return CardAccess("inputs", lambda cls, required: assesses_inputs()(cls))


def model_session(outputs: dict, strict: bool = True) -> CardAccess:
    """The plugin scores rows with a model: the target answers each row, ``outputs`` says where its
    answer carries the label and the probability, or the value (see ``model_session`` the module, and
    ``self.assessed_model()``). ``strict`` (the default) stops the run on the first row the target
    rejects; ``strict=False`` skips and counts such rows."""
    outputs = check_outputs(outputs)

    def apply(cls, required):
        _one_declaration(cls, "model_session")
        cls.target_access = "model_session"
        cls.target_access_required = required
        cls.model_session_outputs = outputs
        cls.model_session_strict = bool(strict)
        # optional in the form, like the other card accesses' target input; ``required`` is what a run needs
        return evaluation_input(name=SYSTEM_INPUT, label="Target of the assessment (the system, or one of its"
                                " components)", input_type=InputType.RESOURCE, required=False)(cls)

    return CardAccess("model_session", apply)


def onnx(input: str = "model") -> LocalModel:
    """A local ONNX model file, read through the plugin's model input ``input``."""
    return LocalModel(input)


def assesses(card: CardAccess | None = None, local_model: LocalModel | None = None):
    """Class decorator that states what the plugin assesses (see the module docstring)."""
    if card is None and local_model is None:
        raise ValueError("declare what the plugin assesses: card=, local_model= or both")

    def decorator(cls):
        # a subclass inherits its base's declaration unchanged, whichever form the base used
        base = declaring_base(cls, "target_kinds", "target_access")
        if base is not None:
            raise inherited_declaration(cls, base)
        if "target_access" in cls.__dict__ or "target_kinds" in cls.__dict__:
            raise ValueError(f"{cls.__name__}: declare one of @assesses, @system_under_test, @assesses_inputs or"
                             " @dataset_through_target, not two")
        if local_model is not None:
            declared = [d for d in getattr(cls, "_input_definitions", []) if d.name == local_model.input]
            if not declared or declared[0].input_type != InputType.MODEL:
                raise ValueError(f"{cls.__name__}: local_model reads the model input {local_model.input!r}: declare"
                                 f" it with @evaluation_input(name={local_model.input!r},"
                                 " input_type=InputType.MODEL) below @assesses")
        if card is not None:
            cls = card.apply(cls, card.required and local_model is None)
        cls.target_kinds = tuple(k for k, v in ((CARD, card), (LOCAL_MODEL, local_model)) if v is not None)
        cls.card_access = card.access if card is not None else None
        cls.card_access_required = card.required if card is not None else None
        cls.local_model_input = local_model.input if local_model is not None else None
        return cls

    return decorator


def target_kinds_of(cls) -> tuple[str, ...]:
    """What the plugin assesses: ("card",), ("local_model",), ("card", "local_model"), or () when it
    declares nothing."""
    kinds = getattr(cls, "target_kinds", None)
    if kinds is not None:
        return tuple(kinds)
    return (CARD,) if target_access_of(cls) is not None else ()


def _card_access_of(cls) -> str | None:
    if getattr(cls, "target_kinds", None) is not None:
        return cls.card_access
    return target_access_of(cls)


def card_protocols_of(cls) -> tuple[str, ...]:
    """The protocols the plugin's tool speaks to a card target's endpoint; () for other accesses."""
    if _card_access_of(cls) != "endpoint":
        return ()
    return tuple(getattr(cls, "system_under_test_protocols", ()))


def card_needs_connection(cls) -> bool:
    """Whether a card target picked for this plugin must have a connection. Under @assesses: every card
    access that calls the target, unless its builder was declared ``required=False`` (a local model too
    only makes the card target optional in the form). For an old declaration alone: its ``required`` flag."""
    access = _card_access_of(cls)
    if getattr(cls, "target_kinds", None) is not None:
        return access in _CALLS_THE_TARGET and bool(cls.card_access_required)
    return access in _CALLS_THE_TARGET and bool(getattr(cls, "target_access_required", True))
