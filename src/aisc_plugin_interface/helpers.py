"""Declare the models a plugin uses as tools: a judge, an attacker, a grader, an embedder or a generator.

    @helper_model("judge", "Judge model", required=True)
    @assesses(card=system_under_test_access(protocols=("openai",)))
    class JailbreakPlugin(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            judge = self.helper("judge")  # HelperModel(role, endpoint_url, model, api_key)

A helper model is never what the plugin assesses (that is @assesses). Each role is an LLM input named
``helper:<role>``, which the engine fills like any LLM input: the endpoint and the name of the project
secret holding the key, with the model picked for the run. ``self.helper(role)`` reads it, the key from
the secret's environment variable (as ``get_secret``), never from the input itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aisc_plugin_interface.decorators.evaluation_input import evaluation_input
from aisc_plugin_interface.models.evaluation_input import InputType

HELPER_ROLES = ("judge", "attacker", "grader", "embedder", "generator")
HELPER_INPUT_PREFIX = "helper:"


@dataclass(frozen=True)
class HelperModel:
    """A helper model resolved for a run; its repr never shows the key."""
    role: str
    endpoint_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)


def helper_input_name(role: str) -> str:
    return HELPER_INPUT_PREFIX + role


def helper_model(role: str, label: str, required: bool = False):
    """Class decorator that declares a helper model for ``role`` (see the module docstring)."""
    if role not in HELPER_ROLES:
        raise ValueError(f"unknown helper role {role!r}: use one of {', '.join(HELPER_ROLES)}")

    def decorator(cls):
        cls = evaluation_input(name=helper_input_name(role), label=label, input_type=InputType.LLM,
                               required=required)(cls)
        # as the inputs: a subclass starts from a copy of the roles it inherits
        if "helper_roles" not in cls.__dict__:
            cls.helper_roles = tuple(getattr(cls, "helper_roles", ()))
        if role not in cls.helper_roles:
            cls.helper_roles = cls.helper_roles + (role,)
        return cls

    return decorator


def helper_roles_of(cls) -> tuple[str, ...]:
    """The helper roles the plugin declares, () when none."""
    return tuple(getattr(cls, "helper_roles", ()))
