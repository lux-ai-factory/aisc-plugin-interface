# Changelog

## 2026-10-25: 0.4.0, upstream 0.3.0 merged; the target's key stays on the platform

New
- `required=` on `dataset_through_target_access` and `system_under_test_access` (default True). With `required=False` a run with no card target, or with one that has no endpoint, runs the plugin on its uploads unchanged, as the older decorators' `required=False`; `@assesses` records `card_access_required` and `card_needs_connection` honours it. Test: `tests/test_assesses.py`.

Changed
- Merged upstream 0.3.0 (its README and guide changes, `pydantic>=2.13.5`, `rich>=15.0.0`). This release is 0.4.0 so that a `>=0.4.0` constraint never resolves PyPI's 0.3.0, which lacks this branch's modules (`connections`, `system_under_test`, `targets`, `assesses`, `helpers`, `model_session`). Test: `tests/test_version.py`.

Unchanged
- `@system_under_test`, `@dataset_through_target`, `@assesses_inputs`.

Check: `uv run --with pytest --with pandas --with onnx --with onnxruntime --with "pydantic>=2.13.5" --with "rich>=15.0.0" python -m pytest -q`.

## 2026-10-24: 0.2.9, what a plugin assesses and its helper models

New
- `@assesses(card=..., local_model=...)` states what a plugin assesses: a target of the AI card (`system_under_test_access`, `dataset_through_target_access`, `inputs_access` or `model_session`, each applying the matching older decorator or adapter) and/or a local ONNX model file (`onnx(input="model")`). The engine reads it with `target_kinds_of`, `card_protocols_of` and `card_needs_connection` to build the evaluation form. Exactly one declaration per class; a subclass inherits it and may not restate it. Tests: `tests/test_assesses.py`, `tests/test_target_declarations.py`.
- `@helper_model(role, label, required=False)` declares a judge, attacker, grader, embedder or generator as the LLM input `helper:<role>`; `self.helper(role)` returns a `HelperModel` with the key read from the project secret, never from a config field. `helper_roles_of(cls)` and `HELPER_ROLES` are exported. Test: `tests/test_helper_model.py`.
- `model_session(outputs={"label": ..., "probability": ...})` and `self.assessed_model()`: one `predict` / `predict_proba` object over the plugin's local ONNX file or, for a card target, the target's connection (each row sent once; binary classification only on card targets). Test: `tests/test_model_session.py`.
- `self.assessed_model(outputs={...})` replaces the declared card outputs for that session only, so a plugin can take the answer paths from its config; validated like a declaration (ValueError naming the key), ignored but still validated by a local session. The guide and docstrings now say that `outputs["classes"]` above 2 refuses a card target, that local labels are class indices only when the ONNX model has a probability output, that `predict_proba` without a probability path raises on a card session, and that `predict` with only a probability path returns 0 or 1.
- PLUGIN_DEVELOPER_GUIDE.md section 13 is rewritten around these.

Unchanged
- `@system_under_test`, `@dataset_through_target` and `@assesses_inputs` used alone behave as before and read as a card declaration with no local model.

Fixes
- The text of a target without an endpoint and the guide name the page "Manage, Components and connections", and the engine's target names are "AI card · System: ... · reached through ..." / "AI card · Component: ... (...) · not connected".

Check: `uv run --with pytest --with pandas --with onnx --with onnxruntime python -m pytest -q`.

## 2026-10-19: 0.2.8, run pass

Security
- The platform's internal routes (resolving a connection or a target's endpoint, issuing a run key) are called with `X-AISC-Run-Pass: <PLATFORM_RUN_PASS>` when the run has a pass, which opens only the targets and connections the run names. Without a pass the `X-AISC-Service-Token` from `PLATFORM_CONNECTIONS_TOKEN` is sent as before (older platforms and setups); with neither, the error names both. Tests: `tests/test_connections.py`, `tests/test_system_under_test.py`.

## 2026-10-16: review fixes

- An A2A answer whose `parts` (of the message, the task's status message or an artifact) is there but not a list is an `EndpointBadResponse`: an int raised `TypeError` out of `call()` and ended the whole dataset run, and a dict or a string was read as no answer. Test: `tests/test_connection_kinds.py`.

## 2026-10-14: review pass (branch pass/libs-2026-10-13)

Areas: `connections.py`, `system_under_test.py` (`@dataset_through_target`), PLUGIN_DEVELOPER_GUIDE.md section 13.

Security
- `target-answers-<dataset>.csv`: a text cell or column name starting with `=`, `+`, `-`, `@`, tab or CR gets a leading `'` (CSV formula injection); numbers unchanged. Test: `tests/test_target_answers_csv.py`.
- Connections: a proxy variable (or `NO_PROXY` alone) no longer switches off the DNS rebinding check; only a socket to a configured proxy's host skips it. Test: `tests/test_connection_network.py`.

Fixes
- A2A and OIP connections with a key header now send it (they sent only a Bearer, or nothing). Test: `tests/test_connection_kinds.py`.
- An A2A answer of the wrong shape, a non-JSON reply or a dropped connection while polling is an `EndpointBadResponse` / `EndpointError`, not an `AttributeError` or `JSONDecodeError`. Test: `tests/test_connection_kinds.py`.
- `@dataset_through_target`: a system's own `refused`, `refusal_reason`, `error` or `answer` field goes under `target.answer.<field>` and no longer stops the run as if every call failed. Test: `tests/test_through_rows.py`.
- `@dataset_through_target`: date, time, Timestamp and Decimal cells are sent as ISO text or numbers instead of failing JSON encoding; the table handed back keeps its types. Test: `tests/test_through_rows.py`.
- `connections.py` docstring names the allow list a platform-resolved client really uses.

Check: `uv run --with pytest --with pandas python -m pytest -q -p no:cacheprovider` (179 passed). On a running stack the fixes arrive only once the plugin images or engine are rebuilt with this version; then a dataset-through-target run's `target-answers-*.csv` shows `'=...` for formula-like cells.
