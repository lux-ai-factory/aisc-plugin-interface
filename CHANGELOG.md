# Changelog

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
