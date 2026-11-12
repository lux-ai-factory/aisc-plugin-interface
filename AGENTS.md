# AGENTS.md — aisc-plugin-interface

A **published ABI**: the base classes and models a plugin author imports.

## Read first

The binding rules live in the parent repository, `lux-ai-factory/aisc`:
`ARCHITECTURE.md` (invariants I1–I11) and `docs/refactor-plan.md`.

## What binds this repo

- **I3** — it must not depend on the engine (`aisc_backend`, `aisc_eval`) or on `aisc-plugin-manager`.
- It ships to third-party plugin authors; a breaking change is a version bump.

## Build and test

```bash
uv sync
uv run ruff check .
uv build
```