"""Constrained-generation layer.

Deterministic scaffolding that wraps the LLM+Skill generation step: feasible-
region integration (pre-generation) and weekly-volume allocation (post form
stage). These are NOT numbered Tools 0-5 — they are the constrained-generation
helpers the orchestrator calls around the LLM, kept as pure functions so they
stay unit-testable and auditable.
"""
