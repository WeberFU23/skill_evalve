---
id: skill_evolve_instructions.merge_redundant_skills
name: merge_redundant_skills
visibility: shared
scope_id: null
tags: [skill-evolution, designer, merge, redundancy]
---

# Merge Redundant Skills

## Description

Skill evolution instruction for consolidating overlapping memory-management skills whose separate presence harms selection clarity or splits learning signal.

## Purpose

Reduce skill-bank redundancy while preserving the useful distinctions that matter for memory construction.

## When to Use

- Two or more skills have highly similar descriptions, usage conditions, and action types.
- The controller alternates between near-duplicate skills without clear reward differences.
- Designer feedback repeatedly proposes similar refinements for multiple skills.
- A smaller merged skill would make selection more stable and explainable.

## How to Apply

- Compare the candidate skills' descriptions, constraints, action types, and observed failure modes.
- Preserve the most useful behavior from each skill in a single clearer instruction.
- Choose one canonical skill name and mark the others as deprecated or aliases when supported.
- Keep the merged skill narrow enough that it still has a clear selection boundary.
- Recompute or refresh embeddings after the merge.

## Constraints

- Do not merge skills that differ in safety-critical constraints, especially deletion behavior.
- Do not merge merely because names sound similar.
- Do not discard a specialized behavior that has strong reward evidence.
- Do not merge shared and scoped skills unless the resulting visibility policy is clear.

## Output Action

Action type: UPDATE only.
