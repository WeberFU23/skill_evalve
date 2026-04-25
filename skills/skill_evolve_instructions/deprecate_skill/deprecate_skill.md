---
id: skill_evolve_instructions.deprecate_skill
name: deprecate_skill
visibility: shared
scope_id: null
tags: [skill-evolution, designer, deprecation, pruning]
---

# Deprecate Skill

## Description

Skill evolution instruction for removing a skill from active selection when it is harmful, obsolete, redundant, or no longer authorized for the current scope.

## Purpose

Keep the skill bank compact, trustworthy, and easier for the controller to search.

## When to Use

- A skill repeatedly receives low reward after sufficient usage.
- The skill causes harmful memory actions such as unsafe deletion or persistent noise.
- The skill has been replaced by a clearer refined or merged skill.
- The skill is no longer valid for the current domain, user, task, or episode scope.

## How to Apply

- Verify that the skill has enough usage or failure evidence to justify deprecation.
- Prefer marking the skill as `deprecated` before hard deletion when checkpoints may reference it.
- Record the reason for deprecation and the recommended replacement skill if one exists.
- Remove deprecated skills from active candidate selection unless explicitly requested for analysis.

## Constraints

- Do not deprecate a rarely used skill solely because it has low usage.
- Do not hard-delete a skill that is still referenced by checkpoints, logs, or aliases.
- Do not deprecate a scoped private skill based only on global aggregate statistics.
- Do not remove safety-oriented constraints just to simplify the skill bank.

## Output Action

Action type: DELETE only.
