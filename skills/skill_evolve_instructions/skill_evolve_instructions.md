---
id: skill_evolve_instructions.skill_evolve_instructions
name: skill_evolve_instructions
visibility: shared
scope_id: null
tags: [skill-evolution, designer, routing]
---

# Skill Evolution Instructions

## Description

A category node for skills that guide how the system should evolve, refine, merge, deprecate, or create memory-management skills from task feedback.

## Routing Guidance

Use this subtree when the designer is analyzing failure cases, reward trends, redundant skills, brittle instructions, or missing memory-management behavior.

## Child Selection Criteria

- Use `propose_new_skill` when failures show a recurring memory need that no existing skill covers well.
- Use `refine_existing_skill` when an existing skill is relevant but too vague, too broad, too narrow, or causing incorrect memory actions.
- Use `merge_redundant_skills` when multiple skills overlap heavily and split usage or reward credit.
- Use `deprecate_skill` when a skill is consistently harmful, obsolete, redundant, or outside the current task scope.

## Constraints

- Do not put task facts, user preferences, or episode-specific memories directly into this subtree.
- Keep evolved nodes focused on how to remember, update, delete, route, or evaluate memory skills.
- Prefer refining an existing skill over creating a new one when the desired behavior is a small correction.
- Prefer deprecation over hard deletion when historical checkpoints or evaluations may still reference the skill.

## Output Action

Action type: NOOP only.
