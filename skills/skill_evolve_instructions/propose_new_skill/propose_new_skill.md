---
id: skill_evolve_instructions.propose_new_skill
name: propose_new_skill
visibility: shared
scope_id: null
tags: [skill-evolution, designer, add-new, coverage]
---

# Propose New Skill

## Description

Skill evolution instruction for creating a new memory-management skill when repeated failures reveal a stable gap in the current skill bank.

## Purpose

Add a reusable skill that captures a missing way of remembering, updating, deleting, or focusing memory construction.

## When to Use

- Failure cases share a recurring pattern that existing skills do not cover.
- The designer repeatedly needs the same instruction but no current skill expresses it.
- The missing behavior is general enough to transfer across examples in the same domain or subtree.
- Existing skills would require large, confusing changes to cover the new behavior.

## How to Apply

- Identify the recurring failure pattern and the missing memory-management behavior.
- Choose the most specific parent subtree where the new skill belongs.
- Write a concise `description` suitable for embedding and retrieval.
- Write execution guidance that states when to use the skill, how to apply it, and what constraints to respect.
- Set `visibility`, `scope_type`, and `scope_id` according to who should be allowed to reuse the skill.
- Set `update_type` to the narrowest valid action type whenever possible.

## Constraints

- Do not create a new skill for a single isolated failure unless it reflects a broader pattern.
- Do not encode private facts or raw memories as skill instructions.
- Do not create a broad catch-all skill that competes with many existing skills.
- Do not duplicate an existing skill with only superficial wording changes.

## Output Action

Action type: INSERT only.
