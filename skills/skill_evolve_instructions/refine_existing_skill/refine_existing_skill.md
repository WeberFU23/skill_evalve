---
id: skill_evolve_instructions.refine_existing_skill
name: refine_existing_skill
visibility: shared
scope_id: null
tags: [skill-evolution, designer, refinement, calibration]
---

# Refine Existing Skill

## Description

Skill evolution instruction for improving an existing memory-management skill when it is relevant but produces incomplete, noisy, or misdirected behavior.

## Purpose

Make an existing skill more precise, reliable, and easier for the controller and executor to use.

## When to Use

- A selected skill addresses the right general need but misses important details.
- The skill causes overly broad memory insertion, unsafe deletion, or weak updates.
- The instruction is ambiguous enough that the executor applies it inconsistently.
- Reward or failure analysis shows the skill is useful but under-specified.

## How to Apply

- Preserve the original intent of the skill unless the failure evidence clearly contradicts it.
- Add concrete use conditions that distinguish this skill from nearby siblings.
- Tighten the `How to Apply` section with explicit steps and decision criteria.
- Add constraints that prevent the observed failure mode.
- Keep the description short and embedding-friendly.
- Increment the version and record the refinement reason in implementation metadata when supported.

## Constraints

- Do not turn a narrow skill into a broad fallback skill.
- Do not silently change the allowed `update_type` unless the failure evidence requires it.
- Do not add task facts, user facts, or episode-specific conclusions to the skill body.
- Do not refine multiple unrelated behaviors into one skill.

## Output Action

Action type: UPDATE only.
