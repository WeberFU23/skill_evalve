---
id: memory_operations.insert_negative_lesson
name: insert_negative_lesson
visibility: shared
scope_id: null
tags: [memory, negative, correction, lesson]
update_type: insert
---

# Insert Negative Lesson

## Description

Memory skill for storing a reusable lesson from a mistake, user correction,
failed answer, or explicit "do not do this again" feedback.

## Purpose

Capture negative memory as a compact guardrail that helps the system avoid
repeating a known error pattern. Store the mistake pattern and correction, not
hidden chain-of-thought.

## When To Use

- The user says the previous answer, memory action, reasoning pattern, or
  behavior was wrong.
- The text contains an explicit correction such as "this is not right",
  "should be", "do not", "avoid", or "next time".
- A failed case reveals a reusable error pattern rather than a one-off typo.
- The lesson can help future answers or memory updates for the same user,
  dataset, domain, or task type.
- During ordinary conversation-memory construction, use this skill only when
  the current text explicitly contains correction, failure, or "do not repeat"
  feedback. Do not infer a negative lesson from normal facts.

## How To Apply

- Insert one concise memory beginning with `[negative]`.
- Include the trigger condition, wrong behavior, correction, and future lesson.
- Prefer reusable lessons over raw transcripts.
- Keep user-specific lessons scoped in wording, for example "For this user..."
  when the correction is personal or private.

## Constraints

- Do not store hidden chain-of-thought or long reasoning traces.
- Do not store insults, irrelevant frustration, or sensitive raw content unless
  it is necessary to define the correction.
- Do not overwrite ordinary factual memories; this skill inserts a separate
  negative lesson.
- Do not choose this skill just because a fact is important. Use `insert` for
  ordinary durable facts.

## Output Action

Action type: INSERT only.
