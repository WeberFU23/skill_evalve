---
id: memory_operations.noop
name: noop
visibility: shared
scope_id: null
tags: [memory, noop, no-change]
update_type: noop
---

# No Memory Change

## Description

Memory skill for deciding that no memory change is needed for the current text
chunk.

## Purpose

Leave the memory bank unchanged when the current chunk has nothing durable to
store, revise, delete, or turn into a negative lesson.

## When To Use

- The text chunk is transient, repetitive, speculative, or already covered by
  retrieved memories.
- The chunk does not clarify, correct, or invalidate any retrieved memory.
- The chunk contains no user preference, durable fact, failed pattern, or future
  instruction worth remembering.

## How To Apply

- Check whether insert, insert_negative_lesson, update, or delete has clear
  evidence.
- Use no action when the best memory behavior is to preserve the current bank.

## Constraints

- Do not use NOOP to avoid storing important durable information.
- Do not emit memory actions from this skill.

## Output Action

Action type: NOOP only.
