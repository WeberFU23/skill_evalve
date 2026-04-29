---
id: memory_operations
name: memory_operations
visibility: shared
scope_id: null
tags: [memory, operations, routing]
---

# Memory Operations

## Description

Root category for executable memory-management skills used by the PPO skill
tree router. Use this subtree when a text chunk may require inserting,
updating, deleting, or intentionally leaving memory unchanged.

## Purpose

Route the current text chunk to the most appropriate durable memory operation.
The children are executable skills that the memory executor can apply directly.

## Routing Guidance

- Continue to `insert` when the current chunk introduces new durable facts,
  preferences, events, plans, constraints, or reusable task details.
- Continue to `insert_negative_lesson` when the chunk contains user correction,
  explicit feedback, a failed answer, or a lesson about what not to repeat.
- Continue to `update` when new text corrects, clarifies, narrows, or extends
  a retrieved memory.
- Continue to `delete` only when a retrieved memory is clearly invalid,
  obsolete, duplicated, or unauthorized for the current scope.
- Continue to `noop` when the chunk is transient, redundant, or unsupported as
  long-term memory.

## Child Selection Criteria

Prefer the most specific child that matches the evidence. If several children
are plausible, include the evidence-preserving operation first and let the
executor emit multiple action blocks only when the selected skills support them.
