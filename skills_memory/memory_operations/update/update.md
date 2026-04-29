---
id: memory_operations.update
name: update
visibility: shared
scope_id: null
tags: [memory, update, correction]
update_type: update
---

# Update Existing Memory

## Description

Memory skill for revising an existing memory when new text clarifies, corrects,
narrows, or extends it.

## Purpose

Improve a retrieved memory so it reflects the latest grounded information while
preserving useful prior context.

## When To Use

- The text chunk corrects, clarifies, extends, or supersedes part of a retrieved
  memory.
- The old and new information refer to the same entity, event, preference,
  object, task, or procedure.
- The retrieved memory is still useful but incomplete, stale, or too vague.

## How To Apply

- Select the retrieved memory that best matches the new evidence.
- Merge valid old details with the new information into one coherent memory.
- Make the updated memory more precise without adding unsupported assumptions.
- Preserve useful context such as time, scope, conditions, or exceptions.

## Constraints

- Do not create a separate new memory when an existing memory should be revised.
- Do not delete information unless the text explicitly shows it is invalid.
- Do not update unrelated memories.

## Output Action

Action type: UPDATE only.
