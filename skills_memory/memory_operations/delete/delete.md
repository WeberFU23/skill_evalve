---
id: memory_operations.delete
name: delete
visibility: shared
scope_id: null
tags: [memory, delete, invalid, obsolete]
update_type: delete
---

# Delete Invalid Memory

## Description

Memory skill for deleting a retrieved memory that is explicitly invalid,
obsolete, duplicated, or unsafe to keep.

## Purpose

Remove a memory only when there is clear evidence that it should no longer be
used.

## When To Use

- The text chunk explicitly contradicts, cancels, or replaces the retrieved
  memory.
- The retrieved memory is a duplicate of a better memory.
- The retrieved memory is clearly stale, misleading, unsafe, or outside the
  authorized scope.

## How To Apply

- Delete only the memory item directly supported by the evidence.
- Prefer update over delete when useful information can be preserved.
- If multiple memories are duplicates, keep the most complete and grounded one.

## Constraints

- If evidence is ambiguous, do not delete.
- Do not delete because a memory is merely unretrieved, rarely used, or
  incomplete.
- Do not delete scoped private memories based only on global aggregate behavior.

## Output Action

Action type: DELETE only.
