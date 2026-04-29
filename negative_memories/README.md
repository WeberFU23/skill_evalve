# Negative Memory Directory

This directory stores optional markdown negative memories. They are disabled by
default and are loaded only when `--enable-negative-memory` is passed.

Only markdown files with frontmatter `type: negative` or a `negative` tag are
loaded. A negative memory should store a reusable mistake pattern, correction,
lesson, and trigger. Do not store hidden chain-of-thought.

Example shape:

```md
---
type: negative
title: "example correction"
date: 2026-04-21
tags: [negative, correction]
visibility: private
scope_id: user_123
---

# Example Correction

## Problem

What went wrong.

## Wrong Behavior

The reusable mistake pattern.

## Correction

What the user or evaluator said is correct.

## Lesson

What future runs should do instead.

## Trigger

When this lesson should be retrieved.
```
