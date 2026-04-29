# Running LoCoMo Skill Tree Experiments

This fork keeps the original MemSkill flat operation-bank path and adds a
directory-backed skill tree path.

## 1. Original MemSkill Path

Run the existing flat operation-bank baseline:

```bash
export DEEPSEEK_API_KEY="your_key"
bash train_locomo.sh
bash eval_locomo.sh
```

## 2. Skill Tree Path

Run the PPO skill-tree router over `skills_memory/`:

```bash
export DEEPSEEK_API_KEY="your_key"
bash train_locomo_skill_tree.sh
bash eval_locomo_skill_tree.sh
```

The important flags are:

```bash
--enable-skill-tree
--skill-tree-dir ./skills_memory
--enable-negative-memory
--negative-memory-dir ./negative_memories
```

`skills_memory/` contains executable memory skills:

- `insert`
- `insert_negative_lesson`
- `update`
- `delete`
- `noop`

## 3. Negative Memories

Negative memories are markdown lessons from mistakes or user corrections. They
are optional prompt guardrails and are loaded only when
`--enable-negative-memory` is passed.

Only markdown files with `type: negative` or a `negative` tag are loaded from
`negative_memories/`. Use the template in `negative_memories/README.md`.

Do not store hidden chain-of-thought. Store the reusable wrong pattern,
correction, trigger, and lesson.

You can write one entry manually:

```bash
python -B record_negative_memory.py \
  --problem "The model answered A when the corrected answer was B." \
  --wrong-behavior "Ignored the user's explicit correction." \
  --correction "Use B when condition X appears." \
  --lesson "Check condition X before reusing answer A." \
  --trigger "Similar questions involving condition X" \
  --user-id "user_123" \
  --tag reasoning_error
```
