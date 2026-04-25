"""
Fallback seed memory skills for the flat OperationBank.

The project is moving toward directory-backed hierarchical skills under
`skills/`. These templates remain as a compact compatibility layer for the
existing OperationBank/controller/executor path.
"""


SKILL_INSERT_DESCRIPTION = (
    "Memory skill for inserting new durable information that is useful beyond "
    "the current text chunk and not already covered by retrieved memories."
)
SKILL_INSERT_TEMPLATE = """Skill: Insert Durable Memory
Purpose: Store new information that is stable, attributable, and likely to help future reasoning.
When to use:
- The text chunk introduces a durable fact, preference, event, plan, constraint, or reusable task detail.
- Retrieved memories do not already contain the same information.
- The information should remain useful after the current chunk is gone.
How to apply:
- Compare the text chunk with retrieved memories to avoid duplicates.
- Keep each memory concise, specific, and grounded in the text.
- Preserve important entities, time, location, task state, cause, outcome, or constraints when they matter.
- Split unrelated facts into separate memory items.
Constraints:
- Do not store raw private or episode-specific details as general skills.
- Do not store trivial, speculative, or one-off details unless they clearly affect future behavior.
- Do not update or delete existing memories.
Action type: INSERT only.
"""


SKILL_UPDATE_DESCRIPTION = (
    "Memory skill for revising an existing memory when new text clarifies, "
    "corrects, narrows, or extends it."
)
SKILL_UPDATE_TEMPLATE = """Skill: Update Existing Memory
Purpose: Improve a retrieved memory so it reflects the latest grounded information.
When to use:
- The text chunk corrects, clarifies, extends, or supersedes part of a retrieved memory.
- The old and new information refer to the same entity, event, preference, object, task, or procedure.
How to apply:
- Select the retrieved memory that best matches the new evidence.
- Merge valid old details with the new information into one coherent memory.
- Make the updated memory more precise without adding unsupported assumptions.
- Preserve useful context such as time, scope, conditions, or exceptions.
Constraints:
- Do not create a separate new memory when an existing memory should be revised.
- Do not delete information unless the text explicitly shows it is invalid.
- Do not update unrelated memories.
Action type: UPDATE only.
"""


SKILL_DELETE_DESCRIPTION = (
    "Memory skill for deleting a retrieved memory that is explicitly invalid, "
    "obsolete, duplicated, or unsafe to keep."
)
SKILL_DELETE_TEMPLATE = """Skill: Delete Invalid Memory
Purpose: Remove a retrieved memory only when there is clear evidence that it should no longer be used.
When to use:
- The text chunk explicitly contradicts, cancels, or replaces the retrieved memory.
- The retrieved memory is a duplicate of a better memory.
- The retrieved memory is clearly stale, misleading, or outside the authorized scope.
How to apply:
- Delete only the memory item directly supported by the evidence.
- Prefer update over delete when useful information can be preserved.
Constraints:
- If evidence is ambiguous, do not delete.
- Do not delete because a memory is merely unretrieved, rarely used, or incomplete.
Action type: DELETE only.
"""


SKILL_NOOP_DESCRIPTION = (
    "Memory skill for deciding that no memory change is needed for the current text chunk."
)
SKILL_NOOP_TEMPLATE = """Skill: No Memory Change
Purpose: Leave the memory bank unchanged when the current chunk has nothing durable to store or revise.
When to use:
- The text chunk is transient, repetitive, speculative, or already covered by retrieved memories.
- The chunk does not clarify, correct, or invalidate any retrieved memory.
How to apply:
- Check whether insert, update, or delete has clear evidence.
- Use no action when the best memory behavior is to preserve the current bank.
Constraints:
- Do not use NOOP to avoid storing important durable information.
- Do not emit memory actions from this skill.
Action type: NOOP only.
"""


INITIAL_OPERATIONS = {
    "insert": {
        "name": "insert",
        "description": SKILL_INSERT_DESCRIPTION,
        "instruction_template": SKILL_INSERT_TEMPLATE,
        "update_type": "insert",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    },
    "update": {
        "name": "update",
        "description": SKILL_UPDATE_DESCRIPTION,
        "instruction_template": SKILL_UPDATE_TEMPLATE,
        "update_type": "update",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    },
    "delete": {
        "name": "delete",
        "description": SKILL_DELETE_DESCRIPTION,
        "instruction_template": SKILL_DELETE_TEMPLATE,
        "update_type": "delete",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    },
    "noop": {
        "name": "noop",
        "description": SKILL_NOOP_DESCRIPTION,
        "instruction_template": SKILL_NOOP_TEMPLATE,
        "update_type": "noop",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    }
}


def get_initial_operations(include_noop: bool = False):
    """Return a deep copy of fallback seed operations."""
    import copy
    ops = copy.deepcopy(INITIAL_OPERATIONS)
    if not include_noop:
        ops.pop("noop", None)
    return ops
