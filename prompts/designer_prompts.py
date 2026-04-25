"""
Designer prompts for evolving memory skills.

These prompts are compatible with the existing flat OperationBank output format,
but the analysis language is aligned with the hierarchical skill-tree design:
skills are routing/execution instructions, while memories contain user/task facts.
"""


DESIGNER_ANALYSIS_PROMPT = """You are an expert analyst for a memory-augmented agent. Analyze the failure cases and decide whether the memory skills should change.

## System Model
- Memory content is stored in a memory bank.
- Memory skills describe how to select, insert, update, delete, route, or compress memory-relevant information.
- A hierarchical skill tree may narrow the candidate skills by path before local skill selection.
- A failure should update skills only when the problem is caused by missing, vague, over-broad, or misrouted memory-skill behavior.

## Failure Types
- storage_failure: Important durable information was never stored.
- memory_quality_failure: A memory exists but is too vague, incomplete, stale, or poorly scoped.
- retrieval_failure: Useful memory exists but was not retrieved; do not change skills unless repeated evidence shows the storage format caused retrieval weakness.
- routing_failure: The wrong skill subtree/path or overly broad skill was selected.
- compression_failure: The selected skills were relevant, but the compressed skill prompt dropped important constraints.

## Current Memory Skills
{operation_bank_description}

## Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Analysis Instructions
This is round 1 of a reflection loop.
1. Identify which failures are actually skill failures rather than retrieval, answer-generation, or data-noise failures.
2. Check whether failures point to an existing skill that should be refined, or a stable sub-pattern that deserves a new child/specialized skill.
3. Keep skill updates about how to remember or route information. Do not encode raw user facts, task answers, or episode-specific memories as skills.
4. Prefer refining an existing skill when the fix is a clearer trigger, step, or constraint.
5. Prefer adding a new skill only when multiple cases reveal a reusable pattern not covered by the current bank.
6. Provide up to {max_changes} recommendations total.
{new_skill_hint}

## Output Format
Return JSON only:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<short name>",
            "affected_cases": [<case numbers>],
            "root_cause": "<storage_failure|memory_quality_failure|retrieval_failure|routing_failure|compression_failure>",
            "explanation": "<why this happened>",
            "potential_fix": "<specific skill-tree or operation-bank change>"
        }}
    ],
    "recommendations": [
        {{
            "action": "<add_new_operation|refine_existing_operation|no_change>",
            "target_operation": "<operation name or null>",
            "rationale": "<why this is justified>",
            "priority": "<high|medium|low>"
        }}
    ],
    "summary": "<1-2 sentence summary>"
}}

Output ONLY the JSON, no other text.
"""


DESIGNER_REFLECTION_PROMPT = """You are in reflection cycle ({reflection_round}/{reflection_round_total}) for memory-skill failure analysis. Critique and improve the previous analysis.

## Previous Analysis
{analysis_feedback}

## Current Memory Skills
{operation_bank_description}

## Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Reflection Instructions
- Re-check whether each proposed change targets a skill failure rather than retrieval or answer-generation noise.
- Strengthen recommendations by naming the trigger, missing constraint, or missing sub-skill.
- Remove recommendations that would store private facts or one-off episode content inside a skill.
- Prefer path-grounded changes: refine the implicated skill or add a child under it.
- Provide up to {max_changes} recommendations total.
{new_skill_hint}

## Output Format
Return JSON only:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<short name>",
            "affected_cases": [<case numbers>],
            "root_cause": "<storage_failure|memory_quality_failure|retrieval_failure|routing_failure|compression_failure>",
            "explanation": "<why this happened>",
            "potential_fix": "<specific skill-tree or operation-bank change>"
        }}
    ],
    "recommendations": [
        {{
            "action": "<add_new_operation|refine_existing_operation|no_change>",
            "target_operation": "<operation name or null>",
            "rationale": "<why this is justified>",
            "priority": "<high|medium|low>"
        }}
    ],
    "summary": "<1-2 sentence summary>"
}}

Output ONLY the JSON, no other text.
"""


DESIGNER_REFINEMENT_PROMPT = """Based on the failure analysis, propose concrete improvements to the memory skill bank.

## Failure Analysis
{analysis_feedback}

## Current Operation Bank
{operation_bank_full}

## Evolution Feedback
{evolution_feedback}

## Task
Propose up to {max_changes} improvements.

Option A - Add New Operation:
Create a specialized memory skill when the analysis shows a reusable capability gap.

Option B - Refine Existing Operation:
Improve an existing skill when its intent is right but the trigger, steps, scope, or constraints are unclear.

Option C - No Change:
Use this when failures are mainly retrieval, answer-generation, data quality, or insufficient evidence.

{new_skill_hint}

## Requirements
1. Skills describe how to remember; they must not contain raw memories, user facts, answers, or one-off episode details.
2. instruction_template must be a skill-style guide and must not include context placeholders.
3. instruction_template must clearly state Purpose, When to use, How to apply, Constraints, and Action type.
4. New operations may use only update_type "insert" or "update".
5. Keep descriptions short and embedding-friendly.
6. Avoid broad catch-all skills and duplicate skills with only cosmetic wording changes.
7. Do not modify the same operation more than once.
8. The number of changes must be <= {max_changes}.

## Output Format
Return one JSON object.

### Changes
{{
    "action": "apply_changes",
    "summary": "<overall rationale>",
    "changes": [
        {{
            "action": "add_new",
            "new_operation": {{
                "name": "<snake_case_name>",
                "description": "<embedding-friendly description>",
                "instruction_template": "Skill: ...\\nPurpose: ...\\nWhen to use:\\n- ...\\nHow to apply:\\n- ...\\nConstraints:\\n- ...\\nAction type: INSERT only.",
                "update_type": "<insert|update>",
                "reasoning": "<how this addresses the failures>"
            }}
        }},
        {{
            "action": "refine_existing",
            "refined_operation": {{
                "name": "<existing_operation_name>",
                "changes": {{
                    "description": "<improved description>",
                    "instruction_template": "<improved skill-style instruction template>"
                }},
                "reasoning": "<how this addresses the failures>"
            }}
        }}
    ]
}}

### No Change
{{
    "action": "no_change",
    "reasoning": "<why no skill change is justified>"
}}

Output ONLY the JSON, no other text.
"""
