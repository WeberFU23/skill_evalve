"""
Designer prompts for interactive or embodied tasks such as ALFWorld.
"""


INTERACTIVE_DESIGNER_ANALYSIS_PROMPT = """You are an expert analyst for a memory-augmented interactive agent. Analyze the failures and decide whether the memory skills should change.

## System Model
- The agent receives partial observations over a trajectory.
- Memory skills decide how to preserve reusable procedures, object states, locations, action preconditions, and constraints.
- Memory content stores concrete episode facts. Skills must remain reusable instructions for how to remember.
- A hierarchical skill tree may first route to an embodied-task subtree, then to a more specific skill.

## Interactive Failure Types
- storage_failure: Important state, procedure, or constraint was never stored.
- memory_quality_failure: Stored memory is too vague, missing steps, missing state changes, or not actionable.
- retrieval_failure: Useful memory exists but was not retrieved for the objective.
- routing_failure: The wrong skill path/subtree was used.
- compression_failure: Relevant selected skills were compressed into a prompt that lost key constraints.
- policy_failure: Memory was adequate, but the action model still chose poorly.

## Current Memory Skills
{operation_bank_description}

## Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Analysis Instructions
This is round 1 of a reflection loop.
1. Determine whether each failure is caused by memory storage/quality/routing, or by retrieval/policy issues.
2. Look for repeated patterns involving object location, state transitions, action preconditions, inventory, containers, devices, or goal ordering.
3. Propose skill changes only for reusable memory-management behavior.
4. Do not hard-code exact room names, object IDs, game files, or episode-specific plans as skills.
5. Prefer refining an implicated skill before adding a new child skill.
6. Provide up to {max_changes} recommendations total.
{new_skill_hint}

## Output Format
Return JSON only:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<short name>",
            "affected_cases": [<case numbers>],
            "root_cause": "<storage_failure|memory_quality_failure|retrieval_failure|routing_failure|compression_failure|policy_failure>",
            "explanation": "<why this happened>",
            "potential_fix": "<specific skill change if justified>"
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


INTERACTIVE_DESIGNER_REFLECTION_PROMPT = """You are in reflection cycle ({reflection_round}/{reflection_round_total}) for interactive memory-skill failure analysis. Critique and improve the previous analysis.

## Previous Analysis
{analysis_feedback}

## Current Memory Skills
{operation_bank_description}

## Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Reflection Instructions
- Verify that proposed changes address reusable memory behavior, not one episode's plan.
- Separate memory failures from policy/action-selection failures.
- Strengthen fixes around object states, locations, preconditions, inventory, and reusable subgoals.
- Remove recommendations unsupported by repeated evidence.
- Provide up to {max_changes} recommendations total.
{new_skill_hint}

## Output Format
Return JSON only:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<short name>",
            "affected_cases": [<case numbers>],
            "root_cause": "<storage_failure|memory_quality_failure|retrieval_failure|routing_failure|compression_failure|policy_failure>",
            "explanation": "<why this happened>",
            "potential_fix": "<specific skill change if justified>"
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


INTERACTIVE_DESIGNER_REFINEMENT_PROMPT = """Based on the failure analysis, propose concrete improvements to the interactive memory skill bank.

## Failure Analysis
{analysis_feedback}

## Current Operation Bank
{operation_bank_full}

## Evolution Feedback
{evolution_feedback}

## Task
Propose up to {max_changes} improvements.

Option A - Add New Operation:
Create a specialized skill for a reusable trajectory-memory need, such as state transitions, object locations, preconditions, inventory, or goal ordering.

Option B - Refine Existing Operation:
Improve a skill whose trigger, scope, steps, or constraints are too weak.

Option C - No Change:
Use this when failures are retrieval, policy, environment stochasticity, or insufficient evidence.

{new_skill_hint}

## Requirements
1. Skills must generalize across task instances and must not hard-code exact room names, object IDs, game files, or private episode facts.
2. instruction_template must be a skill-style guide with Purpose, When to use, How to apply, Constraints, and Action type.
3. instruction_template must not include context placeholders.
4. New operations may use only update_type "insert" or "update".
5. Make descriptions short and embedding-friendly.
6. The number of changes must be <= {max_changes}.
7. Do not modify the same operation more than once.

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
