"""Prompt pool for answer generation and evaluation."""


CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. "
    "The conversation spans multiple days; dates are written at the beginning "
    "of each conversation segment.\n\n"
)


QA_PROMPT = """
Answer the question using the provided context and retrieved memories. Give a short phrase when possible. Use exact words from the context when they are sufficient.

Question: {} Short answer:
"""


LONGMEMEVAL_ANSWER_PROMPT = """
You will receive several historical chats between you and a user. Answer the current question using only relevant chat history and the current date.

History Chats:

{}

Current Date: {}
Question: {}
Short Answer:
"""


LLM_JUDGE_GENERAL_PROMPT = """
You are an objective judge for a QA task.

Evaluate whether the model answer correctly answers the question using the ground truth answers. Ignore wording differences when the meaning is equivalent.

[Question]
{question}

[Ground Truth Answers]
{ground_truth}

[Model Answer]
{model_answer}

Criteria:
1. Correctness: Is the answer factually consistent with any acceptable ground truth?
2. Relevance: Does it directly answer the question without unsupported additions?
3. Completeness: Does it include the essential information needed to answer?

Scoring:
- 1.0: fully correct and complete.
- 0.5: partially correct, incomplete, or slightly imprecise.
- 0.0: incorrect, irrelevant, unsupported, or contradictory.

Return only JSON:
{{
    "explanation": "<brief reason>",
    "score": <0.0 | 0.5 | 1.0>
}}
"""


HOTPOTQA_ANSWER_PROMPT = """Answer the question from the context. The question may require combining multiple pieces of evidence.

{context}

Question: {question}

Instructions:
- Identify the relevant evidence before answering.
- If the answer is present, provide a short precise answer.
- Output only the final answer inside <answer></answer> tags.

<answer>your answer here</answer>"""
