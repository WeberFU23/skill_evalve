#!/usr/bin/env python
"""Write one markdown negative memory entry.

This utility records either structured mistake/correction fields or a raw
user-correction dialogue. It stores reusable error patterns and corrections,
not hidden chain-of-thought.
"""
import argparse
import json
import re
import sys

from src.negative_memory import NegativeMemoryStore


CORRECTION_MARKERS = [
    "not correct",
    "not right",
    "wrong",
    "incorrect",
    "should be",
    "instead",
    "do not",
    "don't",
    "avoid",
    "need to",
    "must",
    "correction",
    "fix",
    "\u4e0d\u5bf9",
    "\u9519",
    "\u5e94\u8be5",
    "\u6539\u6210",
    "\u4e0d\u8981",
    "\u9700\u8981\u6539",
]


def _compact(text: str, max_chars: int = 1600) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def _read_dialogue(args) -> str:
    pieces = []
    if args.dialogue:
        pieces.append(args.dialogue)
    if args.dialogue_file:
        if args.dialogue_file == "-":
            pieces.append(sys.stdin.read())
        else:
            with open(args.dialogue_file, "r", encoding="utf-8") as f:
                pieces.append(f.read())
    return "\n".join(piece.strip() for piece in pieces if piece and piece.strip()).strip()


def _extract_correction(dialogue: str) -> str:
    lines = []
    for raw_line in str(dialogue or "").splitlines():
        split_line = re.sub(
            r"\s+((?:User|Human|Evaluator|Assistant|Model)\s*:)",
            r"\n\1",
            raw_line,
            flags=re.IGNORECASE,
        )
        lines.extend(line.strip() for line in split_line.splitlines() if line.strip())
    marked = []
    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in CORRECTION_MARKERS):
            marked.append(line)
    if marked:
        return _compact(marked[-1], 900)
    if lines:
        return _compact(" ".join(lines[-3:]), 900)
    return ""


def _build_fields(args):
    dialogue = _read_dialogue(args)
    has_structured = all([
        args.problem,
        args.wrong_behavior,
        args.correction,
        args.lesson,
    ])
    if not dialogue and not has_structured:
        raise ValueError(
            "Provide either --dialogue/--dialogue-file or all structured fields: "
            "--problem, --wrong-behavior, --correction, and --lesson."
        )

    extracted_correction = _extract_correction(dialogue)
    problem = args.problem or (
        "User correction dialogue:\n"
        f"{_compact(dialogue, 1400)}"
    )
    wrong_behavior = args.wrong_behavior or (
        "The assistant/model behavior in the dialogue was explicitly corrected "
        "by the user or evaluator."
    )
    correction = args.correction or extracted_correction or (
        "Follow the user/evaluator correction from this dialogue."
    )
    lesson = args.lesson or (
        "When a future case matches this correction, reject the previously "
        f"wrong pattern and apply the corrected behavior: {_compact(correction, 700)}"
    )
    trigger = args.trigger or (
        "Similar user or evaluator corrections"
        if not extracted_correction else
        f"Similar correction: {_compact(extracted_correction, 500)}"
    )
    tags = list(args.tag or [])
    if dialogue and "correction_dialogue" not in tags:
        tags.append("correction_dialogue")
    return {
        "problem": problem,
        "wrong_behavior": wrong_behavior,
        "correction": correction,
        "lesson": lesson,
        "trigger": trigger,
        "user_id": args.user_id,
        "tags": tags,
        "title": args.title,
        "date": args.date,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./negative_memories",
                        help="Directory where negative memories are stored")
    parser.add_argument("--problem",
                        help="The task or situation where the mistake happened")
    parser.add_argument("--wrong-behavior",
                        help="Reusable description of what was wrong")
    parser.add_argument("--correction",
                        help="The user/evaluator correction")
    parser.add_argument("--lesson",
                        help="Future-facing lesson to avoid repeating the mistake")
    parser.add_argument("--dialogue", default=None,
                        help="Raw dialogue containing a user/evaluator correction")
    parser.add_argument("--dialogue-file", default=None,
                        help="Path to a correction dialogue file, or '-' for stdin")
    parser.add_argument("--trigger", default="",
                        help="When this negative memory should be retrieved")
    parser.add_argument("--user-id", default=None,
                        help="Optional user scope. When set, visibility is private.")
    parser.add_argument("--tag", action="append", default=[],
                        help="Extra tag. Can be repeated.")
    parser.add_argument("--title", default=None,
                        help="Optional title")
    parser.add_argument("--date", default=None,
                        help="Optional YYYY-MM-DD date for the markdown entry")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the generated fields without writing a file")
    args = parser.parse_args()

    try:
        fields = _build_fields(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(json.dumps(fields, ensure_ascii=False, indent=2))
        return

    store = NegativeMemoryStore(root_dir=args.dir)
    path = store.write_entry(
        **fields,
    )
    print(path)


if __name__ == "__main__":
    main()
