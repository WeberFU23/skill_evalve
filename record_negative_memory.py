#!/usr/bin/env python
"""Write one markdown negative memory entry.

This is a small utility for manually recording user corrections before the
full interactive collection loop is wired into an application.
"""
import argparse

from src.negative_memory import NegativeMemoryStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./negative_memories",
                        help="Directory where negative memories are stored")
    parser.add_argument("--problem", required=True,
                        help="The task or situation where the mistake happened")
    parser.add_argument("--wrong-behavior", required=True,
                        help="Reusable description of what was wrong")
    parser.add_argument("--correction", required=True,
                        help="The user/evaluator correction")
    parser.add_argument("--lesson", required=True,
                        help="Future-facing lesson to avoid repeating the mistake")
    parser.add_argument("--trigger", default="",
                        help="When this negative memory should be retrieved")
    parser.add_argument("--user-id", default=None,
                        help="Optional user scope. When set, visibility is private.")
    parser.add_argument("--tag", action="append", default=[],
                        help="Extra tag. Can be repeated.")
    parser.add_argument("--title", default=None,
                        help="Optional title")
    args = parser.parse_args()

    store = NegativeMemoryStore(root_dir=args.dir)
    path = store.write_entry(
        problem=args.problem,
        wrong_behavior=args.wrong_behavior,
        correction=args.correction,
        lesson=args.lesson,
        trigger=args.trigger,
        user_id=args.user_id,
        tags=args.tag,
        title=args.title,
    )
    print(path)


if __name__ == "__main__":
    main()
