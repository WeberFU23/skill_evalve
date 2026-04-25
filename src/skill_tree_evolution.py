"""
Path-grounded evolution for directory-backed skill trees.

This module records hard cases together with the skill-tree paths that were
actually used, then asks an LLM designer whether to refine an existing node or
add a child branch under the implicated path.
"""
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

from src.skill_tree import SkillTree, SkillNode


@dataclass
class SkillHardCase:
    """A failed problem plus the skill-tree context used to solve it."""
    problem_id: str
    query: str
    context: str = ""
    prediction: str = ""
    ground_truth: str = ""
    reward: float = 0.0
    is_success: bool = False
    failure_type: str = "unknown"
    selected_skill_paths: List[str] = field(default_factory=list)
    routing_trace: List[Dict[str, Any]] = field(default_factory=list)
    summarized_skill_prompt: str = ""
    retrieved_memories: List[str] = field(default_factory=list)
    memory_actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fail_count: int = 1

    def bucket_key(self) -> Tuple[str, ...]:
        """Group failures by the skill paths that shaped the prompt."""
        if self.selected_skill_paths:
            return tuple(self.selected_skill_paths)
        terminal = self.metadata.get("terminal_skill_path")
        if terminal:
            return (str(terminal),)
        return ("__unrouted__",)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "query": self.query,
            "context": self.context,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "reward": self.reward,
            "is_success": self.is_success,
            "failure_type": self.failure_type,
            "selected_skill_paths": self.selected_skill_paths,
            "routing_trace": self.routing_trace,
            "summarized_skill_prompt": self.summarized_skill_prompt,
            "retrieved_memories": self.retrieved_memories,
            "memory_actions": self.memory_actions,
            "metadata": self.metadata,
            "fail_count": self.fail_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillHardCase":
        return cls(
            problem_id=str(data.get("problem_id", "")),
            query=str(data.get("query", "")),
            context=str(data.get("context", "")),
            prediction=str(data.get("prediction", "")),
            ground_truth=str(data.get("ground_truth", "")),
            reward=float(data.get("reward", 0.0) or 0.0),
            is_success=bool(data.get("is_success", False)),
            failure_type=str(data.get("failure_type", "unknown")),
            selected_skill_paths=list(data.get("selected_skill_paths", []) or []),
            routing_trace=list(data.get("routing_trace", []) or []),
            summarized_skill_prompt=str(data.get("summarized_skill_prompt", "")),
            retrieved_memories=list(data.get("retrieved_memories", []) or []),
            memory_actions=list(data.get("memory_actions", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
            fail_count=int(data.get("fail_count", 1) or 1),
        )


class SkillHardCaseCollector:
    """Rolling pool for failures grounded in selected skill-tree paths."""

    def __init__(self, max_cases: int = 1000,
                 min_reward_for_success: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        self.max_cases = max(0, int(max_cases))
        self.min_reward_for_success = float(min_reward_for_success)
        self.logger = logger or logging.getLogger("AgenticMemory")
        self._cases: Dict[str, SkillHardCase] = {}
        self._lock = threading.RLock()

    def _case_key(self, case: SkillHardCase) -> str:
        if case.problem_id:
            return case.problem_id
        return case.query.strip().lower()

    def add_case(self, case: SkillHardCase):
        """Add a failure case. Successful cases are ignored."""
        if case.is_success or case.reward >= self.min_reward_for_success:
            return

        with self._lock:
            key = self._case_key(case)
            existing = self._cases.get(key)
            if existing is not None:
                existing.fail_count += 1
                existing.prediction = case.prediction
                existing.ground_truth = case.ground_truth
                existing.reward = case.reward
                existing.failure_type = case.failure_type
                existing.selected_skill_paths = case.selected_skill_paths
                existing.routing_trace = case.routing_trace
                existing.summarized_skill_prompt = case.summarized_skill_prompt
                existing.retrieved_memories = case.retrieved_memories
                existing.memory_actions = case.memory_actions
                existing.metadata = case.metadata
            else:
                self._cases[key] = case

            if self.max_cases > 0 and len(self._cases) > self.max_cases:
                sorted_keys = sorted(
                    self._cases.keys(),
                    key=lambda k: self._cases[k].fail_count,
                    reverse=True,
                )
                keep = set(sorted_keys[:self.max_cases])
                for old_key in list(self._cases.keys()):
                    if old_key not in keep:
                        del self._cases[old_key]

    def get_all_cases(self) -> List[SkillHardCase]:
        with self._lock:
            return list(self._cases.values())

    def grouped_by_path(self, min_cases: int = 2) -> Dict[Tuple[str, ...], List[SkillHardCase]]:
        buckets: Dict[Tuple[str, ...], List[SkillHardCase]] = {}
        with self._lock:
            for case in self._cases.values():
                buckets.setdefault(case.bucket_key(), []).append(case)
        return {
            key: cases
            for key, cases in buckets.items()
            if len(cases) >= int(min_cases)
        }

    def clear(self):
        with self._lock:
            self._cases = {}

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_cases": self.max_cases,
                "min_reward_for_success": self.min_reward_for_success,
                "cases": {key: case.to_dict() for key, case in self._cases.items()},
            }

    def load_dict(self, data: Dict[str, Any]):
        with self._lock:
            self.max_cases = int(data.get("max_cases", self.max_cases) or self.max_cases)
            self.min_reward_for_success = float(
                data.get("min_reward_for_success", self.min_reward_for_success)
            )
            self._cases = {
                str(key): SkillHardCase.from_dict(case_data)
                for key, case_data in dict(data.get("cases", {}) or {}).items()
                if isinstance(case_data, dict)
            }


class SkillTreeEvolutionDesigner:
    """
    LLM designer that applies path-grounded updates to skill markdown files.

    Supported actions:
    - no_change
    - refine_node: replace the body or full markdown of an existing node
    - add_child_node: create child_name/child_name.md under a parent node
    """

    def __init__(self, args, tree: SkillTree,
                 logger: Optional[logging.Logger] = None,
                 max_cases_per_prompt: int = 5):
        self.args = args
        self.tree = tree
        self.logger = logger or logging.getLogger("AgenticMemory")
        self.max_cases_per_prompt = max(1, int(max_cases_per_prompt))
        self.designer_model = getattr(args, "designer_model", None) or getattr(args, "model", "")

    def evolve_from_collector(self, collector: SkillHardCaseCollector,
                              min_cases: int = 2) -> List[Dict[str, Any]]:
        """Evolve the tree for each path bucket with enough hard cases."""
        results = []
        for path_key, cases in collector.grouped_by_path(min_cases=min_cases).items():
            result = self.evolve_cases(cases, implicated_paths=list(path_key))
            results.append(result)
        return results

    def evolve_cases(self, cases: List[SkillHardCase],
                     implicated_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        if not cases:
            return {"action": "no_change", "reasoning": "No hard cases provided"}

        implicated_paths = implicated_paths or sorted({
            path for case in cases for path in case.selected_skill_paths
        })
        prompt = self.build_prompt(cases[:self.max_cases_per_prompt], implicated_paths)
        response = self._call_llm(prompt)
        result = self.parse_response(response)
        result["raw_response"] = response
        applied = self.apply_result(result, allowed_paths=set(implicated_paths))
        result["applied"] = applied
        if applied:
            self.tree.load()
        return result

    def build_prompt(self, cases: List[SkillHardCase],
                     implicated_paths: List[str]) -> str:
        node_blocks = []
        for path in implicated_paths:
            try:
                node = self.tree.get_node(path)
            except KeyError:
                continue
            node_blocks.append(self._format_node(node))

        case_blocks = []
        for i, case in enumerate(cases, start=1):
            case_blocks.append(
                f"### Hard Case {i}\n"
                f"Problem ID: {case.problem_id}\n"
                f"Failure type: {case.failure_type}\n"
                f"Reward: {case.reward}\n"
                f"Query:\n{case.query}\n\n"
                f"Context:\n{_truncate(case.context, 1800)}\n\n"
                f"Prediction:\n{_truncate(case.prediction, 800)}\n\n"
                f"Ground truth:\n{_truncate(case.ground_truth, 800)}\n\n"
                f"Selected skill paths: {case.selected_skill_paths}\n"
                f"Routing trace:\n{json.dumps(case.routing_trace, ensure_ascii=False, indent=2)}\n\n"
                f"Compressed skill prompt used:\n{_truncate(case.summarized_skill_prompt, 1800)}\n\n"
                f"Retrieved memories:\n{json.dumps(case.retrieved_memories[:10], ensure_ascii=False, indent=2)}\n"
            )

        return (
            "You are evolving a directory-backed hierarchical skill tree.\n"
            "The hard cases below failed after using the listed skill paths. Decide whether the failure should update the skill tree.\n\n"
            "Rules:\n"
            "- Skills describe how to remember, route, update, delete, or use information. Do not store raw user facts as skill text.\n"
            "- Prefer no_change if failures are caused by retrieval, model reasoning, noisy labels, or insufficient evidence.\n"
            "- Use refine_node when an existing implicated node has the right intent but unclear triggers, steps, or constraints.\n"
            "- Use add_child_node when the hard cases reveal a stable reusable sub-pattern under an implicated node.\n"
            "- Only target one of the implicated paths unless there is an explicit reason.\n"
            "- Return only JSON.\n\n"
            "Allowed JSON formats:\n"
            "{\n"
            '  "action": "no_change",\n'
            '  "reasoning": "why no tree edit is needed"\n'
            "}\n\n"
            "{\n"
            '  "action": "refine_node",\n'
            '  "target_path": "path/of/existing/node",\n'
            '  "reasoning": "why this node should be refined",\n'
            '  "body": "# Existing Title\\n\\n## Description\\n..."\n'
            "}\n\n"
            "{\n"
            '  "action": "add_child_node",\n'
            '  "parent_path": "path/of/parent/node",\n'
            '  "child_name": "snake_case_name",\n'
            '  "reasoning": "why this child is needed",\n'
            '  "visibility": "shared",\n'
            '  "scope_id": null,\n'
            '  "tags": ["tag-a", "tag-b"],\n'
            '  "body": "# Child Title\\n\\n## Description\\n..."\n'
            "}\n\n"
            "Implicated skill nodes:\n"
            f"{chr(10).join(node_blocks) if node_blocks else '(none found)'}\n\n"
            "Hard cases:\n"
            f"{chr(10).join(case_blocks)}"
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start < 0 or end <= 0:
            return {
                "action": "no_change",
                "reasoning": "No JSON object found in LLM response",
            }
        try:
            json_str = cleaned[start:end]
            if repair_json is not None:
                json_str = repair_json(json_str)
            data = json.loads(json_str)
        except Exception as exc:
            self.logger.warning(f"[SkillTreeEvolution] Failed to parse JSON: {exc}")
            return {
                "action": "no_change",
                "reasoning": f"JSON parse error: {exc}",
            }
        if not isinstance(data, dict):
            return {"action": "no_change", "reasoning": "Response JSON is not an object"}
        data["action"] = str(data.get("action", "no_change")).lower().strip()
        return data

    def apply_result(self, result: Dict[str, Any],
                     allowed_paths: Optional[Iterable[str]] = None) -> bool:
        action = str(result.get("action", "no_change")).lower().strip()
        allowed = set(allowed_paths or [])
        if action == "no_change":
            return False
        if action == "refine_node":
            return self._apply_refine_node(result, allowed)
        if action == "add_child_node":
            return self._apply_add_child_node(result, allowed)
        self.logger.warning(f"[SkillTreeEvolution] Unknown action: {action}")
        return False

    def _apply_refine_node(self, result: Dict[str, Any], allowed_paths: set) -> bool:
        target_path = str(result.get("target_path", "")).strip()
        if not target_path:
            self.logger.warning("[SkillTreeEvolution] refine_node missing target_path")
            return False
        if allowed_paths and target_path not in allowed_paths:
            self.logger.warning(f"[SkillTreeEvolution] Refusing non-implicated target: {target_path}")
            return False
        try:
            node = self.tree.get_node(target_path)
        except KeyError:
            self.logger.warning(f"[SkillTreeEvolution] Unknown target node: {target_path}")
            return False

        body = str(result.get("body", "")).strip()
        markdown = str(result.get("markdown", "")).strip()
        if markdown:
            new_text = _ensure_frontmatter(markdown, node.metadata)
        elif body:
            new_text = _format_markdown(node.metadata, body)
        else:
            self.logger.warning("[SkillTreeEvolution] refine_node missing body/markdown")
            return False

        _atomic_write(node.file_path, new_text)
        self.logger.info(f"[SkillTreeEvolution] Refined skill node: {target_path}")
        return True

    def _apply_add_child_node(self, result: Dict[str, Any], allowed_paths: set) -> bool:
        parent_path = str(result.get("parent_path", "")).strip()
        child_name = _safe_node_name(str(result.get("child_name", "")).strip())
        if not parent_path or not child_name:
            self.logger.warning("[SkillTreeEvolution] add_child_node missing parent_path/child_name")
            return False
        if allowed_paths and parent_path not in allowed_paths:
            self.logger.warning(f"[SkillTreeEvolution] Refusing non-implicated parent: {parent_path}")
            return False
        try:
            parent = self.tree.get_node(parent_path)
        except KeyError:
            self.logger.warning(f"[SkillTreeEvolution] Unknown parent node: {parent_path}")
            return False

        body = str(result.get("body", "")).strip()
        if not body:
            self.logger.warning("[SkillTreeEvolution] add_child_node missing body")
            return False

        parent_dir = os.path.dirname(parent.file_path)
        child_dir = os.path.join(parent_dir, child_name)
        child_file = os.path.join(child_dir, f"{child_name}.md")
        if os.path.exists(child_file):
            self.logger.warning(f"[SkillTreeEvolution] Child node already exists: {child_file}")
            return False

        tags = result.get("tags", parent.tags)
        if not isinstance(tags, list):
            tags = [str(tags)]
        parent_id = str(parent.metadata.get("id") or parent.path.replace("/", "."))
        metadata = {
            "id": f"{parent_id}.{child_name}",
            "name": child_name,
            "visibility": str(result.get("visibility") or parent.visibility or "shared"),
            "scope_id": result.get("scope_id", parent.scope_id),
            "tags": [str(tag) for tag in tags],
        }

        os.makedirs(child_dir, exist_ok=True)
        _atomic_write(child_file, _format_markdown(metadata, body))
        self.logger.info(f"[SkillTreeEvolution] Added child skill node: {child_file}")
        return True

    def _call_llm(self, prompt: str) -> str:
        from llm_utils import get_llm_response_via_api

        response, _, _ = get_llm_response_via_api(
            prompt=prompt,
            LLM_MODEL=self.designer_model,
            base_url=self.args.api_base,
            api_key=self.args.api_key,
            MAX_TOKENS=int(getattr(self.args, "max_new_tokens", 2048) or 2048),
            TAU=0.0,
        )
        return response

    def _format_node(self, node: SkillNode) -> str:
        return (
            f"### Node: {node.path}\n"
            f"Metadata:\n{json.dumps(node.metadata, ensure_ascii=False, indent=2)}\n\n"
            f"Content:\n{_truncate(node.instruction_text(), 3000)}\n"
        )


def hard_case_from_selection(problem_id: str, query: str, selection,
                             context: str = "", prediction: str = "",
                             ground_truth: str = "", reward: float = 0.0,
                             is_success: bool = False,
                             failure_type: str = "unknown",
                             summarized_skill_prompt: str = "",
                             retrieved_memories: Optional[List[str]] = None,
                             memory_actions: Optional[List[Dict[str, Any]]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> SkillHardCase:
    """Convenience helper for recording a selector result as a hard case."""
    routing_trace = []
    for step in getattr(selection, "routing_steps", []) or []:
        routing_trace.append({
            "current_path": step.current_path,
            "action": step.action,
            "candidate_paths": step.candidate_paths,
            "selected_path": step.selected_path,
            "log_prob": step.log_prob,
            "value": step.value,
        })
    meta = dict(metadata or {})
    terminal = getattr(selection, "terminal_node", None)
    if terminal is not None:
        meta.setdefault("terminal_skill_path", terminal.path)
    return SkillHardCase(
        problem_id=problem_id,
        query=query,
        context=context,
        prediction=prediction,
        ground_truth=ground_truth,
        reward=reward,
        is_success=is_success,
        failure_type=failure_type,
        selected_skill_paths=list(getattr(selection, "selected_paths", []) or []),
        routing_trace=routing_trace,
        summarized_skill_prompt=summarized_skill_prompt,
        retrieved_memories=list(retrieved_memories or []),
        memory_actions=list(memory_actions or []),
        metadata=meta,
    )


def _format_markdown(metadata: Dict[str, Any], body: str) -> str:
    return f"{_format_frontmatter(metadata)}\n\n{body.strip()}\n"


def _format_frontmatter(metadata: Dict[str, Any]) -> str:
    lines = ["---"]
    for key in ("id", "name", "visibility", "scope_id", "tags"):
        value = metadata.get(key)
        if key == "tags":
            tags = value if isinstance(value, list) else []
            tag_text = ", ".join(str(tag) for tag in tags)
            lines.append(f"tags: [{tag_text}]")
        elif value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _ensure_frontmatter(markdown: str, fallback_metadata: Dict[str, Any]) -> str:
    if markdown.lstrip().startswith("---"):
        return markdown.rstrip() + "\n"
    return _format_markdown(fallback_metadata, markdown)


def _safe_node_name(name: str) -> str:
    name = name.strip().lower().replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _atomic_write(path: str, text: str):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
