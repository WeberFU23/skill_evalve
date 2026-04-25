"""
Operation Bank: Stores and evolves memory operations
"""
import json
import re
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import copy
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_repair import repair_json
from llm_utils import get_llm_response_via_api
from prompts.operation_templates import get_initial_operations


@dataclass
class HardProblem:
    user_key: str
    relative_nodes: List[str]
    problem_description: str
    problem_solution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_key": self.user_key,
            "relative_nodes": list(self.relative_nodes),
            "problem_description": self.problem_description,
            "problem_solution": self.problem_solution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardProblem":
        return cls(
            user_key=str(data.get("user_key", "") or ""),
            relative_nodes=[str(item) for item in data.get("relative_nodes", []) if str(item).strip()],
            problem_description=str(data.get("problem_description", "") or ""),
            problem_solution=str(data.get("problem_solution", "") or ""),
        )


class Operation:
    """Single memory operation"""
    def __init__(self, name: str, description: str,
                 instruction_template: str, update_type: str,
                 meta_info: Optional[Dict] = None,
                 visibility: str = "shared",
                 user_key: Optional[str] = None,
                 children: Optional[List[str]] = None,
                 content: Optional[str] = None):
        self.name = name
        self.description = description
        self.content = content if content is not None else instruction_template
        # Keep instruction_template for backward compatibility with the rest of the pipeline.
        self.instruction_template = self.content
        self.update_type = update_type  # insert, update, delete, noop
        self.visibility = str(visibility or "shared").strip().lower()
        self.user_key = user_key
        self.children = list(children) if children is not None else []
        self.meta_info = meta_info or {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "unknown",
            "last_modified": "unknown"
        }
        # Ensure recent_usage_ema exists for backward compatibility
        if "recent_usage_ema" not in self.meta_info:
            self.meta_info["recent_usage_ema"] = 0.0
        self.embedding = None  # Will be set by operation bank

    @property
    def node_path(self) -> str:
        raw = self.meta_info.get("node_path") or self.meta_info.get("path") or self.name
        raw = str(raw or self.name).strip()
        raw = raw.replace("\\", "/")
        if raw.startswith("./"):
            raw = raw[2:]
        return raw or self.name

    def normalized_children(self) -> List[str]:
        refs = []
        for child in self.children:
            ref = str(child or "").strip().replace("\\", "/")
            if not ref:
                continue
            if ref.startswith("./"):
                ref = ref[2:]
            refs.append(ref)
        return refs

    def get_description_text(self) -> str:
        """Get text description for embedding (no guide characters)"""
        return self.description

    def format_instruction(self, session_text: str, retrieved_memories: str) -> str:
        """Format instruction template with current context"""
        template = self.instruction_template
        if '{session_text}' in template or '{retrieved_memories}' in template:
            try:
                return template.format(
                    session_text=session_text,
                    retrieved_memories=retrieved_memories
                )
            except Exception:
                return template
        return template

    def update_stats(self, reward: float):
        """Update operation statistics when this operation is selected.

        Note: EMA updates are handled separately by OperationBank.update_all_ema()
        which is called at each step during training.

        Args:
            reward: The reward received for this operation
        """
        self.meta_info["usage_count"] += 1

        # Update average reward
        n = self.meta_info["usage_count"]
        old_avg = self.meta_info["avg_reward"]
        new_avg = old_avg + (reward - old_avg) / n
        self.meta_info["avg_reward"] = new_avg

        # Keep recent rewards (last 20)
        self.meta_info["recent_rewards"].append(reward)
        if len(self.meta_info["recent_rewards"]) > 20:
            self.meta_info["recent_rewards"] = self.meta_info["recent_rewards"][-20:]

    def decay_ema(self, ema_alpha: float = 0.1):
        """Decay EMA when this operation is NOT selected

        Should be called for all non-selected operations each step.
        EMA = (1 - alpha) * old_ema (decays toward 0)
        """
        old_ema = self.meta_info.get("recent_usage_ema", 0.0)
        self.meta_info["recent_usage_ema"] = (1.0 - ema_alpha) * old_ema

    def to_dict(self):
        return {
            'name': self.name,
            'visibility': self.visibility,
            'user_key': self.user_key,
            'children': self.children,
            'description': self.description,
            'content': self.content,
            'instruction_template': self.instruction_template,
            'update_type': self.update_type,
            'meta_info': self.meta_info,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data):
        content = data.get('content')
        if content is None:
            content = data.get('instruction_template', '')
        op = cls(
            name=data['name'],
            description=data['description'],
            instruction_template=content,
            update_type=data.get('update_type', 'insert'),
            meta_info=data.get('meta_info', {}),
            visibility=data.get('visibility', 'shared'),
            user_key=data.get('user_key'),
            children=data.get('children', []),
            content=content
        )
        if data.get('embedding') is not None:
            op.embedding = np.array(data['embedding'])
        return op


class OperationBank:
    """
    Operation Bank stores memory operations and supports dynamic evolution
    """
    def __init__(self, encoder=None, max_ops: int = 20,
                 skip_noop: bool = False):
        self.operations: Dict[str, Operation] = {}
        self.encoder = encoder  # OpEncoder object (supports state encoder backbones)
        self.max_ops = max_ops  # Maximum number of operations allowed in the bank
        self.new_operation_names = set()
        self.skip_noop = skip_noop
        self._initialize_with_seeds()

    def _initialize_with_seeds(self):
        """Initialize with seed operations"""
        initial_ops = get_initial_operations(include_noop=not self.skip_noop)
        for op_name, op_data in initial_ops.items():
            operation = Operation(
                name=op_data['name'],
                description=op_data['description'],
                instruction_template=op_data.get('content', op_data.get('instruction_template', '')),
                update_type=op_data['update_type'],
                meta_info=op_data['meta_info'],
                visibility=op_data.get('visibility', 'shared'),
                user_key=op_data.get('user_key'),
                children=op_data.get('children', []),
                content=op_data.get('content', op_data.get('instruction_template', ''))
            )
            self.operations[self._operation_key(operation)] = operation

        # Compute embeddings for seed operations if encoder is available
        if self.encoder is not None:
            self._recompute_embeddings()

    def set_encoder(self, encoder):
        """Set encoder for operation embeddings"""
        self.encoder = encoder
        self._recompute_embeddings()

    def _recompute_embeddings(self):
        """Recompute embeddings for all operations"""
        if self.encoder is None:
            return

        texts = [op.get_description_text() for op in self.operations.values()]
        if len(texts) == 0:
            return

        # Use the encoder to get embeddings
        embeddings = self.encoder.encode(texts)

        # Assign embeddings to operations
        for i, op_key in enumerate(self.operations.keys()):
            self.operations[op_key].embedding = embeddings[i]

    def _normalize_ref(self, ref: str) -> str:
        ref = str(ref or "").strip().replace("\\", "/")
        if ref.startswith("./"):
            ref = ref[2:]
        return ref

    def _operation_key(self, operation: Operation) -> str:
        return self._normalize_ref(operation.node_path)

    def _lookup_operation_key(self, key_or_name: str) -> Optional[str]:
        key = self._normalize_ref(key_or_name)
        if key in self.operations:
            return key
        matches = [op_key for op_key, op in self.operations.items() if op.name == key_or_name]
        if len(matches) == 1:
            return matches[0]
        return None

    def _is_visible_for_user(self, op: Operation, user_key: Optional[str]) -> bool:
        visibility = str(getattr(op, "visibility", "shared") or "shared").strip().lower()
        if visibility == "shared":
            return True
        if visibility != "private":
            return True
        op_user = getattr(op, "user_key", None)
        if op_user in (None, "", "*"):
            return True
        return str(op_user) == str(user_key)

    def _build_visible_index(self, user_key: Optional[str]) -> Dict[str, Any]:
        visible_ops = [op for op in self.get_all_operations() if self._is_visible_for_user(op, user_key)]
        path_map: Dict[str, Operation] = {}
        for op in visible_ops:
            canonical = self._operation_key(op)
            if canonical:
                path_map[canonical] = op
        return {
            "visible_ops": visible_ops,
            "path_map": path_map,
        }

    def _resolve_operation_ref(self, ref: str, index: Dict[str, Any]) -> Optional[Operation]:
        if not ref:
            return None
        norm = self._normalize_ref(ref)
        return index["path_map"].get(norm)

    def _get_root_operations(self, index: Dict[str, Any]) -> List[Operation]:
        referenced = set()
        for op in index["visible_ops"]:
            for child in op.normalized_children():
                child_op = self._resolve_operation_ref(child, index)
                if child_op is not None:
                    referenced.add(child_op.node_path)
        roots = [op for op in index["visible_ops"] if op.node_path not in referenced]
        if not roots:
            roots = list(index["visible_ops"])
        roots.sort(key=lambda op: op.node_path)
        return roots

    def _score_operations(self, query_embedding: np.ndarray,
                          operations: List[Operation],
                          top_k: Optional[int] = None) -> List[Tuple[float, Operation]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query) + 1e-8
        scored: List[Tuple[float, Operation]] = []
        for op in operations:
            if op.embedding is None:
                continue
            emb = np.asarray(op.embedding, dtype=np.float32).reshape(-1)
            score = float(np.dot(query, emb) / ((np.linalg.norm(emb) + 1e-8) * query_norm))
            scored.append((score, op))
        scored.sort(key=lambda item: (-item[0], item[1].node_path, item[1].name))
        if top_k is not None:
            scored = scored[:max(1, int(top_k))]
        return scored

    def _get_child_operations(self, op: Optional[Operation], index: Dict[str, Any]) -> List[Operation]:
        if op is None:
            return self._get_root_operations(index)
        children = []
        seen = set()
        for child_ref in op.normalized_children():
            child = self._resolve_operation_ref(child_ref, index)
            if child is None:
                continue
            if child.node_path in seen:
                continue
            seen.add(child.node_path)
            children.append(child)
        return children

    def _build_routing_prompt(self, session_text: str,
                              current_op: Optional[Operation],
                              path_ops: List[Operation],
                              top_children: List[Operation]) -> str:
        path_section = "(none)"
        if path_ops:
            path_lines = []
            for idx, op in enumerate(path_ops, 1):
                path_lines.append(
                    f"{idx}. path={op.node_path} name={op.name}\n"
                    f"   description={op.description}"
                )
            path_section = "\n".join(path_lines)

        if current_op is None:
            current_block = "path=ROOT\nname=ROOT\ndescription=Virtual root over currently visible skill roots."
        else:
            current_block = (
                f"path={current_op.node_path}\n"
                f"name={current_op.name}\n"
                f"description={current_op.description}\n"
                f"content_preview={current_op.content[:600]}"
            )

        child_section = "(no children)"
        if top_children:
            child_lines = []
            for idx, child in enumerate(top_children, 1):
                child_lines.append(
                    f"{idx}. path={child.node_path} name={child.name}\n"
                    f"   description={child.description}"
                )
            child_section = "\n".join(child_lines)

        return (
            "You are a path-aware skill tree router.\n"
            "Given the query, the current node, the path from root to current node, and a small set of child candidates,\n"
            "decide whether the current node is already sufficient or whether search should continue into exactly one child.\n\n"
            f"Query:\n{session_text}\n\n"
            f"Current Node:\n{current_block}\n\n"
            f"Path From Root To Current:\n{path_section}\n\n"
            f"Top Child Candidates:\n{child_section}\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "decision": "stop" | "continue",\n'
            '  "selected_child_path": "<path or empty>",\n'
            '  "useful_paths": ["<node path>", "..."],\n'
            '  "reasoning": "<brief rationale>"\n'
            "}\n\n"
            "Rules:\n"
            "- Use decision=continue only if one child is clearly needed.\n"
            "- If decision=continue, selected_child_path must match one listed child path.\n"
            "- If decision=stop, useful_paths should contain only nodes from the current path, current node, and listed child candidates.\n"
            "- Include the current node in useful_paths if its content is still useful.\n"
            "- Prefer shorter useful_paths lists when possible.\n"
        )

    def _parse_router_response(self, response: str) -> Dict[str, Any]:
        text = str(response or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if "\n" in text:
                    text = text.split("\n", 1)[1]
        try:
            return json.loads(repair_json(text))
        except Exception:
            return {
                "decision": "stop",
                "selected_child_path": "",
                "useful_paths": [],
                "reasoning": "Failed to parse router response."
            }

    def _call_router_llm(self, prompt: str, args) -> Dict[str, Any]:
        model_name = getattr(args, "tree_router_model", None) or getattr(args, "designer_model", None) or getattr(args, "model", "")
        response, _, _ = get_llm_response_via_api(
            prompt=prompt,
            LLM_MODEL=model_name,
            base_url=args.api_base,
            api_key=args.api_key,
            MAX_TOKENS=min(1024, getattr(args, "max_new_tokens", 1024)),
            TAU=0.0,
            MAX_TRIALS=10,
            TIME_GAP=3,
        )
        return self._parse_router_response(response)

    def _build_summary_prompt(self, session_text: str, selected_ops: List[Operation]) -> str:
        node_blocks = []
        for idx, op in enumerate(selected_ops, 1):
            node_blocks.append(
                f"[Node {idx}] path={op.node_path} name={op.name}\n"
                f"Description: {op.description}\n"
                f"Content:\n{op.content}"
            )
        nodes_text = "\n\n".join(node_blocks) if node_blocks else "(none)"
        return (
            "You are summarizing routed skill nodes for a downstream reasoning/execution model.\n"
            "Compress only the parts of the skills that are useful for the current query.\n"
            "Keep concrete guidance, constraints, and discriminative triggers. Drop unrelated detail.\n\n"
            f"Query:\n{session_text}\n\n"
            f"Selected Skill Nodes:\n{nodes_text}\n\n"
            "Return a concise structured summary with these sections if relevant:\n"
            "Relevant skills:\n"
            "When useful:\n"
            "How to apply:\n"
            "Constraints:\n"
        )

    def _call_summary_llm(self, prompt: str, args) -> str:
        model_name = getattr(args, "tree_summary_model", None) or getattr(args, "tree_router_model", None) or getattr(args, "designer_model", None) or getattr(args, "model", "")
        response, _, _ = get_llm_response_via_api(
            prompt=prompt,
            LLM_MODEL=model_name,
            base_url=args.api_base,
            api_key=args.api_key,
            MAX_TOKENS=min(1024, getattr(args, "max_new_tokens", 1024)),
            TAU=0.0,
            MAX_TRIALS=10,
            TIME_GAP=3,
        )
        return str(response or "").strip()

    def _normalize_path_list(self, refs: List[str]) -> List[str]:
        normalized = []
        seen = set()
        for ref in refs:
            norm = self._normalize_ref(ref)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            normalized.append(norm)
        return normalized

    def create_hard_problem(self, user_key: Optional[str],
                            relative_nodes: Optional[List[str]],
                            problem_description: str,
                            problem_solution: str) -> HardProblem:
        resolved_nodes = []
        seen = set()
        for node in relative_nodes or []:
            norm = self._normalize_ref(node)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            resolved_nodes.append(norm)
        return HardProblem(
            user_key=str(user_key or ""),
            relative_nodes=resolved_nodes,
            problem_description=str(problem_description or "").strip(),
            problem_solution=str(problem_solution or "").strip(),
        )

    def hard_problem_from_routing(self, user_key: Optional[str], routing_meta: Optional[Dict[str, Any]],
                                  problem_description: str, problem_solution: str) -> HardProblem:
        relative_nodes = []
        if isinstance(routing_meta, dict):
            relative_nodes = routing_meta.get("selected_paths", []) or []
        return self.create_hard_problem(
            user_key=user_key,
            relative_nodes=relative_nodes,
            problem_description=problem_description,
            problem_solution=problem_solution
        )

    def _build_hard_problem_update_prompt(self, hard_problem: HardProblem,
                                          indexed_nodes: List[Operation]) -> str:
        node_blocks = []
        for idx, op in enumerate(indexed_nodes, 1):
            node_blocks.append(
                f"[Node {idx}] path={op.node_path} name={op.name}\n"
                f"visibility={op.visibility} user_key={op.user_key}\n"
                f"update_type={op.update_type}\n"
                f"children={json.dumps(op.normalized_children(), ensure_ascii=True)}\n"
                f"description={op.description}\n"
                f"content=\n{op.content}"
            )
        nodes_text = "\n\n".join(node_blocks) if node_blocks else "(No relative nodes provided)"
        return (
            "You are updating a hierarchical skill tree from a solved hard problem.\n"
            "You may refine provided relative nodes, or add a new child branch under one of them.\n"
            "Only change the local neighborhood around the provided relative nodes.\n\n"
            f"user_key: {hard_problem.user_key}\n"
            f"relative_nodes: {json.dumps(hard_problem.relative_nodes, ensure_ascii=True)}\n"
            f"problem_description:\n{hard_problem.problem_description}\n\n"
            f"problem_solution:\n{hard_problem.problem_solution}\n\n"
            f"Relative Skill Nodes:\n{nodes_text}\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "action": "apply_changes" | "no_change",\n'
            '  "reasoning": "<brief rationale>",\n'
            '  "changes": [\n'
            "    {\n"
            '      "action": "refine_node",\n'
            '      "target_path": "<existing node path>",\n'
            '      "changes": {\n'
            '        "description": "<optional updated description>",\n'
            '        "content": "<optional updated content>",\n'
            '        "children": ["<optional replacement child paths>"]\n'
            "      },\n"
            '      "reasoning": "<why>"\n'
            "    },\n"
            "    {\n"
            '      "action": "add_child",\n'
            '      "parent_path": "<existing parent node path>",\n'
            '      "new_node": {\n'
            '        "name": "<node name>",\n'
            '        "visibility": "shared" | "private",\n'
            '        "user_key": "<null or user key>",\n'
            '        "description": "<description used for embedding>",\n'
            '        "content": "<skill body>",\n'
            '        "children": [],\n'
            '        "update_type": "<insert|update|delete|noop>",\n'
            '        "node_path": "<new child path under parent>"\n'
            "      },\n"
            '      "reasoning": "<why>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- If no local tree change is needed, return no_change.\n"
            "- Add children only under one of the provided relative nodes.\n"
            "- Keep updates local to the provided nodes.\n"
            "- For private nodes, set user_key to the hard problem user_key.\n"
            "- Do not invent unrelated branches.\n"
        )

    def _parse_hard_problem_update_response(self, response: str) -> Dict[str, Any]:
        text = str(response or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if "\n" in text:
                    text = text.split("\n", 1)[1]
        try:
            parsed = json.loads(repair_json(text))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {
            "action": "no_change",
            "reasoning": "Failed to parse hard problem update response.",
            "changes": []
        }

    def _call_hard_problem_updater_llm(self, prompt: str, args) -> Dict[str, Any]:
        model_name = (
            getattr(args, "hard_problem_updater_model", None)
            or getattr(args, "tree_router_model", None)
            or getattr(args, "designer_model", None)
            or getattr(args, "model", "")
        )
        response, _, _ = get_llm_response_via_api(
            prompt=prompt,
            LLM_MODEL=model_name,
            base_url=args.api_base,
            api_key=args.api_key,
            MAX_TOKENS=min(1536, getattr(args, "max_new_tokens", 1024)),
            TAU=0.0,
            MAX_TRIALS=10,
            TIME_GAP=3,
        )
        return self._parse_hard_problem_update_response(response)

    def _next_child_path(self, parent: Operation, node_name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_/-]+", "_", str(node_name or "").strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_/")
        if not slug:
            slug = "new_skill"
        parent_path = self._normalize_ref(parent.node_path)
        if parent_path.endswith(".md"):
            parent_path = parent_path[:-3]
        if parent_path.endswith("/skill"):
            parent_path = parent_path[:-len("/skill")]
        base = f"{parent_path}/{slug}/skill.md"
        candidate = base
        suffix = 1
        existing = {self._normalize_ref(op.node_path) for op in self.get_all_operations()}
        while candidate in existing:
            suffix += 1
            candidate = f"{parent_path}/{slug}_{suffix}/skill.md"
        return candidate

    def apply_hard_problem_update(self, hard_problem_input: Any, args) -> Dict[str, Any]:
        hard_problem = (
            hard_problem_input
            if isinstance(hard_problem_input, HardProblem)
            else HardProblem.from_dict(hard_problem_input)
        )
        index = self._build_visible_index(hard_problem.user_key)
        relative_ops = []
        seen = set()
        for ref in hard_problem.relative_nodes:
            op = self._resolve_operation_ref(ref, index)
            if op is None or op.node_path in seen:
                continue
            seen.add(op.node_path)
            relative_ops.append(op)

        prompt = self._build_hard_problem_update_prompt(hard_problem, relative_ops)
        updater_result = self._call_hard_problem_updater_llm(prompt, args)

        changes = updater_result.get("changes", [])
        if updater_result.get("action") == "no_change" or not isinstance(changes, list):
            updater_result["applied_changes"] = []
            return updater_result

        applied_changes = []
        new_names = []
        for change in changes:
            if not isinstance(change, dict):
                continue
            action = str(change.get("action", "")).strip().lower()
            if action == "refine_node":
                target = self._resolve_operation_ref(change.get("target_path", ""), index)
                if target is None:
                    continue
                payload = change.get("changes", {})
                if not isinstance(payload, dict):
                    continue
                allowed = {}
                for key in ("description", "content", "children", "visibility", "user_key"):
                    if key in payload:
                        allowed[key] = payload[key]
                if not allowed:
                    continue
                self.update_operation(target.node_path, **allowed)
                applied_changes.append({
                    "action": "refine_node",
                    "target_path": target.node_path,
                    "changes": allowed
                })
                new_names.append(target.node_path)

            elif action == "add_child":
                parent = self._resolve_operation_ref(change.get("parent_path", ""), index)
                if parent is None:
                    continue
                new_node = change.get("new_node", {})
                if not isinstance(new_node, dict):
                    continue
                node_name = str(new_node.get("name", "")).strip()
                description = str(new_node.get("description", "")).strip()
                content = str(new_node.get("content", "")).strip()
                if not node_name or not description or not content:
                    continue
                visibility = str(new_node.get("visibility", "private" if hard_problem.user_key else "shared")).strip().lower()
                user_key = new_node.get("user_key")
                if visibility == "private" and user_key in (None, ""):
                    user_key = hard_problem.user_key
                update_type = str(new_node.get("update_type", parent.update_type or "insert")).strip().lower() or "insert"
                node_path = self._normalize_ref(new_node.get("node_path", "")) or self._next_child_path(parent, node_name)
                child_paths = [str(item) for item in new_node.get("children", []) if str(item).strip()]
                new_op = Operation(
                    name=node_name,
                    description=description,
                    instruction_template=content,
                    update_type=update_type,
                    visibility=visibility,
                    user_key=user_key,
                    children=child_paths,
                    content=content,
                    meta_info={
                        'usage_count': 0,
                        'avg_reward': 0.0,
                        'recent_rewards': [],
                        'recent_usage_ema': 0.0,
                        'created_at': 'hard_problem',
                        'last_modified': 'hard_problem',
                        'node_path': node_path,
                    }
                )
                if not self.add_operation(new_op):
                    continue
                parent_children = list(parent.children)
                if node_path not in [self._normalize_ref(item) for item in parent_children]:
                    parent_children.append(node_path)
                    self.update_operation(parent.node_path, children=parent_children)
                applied_changes.append({
                    "action": "add_child",
                    "parent_path": parent.node_path,
                    "new_node_path": node_path,
                    "new_node_name": node_name
                })
                new_names.extend([new_op.node_path, parent.node_path])

        if new_names:
            self.set_new_operation_names(new_names)
        updater_result["applied_changes"] = applied_changes
        return updater_result

    def route_skill_tree(self, session_text: str, query_embedding: np.ndarray,
                         user_key: Optional[str], args) -> Dict[str, Any]:
        index = self._build_visible_index(user_key)
        visible_ops = index["visible_ops"]
        if not visible_ops:
            return {
                "candidate_ops": [],
                "selected_paths": [],
                "routing_summary": "",
                "trace": [],
                "selection_mode": "empty"
            }

        top_k = max(1, int(getattr(args, "tree_routing_top_k", 3)))
        max_depth = max(1, int(getattr(args, "tree_routing_max_depth", 6)))
        current_op: Optional[Operation] = None
        path_ops: List[Operation] = []
        trace: List[Dict[str, Any]] = []
        useful_paths: List[str] = []

        for _ in range(max_depth):
            children = self._get_child_operations(current_op, index)
            scored_children = self._score_operations(query_embedding, children, top_k=top_k)
            top_children = [op for _, op in scored_children]
            current_context_paths = [op.node_path for op in path_ops]
            if current_op is not None and (not current_context_paths or current_context_paths[-1] != current_op.node_path):
                current_context_paths = current_context_paths + [current_op.node_path]
            local_allowed_paths = set(
                self._normalize_path_list(current_context_paths + [op.node_path for op in top_children])
            )

            prompt = self._build_routing_prompt(
                session_text=session_text,
                current_op=current_op,
                path_ops=path_ops,
                top_children=top_children
            )
            try:
                decision = self._call_router_llm(prompt, args)
            except Exception as exc:
                decision = {
                    "decision": "stop",
                    "selected_child_path": "",
                    "useful_paths": [op.node_path for op in path_ops] or ([current_op.node_path] if current_op else []),
                    "reasoning": f"Router fallback due to API error: {exc}"
                }

            trace.append({
                "current_path": current_op.node_path if current_op is not None else "ROOT",
                "top_children": [op.node_path for op in top_children],
                "decision": decision
            })

            if str(decision.get("decision", "stop")).strip().lower() != "continue":
                candidate_useful_paths = self._normalize_path_list(decision.get("useful_paths", []) or [])
                useful_paths = [p for p in candidate_useful_paths if p in local_allowed_paths]
                if not useful_paths:
                    useful_paths = [p for p in self._normalize_path_list(current_context_paths) if p in local_allowed_paths]
                break

            allowed_child_paths = {self._normalize_ref(op.node_path) for op in top_children}
            selected_child_path = self._normalize_ref(decision.get("selected_child_path", ""))
            if selected_child_path not in allowed_child_paths:
                useful_paths = [p for p in self._normalize_path_list(current_context_paths) if p in local_allowed_paths]
                break
            selected_child = self._resolve_operation_ref(selected_child_path, index)
            if selected_child is None:
                useful_paths = [p for p in self._normalize_path_list(current_context_paths) if p in local_allowed_paths]
                break

            current_op = selected_child
            path_ops = path_ops + [selected_child]

            if not self._get_child_operations(current_op, index):
                useful_paths = [op.node_path for op in path_ops]
                break

        if not useful_paths:
            useful_paths = [op.node_path for op in path_ops] or (
                [current_op.node_path] if current_op is not None else [op.node_path for op in self._get_root_operations(index)[:top_k]]
            )

        selected_ops: List[Operation] = []
        seen = set()
        for ref in useful_paths:
            op = self._resolve_operation_ref(ref, index)
            if op is None:
                continue
            if op.node_path in seen:
                continue
            seen.add(op.node_path)
            selected_ops.append(op)

        if not selected_ops:
            selected_ops = [op for _, op in self._score_operations(query_embedding, visible_ops, top_k=top_k)]
        if not selected_ops:
            selected_ops = visible_ops[:top_k]

        try:
            summary_prompt = self._build_summary_prompt(session_text, selected_ops)
            routing_summary = self._call_summary_llm(summary_prompt, args)
        except Exception as exc:
            routing_summary = f"Skill summary unavailable: {exc}"

        return {
            "candidate_ops": selected_ops,
            "selected_paths": [op.node_path for op in selected_ops],
            "routing_summary": routing_summary,
            "trace": trace,
            "selection_mode": "tree_routing"
        }

    def load_from_templates(self, templates: Dict[str, Dict]):
        """Replace operation bank using a template dict (sanity check helper)."""
        self.operations = {}
        self.new_operation_names = set()

        for op_name, op_data in templates.items():
            operation = Operation(
                name=op_data['name'],
                description=op_data['description'],
                instruction_template=op_data.get('content', op_data.get('instruction_template', '')),
                update_type=op_data['update_type'],
                meta_info=copy.deepcopy(op_data.get('meta_info', {})),
                visibility=op_data.get('visibility', 'shared'),
                user_key=op_data.get('user_key'),
                children=copy.deepcopy(op_data.get('children', [])),
                content=op_data.get('content', op_data.get('instruction_template', ''))
            )
            self.operations[self._operation_key(operation)] = operation

        if self.encoder is not None:
            self._recompute_embeddings()

    def set_new_operation_names(self, names: List[str]):
        """Set which operation names are treated as new for exploration bias."""
        self.new_operation_names = {
            self._normalize_ref(name) for name in names if self._normalize_ref(name)
        }

    def get_new_action_indices(self, candidate_ops: Optional[List[Operation]] = None) -> List[int]:
        """Get indices of new actions in the candidate list."""
        if candidate_ops is None:
            candidate_ops = self.get_candidate_operations()
        return [i for i, op in enumerate(candidate_ops) if self._operation_key(op) in self.new_operation_names]

    def get_candidate_operations(self, session_text: Optional[str] = None,
                                 query_embedding: Optional[np.ndarray] = None,
                                 user_key: Optional[str] = None,
                                 args=None,
                                 return_metadata: bool = False):
        """
        Get all operations in the bank for controller to select from.

        Returns:
            List of all Operation objects
        """
        if len(self.operations) == 0:
            return ([], {}) if return_metadata else []

        if session_text is not None and query_embedding is not None and args is not None:
            routed = self.route_skill_tree(
                session_text=session_text,
                query_embedding=query_embedding,
                user_key=user_key,
                args=args
            )
            if return_metadata:
                return routed["candidate_ops"], routed
            return routed["candidate_ops"]

        # Return all operations sorted by name for stable ordering.
        # This ensures action_idx mapping is deterministic across calls.
        # Note: Even though we store op_embeddings per step (making PPO self-consistent),
        # stable ordering is still good practice to avoid subtle bugs when Designer
        # adds/removes operations between episodes.
        names = sorted(self.operations.keys())
        ops = [self.operations[name] for name in names]
        if user_key is not None:
            ops = [op for op in ops if self._is_visible_for_user(op, user_key)]
        if return_metadata:
            return ops, {"selection_mode": "flat", "routing_summary": "", "selected_paths": [], "trace": []}
        return ops

    def add_operation(self, operation: Operation) -> bool:
        """
        Add a new operation to the bank.

        If the bank is at capacity (max_ops), the operation with the lowest
        avg_reward will be replaced.

        Args:
            operation: The operation to add

        Returns:
            True if operation was added successfully
        """
        operation_key = self._operation_key(operation)

        # If operation already exists, just update it
        if operation_key in self.operations:
            self.operations[operation_key] = operation
            if self.encoder is not None:
                self._recompute_embeddings()
            return True

        # Check if at capacity
        if len(self.operations) >= self.max_ops:
            # Find the operation with lowest avg_reward (that has been used at least once)
            worst_op_name = None
            worst_reward = float('inf')

            for name, op in self.operations.items():
                # Only consider operations that have been used
                if op.meta_info.get('usage_count', 0) > 0:
                    avg_reward = op.meta_info.get('avg_reward', 0.0)
                    if avg_reward < worst_reward:
                        worst_reward = avg_reward
                        worst_op_name = name

            # If no used operations found, pick the one with lowest usage count
            if worst_op_name is None:
                min_usage = float('inf')
                for name, op in self.operations.items():
                    usage = op.meta_info.get('usage_count', 0)
                    if usage < min_usage:
                        min_usage = usage
                        worst_op_name = name

            # Replace worst operation
            if worst_op_name is not None:
                del self.operations[worst_op_name]
            else:
                return False

        self.operations[operation_key] = operation
        self.new_operation_names.add(operation_key)
        # Recompute embeddings if encoder is available
        if self.encoder is not None:
            self._recompute_embeddings()
        return True

    def update_operation(self, node_path: str, **kwargs):
        """Update an existing operation"""
        key = self._lookup_operation_key(node_path)
        if key is None:
            raise KeyError(f"Operation {node_path} not found")

        op = self.operations[key]
        for key, value in kwargs.items():
            if hasattr(op, key):
                setattr(op, key, value)

        if 'content' in kwargs and 'instruction_template' not in kwargs:
            op.instruction_template = op.content
        elif 'instruction_template' in kwargs and 'content' not in kwargs:
            op.content = op.instruction_template

        new_key = self._operation_key(op)
        if new_key != key:
            del self.operations[key]
            self.operations[new_key] = op
            self.new_operation_names.discard(key)
        else:
            self.operations[new_key] = op

        # Recompute embeddings if encoder is available
        if self.encoder is not None:
            self._recompute_embeddings()
        self.new_operation_names.add(new_key)

    def remove_operation(self, node_path: str):
        """Remove an operation from the bank"""
        key = self._lookup_operation_key(node_path)
        if key in self.operations:
            del self.operations[key]
            self.new_operation_names.discard(key)

    def get_operation(self, node_path: str) -> Operation:
        """Get operation by canonical node path."""
        key = self._lookup_operation_key(node_path)
        if key is None or key not in self.operations:
            raise KeyError(f"Operation {node_path} not found")
        return self.operations[key]

    def get_all_operations(self) -> List[Operation]:
        """Get all operations"""
        return list(self.operations.values())

    def get_operation_stats(self) -> Dict:
        """Get statistics for all operations"""
        stats = {}
        for name, op in self.operations.items():
            stats[name] = {
                'usage_count': op.meta_info['usage_count'],
                'avg_reward': op.meta_info['avg_reward'],
                'recent_rewards': op.meta_info['recent_rewards'],
                'recent_usage_ema': op.meta_info.get('recent_usage_ema', 0.0)
            }
        return stats

    def update_all_ema(self, selected_op_name: str, ema_alpha: float = 0.1):
        """Update EMA for all operations after a selection.

        - Selected operation: EMA spikes toward 1.0
        - Non-selected operations: EMA decays toward 0.0

        Args:
            selected_op_name: Name of the operation that was selected
            ema_alpha: EMA smoothing factor
        """
        for name, op in self.operations.items():
            if name == selected_op_name:
                # Spike: EMA = alpha * 1.0 + (1 - alpha) * old_ema
                old_ema = op.meta_info.get("recent_usage_ema", 0.0)
                op.meta_info["recent_usage_ema"] = ema_alpha * 1.0 + (1.0 - ema_alpha) * old_ema
            else:
                op.decay_ema(ema_alpha)

    def batch_update_ema(self, op_usage_counts: Dict[str, int], total_steps: int, ema_alpha: float = 0.1):
        """Update EMA based on batch-level usage counts.

        This is used for parallel episode collection where we aggregate usage
        across all episodes and update EMA once at the end of the batch.

        The update simulates `total_steps` EMA updates where each op's selection
        frequency determines how often it spikes vs decays.

        Args:
            op_usage_counts: Dict mapping op_name -> number of times selected in batch
            total_steps: Total number of steps across all episodes in the batch
            ema_alpha: EMA smoothing factor
        """
        if total_steps == 0:
            return

        for name, op in self.operations.items():
            old_ema = op.meta_info.get("recent_usage_ema", 0.0)
            count = op_usage_counts.get(name, 0)

            # Approximate EMA after `total_steps` updates:
            # - Op was selected `count` times (spike to 1.0)
            # - Op was not selected `total_steps - count` times (decay)
            #
            # The closed-form approximation for repeated EMA updates:
            # After n steps with selection frequency f = count/total_steps:
            # EMA converges toward f, with decay rate (1-alpha)^n toward old value
            #
            # new_ema = target * (1 - decay_factor) + old_ema * decay_factor
            # where target = selection_frequency, decay_factor = (1-alpha)^total_steps

            selection_freq = count / total_steps
            decay_factor = (1.0 - ema_alpha) ** total_steps

            # The EMA update formula for batch:
            # new_ema blends toward selection_freq, with old_ema decayed
            new_ema = selection_freq * (1.0 - decay_factor) + old_ema * decay_factor

            op.meta_info["recent_usage_ema"] = new_ema

    def __len__(self):
        return len(self.operations)

    def to_dict(self):
        """Serialize to dict"""
        return {
            'operations': {name: op.to_dict() for name, op in self.operations.items()},
            'max_ops': self.max_ops,
            'skip_noop': self.skip_noop
        }

    @classmethod
    def from_dict(cls, data, encoder=None):
        """Deserialize from dict"""
        # Create bank without initializing seeds (we'll load operations from data)
        bank = cls.__new__(cls)
        bank.operations = {}
        bank.encoder = encoder
        bank.max_ops = data.get('max_ops', 20)
        bank.skip_noop = data.get('skip_noop', False)
        bank.new_operation_names = set()

        for name, op_data in data.get('operations', {}).items():
            op = Operation.from_dict(op_data)
            key = bank._operation_key(op) or bank._normalize_ref(name)
            bank.operations[key] = op

        # Recompute embeddings with current encoder for consistency
        if bank.encoder is not None:
            bank._recompute_embeddings()

        return bank

    def copy(self):
        """Create a deep copy of the operation bank"""
        return copy.deepcopy(self)
