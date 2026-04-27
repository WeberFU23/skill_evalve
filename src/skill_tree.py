"""
Hierarchical skill tree loading and PPO-compatible routing.

The tree layout convention is:
    skills/domain/domain.md
    skills/domain/child/child.md

Each directory may contain one node markdown file named after the directory.
Other markdown files are also loaded as child nodes for compatibility, but the
preferred shape is one node file per directory.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # Tree loading and embedding fallback do not require torch.
    torch = None


STOP_ACTION_NAME = "__stop__"
STOP_ACTION_DESCRIPTION = (
    "Stop searching and use the current skill node or selected path because "
    "further specialization is unnecessary for the current context."
)


@dataclass
class SkillNode:
    """A node loaded from a skill markdown file."""
    id: str
    name: str
    path: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    body: str = ""
    parent: Optional["SkillNode"] = None
    children: List["SkillNode"] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    @property
    def visibility(self) -> str:
        return str(self.metadata.get("visibility", "shared")).lower()

    @property
    def scope_id(self) -> Optional[str]:
        value = self.metadata.get("scope_id")
        if value is None:
            return None
        value = str(value).strip()
        if value.lower() in ("", "null", "none"):
            return None
        return value

    @property
    def tags(self) -> List[str]:
        tags = self.metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split(",") if tag.strip()]
        return []

    def is_visible(self, scope_ids: Optional[Iterable[str]] = None) -> bool:
        """Return whether this node is visible under the requested scopes."""
        if self.visibility == "shared":
            return True
        allowed_scopes = {str(scope) for scope in (scope_ids or []) if scope is not None}
        return self.scope_id is not None and self.scope_id in allowed_scopes

    def description_text(self) -> str:
        """Text used for embedding retrieval and routing action embeddings."""
        description = _extract_section(self.body, "Description")
        purpose = _extract_section(self.body, "Purpose")
        routing = _extract_section(self.body, "Routing Guidance")
        child_criteria = _extract_section(self.body, "Child Selection Criteria")
        title = _extract_title(self.body) or self.name
        tag_text = ", ".join(self.tags)
        parts = [self.name, title]
        if tag_text:
            parts.append(f"Tags: {tag_text}")
        if description:
            parts.append(description)
        if purpose:
            parts.append(purpose)
        if routing:
            parts.append(routing)
        if child_criteria:
            parts.append(child_criteria)
        return "\n".join(part for part in parts if part).strip()

    def instruction_text(self) -> str:
        """Full node text suitable for prompt assembly."""
        return self.body.strip()

    @property
    def update_type(self) -> str:
        """Executor action type declared by this skill node, if any."""
        return _extract_update_type(self.metadata, self.body)

    def is_executable(self) -> bool:
        """Whether this node can be handed to the memory executor."""
        return self.update_type in {"insert", "update", "delete", "noop"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "children": [child.path for child in self.children],
        }


@dataclass
class RoutingStep:
    """One tree-routing decision."""
    current_path: str
    action: str
    candidate_paths: List[str]
    selected_path: Optional[str]
    log_prob: float
    value: float
    action_embeddings: Optional[np.ndarray] = None


@dataclass
class SkillTreeSelection:
    """Result returned by hierarchical routing."""
    selected_nodes: List[SkillNode]
    path_nodes: List[SkillNode]
    terminal_node: SkillNode
    routing_steps: List[RoutingStep]
    stopped: bool

    @property
    def selected_paths(self) -> List[str]:
        return [node.path for node in self.selected_nodes]

    def prompt_context(self, max_chars_per_node: int = 2000) -> str:
        """Assemble selected skill node contents for a downstream prompt."""
        blocks = []
        for node in self.selected_nodes:
            text = node.instruction_text()
            if len(text) > max_chars_per_node:
                text = text[:max_chars_per_node] + "\n...[truncated]"
            blocks.append(f"## Skill: {node.path}\n{text}")
        return "\n\n".join(blocks)


class SkillTree:
    """Load and query a directory-backed skill tree."""

    def __init__(self, root_dir: str = "skills", encoder=None):
        self.root_dir = os.path.abspath(root_dir)
        self.encoder = encoder
        self.roots: List[SkillNode] = []
        self.nodes_by_path: Dict[str, SkillNode] = {}
        self.nodes_by_id: Dict[str, SkillNode] = {}
        self.load()

    def load(self):
        self.roots = []
        self.nodes_by_path = {}
        self.nodes_by_id = {}

        if not os.path.isdir(self.root_dir):
            return

        for entry in sorted(os.listdir(self.root_dir)):
            abs_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(abs_path):
                continue
            root = self._load_dir(abs_path, parent=None)
            if root is not None:
                self.roots.append(root)

        if self.encoder is not None:
            self.recompute_embeddings()

    def _load_dir(self, dir_path: str, parent: Optional[SkillNode]) -> Optional[SkillNode]:
        dir_name = os.path.basename(dir_path)
        node_file = os.path.join(dir_path, f"{dir_name}.md")
        node = None

        if os.path.exists(node_file):
            node = self._load_node_file(node_file, parent=parent)
        elif parent is None:
            # Top-level directories should have a node file. If one is missing,
            # create a lightweight synthetic root so descendants remain reachable.
            rel_path = os.path.relpath(dir_path, self.root_dir).replace(os.sep, "/")
            node = SkillNode(
                id=rel_path.replace("/", "."),
                name=dir_name,
                path=rel_path,
                file_path="",
                metadata={"visibility": "shared", "scope_id": None, "tags": []},
                body=f"# {dir_name}\n\n## Description\n\nSynthetic skill category node.",
                parent=parent,
            )
            self._register_node(node)

        if node is None:
            return None

        for child_name in sorted(os.listdir(dir_path)):
            child_path = os.path.join(dir_path, child_name)
            if os.path.isdir(child_path):
                child = self._load_dir(child_path, parent=node)
                if child is not None:
                    node.children.append(child)
            elif child_name.endswith(".md") and child_path != node_file:
                child = self._load_node_file(child_path, parent=node)
                node.children.append(child)

        return node

    def _load_node_file(self, file_path: str, parent: Optional[SkillNode]) -> SkillNode:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        metadata, body = _split_frontmatter(raw)
        rel_file = os.path.relpath(file_path, self.root_dir)
        rel_file = rel_file.replace(os.sep, "/")
        path = rel_file[:-3] if rel_file.endswith(".md") else rel_file
        name = str(metadata.get("name") or os.path.splitext(os.path.basename(file_path))[0])
        node_id = str(metadata.get("id") or path.replace("/", "."))

        node = SkillNode(
            id=node_id,
            name=name,
            path=path,
            file_path=file_path,
            metadata=metadata,
            body=body,
            parent=parent,
        )
        self._register_node(node)
        return node

    def _register_node(self, node: SkillNode):
        self.nodes_by_path[node.path] = node
        self.nodes_by_id[node.id] = node

    def recompute_embeddings(self):
        if self.encoder is None:
            return
        nodes = list(self.nodes_by_path.values())
        if not nodes:
            return
        texts = [node.description_text() for node in nodes]
        embeddings = self.encoder.encode(texts)
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

    def visible_roots(self, scope_ids: Optional[Iterable[str]] = None) -> List[SkillNode]:
        return [node for node in self.roots if node.is_visible(scope_ids)]

    def get_node(self, path_or_id: str) -> SkillNode:
        if path_or_id in self.nodes_by_path:
            return self.nodes_by_path[path_or_id]
        if path_or_id in self.nodes_by_id:
            return self.nodes_by_id[path_or_id]
        raise KeyError(f"Skill node not found: {path_or_id}")


class SkillTreeSelector:
    """
    Hierarchical selector using embedding top-k pruning plus an optional PPO controller.

    The controller action space at each node is:
        [STOP, child_1, child_2, ..., child_k]
    where child_i are the embedding top-k visible children.
    """

    def __init__(self, tree: SkillTree, encoder=None, controller=None,
                 device: str = "cpu", top_k: int = 3, max_depth: int = 4):
        self.tree = tree
        self.encoder = encoder or tree.encoder
        self.controller = controller
        self.device = device
        self.top_k = top_k
        self.max_depth = max_depth

    def select(self, query: str, state_embedding: Optional[np.ndarray] = None,
               scope_ids: Optional[Iterable[str]] = None,
               start_path: Optional[str] = None,
               deterministic: bool = True) -> SkillTreeSelection:
        """Route through the tree and return selected skill nodes."""
        if self.encoder is None:
            raise ValueError("SkillTreeSelector requires an encoder")

        query_embedding = self._encode_query(query)
        if state_embedding is None:
            state_embedding = self._build_routing_state(query, [], None)

        if start_path is None:
            roots = self.tree.visible_roots(scope_ids)
            if not roots:
                raise ValueError("No visible skill roots")
            current = self._select_start_root(query_embedding, roots)
        else:
            current = self.tree.get_node(start_path)
            if not current.is_visible(scope_ids):
                raise ValueError(f"Start node is not visible: {start_path}")

        path_nodes = [current]
        routing_steps: List[RoutingStep] = []
        stopped = False

        for _ in range(self.max_depth):
            children = self._top_k_children(current, query_embedding, scope_ids)
            action_nodes = [None] + children
            action_embeddings = self._action_embeddings(action_nodes)
            route_state = self._build_routing_state(query, path_nodes, current, state_embedding)

            action_idx, log_prob, value = self._choose_action(
                route_state, action_embeddings, deterministic=deterministic
            )
            candidate_paths = [STOP_ACTION_NAME] + [child.path for child in children]

            if action_idx == 0 or len(children) == 0:
                routing_steps.append(RoutingStep(
                    current_path=current.path,
                    action=STOP_ACTION_NAME,
                    candidate_paths=candidate_paths,
                    selected_path=None,
                    log_prob=log_prob,
                    value=value,
                    action_embeddings=np.asarray(action_embeddings),
                ))
                stopped = True
                break

            selected_child = children[action_idx - 1]
            routing_steps.append(RoutingStep(
                current_path=current.path,
                action="select_child",
                candidate_paths=candidate_paths,
                selected_path=selected_child.path,
                log_prob=log_prob,
                value=value,
                action_embeddings=np.asarray(action_embeddings),
            ))
            current = selected_child
            path_nodes.append(current)

        selected_nodes = self._assemble_selected_nodes(
            path_nodes=path_nodes,
            terminal_node=current,
            query_embedding=query_embedding,
            scope_ids=scope_ids,
        )
        return SkillTreeSelection(
            selected_nodes=selected_nodes,
            path_nodes=path_nodes,
            terminal_node=current,
            routing_steps=routing_steps,
            stopped=stopped,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        emb = self.encoder.encode(query)
        if emb.ndim == 2:
            emb = emb[0]
        return emb

    def _select_start_root(self, query_embedding: np.ndarray,
                           roots: List[SkillNode]) -> SkillNode:
        if len(roots) == 1:
            return roots[0]
        ranked = self._rank_nodes(query_embedding, roots)
        return ranked[0]

    def _top_k_children(self, node: SkillNode, query_embedding: np.ndarray,
                        scope_ids: Optional[Iterable[str]]) -> List[SkillNode]:
        visible = [child for child in node.children if child.is_visible(scope_ids)]
        if not visible:
            return []
        ranked = self._rank_nodes(query_embedding, visible)
        return ranked[:max(1, int(self.top_k))]

    def _rank_nodes(self, query_embedding: np.ndarray,
                    nodes: List[SkillNode]) -> List[SkillNode]:
        for node in nodes:
            if node.embedding is None:
                node.embedding = self.encoder.encode(node.description_text())
                if node.embedding.ndim == 2:
                    node.embedding = node.embedding[0]
        q = _normalize(query_embedding)
        scored = []
        for node in nodes:
            scored.append((float(np.dot(q, _normalize(node.embedding))), node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored]

    def _action_embeddings(self, action_nodes: List[Optional[SkillNode]]) -> np.ndarray:
        texts = []
        for node in action_nodes:
            if node is None:
                texts.append(STOP_ACTION_DESCRIPTION)
            else:
                texts.append(node.description_text())
        return self.encoder.encode(texts)

    def _build_routing_state(self, query: str, path_nodes: List[SkillNode],
                             current: Optional[SkillNode],
                             fallback: Optional[np.ndarray] = None) -> np.ndarray:
        if fallback is not None:
            arr = np.asarray(fallback)
            if arr.ndim == 2:
                arr = arr[0]
            return arr

        path_text = " > ".join(node.name for node in path_nodes) if path_nodes else "(root)"
        current_text = current.description_text() if current is not None else ""
        state_text = (
            f"Query:\n{query}\n\n"
            f"Current skill path: {path_text}\n\n"
            f"Current skill node:\n{current_text}"
        )
        emb = self.encoder.encode(state_text)
        if emb.ndim == 2:
            emb = emb[0]
        return emb

    def _choose_action(self, state_embedding: np.ndarray, action_embeddings: np.ndarray,
                       deterministic: bool) -> Tuple[int, float, float]:
        if self.controller is None:
            # Embedding-only fallback: continue to best child when it is more
            # similar than STOP; otherwise stop. This keeps the selector usable
            # before PPO routing is trained.
            sims = np.dot(
                np.vstack([_normalize(emb) for emb in action_embeddings]),
                _normalize(state_embedding),
            )
            action_idx = int(np.argmax(sims))
            return action_idx, 0.0, 0.0

        if torch is None:
            raise ImportError("Using a PPO controller for skill-tree routing requires torch")

        state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action_embeddings, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            state_h = self.controller.encode_state(state_tensor.unsqueeze(0))
            op_h = self.controller.encode_ops(action_tensor.unsqueeze(0))
            logits = self.controller.get_action_logits(state_h, op_h)[0]
            value = self.controller.get_value(state_h)[0]
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _assemble_selected_nodes(self, path_nodes: List[SkillNode],
                                 terminal_node: SkillNode,
                                 query_embedding: np.ndarray,
                                 scope_ids: Optional[Iterable[str]]) -> List[SkillNode]:
        selected: List[SkillNode] = []
        seen = set()

        def add(node: SkillNode):
            if node.path not in seen:
                seen.add(node.path)
                selected.append(node)

        for node in path_nodes:
            add(node)

        # Include the best visible children of the terminal node as local
        # executable hints. A later operation adapter can decide which nodes are
        # executable versus inherited context.
        for child in self._top_k_children(terminal_node, query_embedding, scope_ids):
            add(child)

        return selected


def _split_frontmatter(raw: str) -> Tuple[Dict[str, Any], str]:
    if not raw.startswith("---"):
        return {}, raw
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, flags=re.DOTALL)
    if not match:
        return {}, raw
    return _parse_simple_yaml(match.group(1)), match.group(2)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in ("null", "none", "~"):
            data[key] = None
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                data[key] = [item.strip().strip("'\"") for item in inner.split(",")]
        else:
            data[key] = value.strip("'\"")
    return data


def _extract_title(body: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_section(body: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, body, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_update_type(metadata: Dict[str, Any], body: str) -> str:
    value = metadata.get("update_type") or metadata.get("action_type")
    if value is not None:
        value = str(value).strip().lower()
        if value in {"insert", "update", "delete", "noop"}:
            return value

    output_action = _extract_section(body, "Output Action")
    search_text = output_action if output_action else body
    match = re.search(
        r"action\s+type\s*:\s*(insert|update|delete|noop)\b",
        search_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    return ""


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    norm = np.linalg.norm(arr)
    if norm <= 1e-8:
        return arr
    return arr / norm
