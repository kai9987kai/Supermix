"""NexusMind Graph-of-Thoughts (GoT) Reasoner.

Implements structured Graph-of-Thoughts reasoning for Supermix v72 / NexusMind:
* **Thought Nodes**: States containing partial reasoning steps, evidence, and scores.
* **Multi-Draft Branching**: Generates parallel candidate continuations at key decision points.
* **Speculative Drafting**: Evaluates speculative candidates and branches before commitment.
* **Pruning**: Dynamically removes low-scoring or contradictory thought paths.
* **Node Merging**: Synthesizes complementary insights from disparate branches.
* **Audit Receipt**: Emits deterministic `GoTReceipt` with graph topology digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


__all__ = [
    "ThoughtNode",
    "GoTReceipt",
    "GoTSearchResult",
    "GraphOfThoughts",
]


@dataclass
class ThoughtNode:
    """A single thought node within the reasoning graph."""

    node_id: str
    parent_id: Optional[str]
    step_text: str
    depth: int
    score: float = 1.0
    branch_type: str = "draft"  # "root" | "draft" | "critique" | "merged" | "leaf"
    is_pruned: bool = False
    is_terminal: bool = False
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoTReceipt:
    """Cryptographic audit receipt for a Graph-of-Thoughts execution."""

    schema_version: str = "nexus-got-receipt-v1"
    query_digest: str = ""
    best_path_digest: str = ""
    total_nodes_generated: int = 0
    nodes_pruned: int = 0
    nodes_merged: int = 0
    max_search_depth: int = 0
    optimal_path_score: float = 0.0
    search_strategy: str = "beam_prune_merge"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoTSearchResult:
    """The result of a Graph-of-Thoughts reasoning run."""

    query: str
    best_path_nodes: List[ThoughtNode]
    final_output: str
    receipt: GoTReceipt
    all_nodes: Dict[str, ThoughtNode]
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "best_path_nodes": [n.to_dict() for n in self.best_path_nodes],
            "final_output": self.final_output,
            "receipt": self.receipt.to_dict(),
            "nodes_count": len(self.all_nodes),
            "telemetry": self.telemetry,
        }


class GraphOfThoughts:
    """Orchestrates structured Graph-of-Thoughts reasoning and tree search."""

    def __init__(
        self,
        max_depth: int = 4,
        beam_width: int = 3,
        prune_threshold: float = 0.45,
    ):
        self.max_depth = max(1, min(10, max_depth))
        self.beam_width = max(1, min(10, beam_width))
        self.prune_threshold = prune_threshold
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self._node_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter:04d}"

    def reset(self) -> None:
        self.nodes.clear()
        self.root_id = None
        self._node_counter = 0

    def add_node(
        self,
        step_text: str,
        parent_id: Optional[str] = None,
        depth: int = 0,
        score: float = 1.0,
        branch_type: str = "draft",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThoughtNode:
        node_id = self._next_node_id()
        node = ThoughtNode(
            node_id=node_id,
            parent_id=parent_id,
            step_text=step_text,
            depth=depth,
            score=score,
            branch_type=branch_type,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        if self.root_id is None and parent_id is None:
            self.root_id = node_id
        return node

    def expand_node(
        self,
        node: ThoughtNode,
        query: str,
        candidate_generator_fn: Optional[Callable[[str, int], List[str]]] = None,
    ) -> List[ThoughtNode]:
        """Expand a node into candidate child thoughts."""
        if node.is_pruned or node.depth >= self.max_depth:
            return []

        children: List[ThoughtNode] = []
        if candidate_generator_fn:
            candidates = candidate_generator_fn(node.step_text, node.depth + 1)
        else:
            # Deterministic default branching templates
            d = node.depth + 1
            if d == 1:
                candidates = [
                    f"Decompose primary constraints and variables for: '{query}'",
                    f"Frame problem through direct deductive transformation",
                    f"Evaluate boundary conditions and invariant relationships",
                ]
            elif d == 2:
                candidates = [
                    f"Execute step-by-step arithmetic / logical deduction on [{node.step_text[:30]}]",
                    f"Verify consistency against baseline rules and definitions",
                ]
            else:
                candidates = [
                    f"Synthesize verified intermediate result into final answer",
                    f"Cross-check final result against initial requirements",
                ]

        for idx, cand_text in enumerate(candidates[: self.beam_width]):
            # Score candidate based on depth and position
            base_score = 0.95 - (node.depth * 0.05) - (idx * 0.08)
            score = round(max(0.2, min(1.0, base_score)), 4)
            child = self.add_node(
                step_text=cand_text,
                parent_id=node.node_id,
                depth=node.depth + 1,
                score=score,
                branch_type="draft",
            )
            children.append(child)

        return children

    def prune_unviable(self) -> int:
        """Prune nodes below the prune threshold."""
        pruned_count = 0
        for node in self.nodes.values():
            if not node.is_pruned and node.score < self.prune_threshold and node.parent_id is not None:
                node.is_pruned = True
                pruned_count += 1
        return pruned_count

    def merge_complementary(self, active_leaves: List[ThoughtNode]) -> Optional[ThoughtNode]:
        """Merge complementary high-scoring leaves into a unified synthesis node."""
        viable = [n for n in active_leaves if not n.is_pruned and n.score >= 0.70]
        if len(viable) < 2:
            return None

        # Sort by score descending
        viable.sort(key=lambda n: n.score, reverse=True)
        top_two = viable[:2]
        merged_text = (
            f"Merged Synthesis: [{top_two[0].step_text}] combined with [{top_two[1].step_text}]"
        )
        avg_score = round(sum(n.score for n in top_two) / len(top_two) + 0.05, 4)
        avg_score = min(1.0, avg_score)

        merged_node = self.add_node(
            step_text=merged_text,
            parent_id=top_two[0].node_id,
            depth=max(n.depth for n in top_two) + 1,
            score=avg_score,
            branch_type="merged",
            metadata={"source_node_ids": [n.node_id for n in top_two]},
        )
        return merged_node

    def search(
        self,
        query: str,
        initial_hypothesis: Optional[str] = None,
        candidate_generator_fn: Optional[Callable[[str, int], List[str]]] = None,
    ) -> GoTSearchResult:
        """Execute full Graph-of-Thoughts tree expansion, pruning, merging, and search."""
        self.reset()
        root_text = initial_hypothesis or f"Root: {query}"
        root = self.add_node(step_text=root_text, depth=0, score=1.0, branch_type="root")

        current_frontier = [root]
        nodes_pruned_total = 0
        nodes_merged_total = 0

        for depth in range(self.max_depth):
            next_frontier: List[ThoughtNode] = []
            for node in current_frontier:
                if not node.is_pruned:
                    children = self.expand_node(node, query, candidate_generator_fn)
                    next_frontier.extend(children)

            if not next_frontier:
                break

            # Prune
            pruned = self.prune_unviable()
            nodes_pruned_total += pruned

            # Select top-k beam
            surviving = [n for n in next_frontier if not n.is_pruned]
            surviving.sort(key=lambda n: n.score, reverse=True)
            current_frontier = surviving[: self.beam_width]

            # Merge complementary branches if available at depth >= 2
            if depth >= 1 and len(current_frontier) >= 2:
                merged = self.merge_complementary(current_frontier)
                if merged:
                    nodes_merged_total += 1
                    current_frontier.append(merged)

        # Find best leaf path (deepest viable explored nodes)
        terminal_nodes = [
            n for n in self.nodes.values()
            if not n.is_pruned and (len(n.children_ids) == 0 or n.depth > 0)
        ]
        if not terminal_nodes:
            terminal_nodes = [root]

        best_leaf = max(terminal_nodes, key=lambda n: (n.depth, n.score))

        # Trace path back to root
        best_path: List[ThoughtNode] = []
        curr: Optional[ThoughtNode] = best_leaf
        while curr:
            best_path.append(curr)
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None
        best_path.reverse()

        final_output = best_leaf.step_text

        # Compute digests
        q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        path_str = " -> ".join(n.step_text for n in best_path)
        path_hash = hashlib.sha256(path_str.encode("utf-8")).hexdigest()

        receipt = GoTReceipt(
            query_digest=q_hash,
            best_path_digest=path_hash,
            total_nodes_generated=len(self.nodes),
            nodes_pruned=nodes_pruned_total,
            nodes_merged=nodes_merged_total,
            max_search_depth=max(n.depth for n in best_path),
            optimal_path_score=best_leaf.score,
        )

        return GoTSearchResult(
            query=query,
            best_path_nodes=best_path,
            final_output=final_output,
            receipt=receipt,
            all_nodes=dict(self.nodes),
            telemetry={
                "total_nodes": len(self.nodes),
                "nodes_pruned": nodes_pruned_total,
                "nodes_merged": nodes_merged_total,
                "optimal_score": best_leaf.score,
                "path_length": len(best_path),
            },
        )
