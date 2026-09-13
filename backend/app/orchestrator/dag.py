"""
DAG Execution Engine.
Constructs Directed Acyclic Graphs from AnalysisPlan operations,
performs topological sorting, detects cycles, and manages node invalidation.
"""
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from app.contracts import OperationSpec


class DAGCycleError(Exception):
    pass


class DAGReferenceError(Exception):
    pass


class ExecutionDAG:
    def __init__(self, operations: List[OperationSpec]):
        self.operations: Dict[str, OperationSpec] = {op.step_id: op for op in operations}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # u -> set of v (u must run before v)
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.reverse_adj: Dict[str, Set[str]] = defaultdict(set)  # v -> set of u (v depends on u)

        self._build_graph()

    def _build_graph(self) -> None:
        # Initialize in-degree for all operations
        for step_id in self.operations:
            self.in_degree[step_id] = 0

        for step_id, op in self.operations.items():
            for dep in op.depends_on:
                if dep not in self.operations:
                    raise DAGReferenceError(
                        f"Operation '{step_id}' depends on non-existent operation '{dep}'."
                    )
                # dep must run before step_id
                self.adjacency[dep].add(step_id)
                self.reverse_adj[step_id].add(dep)
                self.in_degree[step_id] += 1

    def topological_sort(self) -> List[str]:
        """
        Kahn's algorithm for topological sorting.
        Raises DAGCycleError if cyclic dependencies are detected.
        """
        in_deg = dict(self.in_degree)
        queue = deque([node for node, deg in in_deg.items() if deg == 0])
        ordered: List[str] = []

        while queue:
            node = queue.popleft()
            ordered.append(node)

            for neighbor in self.adjacency[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(self.operations):
            unresolved = [n for n, deg in in_deg.items() if deg > 0]
            raise DAGCycleError(
                f"Cyclic dependency detected in analytical plan involving nodes: {unresolved}"
            )

        return ordered

    def get_independent_batches(self) -> List[List[str]]:
        """
        Returns execution levels: operations in the same batch can run concurrently.
        """
        in_deg = dict(self.in_degree)
        current_level = [node for node, deg in in_deg.items() if deg == 0]
        batches: List[List[str]] = []
        processed_count = 0

        while current_level:
            batches.append(sorted(current_level))
            processed_count += len(current_level)
            next_level = []

            for node in current_level:
                for neighbor in self.adjacency[node]:
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0:
                        next_level.append(neighbor)

            current_level = next_level

        if processed_count != len(self.operations):
            unresolved = [n for n, deg in in_deg.items() if deg > 0]
            raise DAGCycleError(f"Cyclic dependency in plan: {unresolved}")

        return batches

    def get_downstream_dependents(self, node_id: str) -> Set[str]:
        """
        Collect all transitive downstream nodes that depend on node_id.
        Used for downstream result invalidation.
        """
        visited: Set[str] = set()
        queue = deque([node_id])

        while queue:
            curr = queue.popleft()
            for child in self.adjacency[curr]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return visited
