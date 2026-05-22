#!/usr/bin/env python3
"""Case 1: ToF2DG + MetaTree export over SNAP cit-HepPh."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "topolayout-dg-matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    import igraph as ig
except ImportError as exc:
    raise SystemExit(
        "python-igraph is required. Create the environment with "
        "`conda env create -f environment.yml` and activate `topolayout-dg`."
    ) from exc


META_TYPES = ["ISOLATED", "CYCLE", "COMPLETE", "SCC", "TREE", "DAG", "BISECTION"]
META_COLORS = {
    "ISOLATED": "#9ca3af",
    "CYCLE": "#f59e0b",
    "COMPLETE": "#ef4444",
    "SCC": "#22c55e",
    "TREE": "#14b8a6",
    "DAG": "#3b82f6",
    "BISECTION": "#a855f7",
}


@dataclass
class MetaNode:
    numeric_id: int
    id: str
    type: str
    vertices: List[int]
    edge_count: int
    parent: Optional[int]
    root_vertex: Optional[int]
    children_meta: List[int] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0


class TopoLayoutBuilder:
    """ToF2DG structural decomposition with nested geometric boxes.

    igraph is used for weak-component, strong-component, and induced-subgraph
    operations. The SCC phase corresponds to Tarjan-style SCC decomposition in
    the target procedure, delegated to igraph's optimized implementation.
    """

    def __init__(
        self,
        vertex_ids: Sequence[str],
        edges: Sequence[Tuple[int, int]],
        vertex_types: Optional[Sequence[str]] = None,
        scc_limit: int = 100,
    ) -> None:
        self.vertex_ids = [str(vertex_id) for vertex_id in vertex_ids]
        self.vertex_types = list(vertex_types) if vertex_types else ["Paper"] * len(self.vertex_ids)
        self.edges = list(dict.fromkeys((int(source), int(target)) for source, target in edges))
        self.scc_limit = scc_limit
        self.n = len(self.vertex_ids)

        self.graph = ig.Graph(n=self.n, edges=self.edges, directed=True)
        self.graph.vs["name"] = self.vertex_ids

        self.out_adj: List[List[int]] = [[] for _ in range(self.n)]
        self.in_adj: List[List[int]] = [[] for _ in range(self.n)]
        for source, target in self.edges:
            self.out_adj[source].append(target)
            self.in_adj[target].append(source)

        self.meta_nodes: List[MetaNode] = []
        self.root_meta_ids: List[int] = []
        self.vertex_owner: List[Optional[int]] = [None] * self.n
        self.vertex_local_x: List[float] = [0.0] * self.n
        self.vertex_local_y: List[float] = [0.0] * self.n
        self.meta_edges: List[Dict[str, object]] = []

    def build(self) -> None:
        all_vertices = list(range(self.n))
        self.root_meta_ids = self.tof2dg(all_vertices, parent=None)
        self.assign_fallback_isolated_vertices()
        self.compute_meta_edges()
        self.compute_nested_layout()

    def tof2dg(self, vertices: Sequence[int], parent: Optional[int]) -> List[int]:
        roots: List[int] = []
        for component in self.weak_components(vertices):
            if len(component) == 1:
                vertex = component[0]
                meta_id = self.create_meta_node("ISOLATED", [vertex], parent, root_vertex=vertex, edge_count=0)
                self.assign_owner([vertex], meta_id)
                roots.append(meta_id)
            else:
                roots.extend(self.process_connected_subgraph(component, parent))
        return roots

    def process_connected_subgraph(self, vertices: Sequence[int], parent: Optional[int]) -> List[int]:
        roots: List[int] = []
        covered_by_scc: Set[int] = set()
        nontrivial_sccs = sorted(
            (component for component in self.strong_components(vertices) if len(component) > 1),
            key=lambda component: (-len(component), component[0]),
        )

        for component in nontrivial_sccs:
            edge_count = self.count_edges_inside(component)
            meta_type = self.classify_scc(len(component), edge_count)
            meta_id = self.create_meta_node(meta_type, component, parent, root_vertex=min(component), edge_count=edge_count)
            roots.append(meta_id)
            covered_by_scc.update(component)

            if meta_type == "SCC" and len(component) > self.scc_limit:
                left, right = self.algebraic_bisect(component)
                for side in (left, right):
                    if not side:
                        continue
                    child_id = self.create_meta_node(
                        "BISECTION",
                        side,
                        meta_id,
                        root_vertex=min(side),
                        edge_count=self.count_edges_inside(side),
                    )
                    self.meta_nodes[meta_id].children_meta.append(child_id)
                    grandchildren = self.tof2dg(side, parent=child_id)
                    self.meta_nodes[child_id].children_meta.extend(grandchildren)
            else:
                self.assign_owner(component, meta_id)

        residual = [vertex for vertex in vertices if vertex not in covered_by_scc]
        for component in self.weak_components(residual):
            if not component:
                continue
            if len(component) == 1:
                vertex = component[0]
                if self.vertex_owner[vertex] is None:
                    meta_id = self.create_meta_node("ISOLATED", [vertex], parent, root_vertex=vertex, edge_count=0)
                    self.assign_owner([vertex], meta_id)
                    roots.append(meta_id)
                continue

            if self.is_tree(component):
                meta_type = "TREE"
            elif self.is_dag(component):
                meta_type = "DAG"
            else:
                meta_type = "SCC"

            meta_id = self.create_meta_node(
                meta_type,
                component,
                parent,
                root_vertex=self.choose_root_vertex(component),
                edge_count=self.count_edges_inside(component),
            )
            self.assign_owner(component, meta_id)
            roots.append(meta_id)

        return roots

    def weak_components(self, vertices: Sequence[int]) -> List[List[int]]:
        return self.igraph_components(vertices, mode="weak")

    def strong_components(self, vertices: Sequence[int]) -> List[List[int]]:
        return self.igraph_components(vertices, mode="strong")

    def igraph_components(self, vertices: Sequence[int], mode: str) -> List[List[int]]:
        ordered = sorted(vertices)
        if not ordered:
            return []
        if len(ordered) == self.n:
            components = self.graph.connected_components(mode=mode)
            return [sorted(component) for component in components]
        subgraph = self.graph.induced_subgraph(ordered)
        components = subgraph.connected_components(mode=mode)
        return [[ordered[local_index] for local_index in sorted(component)] for component in components]

    def classify_scc(self, vertex_count: int, edge_count: int) -> str:
        if edge_count == vertex_count:
            return "CYCLE"
        if edge_count > 2 * vertex_count:
            return "COMPLETE"
        return "SCC"

    def create_meta_node(
        self,
        meta_type: str,
        vertices: Sequence[int],
        parent: Optional[int],
        root_vertex: Optional[int],
        edge_count: Optional[int] = None,
    ) -> int:
        numeric_id = len(self.meta_nodes)
        node = MetaNode(
            numeric_id=numeric_id,
            id=f"meta_{numeric_id}",
            type=meta_type,
            vertices=sorted(vertices),
            edge_count=self.count_edges_inside(vertices) if edge_count is None else edge_count,
            parent=parent,
            root_vertex=root_vertex,
        )
        self.meta_nodes.append(node)
        return numeric_id

    def assign_owner(self, vertices: Iterable[int], meta_id: int) -> None:
        for vertex in vertices:
            self.vertex_owner[vertex] = meta_id

    def assign_fallback_isolated_vertices(self) -> None:
        for vertex, owner in enumerate(self.vertex_owner):
            if owner is not None:
                continue
            meta_id = self.create_meta_node("ISOLATED", [vertex], parent=None, root_vertex=vertex, edge_count=0)
            self.root_meta_ids.append(meta_id)
            self.assign_owner([vertex], meta_id)

    def count_edges_inside(self, vertices: Sequence[int]) -> int:
        vertex_set = set(vertices)
        count = 0
        for source in vertices:
            for target in self.out_adj[source]:
                if target in vertex_set:
                    count += 1
        return count

    def is_dag(self, vertices: Sequence[int]) -> bool:
        vertex_set = set(vertices)
        indegree = {vertex: 0 for vertex in vertex_set}
        for source in vertex_set:
            for target in self.out_adj[source]:
                if target in vertex_set:
                    indegree[target] += 1

        queue = deque(sorted(vertex for vertex, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            for target in self.out_adj[source]:
                if target in vertex_set:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
        return visited == len(vertex_set)

    def is_tree(self, vertices: Sequence[int]) -> bool:
        if len(vertices) <= 1 or not self.is_dag(vertices):
            return False
        vertex_set = set(vertices)
        undirected_edges = {
            tuple(sorted((source, target)))
            for source in vertex_set
            for target in self.out_adj[source]
            if target in vertex_set and source != target
        }
        return len(undirected_edges) == len(vertex_set) - 1

    def choose_root_vertex(self, vertices: Sequence[int]) -> int:
        vertex_set = set(vertices)
        indegree = {vertex: 0 for vertex in vertex_set}
        outdegree = {vertex: 0 for vertex in vertex_set}
        for source in vertex_set:
            for target in self.out_adj[source]:
                if target in vertex_set:
                    indegree[target] += 1
                    outdegree[source] += 1
        candidates = [vertex for vertex, degree in indegree.items() if degree == 0] or list(vertex_set)
        return min(candidates, key=lambda vertex: (-outdegree[vertex], vertex))

    def algebraic_bisect(self, vertices: Sequence[int]) -> Tuple[List[int], List[int]]:
        """Spectral-style bisection fallback using graph-geodesic anchors.

        For reproducibility and no SciPy dependency, this approximates an
        algebraic bisection by projecting vertices onto the line between two
        farthest geodesic anchors in the induced undirected graph, then applying
        local cut refinement.
        """
        ordered = sorted(vertices)
        if len(ordered) < 2:
            return ordered, []
        vertex_set = set(ordered)
        first = ordered[0]
        distances = self.bfs_distances(first, vertex_set)
        anchor_a = max(ordered, key=lambda vertex: distances.get(vertex, -1))
        distances_a = self.bfs_distances(anchor_a, vertex_set)
        anchor_b = max(ordered, key=lambda vertex: distances_a.get(vertex, -1))
        distances_b = self.bfs_distances(anchor_b, vertex_set)

        scored = []
        for vertex in ordered:
            da = distances_a.get(vertex, len(ordered))
            db = distances_b.get(vertex, len(ordered))
            scored.append((da - db, vertex))
        scored.sort()
        midpoint = len(scored) // 2
        left = {vertex for _, vertex in scored[:midpoint]}
        right = {vertex for _, vertex in scored[midpoint:]}
        if not left or not right:
            midpoint = len(ordered) // 2
            left = set(ordered[:midpoint])
            right = set(ordered[midpoint:])
        left, right = self.refine_cut(left, right, rounds=10)
        return sorted(left), sorted(right)

    def bfs_distances(self, start: int, vertex_set: Set[int]) -> Dict[int, int]:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            next_distance = distances[vertex] + 1
            for neighbor in self.undirected_neighbors(vertex):
                if neighbor in vertex_set and neighbor not in distances:
                    distances[neighbor] = next_distance
                    queue.append(neighbor)
        return distances

    def undirected_neighbors(self, vertex: int) -> Iterable[int]:
        yield from self.out_adj[vertex]
        yield from self.in_adj[vertex]

    def refine_cut(self, left: Set[int], right: Set[int], rounds: int) -> Tuple[Set[int], Set[int]]:
        if not left or not right:
            return left, right
        for _ in range(rounds):
            best_vertex: Optional[int] = None
            best_gain = 0.0
            best_destination_left = False
            for source_set, target_set, move_to_left in ((left, right, False), (right, left, True)):
                if len(source_set) <= 1:
                    continue
                for vertex in source_set:
                    external = sum(1 for neighbor in self.undirected_neighbors(vertex) if neighbor in target_set)
                    internal = sum(1 for neighbor in self.undirected_neighbors(vertex) if neighbor in source_set)
                    balance_penalty = abs((len(source_set) - 1) - (len(target_set) + 1)) * 0.05
                    gain = external - internal - balance_penalty
                    if gain > best_gain:
                        best_gain = gain
                        best_vertex = vertex
                        best_destination_left = move_to_left
            if best_vertex is None:
                break
            if best_destination_left:
                right.remove(best_vertex)
                left.add(best_vertex)
            else:
                left.remove(best_vertex)
                right.add(best_vertex)
        return left, right

    def compute_meta_edges(self) -> None:
        aggregate: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        for source, target in self.edges:
            source_meta = self.vertex_owner[source]
            target_meta = self.vertex_owner[target]
            if source_meta is None or target_meta is None or source_meta == target_meta:
                continue
            aggregate[(source_meta, target_meta)] += 1
        self.meta_edges = [
            {
                "source": self.meta_nodes[source].id,
                "target": self.meta_nodes[target].id,
                "source_numeric": source,
                "target_numeric": target,
                "edge_count": count,
            }
            for (source, target), count in sorted(aggregate.items())
        ]

    def compute_nested_layout(self) -> None:
        total_weight = sum(self.meta_weight(meta_id) for meta_id in self.root_meta_ids)
        root_width = max(4200.0, math.sqrt(max(1, self.n)) * 42.0)
        root_height = max(2600.0, root_width * 0.62, total_weight * 3.2)
        self.layout_meta_boxes(self.root_meta_ids, 0.0, 0.0, root_width, root_height, depth=0)

    def layout_meta_boxes(
        self,
        meta_ids: Sequence[int],
        x: float,
        y: float,
        width: float,
        height: float,
        depth: int,
    ) -> None:
        if not meta_ids:
            return
        ordered = sorted(meta_ids, key=lambda meta_id: (-self.meta_weight(meta_id), meta_id))
        total = sum(self.meta_weight(meta_id) for meta_id in ordered) or 1.0
        cursor = x if depth % 2 == 0 else y
        for index, meta_id in enumerate(ordered):
            weight = self.meta_weight(meta_id)
            if depth % 2 == 0:
                segment = width * weight / total
                box = (cursor, y, segment if index < len(ordered) - 1 else x + width - cursor, height)
                cursor += segment
            else:
                segment = height * weight / total
                box = (x, cursor, width, segment if index < len(ordered) - 1 else y + height - cursor)
                cursor += segment
            self.assign_meta_box(meta_id, box, depth)

    def assign_meta_box(self, meta_id: int, box: Tuple[float, float, float, float], depth: int) -> None:
        node = self.meta_nodes[meta_id]
        x, y, width, height = box
        pad = min(22.0, max(7.0, min(width, height) * 0.045))
        node.x = round(x + pad, 3)
        node.y = round(y + pad, 3)
        node.width = round(max(28.0, width - 2 * pad), 3)
        node.height = round(max(28.0, height - 2 * pad), 3)
        if node.children_meta:
            self.layout_meta_boxes(node.children_meta, node.x, node.y, node.width, node.height, depth + 1)
        else:
            self.layout_vertices_inside(node)

    def meta_weight(self, meta_id: int) -> float:
        return max(1.0, math.sqrt(max(1, len(self.meta_nodes[meta_id].vertices))))

    def layout_vertices_inside(self, node: MetaNode) -> None:
        if node.type in {"CYCLE", "COMPLETE"}:
            self.layout_circle(node)
        elif node.type == "TREE":
            self.layout_tree(node)
        elif node.type == "DAG":
            self.layout_dag(node)
        elif len(node.vertices) == 1:
            vertex = node.vertices[0]
            self.vertex_local_x[vertex] = node.width / 2.0
            self.vertex_local_y[vertex] = node.height / 2.0
        else:
            self.layout_phyllotaxis(node)

    def layout_circle(self, node: MetaNode) -> None:
        vertices = sorted(node.vertices)
        center_x = node.width / 2.0
        center_y = node.height / 2.0
        radius = max(1.0, min(node.width, node.height) * 0.43)
        for index, vertex in enumerate(vertices):
            angle = 2.0 * math.pi * index / max(1, len(vertices))
            self.vertex_local_x[vertex] = center_x + radius * math.cos(angle)
            self.vertex_local_y[vertex] = center_y + radius * math.sin(angle)

    def layout_tree(self, node: MetaNode) -> None:
        vertices = sorted(node.vertices)
        vertex_set = set(vertices)
        root = node.root_vertex if node.root_vertex in vertex_set else vertices[0]
        distances = self.bfs_distances(root, vertex_set)
        levels: DefaultDict[int, List[int]] = defaultdict(list)
        for vertex in vertices:
            levels[distances.get(vertex, 0)].append(vertex)
        max_level = max(levels) if levels else 0
        center_x = node.width / 2.0
        center_y = node.height / 2.0
        max_radius = min(node.width, node.height) * 0.44
        for level, ring_vertices in levels.items():
            radius = 0.0 if level == 0 else max_radius * level / max(1, max_level)
            for index, vertex in enumerate(sorted(ring_vertices)):
                angle = 2.0 * math.pi * index / max(1, len(ring_vertices))
                self.vertex_local_x[vertex] = center_x + radius * math.cos(angle)
                self.vertex_local_y[vertex] = center_y + radius * math.sin(angle)

    def layout_dag(self, node: MetaNode) -> None:
        vertices = sorted(node.vertices)
        vertex_set = set(vertices)
        indegree = {vertex: 0 for vertex in vertices}
        rank = {vertex: 0 for vertex in vertices}
        for source in vertices:
            for target in self.out_adj[source]:
                if target in vertex_set:
                    indegree[target] += 1
        queue = deque(sorted(vertex for vertex, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            for target in self.out_adj[source]:
                if target in vertex_set:
                    rank[target] = max(rank[target], rank[source] + 1)
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
        if visited != len(vertices):
            self.layout_phyllotaxis(node)
            return
        layers: DefaultDict[int, List[int]] = defaultdict(list)
        for vertex in vertices:
            layers[rank[vertex]].append(vertex)
        max_rank = max(layers) if layers else 0
        for layer, layer_vertices in layers.items():
            ordered = sorted(layer_vertices)
            x = node.width * (layer + 0.5) / (max_rank + 1)
            for row, vertex in enumerate(ordered):
                y = node.height * (row + 0.5) / max(1, len(ordered))
                self.vertex_local_x[vertex] = min(node.width - 2.0, max(2.0, x))
                self.vertex_local_y[vertex] = min(node.height - 2.0, max(2.0, y))

    def layout_phyllotaxis(self, node: MetaNode) -> None:
        vertices = sorted(node.vertices)
        center_x = node.width / 2.0
        center_y = node.height / 2.0
        radius = min(node.width, node.height) * 0.45
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index, vertex in enumerate(vertices):
            scaled_radius = radius * math.sqrt((index + 0.5) / max(1, len(vertices)))
            angle = index * golden_angle
            self.vertex_local_x[vertex] = center_x + scaled_radius * math.cos(angle)
            self.vertex_local_y[vertex] = center_y + scaled_radius * math.sin(angle)

    def root_extent(self) -> Tuple[float, float]:
        width = 0.0
        height = 0.0
        for meta_id in self.root_meta_ids:
            node = self.meta_nodes[meta_id]
            width = max(width, node.x + node.width)
            height = max(height, node.y + node.height)
        return round(width, 3), round(height, 3)

    def meta_to_tree(self, meta_id: int) -> Dict[str, object]:
        node = self.meta_nodes[meta_id]
        if node.children_meta:
            children = [self.meta_to_tree(child_id) for child_id in node.children_meta]
        else:
            children = [self.leaf_to_tree(vertex, node) for vertex in node.vertices]
        return {
            "id": node.id,
            "kind": "meta",
            "type": node.type,
            "numeric_id": node.numeric_id,
            "x": node.x,
            "y": node.y,
            "width": node.width,
            "height": node.height,
            "cx": round(node.x + node.width / 2.0, 3),
            "cy": round(node.y + node.height / 2.0, 3),
            "vertex_count": len(node.vertices),
            "edge_count": node.edge_count,
            "parent": self.meta_nodes[node.parent].id if node.parent is not None else "root_graph",
            "children": children,
        }

    def leaf_to_tree(self, vertex: int, owner: MetaNode) -> Dict[str, object]:
        local_x = self.vertex_local_x[vertex]
        local_y = self.vertex_local_y[vertex]
        return {
            "id": self.vertex_ids[vertex],
            "kind": "leaf",
            "type": self.vertex_types[vertex],
            "vertex_index": vertex,
            "meta_id": owner.id,
            "local_x": round(local_x, 3),
            "local_y": round(local_y, 3),
            "x": round(owner.x + local_x, 3),
            "y": round(owner.y + local_y, 3),
            "cx": round(owner.x + local_x, 3),
            "cy": round(owner.y + local_y, 3),
            "children": [],
        }

    def to_graph_data(self, case_id: str, title: str) -> Dict[str, object]:
        width, height = self.root_extent()
        tree = {
            "id": "root_graph",
            "kind": "root",
            "type": "MACRO_FLOW",
            "x": 0.0,
            "y": 0.0,
            "width": width,
            "height": height,
            "cx": round(width / 2.0, 3),
            "cy": round(height / 2.0, 3),
            "children": [self.meta_to_tree(meta_id) for meta_id in self.root_meta_ids],
        }

        return {
            "format": "topolayout-dg-heb-v1",
            "case_id": case_id,
            "title": title,
            "meta_types": META_TYPES,
            "meta_colors": META_COLORS,
            "stats": {
                "vertices": self.n,
                "edges": len(self.edges),
                "meta_nodes": len(self.meta_nodes),
                "meta_edges": len(self.meta_edges),
                "scc_limit": self.scc_limit,
            },
            "tree": tree,
            "edges": {
                "source": [self.vertex_ids[source] for source, _ in self.edges],
                "target": [self.vertex_ids[target] for _, target in self.edges],
            },
            "meta_edges": self.meta_edges,
        }


def read_citation_graph(path: Path) -> Tuple[List[str], List[Tuple[int, int]]]:
    temp_path = ""
    try:
        with path.open("r", encoding="utf-8") as infile:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".ncol") as temp:
                temp_path = temp.name
                for line in infile:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        temp.write(stripped)
                        temp.write("\n")
        graph = ig.Graph.Read_Ncol(temp_path, directed=True)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    vertex_ids = [str(name) for name in graph.vs["name"]]
    edges = [(int(source), int(target)) for source, target in graph.get_edgelist()]
    return vertex_ids, edges


def write_topolayout_json(builder: TopoLayoutBuilder, output_path: Path, case_id: str, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(builder.to_graph_data(case_id, title), handle, separators=(",", ":"))


def run_pipeline(input_path: Path, output_path: Path, scc_limit: int) -> None:
    started = time.perf_counter()
    vertex_ids, edges = read_citation_graph(input_path)
    builder = TopoLayoutBuilder(vertex_ids, edges, scc_limit=scc_limit)
    builder.build()
    write_topolayout_json(builder, output_path, "case_1_citation", "Case 1: cit-HepPh HEB")
    elapsed = time.perf_counter() - started
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Case 1 complete: {len(vertex_ids):,} vertices, {len(edges):,} directed edges")
    print(f"MetaNodes: {len(builder.meta_nodes):,}; MetaEdges: {len(builder.meta_edges):,}")
    print(f"Wrote {output_path} ({size_mb:.2f} MiB) in {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parents[1] / "cit-HepPh.txt"
    default_output = Path(__file__).resolve().parent / "graph_data.json"
    parser = argparse.ArgumentParser(description="Run ToF2DG + HEB tree export over SNAP cit-HepPh.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--scc-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input, args.output, args.scc_limit)


if __name__ == "__main__":
    main()
