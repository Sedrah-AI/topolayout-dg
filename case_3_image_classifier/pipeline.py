#!/usr/bin/env python3
"""LayMan-style hierarchical graph prototype for a residual image classifier.

The script traces a PyTorch model with torch.fx, records runtime tensor values
and gradients, builds a structural module tree, overlays execution dataflow,
and exports a browser visualization that highlights failure propagation paths
down to sampled activation-neuron and weight-element nodes.

Cycle removal is intentionally skipped for this prototype. The traced execution
graph is assumed to be a DAG.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import operator
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.fx import GraphModule, Interpreter, Node, symbolic_trace
except ImportError as exc:
    raise SystemExit(
        "case_3_image_classifier requires PyTorch. Create the environment with "
        "`conda env create -f environment.yml` and activate `topolayout-dg`."
    ) from exc


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            ResidualBlock(16)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class RecordingInterpreter(Interpreter):
    """Runs an FX graph while retaining intermediate tensor gradients."""

    def __init__(self, module: GraphModule) -> None:
        super().__init__(module)
        self.values: Dict[str, Any] = {}

    def run_node(self, node: Node) -> Any:
        result = super().run_node(node)
        self.values[node.name] = result
        if isinstance(result, torch.Tensor) and result.requires_grad:
            result.retain_grad()
        return result


def safe_name(value: Any) -> str:
    text = str(value)
    return text.replace(" ", "_").replace("/", "_").replace(":", "_")


def module_node_id(path: str) -> str:
    return "module:root" if path == "" else f"module:{path}"


def op_node_id(name: str) -> str:
    return f"op:{name}"


def param_node_id(name: str) -> str:
    return f"param:{name}"


def weight_node_id(name: str, rank: int) -> str:
    return f"weight:{name}:{rank}"


def neuron_node_id(name: str, rank: int) -> str:
    return f"neuron:{name}:{rank}"


def shape_of(value: Any) -> Optional[List[int]]:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def tensor_stats(value: Any) -> Dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {}
    detached = value.detach()
    stats: Dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "mean": float(detached.float().mean().item()),
        "mean_abs": float(detached.float().abs().mean().item()),
        "max_abs": float(detached.float().abs().max().item()),
        "nan_count": int(torch.isnan(detached).sum().item()) if detached.is_floating_point() else 0,
    }
    grad = getattr(value, "grad", None)
    if isinstance(grad, torch.Tensor):
        grad_detached = grad.detach()
        stats.update(
            {
                "grad_mean_abs": float(grad_detached.float().abs().mean().item()),
                "grad_max_abs": float(grad_detached.float().abs().max().item()),
            }
        )
    else:
        stats.update({"grad_mean_abs": 0.0, "grad_max_abs": 0.0})
    stats["failure_score"] = stats["mean_abs"] * stats["grad_mean_abs"] + stats["grad_max_abs"]
    return stats


def iter_node_dependencies(value: Any) -> Iterable[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_node_dependencies(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_node_dependencies(item)


def common_module_parent(paths: Sequence[str]) -> str:
    clean = [path for path in paths if path]
    if not clean:
        return ""
    split_paths = [path.split(".") for path in clean]
    prefix: List[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    if prefix:
        return ".".join(prefix)
    longest = max(clean, key=lambda value: value.count("."))
    return ".".join(longest.split(".")[:-1])


def flatten_topk(tensor: torch.Tensor, k: int) -> List[Tuple[float, Tuple[int, ...]]]:
    if tensor.numel() == 0:
        return []
    flat = tensor.detach().float().abs().reshape(-1)
    count = min(k, flat.numel())
    values, indices = torch.topk(flat, count)
    result: List[Tuple[float, Tuple[int, ...]]] = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        unravelled = []
        remainder = int(flat_index)
        for dim in reversed(tensor.shape):
            unravelled.append(remainder % int(dim))
            remainder //= int(dim)
        result.append((float(value), tuple(reversed(unravelled))))
    return result


def add_child(tree_nodes: Dict[str, Dict[str, Any]], parent_id: str, child: Dict[str, Any]) -> None:
    tree_nodes[child["id"]] = child
    tree_nodes[parent_id].setdefault("children", []).append(child)


def create_tree_node(
    node_id: str,
    label: str,
    kind: str,
    node_type: str,
    module_path: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "kind": kind,
        "type": node_type,
        "module_path": module_path,
        "details": details or {},
        "children": [],
    }


def build_module_tree(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    tree_nodes: Dict[str, Dict[str, Any]] = {}
    root = create_tree_node("module:root", "ImageClassifier", "module", model.__class__.__name__, "")
    tree_nodes[root["id"]] = root

    def visit(module: nn.Module, path: str, parent_id: str) -> None:
        for name, child in module.named_children():
            child_path = name if path == "" else f"{path}.{name}"
            node = create_tree_node(
                module_node_id(child_path),
                child_path,
                "module",
                child.__class__.__name__,
                child_path,
                {"parameter_count": sum(param.numel() for param in child.parameters(recurse=False))},
            )
            add_child(tree_nodes, parent_id, node)
            visit(child, child_path, node["id"])

    visit(model, "", root["id"])
    return tree_nodes


def layout_tree(node: Dict[str, Any], x: float, y: float, width: float, height: float, depth: int = 0) -> None:
    node["x"] = round(x, 3)
    node["y"] = round(y, 3)
    node["width"] = round(width, 3)
    node["height"] = round(height, 3)
    node["cx"] = round(x + width / 2.0, 3)
    node["cy"] = round(y + height / 2.0, 3)

    children = node.get("children", [])
    if not children:
        return

    pad = min(22.0, max(8.0, min(width, height) * 0.05))
    inner_x = x + pad
    inner_y = y + pad + 18.0
    inner_w = max(8.0, width - 2 * pad)
    inner_h = max(8.0, height - 2 * pad - 18.0)
    weights = [subtree_weight(child) for child in children]
    total = sum(weights) or 1.0
    cursor = inner_x if depth % 2 == 0 else inner_y

    for index, child in enumerate(children):
        fraction = weights[index] / total
        if depth % 2 == 0:
            segment = inner_w * fraction
            child_w = segment if index < len(children) - 1 else inner_x + inner_w - cursor
            layout_tree(child, cursor, inner_y, max(8.0, child_w), inner_h, depth + 1)
            cursor += segment
        else:
            segment = inner_h * fraction
            child_h = segment if index < len(children) - 1 else inner_y + inner_h - cursor
            layout_tree(child, inner_x, cursor, inner_w, max(8.0, child_h), depth + 1)
            cursor += segment


def subtree_weight(node: Dict[str, Any]) -> float:
    children = node.get("children", [])
    if not children:
        return 1.0
    return max(1.0, sum(subtree_weight(child) for child in children))


def path_to_output(start: str, output_id: str, adjacency: Dict[str, List[str]]) -> List[str]:
    queue = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == output_id:
            return path
        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(path + [neighbor])
    return [start]


def build_layman_graph(
    model: ImageClassifier,
    traced: GraphModule,
    values: Dict[str, Any],
    loss: torch.Tensor,
    target_class: int,
    top_k: int,
) -> Dict[str, Any]:
    named_modules = dict(model.named_modules())
    tree_nodes = build_module_tree(model)
    fx_owner: Dict[str, str] = {}
    op_nodes: Dict[str, Dict[str, Any]] = {}
    dataflow_edges: List[Dict[str, Any]] = []
    forward_adjacency: DefaultDict[str, List[str]] = defaultdict(list)
    call_module_nodes_by_target: DefaultDict[str, List[str]] = defaultdict(list)

    for fx_node in traced.graph.nodes:
        node_id = op_node_id(fx_node.name)
        dependencies = list(iter_node_dependencies(fx_node.args)) + list(iter_node_dependencies(fx_node.kwargs))

        if fx_node.op == "placeholder":
            owner = ""
            label = "input image"
            node_type = "Input"
        elif fx_node.op == "output":
            owner = ""
            label = "classifier logits"
            node_type = "Output"
        elif fx_node.op == "call_module":
            owner = str(fx_node.target)
            module = named_modules[owner]
            label = f"{owner} ({module.__class__.__name__})"
            node_type = module.__class__.__name__
            call_module_nodes_by_target[owner].append(node_id)
        elif fx_node.op == "call_function":
            dependency_owners = [fx_owner.get(dep.name, "") for dep in dependencies]
            owner = common_module_parent(dependency_owners)
            label = getattr(fx_node.target, "__name__", safe_name(fx_node.target))
            node_type = "Function"
        elif fx_node.op == "call_method":
            dependency_owners = [fx_owner.get(dep.name, "") for dep in dependencies]
            owner = common_module_parent(dependency_owners)
            label = str(fx_node.target)
            node_type = "Method"
        else:
            owner = ""
            label = fx_node.name
            node_type = fx_node.op

        fx_owner[fx_node.name] = owner
        stats = tensor_stats(values.get(fx_node.name))
        op_node = create_tree_node(node_id, label, "op", node_type, owner, {"fx_op": fx_node.op, **stats})
        op_nodes[node_id] = op_node
        add_child(tree_nodes, module_node_id(owner), op_node)

        for dep in dependencies:
            source_id = op_node_id(dep.name)
            if source_id == node_id:
                continue
            edge = {
                "source": source_id,
                "target": node_id,
                "kind": "activation",
                "score": float(stats.get("failure_score", 0.0)),
            }
            dataflow_edges.append(edge)
            forward_adjacency[source_id].append(node_id)

    output_node_id = "op:output"

    for param_name, parameter in model.named_parameters():
        owner_path = ".".join(param_name.split(".")[:-1])
        grad = parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter.detach())
        grad_abs = grad.float().abs()
        param_score = float(grad_abs.norm().item())
        param_node = create_tree_node(
            param_node_id(param_name),
            param_name.split(".")[-1],
            "param",
            "Parameter",
            owner_path,
            {
                "full_name": param_name,
                "shape": list(parameter.shape),
                "value_norm": float(parameter.detach().float().norm().item()),
                "grad_norm": param_score,
                "grad_max_abs": float(grad_abs.max().item()) if grad_abs.numel() else 0.0,
                "failure_score": param_score,
            },
        )
        add_child(tree_nodes, module_node_id(owner_path), param_node)

        for rank, (score, index_tuple) in enumerate(flatten_topk(grad_abs, top_k)):
            weight_node = create_tree_node(
                weight_node_id(param_name, rank),
                f"w{list(index_tuple)}",
                "weight",
                "WeightElement",
                owner_path,
                {
                    "parameter": param_name,
                    "index": list(index_tuple),
                    "grad_abs": score,
                    "failure_score": score,
                },
            )
            add_child(tree_nodes, param_node["id"], weight_node)
            dataflow_edges.append({"source": weight_node["id"], "target": param_node["id"], "kind": "weight-element", "score": score})
            forward_adjacency[weight_node["id"]].append(param_node["id"])

        for target_op in call_module_nodes_by_target.get(owner_path, []):
            dataflow_edges.append({"source": param_node["id"], "target": target_op, "kind": "parameter", "score": param_score})
            forward_adjacency[param_node["id"]].append(target_op)

    for fx_name, value in values.items():
        if fx_name == "output" or not isinstance(value, torch.Tensor):
            continue
        grad = getattr(value, "grad", None)
        if not isinstance(grad, torch.Tensor):
            continue
        influence = (value.detach().float().abs() * grad.detach().float().abs()).reshape(value.shape)
        owner = fx_owner.get(fx_name, "")
        op_id = op_node_id(fx_name)
        for rank, (score, index_tuple) in enumerate(flatten_topk(influence, top_k)):
            neuron_node = create_tree_node(
                neuron_node_id(fx_name, rank),
                f"a{list(index_tuple)}",
                "neuron",
                "ActivationNeuron",
                owner,
                {
                    "fx_node": fx_name,
                    "activation_index": list(index_tuple),
                    "influence": score,
                    "failure_score": score,
                },
            )
            add_child(tree_nodes, op_id, neuron_node)
            dataflow_edges.append({"source": neuron_node["id"], "target": op_id, "kind": "neuron", "score": score})
            forward_adjacency[neuron_node["id"]].append(op_id)

    failure_seeds = []
    for node in tree_nodes.values():
        score = float(node.get("details", {}).get("failure_score", 0.0))
        if node["kind"] in {"weight", "neuron", "param"} and score > 0.0:
            failure_seeds.append((score, node["id"], node["kind"], node["label"]))
    failure_seeds.sort(reverse=True)

    failure_paths = []
    for rank, (score, seed_id, kind, label) in enumerate(failure_seeds[:12]):
        path = path_to_output(seed_id, output_node_id, forward_adjacency)
        failure_paths.append(
            {
                "rank": rank + 1,
                "source": seed_id,
                "source_label": label,
                "source_kind": kind,
                "target": output_node_id,
                "score": score,
                "nodes": path,
                "edges": [[path[index], path[index + 1]] for index in range(len(path) - 1)],
            }
        )

    root = tree_nodes["module:root"]
    layout_tree(root, 0.0, 0.0, 1800.0, 1120.0)

    logits = values["output"].detach()
    probabilities = F.softmax(logits, dim=-1)
    predicted_class = int(torch.argmax(probabilities, dim=-1).item())
    target_probability = float(probabilities[0, target_class].item())
    predicted_probability = float(probabilities[0, predicted_class].item())

    return {
        "format": "layman-image-classifier-v1",
        "case_id": "case_3_image_classifier",
        "title": "Case 3: LayMan Image Classifier Failure Trace",
        "assumption": "DAG execution graph; cycle removal preprocessing skipped.",
        "model": {
            "name": "ImageClassifier",
            "input_shape": [1, 3, 32, 32],
            "target_class": target_class,
            "predicted_class": predicted_class,
            "target_probability": target_probability,
            "predicted_probability": predicted_probability,
            "loss": float(loss.detach().item()),
        },
        "stats": {
            "tree_nodes": len(tree_nodes),
            "dataflow_edges": len(dataflow_edges),
            "failure_paths": len(failure_paths),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "tree": root,
        "dataflow_edges": dataflow_edges,
        "failure_paths": failure_paths,
        "execution_order": [op_node_id(node.name) for node in traced.graph.nodes],
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LayMan Image Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { width: 100%; height: 100%; overflow: hidden; background: #111317; color: #eef2f7; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        #stage { display: block; width: 100vw; height: 100vh; background: #111317; }
        #panel { position: fixed; top: 16px; left: 16px; width: min(380px, calc(100vw - 32px)); padding: 14px; border: 1px solid rgba(255,255,255,0.16); border-radius: 8px; background: rgba(18,21,27,0.9); box-shadow: 0 18px 46px rgba(0,0,0,0.38); backdrop-filter: blur(14px); }
        #title { margin-bottom: 10px; font-size: 13px; font-weight: 700; color: #fff; }
        .row { display: grid; grid-template-columns: 1fr auto; gap: 12px; padding: 6px 0; border-top: 1px solid rgba(255,255,255,0.08); font-size: 12px; line-height: 1.25; }
        .label { color: #aeb7c3; }
        .value { color: #f8fafc; font-variant-numeric: tabular-nums; text-align: right; }
        #details { margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); color: #d7dee8; font-size: 12px; line-height: 1.45; max-height: 34vh; overflow: auto; }
        #loading { position: fixed; inset: 0; display: grid; place-items: center; background: #111317; color: #dbe3ee; font-size: 13px; z-index: 10; }
        #loading[hidden] { display: none; }
    </style>
</head>
<body>
    <canvas id="stage"></canvas>
    <div id="loading">Loading graph_data.json</div>
    <aside id="panel">
        <div id="title">LayMan Image Classifier</div>
        <div class="row"><span class="label">Loss</span><span class="value" id="loss">0</span></div>
        <div class="row"><span class="label">Prediction</span><span class="value" id="prediction">0</span></div>
        <div class="row"><span class="label">Tree Nodes</span><span class="value" id="tree-nodes">0</span></div>
        <div class="row"><span class="label">Dataflow Edges</span><span class="value" id="dataflow-edges">0</span></div>
        <div class="row"><span class="label">Failure Paths</span><span class="value" id="failure-paths">0</span></div>
        <div id="details">Click any module, op, neuron, or weight node.</div>
    </aside>
    <script src="https://cdn.jsdelivr.net/npm/pixi.js@7.4.2/dist/pixi.min.js"></script>
    <script>
        const app = new PIXI.Application({ view: document.getElementById("stage"), resizeTo: window, backgroundColor: 0x111317, antialias: true, autoDensity: true, resolution: window.devicePixelRatio || 1 });
        const world = new PIXI.Container();
        const edgeLayer = new PIXI.Container();
        const boxLayer = new PIXI.Container();
        const nodeLayer = new PIXI.Container();
        world.addChild(edgeLayer, boxLayer, nodeLayer);
        app.stage.addChild(world);
        app.stage.interactive = true;
        app.stage.hitArea = app.screen;

        const nodeById = new Map();
        const visualNodes = [];
        let graphData = null;
        let camera = { scale: 1, x: 0, y: 0 };

        const colors = {
            module: 0x64748b, op: 0x3b82f6, param: 0xf59e0b,
            weight: 0xef4444, neuron: 0x22c55e
        };

        function indexTree(node) {
            nodeById.set(node.id, node);
            visualNodes.push(node);
            for (const child of node.children || []) indexTree(child);
        }

        function setCameraToRoot(root) {
            const padding = 60;
            camera.scale = Math.min((app.screen.width - padding * 2) / root.width, (app.screen.height - padding * 2) / root.height);
            camera.scale = Math.max(0.05, camera.scale);
            camera.x = app.screen.width / 2 - (root.x + root.width / 2) * camera.scale;
            camera.y = app.screen.height / 2 - (root.y + root.height / 2) * camera.scale;
            world.scale.set(camera.scale);
            world.position.set(camera.x, camera.y);
        }

        function drawTree(node, depth = 0) {
            const color = colors[node.kind] || 0x94a3b8;
            if (node.kind === "module" || node.kind === "op" || node.kind === "param") {
                const g = new PIXI.Graphics();
                g.lineStyle(Math.max(1 / camera.scale, 1.2), color, node.kind === "module" ? 0.55 : 0.85);
                g.beginFill(color, node.kind === "module" ? 0.035 : 0.08);
                g.drawRect(node.x, node.y, node.width, node.height);
                g.endFill();
                g.interactive = true;
                g.cursor = "pointer";
                g.on("pointertap", (event) => { event.stopPropagation(); showDetails(node); });
                boxLayer.addChild(g);
                if (node.width > 70 && node.height > 28) {
                    const label = new PIXI.Text(node.label, { fontFamily: "Inter, Arial", fontSize: 10, fill: 0xe5edf8 });
                    label.x = node.x + 6;
                    label.y = node.y + 5;
                    label.eventMode = "none";
                    boxLayer.addChild(label);
                }
            } else {
                const g = new PIXI.Graphics();
                const radius = node.kind === "weight" ? 3.2 : 3.8;
                g.beginFill(color, 0.95);
                g.drawCircle(node.cx, node.cy, radius / Math.max(camera.scale, 0.1));
                g.endFill();
                g.interactive = true;
                g.cursor = "pointer";
                g.on("pointertap", (event) => { event.stopPropagation(); showDetails(node); });
                nodeLayer.addChild(g);
            }
            for (const child of node.children || []) drawTree(child, depth + 1);
        }

        function failureEdgeSet() {
            const result = new Set();
            for (const path of graphData.failure_paths) {
                for (const edge of path.edges) result.add(edge[0] + "->" + edge[1]);
            }
            return result;
        }

        function drawEdges() {
            const hot = failureEdgeSet();
            const g = new PIXI.Graphics();
            for (const edge of graphData.dataflow_edges) {
                const source = nodeById.get(edge.source);
                const target = nodeById.get(edge.target);
                if (!source || !target) continue;
                const key = edge.source + "->" + edge.target;
                const highlighted = hot.has(key);
                g.lineStyle(highlighted ? 2.8 / camera.scale : 1.0 / camera.scale, highlighted ? 0xff4d4d : 0x5eead4, highlighted ? 0.95 : 0.22);
                const dx = target.cx - source.cx;
                const c1x = source.cx + dx * 0.45;
                const c2x = source.cx + dx * 0.55;
                g.moveTo(source.cx, source.cy);
                g.bezierCurveTo(c1x, source.cy, c2x, target.cy, target.cx, target.cy);
            }
            edgeLayer.addChild(g);
        }

        function showDetails(node) {
            const escaped = (value) => String(value).replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]));
            const details = Object.entries(node.details || {}).map(([key, value]) => `<div><span class="label">${escaped(key)}</span>: ${escaped(JSON.stringify(value))}</div>`).join("");
            document.getElementById("details").innerHTML = `<strong>${escaped(node.label)}</strong><br>${escaped(node.kind)} / ${escaped(node.type)}<br>${details}`;
        }

        async function boot() {
            const response = await fetch("graph_data.json", { cache: "no-store" });
            if (!response.ok) throw new Error(`graph_data.json returned HTTP ${response.status}`);
            graphData = await response.json();
            indexTree(graphData.tree);
            setCameraToRoot(graphData.tree);
            drawEdges();
            drawTree(graphData.tree);
            document.getElementById("loss").textContent = graphData.model.loss.toFixed(5);
            document.getElementById("prediction").textContent = `${graphData.model.predicted_class} / target ${graphData.model.target_class}`;
            document.getElementById("tree-nodes").textContent = graphData.stats.tree_nodes;
            document.getElementById("dataflow-edges").textContent = graphData.stats.dataflow_edges;
            document.getElementById("failure-paths").textContent = graphData.stats.failure_paths;
            document.getElementById("loading").hidden = true;
        }

        boot().catch((error) => {
            document.getElementById("loading").textContent = error.message;
            console.error(error);
        });
    </script>
</body>
</html>
"""


def write_html(path: Path) -> None:
    path.write_text(HTML_TEMPLATE, encoding="utf-8")


def run_pipeline(output_path: Path, html_path: Path, seed: int, target_class: int, top_k: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    model = ImageClassifier()
    model.train()
    traced = symbolic_trace(model)

    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    target = torch.tensor([target_class], dtype=torch.long)
    interpreter = RecordingInterpreter(traced)
    logits = interpreter.run(image)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    graph_data = build_layman_graph(model, traced, interpreter.values, loss, target_class, top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
    write_html(html_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {html_path}")
    print(f"Tree nodes: {graph_data['stats']['tree_nodes']}")
    print(f"Dataflow edges: {graph_data['stats']['dataflow_edges']}")
    print(f"Failure paths: {graph_data['stats']['failure_paths']}")


def parse_args() -> argparse.Namespace:
    case_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Trace ImageClassifier and export a LayMan failure visualizer.")
    parser.add_argument("--output", type=Path, default=case_dir / "graph_data.json")
    parser.add_argument("--html", type=Path, default=case_dir / "index.html")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-class", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2, help="Activation/weight elements retained per tensor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.output, args.html, args.seed, args.target_class, args.top_k)


if __name__ == "__main__":
    main()
