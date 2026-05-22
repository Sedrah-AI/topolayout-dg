#!/usr/bin/env python3
"""Case 2: PyTorch -> ONNX -> ToF2DG over a Transformer layer."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_1_citation.pipeline import TopoLayoutBuilder, write_topolayout_json


def export_transformer_onnx(output_path: Path, sequence_length: int, hidden_size: int, heads: int) -> None:
    try:
        import onnx  # noqa: F401
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(
            "Case 2 requires pytorch and onnx. Create the Conda environment with "
            "`conda env create -f environment.yml` and run inside `conda activate topolayout-dg`."
        ) from exc

    class LiveTransformerLayer(nn.Module):
        def __init__(self, width: int, num_heads: int) -> None:
            super().__init__()
            if width % num_heads != 0:
                raise ValueError("hidden_size must be divisible by heads")
            self.width = width
            self.num_heads = num_heads
            self.head_dim = width // num_heads
            self.qkv = nn.Linear(width, width * 3)
            self.proj = nn.Linear(width, width)
            self.norm_1 = nn.LayerNorm(width)
            self.fc_1 = nn.Linear(width, width * 4)
            self.fc_2 = nn.Linear(width * 4, width)
            self.norm_2 = nn.LayerNorm(width)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            batch, seq, width = tokens.shape
            qkv = self.qkv(tokens)
            qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query = qkv[0]
            key = qkv[1]
            value = qkv[2]
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, value)
            context = context.transpose(1, 2).reshape(batch, seq, width)
            attention = self.proj(context)
            residual_1 = self.norm_1(tokens + attention)
            hidden = self.fc_2(F.gelu(self.fc_1(residual_1)))
            return self.norm_2(residual_1 + hidden)

    torch.manual_seed(7)
    model = LiveTransformerLayer(hidden_size, heads).eval()
    dummy_tokens = torch.randn(1, sequence_length, hidden_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_tokens,
        output_path,
        input_names=["tokens"],
        output_names=["encoded"],
        opset_version=17,
        do_constant_folding=True,
    )


def onnx_to_operation_graph(onnx_path: Path) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {initializer.name for initializer in graph.initializer}

    vertex_ids: List[str] = []
    vertex_types: List[str] = []
    edges: List[Tuple[int, int]] = []
    tensor_producer: Dict[str, int] = {}

    def add_vertex(vertex_id: str, vertex_type: str) -> int:
        vertex_ids.append(vertex_id)
        vertex_types.append(vertex_type)
        return len(vertex_ids) - 1

    for graph_input in graph.input:
        if graph_input.name in initializers:
            continue
        vertex_index = add_vertex(f"input_{graph_input.name}", "TensorInput")
        tensor_producer[graph_input.name] = vertex_index

    node_indices: List[int] = []
    for index, node in enumerate(graph.node):
        base_name = node.name if node.name else f"{node.op_type}_{index}"
        vertex_index = add_vertex(f"op_{index}_{base_name}", node.op_type)
        node_indices.append(vertex_index)
        for tensor_name in node.input:
            if tensor_name in tensor_producer:
                edges.append((tensor_producer[tensor_name], vertex_index))
        for tensor_name in node.output:
            if tensor_name:
                tensor_producer[tensor_name] = vertex_index

    for graph_output in graph.output:
        vertex_index = add_vertex(f"output_{graph_output.name}", "TensorOutput")
        if graph_output.name in tensor_producer:
            edges.append((tensor_producer[graph_output.name], vertex_index))

    if not edges and len(vertex_ids) > 1:
        edges.extend((index, index + 1) for index in range(len(vertex_ids) - 1))

    return vertex_ids, vertex_types, list(dict.fromkeys(edges))


def run_pipeline(
    output_path: Path,
    onnx_path: Path,
    sequence_length: int,
    hidden_size: int,
    heads: int,
    scc_limit: int,
) -> None:
    started = time.perf_counter()
    export_transformer_onnx(onnx_path, sequence_length, hidden_size, heads)
    vertex_ids, vertex_types, edges = onnx_to_operation_graph(onnx_path)
    builder = TopoLayoutBuilder(vertex_ids, edges, vertex_types=vertex_types, scc_limit=scc_limit)
    builder.build()
    write_topolayout_json(builder, output_path, "case_2_transformer", "Case 2: Transformer Layer")
    elapsed = time.perf_counter() - started
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Case 2 complete: {len(vertex_ids):,} operation vertices, {len(edges):,} tensor-flow edges")
    print(f"MetaNodes: {len(builder.meta_nodes):,}; MetaEdges: {len(builder.meta_edges):,}")
    print(f"Wrote {output_path} ({size_mb:.2f} MiB) in {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    case_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export a Transformer layer to ONNX and run ToF2DG.")
    parser.add_argument("--output", type=Path, default=case_dir / "graph_data.json")
    parser.add_argument("--onnx", type=Path, default=case_dir / "transformer_layer.onnx")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--scc-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        output_path=args.output,
        onnx_path=args.onnx,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        heads=args.heads,
        scc_limit=args.scc_limit,
    )


if __name__ == "__main__":
    main()
