# TopoLayout-DG Case Studies

This repository contains two reproducible TopoLayout-DG case studies with
hierarchical edge bundling:

- Case 1: macro-topology validation on SNAP `cit-HepPh`.
- Case 2: PyTorch -> ONNX -> ToF2DG profiling of a live Transformer layer.

The backend exports `graph_data.json` with two primary structures:

- `tree`: the strict MetaNode hierarchy down to leaf vertices, including nested
  bounding boxes and local coordinates.
- `edges`: raw directed leaf connectivity as source and target leaf ID arrays.

The frontend parses the tree, computes lowest-common-ancestor control paths for
Danny Holten-style hierarchical edge bundling, samples cubic B-splines, and
renders red-to-green directed flow curves through PixiJS WebGL meshes. Click a
MetaNode cell to drill down; press `Escape` or click empty space to return to
the global bundled view.

## Setup

```bash
conda env create -f environment.yml
conda activate topolayout-dg
```

## Run

```bash
./run_case_studies.sh
```

The script regenerates both case payloads and serves:

```text
Case 1: http://127.0.0.1:8001/index.html
Case 2: http://127.0.0.1:8002/index.html
```

It keeps both servers alive until `Ctrl-C`.

## Sample Payload

`case_1_citation/graph_data.json` is a committed lightweight sample with 100
nodes and 500 directed edges. It lets the PixiJS HEB visualization open
immediately after cloning. The full `cit-HepPh.txt` dataset, generated ONNX
models, and large generated `graph_data.json` files are excluded by
`.gitignore`.

## Structure

```text
topolayout-dg/
+-- environment.yml
+-- run_case_studies.sh
+-- case_1_citation/
|   +-- pipeline.py
|   +-- index.html
|   +-- graph_data.json
+-- case_2_transformer/
    +-- pipeline.py
    +-- index.html
    +-- graph_data.json
    +-- transformer_layer.onnx
```
