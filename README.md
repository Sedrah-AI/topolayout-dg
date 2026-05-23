# TopoLayout-DG Case Studies

This repository contains three reproducible TopoLayout-DG case studies with
hierarchical edge bundling:

- Case 1: macro-topology validation on SNAP `cit-HepPh`.
- Case 2: PyTorch -> ONNX -> ToF2DG profiling of a live Transformer layer.
- Case 3: LayMan-style PyTorch image-classifier hierarchy plus failure
  propagation tracing down to sampled activation-neuron and weight elements.

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

The script regenerates the available case payloads and serves:

```text
Case 1: http://127.0.0.1:8001/index.html
Case 2: http://127.0.0.1:8002/index.html
Case 3: http://127.0.0.1:8003/index.html
```

It keeps the local servers alive until `Ctrl-C`.
If `cit-HepPh.txt` is not present, Case 1 serves the committed sample payload
and still runs Cases 2 and 3.

Case 3 also includes a fully self-contained Cytoscape/Tailwind LayMan GUI with
an embedded image-classifier schema:

```text
http://127.0.0.1:8003/layman_cytoscape.html
```

## Sample Payload

`case_1_citation/sample_graph_data.json` is a committed lightweight sample with
100 nodes and 500 directed edges. It lets the PixiJS HEB visualization open
without committing the full dataset output.

```text
http://127.0.0.1:8001/index.html?data=sample_graph_data.json
```

The full `cit-HepPh.txt` dataset, generated ONNX models, and large generated
`graph_data.json` files are excluded by `.gitignore`.

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
+-- case_3_image_classifier/
    +-- pipeline.py
    +-- index.html
    +-- layman_cytoscape.html
    +-- graph_data.json
```
