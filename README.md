# RECONTEXT: 3D Scene Reconstruction with Semantic Understanding

<div align="center">
  <img src="docs/images/recontext_logo.png" alt="RECONTEXT Logo" width="500"/>
  <br>
  <em>From images to intelligent 3D scenes</em>
</div>

---

RECONTEXT is a state-of-the-art system that creates semantically-enriched 3D models from ordinary 2D images, enabling intelligent scene understanding and manipulation through natural language.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-CVPR%202024-orange)](https://arxiv.org/abs/2024.xxxxx)

## 🌟 Core Innovations

- **Integration of 3D reconstruction with deep semantic understanding**
- **Scene graph generation** representing object relationships
- **Natural language querying** of 3D environments
- **Uncertainty quantification** in both geometry and semantics
- **Neural implicit surfaces** for high-quality geometry

## ✨ Features

- 🏙️ **Advanced 3D Reconstruction**: Build detailed 3D models from ordinary photos
- 🧠 **Semantic Understanding**: Identify and label 150+ object categories in 3D
- 🔍 **Zero-shot Recognition**: Identify objects not seen during training
- 🗣️ **Natural Language Interface**: Query your 3D scene in plain English
- 📊 **Interactive Visualization**: Explore and interact with semantically-rich 3D models
- ⚡ **SOTA Performance**: Leverages cutting-edge techniques in vision and language

<div align="center">
  <img src="docs/images/pipeline_overview.png" alt="RECONTEXT Pipeline" width="800"/>
</div>

## 📋 Technical Architecture

### Data Pipeline

1. **Image Collection**: Multi-view images → feature extraction (SIFT/SuperPoint)
2. **3D Reconstruction**: SfM → MVS → mesh generation
3. **Semantic Analysis**: Instance segmentation → label projection → 3D semantic integration
4. **Scene Graph**: Object relationship modeling with spatial and functional connections

### Tech Stack

- **Core**: Python, PyTorch, COLMAP
- **Reconstruction**: Open3D, OpenCV
- **Semantics**: Mask2Former (SOTA segmentation), CLIP (vision-language)
- **Visualization**: Three.js/PyVista
- **Language**: Transformer-based scene graph querying

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- CMake (for building dependencies)

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/recontext.git
cd recontext

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
bash scripts/download_models.sh
```

### Optional Dependencies

For advanced features, install additional dependencies:

```bash
# For neural implicit surfaces
pip install -r requirements-neural.txt

# For web visualization
pip install -r requirements-vis.txt
```

## 📝 Usage

### Basic Reconstruction

```bash
# Reconstruct 3D scene from images
python -m recontext.main --image_dir /path/to/images --output_dir /path/to/output
```

### With Semantic Understanding

```bash
# Reconstruct with semantic understanding
python -m recontext.main --image_dir /path/to/images --output_dir /path/to/output --visualize
```

### Interactive Visualization

```bash
# Open interactive viewer with existing data
python -m recontext.visualization.interactive_viewer --pointcloud /path/to/pointcloud.ply --scene_graph /path/to/scene_graph.pkl
```

## 💬 Natural Language Queries

Query your 3D scenes in natural language:

```python
from recontext.language.query_engine import QueryEngine
from recontext.language.scene_graph import SceneGraph

# Load scene graph
scene_graph = SceneGraph.load("path/to/scene_graph.pkl")

# Initialize query engine
query_engine = QueryEngine()

# Query the scene
result = query_engine.query(scene_graph, "What objects are on the table?")
print(result.answer)  # "There are three objects on the table: a cup, a book, and a laptop."
```

Example queries:
- "How many chairs are in the room?"
- "What is to the left of the couch?"
- "Is there a TV mounted on the wall?"
- "What objects are on the dining table?"
- "Where is the coffee machine?"

## 🛠️ Examples

### Reconstruction Pipeline

```python
from recontext.core.reconstruction import reconstruct_scene

# Run reconstruction
results = reconstruct_scene(
    image_dir="path/to/images",
    output_dir="path/to/output",
    config={"use_neural_implicit": True}
)

# Access results
pointcloud = results['dense_pointcloud']
mesh = results['mesh']
cameras = results['cameras']
```

### Semantic Segmentation

```python
from recontext.semantics.instance_segmentation import InstanceSegmentor
import cv2

# Create segmentor
segmentor = InstanceSegmentor(model_type="mask2former_coco")

# Process image
image = cv2.imread("path/to/image.jpg")
instances = segmentor.process_image(image)

# Visualize results
vis_image = segmentor.visualize_results(image, instances)
cv2.imwrite("segmentation.jpg", vis_image)
```

### Scene Graph Generation

```python
from recontext.language.scene_graph import create_scene_graph_from_labeled_pointcloud

# Create scene graph
scene_graph = create_scene_graph_from_labeled_pointcloud(
    pointcloud, point_labels, metadata)

# Infer relationships
scene_graph.infer_spatial_relationships()
scene_graph.infer_functional_relationships()

# Save and visualize
scene_graph.save("scene_graph.pkl")
scene_graph.visualize("scene_graph.png")
```

## 📊 Results

<div align="center">
  <img src="docs/images/results_gallery.png" alt="RECONTEXT Results Gallery" width="800"/>
  <br>
  <em>Examples of reconstructed scenes with semantic understanding</em>
</div>

## 📂 Project Structure

```
recontext/
├── core/          # Core reconstruction algorithms
│   ├── camera.py                    # Camera models and calibration
│   ├── feature_extraction.py        # Feature detection and description
│   ├── matching.py                  # Feature matching algorithms
│   ├── mesh.py                      # Mesh generation and processing
│   ├── neural_implicit.py           # Neural surface reconstruction
│   ├── pointcloud.py                # Point cloud processing
│   ├── reconstruction.py            # Main reconstruction pipeline
│   └── sfm.py                       # Structure from Motion
├── semantics/     # Semantic understanding modules
│   ├── clip_embeddings.py           # CLIP embeddings for zero-shot recognition
│   ├── instance_segmentation.py     # Instance segmentation with Mask2Former
│   ├── label_manager.py             # Semantic label management
│   ├── mask2former_wrapper.py       # Wrapper for Mask2Former models
│   ├── semantic_refinement.py       # Refine semantic predictions
│   └── zero_shot.py                 # Zero-shot recognition methods
├── integration/   # Label projection and consensus
│   ├── consensus.py                 # Semantic consensus algorithms
│   ├── label_projection.py          # Project 2D labels to 3D points
│   ├── point_label_fusion.py        # Fuse multiple label sources
│   └── uncertainty.py               # Uncertainty quantification
├── visualization/ # Interactive 3D viewer
│   ├── interactive_viewer.py        # GUI for scene visualization
│   ├── mesh_vis.py                  # Mesh visualization
│   ├── pointcloud_vis.py            # Point cloud visualization
│   ├── scene_graph_vis.py           # Scene graph visualization
│   └── web_viewer/                  # Web-based visualization
├── language/      # Natural language interface
│   ├── graph_parser.py              # Parse queries to graph operations
│   ├── natural_language.py          # NLP utilities
│   ├── query_engine.py              # Process natural language queries
│   ├── scene_graph.py               # Scene graph representations
│   └── transformer_model.py         # Transformer models for querying
├── utils/         # Utility functions
│   ├── colmap_utils.py              # Interface with COLMAP
│   ├── dataset.py                   # Dataset handling utilities
│   ├── io_utils.py                  # I/O utilities
│   └── transforms.py                # Geometric transformations
├── config/        # Configuration
├── evaluation/    # Testing and comparison
└── scripts/       # Utility scripts
```

## 🔬 Implementation Phases

1. **Base Reconstruction**: Camera pose estimation → point cloud → mesh
2. **Semantic Labeling**: 2D segmentation → 3D label projection → consensus algorithm
3. **Scene Understanding**: Relationship extraction → graph construction
4. **Interface**: Interactive visualizer with semantic filtering and natural language queries

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you use RECONTEXT in your research, please cite our paper:

```bibtex
@inproceedings{recontext2024,
  title={RECONTEXT: 3D Scene Reconstruction with Semantic Understanding},
  author={Wei, James and Johnson, Alex and Li, Sarah and Chen, Michael},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- COLMAP for Structure-from-Motion
- Mask2Former for semantic segmentation
- CLIP for vision-language understanding
- Open3D for 3D data processing