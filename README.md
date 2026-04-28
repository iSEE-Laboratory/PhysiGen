<img width="5047" height="105" alt="image" src="https://github.com/user-attachments/assets/cf3ae760-e30d-49ef-96f0-c7e636431b2f" /><div align="center">


<div align="center">
  <img src="fig/title.png" width="100%">
</div>

</div>

---

## 📢 News
- **[2026.01.18]** 🎉 IRG-MotionLLM: our new work with Tongyi!
- Can we seamlessly interleave motion generation, assessment, and refinement in a unified model to progressively improve motion generation? Can learning motion assessment & refinement bring cross-task, cross-model synergies?  Our answer is YES! We propose the first model supporting natively text-motion interleaved reasoning for text-to-motion generation. We demonstrate its advanced performance and emerging properties.
- https://github.com/HumanMLLM/IRG-MotionLLM
- https://arxiv.org/abs/2512.10730

<img width="4663" height="105" alt="image" src="https://github.com/user-attachments/assets/b5087f9a-bb42-4413-8ed3-19ef264418e0" />

- **[2026.01.18]** 🎉 PhysiGen is accepted to **ICASSP 2026**!
- 🚧 Code and models coming soon...

---

## 📝 TODO

- ⬜ Release training & inference code
- ⬜ Release trained model checkpoints
- ⬜ Support plug-in integration with more generative models
- ⬜ One-click online demo

---



## 🔍 Overview

<div align="center">
<img src="fig/model.png" width="90%">
</div>

> **PhysiGen** is a plug-and-play, computationally efficient optimization strategy that explicitly integrates collision-aware physical constraints into human-human interaction generation.

Generating realistic multi-person interaction sequences remains challenging due to pervasive **body interpenetration** — a problem that spans from data acquisition to generated results. Existing approaches either ignore this issue or rely on computationally expensive mesh-level SDF losses (e.g., inflating training time from **3 days → 14 days**).

PhysiGen addresses this by:
- 🔷 **Simplifying** high-resolution human body meshes into geometric primitives (cylinders/cuboids) for efficient collision detection
- 🔷 **Computing** physics-inspired guidance directions via antipodal point construction to resolve penetration
- 🔷 **Integrating** seamlessly into existing models as a plug-and-play module — no architectural changes required

**Key results on InterHuman & Inter-X:**
| Model | Collision Distance ↓ | Collision Rate ↓ | Top-1 R-Precision ↑ |
|---|---|---|---|
| InterGen | 3.905 | 0.2270 | 0.371 |
| InterGen + PhysiGen | **1.836** | **0.1878** | **0.485** |
| in2IN | 3.142 | 0.1863 | 0.455 |
| in2IN + PhysiGen | **2.005** | **0.1503** | **0.481** |

---

## 🛠️ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/iSEE-Laboratory/PhysiGen.git
cd PhysiGen
```

Create conda environment:

```bash
conda create -n physigen python=3.9
conda activate physigen
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📋 Prerequisites

### Download Assets & Checkpoints

Download the required assets and pretrained checkpoints:

```bash
bash scripts/download_assets.sh
```

### Datasets

We evaluate on two datasets:

- **InterHuman** — 7,779 two-person interaction sequences with text annotations. Download from [InterGen](https://github.com/tr3e/InterGen).
- **Inter-X** — 11,388 interaction sequences using SMPL-X. Download from [Inter-X](https://github.com/liangxuy/Inter-X).

Organize the data as follows:

```
data/
├── InterHuman/
│   ├── motions/
│   └── texts/
└── Inter-X/
    ├── motions/
    └── texts/
```

---

## ▶️ Usage

### 🏋️ Training

**Train from scratch** (PhysiGen integrated into full training):

```bash
conda activate physigen
python train.py --dataset interhuman --mode scratch
```

**Adaptation** (fine-tune a pretrained model with PhysiGen):

```bash
conda activate physigen
python train.py --dataset interhuman --mode adaption --ckpt path/to/pretrained.ckpt
```

### 🎯 Inference

Generate interaction motions from text:

```bash
conda activate physigen
python generate.py --text "Two people shake hands and then hug each other." --output ./output
```

### 📏 Evaluation

Evaluate on the InterHuman test set:

```bash
conda activate physigen
python eval.py --dataset interhuman --ckpt path/to/checkpoint.ckpt
```

Evaluate on the Inter-X test set:

```bash
conda activate physigen
python eval.py --dataset inter-x --ckpt path/to/checkpoint.ckpt
```

### 👁️ Visualization

Visualize generated motion sequences:

```bash
conda activate physigen
python scripts/visualize.py --motion_path ./output/motion.npy --text "your text prompt"
```

---

## 📊 Results

### Quantitative Comparison

<div align="center">
<img src="assets/quantitative.png" width="85%">
</div>

### Qualitative Comparison

<div align="center">
<img src="assets/qualitative.png" width="90%">
</div>

PhysiGen significantly reduces interpenetration while maintaining semantic consistency with the text prompt. Red dashed boxes in the figure above highlight severe collision artifacts in baseline methods.

### Computational Cost

PhysiGen introduces minimal overhead compared to SDF-based losses:

| Method | Memory (MB) | Time per batch (s) |
|---|---|---|
| Baseline | 15,107 | — |
| PhysiGen (50×19 pts) | 20,219 | **0.053** |
| SDF Loss (128 pts) | 24,152 | 0.352 |
| SDF Loss (6890 pts) | 24,156 | 3.734 |

---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{Lei2026PhysigenIC,
  title={Physigen: Integrating Collision-Aware Physical Constraints for High-Fidelity Human-Human Interaction Generation},
  author={Nan Lei and Yuan-Ming Li and Ling-an Zeng and Liangliang Xu and Zhi-Wei Xia and Huihui Huang and Fa-Ting Hong and Wei-Shi Zheng},
  journal={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  url={https://api.semanticscholar.org/CorpusID:287685790}
}
```

---

## 📜 License

This project is released under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgement

We thank the following open-source projects for their contributions to the community:

- [InterGen](https://github.com/tr3e/InterGen) — two-person interaction generation framework and InterHuman dataset
- [Inter-X](https://github.com/liangxuy/Inter-X) — large-scale human interaction dataset
- [in2IN](https://github.com/pabloruizponce/in2IN) — individual-aware interaction generation
- [TIMotion](https://github.com/AIGC-Explorer/TIMotion) — temporal and interactive motion generation framework

---

<div align="center">
<sup>If you have any questions, please open an issue or contact us at lein7@mail2.sysu.edu.cn</sup>
</div>
```

