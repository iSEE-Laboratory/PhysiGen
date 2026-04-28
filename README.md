---

## ▶️ Usage

### 🏋️ Training

**Train from scratch:**

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

```bash
conda activate physigen
python generate.py --text "Two people shake hands and then hug each other." --output ./output
```

### 📏 Evaluation

```bash
# InterHuman
conda activate physigen
python eval.py --dataset interhuman --ckpt path/to/checkpoint.ckpt

# Inter-X
conda activate physigen
python eval.py --dataset inter-x --ckpt path/to/checkpoint.ckpt
```

### 👁️ Visualization

```bash
conda activate physigen
python scripts/visualize.py --motion_path ./output/motion.npy --text "your text prompt"
```

---

## 📊 Results

### Qualitative Comparison

<div align="center">
<img src="fig/qualitative.png" width="90%">
</div>

PhysiGen significantly reduces interpenetration while maintaining semantic consistency with the text prompt. Red dashed boxes highlight severe collision artifacts in baseline methods.

### Computational Cost

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

We thank the following open-source projects for their contributions:

- [InterGen](https://github.com/tr3e/InterGen) — two-person interaction generation framework and InterHuman dataset
- [Inter-X](https://github.com/liangxuy/Inter-X) — large-scale human interaction dataset
- [in2IN](https://github.com/pabloruizponce/in2IN) — individual-aware interaction generation
- [TIMotion](https://github.com/AIGC-Explorer/TIMotion) — temporal and interactive motion generation framework

---

<div align="center">
<sup>If you have any questions, please open an issue or contact us at lein7@mail2.sysu.edu.cn</sup>
</div>
