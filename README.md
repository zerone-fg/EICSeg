# EICSeg: Universal Medical Image Segmentation via Explicit In-Context Learning

[Shiao Xie](https://example.com), [Liangjun Zhang](https://example.com)  
<sup>1 Zhejiang University</sup>, <sup>2 Hikvision Research Institute</sup>  


👉 [**Paper**](#) | [**Demo**](#) | [**Project Page**](#)


## 🧠 Overview

We present **EICSeg**, a generalist model for universal medical image segmentation.
With a single unified architecture, EICSeg enables fully automatic segmentation without any manual interaction via in-context inference, supporting a wide range of tasks including but not limited to:

- 🧩 Few-shot scenarios
- 🌀 In-domain and out-of-domain generalization

EICSeg is evaluated across diverse benchmarks, showing strong performance in [automatic medical image segmentation tasks].
- MoNuSeg
- STARE
- PanDental
- Spine
- ACDC
- Cervix
- SCD
- WBC
- HipXray
---

## 📦 Features

- ✅ Unified framework for **multi-modal** inference
- 🧩 Supports **in-context prompts** for flexible adaptation
- ⚡ Exploring the integration of different functional VFMs

---
## ✅ To Do List

We are actively working on releasing more components of the EICSeg project. The following items will be gradually made available:

- [ ] **Model Checkpoints**  
  Pre-trained weights for various datasets and configurations.

- [x] **Training Scripts**  
  Full training pipeline including dataset loading, loss functions, and training configuration.

- [x] **Network Architecture Details**  
  A detailed breakdown of our unified architecture, including the explicit in-context prompt design.

- [x] **Evaluation Script**  
Standardized scripts to validate model performance on benchmarks (e.g., metrics like Dice Score)

- [ ] **Data Preparation Instructions**  
  Guidance for dataset preprocessing and formatting, covering all supported benchmarks.


Stay tuned! 🔧 If you’re interested in contributing or collaborating, feel free to reach out via issues or email.


## 📂 Get Started

```bash
#### Installation
git clone [https://github.com/zerone-fg/EICSeg.git](https://github.com/zerone-fg/EICSeg.git)
cd EICSeg
conda env create -f environment.yaml

#### For evaluation
python inference.py --config configs/demo.yaml --input demo_inputs

#### For training
python main_train_medical.py --config configs/demo.yaml --input demo_inputs

