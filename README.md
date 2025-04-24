# EICSeg: Universal Medical Image Segmentation via Explicit In-Context Learning

[Shiao Xie](https://example.com), [Liangjun Zhang](https://example.com), [Ziwei Niu](https://example.com)  
<sup>1 Zhejiang University</sup>, <sup>2 Hikvision Research Institute</sup>, <sup>3 Zhejiang University</sup>

👉 [**Paper**](#) | [**Demo**](#) | [**Project Page**](#)


## 🧠 Overview

We present **EICSeg**, a generalist model for universal medical image segmentation.
With a single unified architecture, EICSeg enables fully automatic segmentation without any manual interaction via in-context inference, supporting a wide range of tasks including but not limited to:

- 🧩 Few-shot scenarios
- 🌀 In-domain and out-of-domain generalization

EICSeg is evaluated across diverse benchmarks (e.g., MoNuSeg, Cervix, ACDC), showing strong performance in [automatic medical image segmentation tasks].

---

## 📦 Features

- ✅ Unified framework for **multi-modal** inference
- 🧩 Supports **in-context prompts** for flexible adaptation
- ⚡ Deployable to edge devices with **lightweight variants**
- 🔁 Built-in support for **continual learning** and **domain adaptation**

---

## 📂 Get Started

```bash
#### Installation
git clone https://github.com/your-org/ProjectName.git
cd ProjectName
conda env create -f environment.yaml

#### For evaluation
python inference.py --config configs/demo.yaml --input demo_inputs

#### For training
python main_train_medical.py --config configs/demo.yaml --input demo_inputs

