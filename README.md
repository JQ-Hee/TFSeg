# TFSeg

This repository corresponds to the accepted paper:  
**“Training-Free Breast Ultrasound Image Segmentation with Retrieval-based SAM2.”**

---

## Overview

This project proposes a **training-free breast ultrasound image segmentation** framework based on **Segment Anything Model v2 (SAM2)**.  

---

## Environment and Dependencies

The environment consists of two main components:

1. **[SSCD-Copy-Detection](https://github.com/facebookresearch/sscd-copy-detection)**  
   - Used for generating reference–target image sequences.
2. **[Segment Anything Model v2 (SAM2)](https://github.com/facebookresearch/sam2)**  
   - Used for image segmentation.

You can set up these two environments manually, or use the provided `environment.yaml` file for a one-step installation.

---

## Data Source

Experiments are conducted using the **BUSI (Breast Ultrasound Images) dataset**.

---

## Acknowledgment

Thanks to the open-source of the following projects:

- [SSCD-Copy-Detection](https://github.com/facebookresearch/sscd-copy-detection)  
- [Segment Anything Model v2 (SAM2)](https://github.com/facebookresearch/segment-anything)  
- [BUSI Dataset](https://doi.org/10.1016/j.dib.2019.104863)  

