# Deep Learning for Urban Land Cover Classification

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Computer Vision](https://img.shields.io/badge/Topic-Semantic_Segmentation-blue)

## Project Overview
This repository hosts a deep learning framework designed to perform semantic segmentation on high-resolution orthophotos (DOP). The primary objective is the automated detection of sealed surfaces (imperviousness) to support urban drainage tax assessment and environmental monitoring.

## Objectives
* **Automation:** Replace manual digitizing of sealed surfaces with AI-driven inference.
* **Accuracy:** Achieve a high Intersection over Union (IoU) score on the "Impervious" class.
* **Scalability:** Enable processing of large-scale raster datasets using tile-based inference.

## Methodology
1.  **Data Preparation:** Tiling large orthophotos into 512x512 patches with overlap.
2.  **Model Architecture:** Implementation of a U-Net with a ResNet-34 backbone pre-trained on ImageNet.
3.  **Training:** Supervised training using labeled masks (Background vs. Sealed).
4.  **Post-Processing:** Morphological operations to clean prediction noise and vectorization of results for QGIS.

## Tech Stack
* **Deep Learning:** PyTorch, Segmentation Models PyTorch (SMP)
* **Image Processing:** OpenCV, Rasterio, Albumentations
* **GIS Integration:** QGIS (for visualization of result vectors)

## Dataset
* **Input:** Digital Orthophotos (DOP20)
* **Labels:** Cadastral map derived ground truth

---
*Melih Levent Aslan | M.Sc. Geodetic Engineering Student, University of Bonn*
