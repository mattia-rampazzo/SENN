# Self-Explaining Neural Networks: CIFAR-10 Extension and Comparative Analysis

This repository is a fork of [Self-Explaining Neural Networks: A review with extensions](https://github.com/dtak/senn) and expands upon the original work by adapting it to the CIFAR-10 dataset.

## Overview

The original repository contains the implementation for reproducing the results from the paper _"Towards Robust Interpretability with Self-Explaining Neural Networks"_ [[1]](#references). It introduces the SENN (Self-Explaining Neural Network) framework — a model designed to be inherently interpretable. The authors critically analyzed its performance, highlighted several limitations (e.g., unstable explanations and reduced predictive accuracy), and proposed improvements to enhance explanation quality and model robustness.
This project was developed as the **final project for the course "Machine Learning for NLP II"** at the **University of Trento**.


## What's New in This Fork

In this fork, the original work is extended in the following ways:

- **CIFAR-10 Compatibility**: Added complete support for training and evaluating SENN on the CIFAR-10 dataset.
- **Architecture Adaptation**: Adopted the original SENN structure — specifically the **Conceptizer** and **Parameterizer** modules — to suit image data from CIFAR-10.
- **Comparative Interpretability Study**: Included a comprehensive analysis comparing SENN to two widely-used post-hoc interpretability methods: **LIME** and **Integrated Gradients**.
- **Notebook Report**: The file `report.ipynb` contains all results, visualizations, and discussion regarding the effectiveness of ante-hoc (SENN) versus post-hoc (LIME, IG) explanations on CIFAR-10.
