# Multimodal E-commerce Product Classification

This project implements and evaluates unimodal and multimodal deep learning models
for e-commerce product category classification using both images and text.
It was developed as part of a Neural Networks coursework (Week 10 milestone).

## Overview

Three architectures are explored and compared:

1. **Image-only model** using ResNet18
2. **Text-only model** using DistilBERT
3. **Multimodal model** using late fusion (feature concatenation)

The goal is to analyse how visual and textual information contribute individually
and jointly to product classification performance.

## Models

- **Image-only (ResNet18):** Extracts visual features from product images.
- **Text-only (DistilBERT):** Encodes semantic information from product titles
  and descriptions.
- **Multimodal (Concat):** Combines image and text embeddings via late fusion
  followed by a shared classification head.

## Implementation

- Frameworks: PyTorch, Torchvision, Hugging Face Transformers
- Image preprocessing: resize to 224×224 and ImageNet normalization
- Text preprocessing: tokenization with max sequence length of 64
- Optimizer: AdamW
- Loss function: Cross-entropy loss

## Evaluation

Models are evaluated on a validation set using:

- Accuracy
- Macro-averaged Precision
- Macro-averaged Recall
- Macro-averaged F1-score

The multimodal model achieves the best overall performance, demonstrating that
visual features provide complementary information to text in e-commerce
classification tasks.

## Results

Evaluation artefacts (confusion matrix, metrics, and classification report)
are available in the `results/` directory.

## Repository Purpose

This repository is maintained to support reproducibility, version control,
and coursework assessment requirements. Large datasets and credentials are
intentionally excluded.
