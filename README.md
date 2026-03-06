Multimodal Deep Learning for Streptococcal Pharyngitis Detection

This repository contains a reproducible multimodal deep learning pipeline for classifying streptococcal pharyngitis (strep throat) from:

throat images

structured clinical symptom features

The project was developed as an end-to-end take-home assignment with emphasis on:

multimodal modeling

rigorous evaluation

reproducibility

MLOps-based experiment tracking

Project Overview

Diagnosis of strep throat often depends on both visual throat examination and clinical symptoms such as fever, sore throat, congestion, and recent exposure history. This project investigates whether deep learning can combine these complementary signals into a single predictive model.

The central hypothesis is that:

multimodal fusion of image and clinical features outperforms single-modality approaches

Modeling Approach

Three multimodal architectures were implemented and evaluated.

1. Gated ResNet Fusion

A pretrained ResNet image encoder extracts visual features, while a small MLP encodes clinical symptoms. The clinical embedding generates a gating signal that modulates the image representation before classification.

Why it matters:

simple and stable multimodal fusion

clinically intuitive conditioning

effective for small datasets

2. FiLM-Conditioned ResNet Fusion

This model uses Feature-wise Linear Modulation (FiLM) to condition image features on clinical variables. The symptom embedding produces learned scale and shift parameters that adjust the visual embedding.

Why it matters:

stronger cross-modal interaction than simple concatenation

parameter-efficient conditioning

well aligned with modern multimodal learning literature

3. Perforated / Dendritic Fusion Network

This architecture introduces a biologically inspired refinement mechanism. A base multimodal network is trained first, then lightweight dendritic adapters refine the fused hidden representation in a second training phase.

Why it matters:

adds capacity without retraining the full backbone

designed for small-data settings

explores adapter-style multimodal refinement

MLOps and Experiment Tracking

Experiments were tracked using Weights & Biases (W&B) to ensure full reproducibility and systematic comparison across runs.

What was tracked

model architecture

backbone type

optimizer

batch size

learning rate

dropout

weight decay

class weighting

cross-validation metrics

held-out test metrics

learning curves

generated plot artifacts

MLOps advantages in this project

Reproducibility: every run is versioned with full configuration

Comparability: results across architectures and hyperparameters are easy to compare

Auditability: plots, metrics, and model choices are transparent

Scalability: Bayesian sweeps support efficient hyperparameter exploration

Traceability: selected models can be tied back to exact configurations and metrics

Training and Evaluation Strategy

The project uses a strict evaluation design to avoid leakage.

Held-out test split is created first

5-fold stratified cross-validation is performed on the training pool

Hyperparameters are explored with Bayesian sweeps

Final model is retrained and evaluated on the held-out test set

This ensures the test set remains unbiased during model selection.

Metrics

The pipeline tracks both threshold-based and ranking-based metrics.

Cross-validation metrics

validation loss

accuracy

precision

recall

Final test metrics

accuracy

precision

recall

specificity

AUROC

AUPRC

loss

Training artifacts

mean CV loss plot

mean CV accuracy plot

mean CV precision plot

mean CV recall plot

Repository Structure
.
├── data/
│   ├── dataset_120.csv
│   └── images/
├── plots/
├── artifacts/
├── notebooks/
├── src/
├── README.md
└── requirements.txt

Adjust the structure to match your final repo layout.

Dataset Format

The code expects a CSV with:

ImageName

label

symptom columns such as:

Hoarseness

Rhinorrhea

sorethroat

Congestion

Knownrecentcontact

Headache

Fever

Labels are mapped as:

Positive -> 1

Negative -> 0

Installation

Create an environment and install dependencies:

pip install torch torchvision pandas numpy scikit-learn pillow matplotlib wandb

Login to W&B if you want sweep tracking:

wandb login
Running the Pipeline

Update the dataset and image paths in the training script, then run:

python train.py

If using a notebook, execute the notebook cells in order after editing paths.

Hyperparameter Optimization

The project supports Bayesian sweeps over:

model type

backbone

optimizer

learning rate

batch size

dropout

weight decay

fusion hidden dimension

early stopping settings
