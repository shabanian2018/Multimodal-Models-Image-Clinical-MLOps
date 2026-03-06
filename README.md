# 🧪 Strep-Multimodal-MLOps
# Multimodal Deep Learning for Streptococcal Pharyngitis Detection  

**End-to-end MLOps pipeline for strep throat classification from throat images + clinical symptoms**

---

## 🔍 Project Overview  

Diagnosis of streptococcal pharyngitis (“strep throat”) in practice relies on **both**:

- Visual inspection of the **throat** (tonsillar exudate, redness, swelling)  
- Structured **clinical symptoms** (fever, sore throat, congestion, recent contact, etc.)

This repository implements a **multimodal deep learning pipeline** that combines:

- 📷 **Throat images**  
- 📊 **Tabular clinical features**  

to predict **strep-positive vs strep-negative**, with:

- ✅ Strict held-out **test set**  
- ✅ **5-fold cross-validation** on the training pool  
- ✅ **Bayesian hyperparameter sweeps**  
- ✅ Full **MLOps tracking** via [Weights & Biases (W&B)](https://wandb.ai)  

> **Hypothesis**  
> Multimodal fusion of **image + clinical features** provides more robust and clinically meaningful performance than image-only models.

---

## ⚙️ Modeling Approach  

Three multimodal architectures are implemented and compared. All image encoders are **pretrained ResNets** (`resnet18` / `resnet50`) from `torchvision`.

### 1️⃣ Gated ResNet Fusion  

**Idea:** Clinical symptoms “gate” the image representation.

- **Image encoder:** ResNet-18 / ResNet-50 (pretrained on ImageNet)  
- **Clinical encoder:** MLP → 64-dim embedding  
- **Fusion:**  
  - Clinical embedding → gating vector `g`  
  - Gated image feature:  

    \[
    \tilde{z}_{\text{img}} = z_{\text{img}} \odot g(x_{\text{clinical}})
    \]

  - Fused feature:  

    \[
    z_{\text{fusion}} = [\tilde{z}_{\text{img}}, z_{\text{clinical}}]
    \]

- **Classifier:** Fully connected head on `z_fusion`  

**Why this model?**

- Simple, stable **late fusion**  
- Clinically intuitive conditioning (symptoms modulate how the visual signal is interpreted)  
- Well-suited for **small datasets** with pretrained CNN backbones  

---

### 2️⃣ FiLM-Conditioned ResNet Fusion  

**Idea:** Use **Feature-wise Linear Modulation (FiLM)** to condition image features on symptoms.

- **Image encoder:** ResNet-18 / ResNet-50  
- **Clinical encoder:** MLP → 128-dim embedding  
- **Fusion:**
  - Clinical embedding → FiLM parameters `γ` (scale), `β` (shift)  
  - Modulated feature:  

    \[
    z_{\text{img}}' = \gamma(x_{\text{clinical}}) \odot z_{\text{img}} + \beta(x_{\text{clinical}})
    \]

  - Fused feature:  

    \[
    z_{\text{fusion}} = [z_{\text{img}}', z_{\text{clinical}}]
    \]

- **Classifier:** Fully connected head  

**Why this model?**

- Richer **cross-modal interaction** than pure concatenation  
- Parameter-efficient conditioning  
- Aligned with modern **multimodal learning** literature (vision–language, CLIP-style conditioning)  

---

### 3️⃣ Perforated / Dendritic Fusion Network  

**Idea:** Add a **biologically inspired “dendritic” refinement** on top of a standard multimodal MLP.

- **Base multimodal backbone:**
  - Concatenate image embedding + clinical embedding  
  - Two-layer MLP (hidden fusion dimension 128–256)  
- **Dendritic adapters (perforated layers):**
  - Small linear “dendrite” modules that refine hidden activations:  

    \[
    h' = h + D(h_{\text{detached}})
    \]

- **Two-phase training:**
  1. **Phase 1 – Backbone learning**  
     - Train multimodal network end-to-end  
     - Early stopping on validation loss  
  2. **Phase 2 – Dendritic refinement**  
     - Freeze backbone parameters  
     - Train only dendritic adapters on the same loss  

**Why this model?**

- Adds capacity with **very few additional parameters**  
- Designed for **small-data regimes**  
- Inspired by **dendritic computation** and adapter-style fine-tuning in modern deep learning  

---

## 🧬 Data & Preprocessing  

### Dataset Format  

The pipeline expects a CSV file with the following columns.

#### ✅ Required Columns  

- `ImageName`  
  - Filename or relative path to the throat image.  
- `label`  
  - String labels mapped internally to binary targets:  

    ```text
    Positive → 1
    Negative → 0
    ```

#### 📊 Symptom Features  

Example symptom variables (can be extended if needed):

- `Hoarseness`  
- `Rhinorrhea`  
- `sorethroat`  
- `Congestion`  
- `Knownrecentcontact`  
- `Headache`  
- `Fever`  

All symptom columns are:

- Cast to numeric (`float32`)  
- Non-numeric values coerced to `NaN` and imputed with `0.0`  

> This design keeps the code robust to missing or slightly noisy symptom encodings.

---

## 📁 Repository Structure  

Example structure (adapt to your final layout):

```text
.
├── data/
│   ├── dataset_120.csv         # Tabular file: image names, labels, symptoms
│   └── images/                 # Throat image files referenced by ImageName
├── plots/                      # Generated CV mean plots (loss/acc/precision/recall)
├── artifacts/                  # (Optional) Saved models, checkpoints, or W&B exports
├── notebooks/                  # Exploratory notebooks / EDA / prototyping
├── src/
│   └── train_multimodal.py     # Main training + CV + test + W&B sweep script
├── README.md                   # This file
└── requirements.txt            # Python dependencies
