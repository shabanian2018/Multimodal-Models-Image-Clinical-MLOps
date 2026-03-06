# 🧪 Strep-Multimodal-MLOps
# Multimodal Deep Learning for Streptococcal Pharyngitis Detection  

**End-to-end MLOps pipeline for strep throat classification from throat images + clinical symptoms**

---

## 🔍 Project Overview  

Diagnosis of streptococcal pharyngitis (“strep throat”) in practice relies on **both**:

- Visual inspection of the **throat image** (tonsillar exudate, redness, swelling)  
- Structured **clinical symptoms** (fever, sore throat, congestion, known contact, etc.)

This repository implements a **multimodal deep learning pipeline** that combines:

- 📷 **Throat images**  
- 📊 **Tabular clinical features**  

to predict **strep-positive vs strep-negative**, with:

- ✅ Strict held-out **test set**  
- ✅ **5-fold cross-validation** on the training pool  
- ✅ **Bayesian hyperparameter sweeps**  
- ✅ Full **MLOps tracking** via Weights & Biases (W&B)  

> Hypothesis:  
> **Image + clinical features** → better performance and more clinically realistic decision-making than image-only models.

---

## ⚙️ Modeling Approach  

Three multimodal architectures are implemented and compared. All image encoders are **pretrained ResNets** (18 / 50) from `torchvision`.

### 1️⃣ Gated ResNet Fusion  

**Idea:** Clinical symptoms “gate” the image representation.

- **Image encoder:** ResNet-18 / ResNet-50 (pretrained)  
- **Clinical encoder:** MLP → 64-dim embedding  
- **Fusion:**  
  - Clinical embedding → gating vector `g`  
  - Gated image feature: `img_mod = img_feat ⊙ g`  
  - Concatenate `[img_mod, sym_feat]` → classifier head  

**Why this model?**

- Simple, stable **late fusion**  
- Clinically intuitive: symptoms modulate how the image is interpreted  
- Works well for **small datasets** with strong priors from the pretrained CNN  

---

### 2️⃣ FiLM-Conditioned ResNet Fusion  

**Idea:** Use **Feature-wise Linear Modulation (FiLM)** to condition image features on symptoms.

- **Image encoder:** ResNet-18 / ResNet-50  
- **Clinical encoder:** MLP → 128-dim embedding  
- **Fusion:**
  - Clinical embedding → `γ` (scale), `β` (shift)  
  - Modulated feature: `img_mod = γ ⊙ img_feat + β`  
  - Concatenate `[img_mod, sym_feat]` → classifier  

**Why this model?**

- Stronger **cross-modal interaction** than pure concatenation  
- Parameter-efficient conditioning  
- Aligned with **modern multimodal learning** (vision–language, CLIP-style conditioning)  

---

### 3️⃣ Perforated / Dendritic Fusion Network  

**Idea:** Add a **biologically inspired “dendritic” refinement** on top of a standard multimodal MLP.

- **Base multimodal network:**  
  - Concatenate image + clinical embeddings  
  - Two-layer MLP (fusion hidden size 128–256)  
- **Dendritic adapters:**  
  - Small residual “dendrite” layers on fused hidden states  
  - Trained in a **second phase** while backbone is frozen  

**Two-phase training:**

1. **Phase 1 – Backbone learning**  
   - Train multimodal fusion network end-to-end  
   - Early stopping on validation loss  

2. **Phase 2 – Dendritic refinement**  
   - Freeze backbone weights  
   - Train only dendritic adapters:  
     - `h₁' = h₁ + D₁(h₁)`  
     - `h₂' = h₂ + D₂(h₂)`  

**Why this model?**

- Adds capacity with **very few extra parameters**  
- Designed for **small clinical datasets**  
- Inspired by dendritic computation and adapter-style fine-tuning  

---

## 🧬 Data & Preprocessing  

### Inputs  

- CSV file (e.g., `dataset_120.csv`) with columns:
  - `ImageName`  
  - `label` (e.g., `"Positive"` / `"Negative"`)  
  - Symptom columns:
    - `Hoarseness`  
    - `Rhinorrhea`  
    - `sorethroat`  
    - `Congestion`  
    - `Knownrecentcontact`  
    - `Headache`  
    - `Fever`  

- Image directory:
  - Contains the corresponding **throat images** referenced by `ImageName`.

### Label Mapping  

```text
Positive → 1
Negative → 0
