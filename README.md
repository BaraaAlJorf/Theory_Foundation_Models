# Theory_Foundation_Models

This repository contains code for the project **"Effectiveness of Layered Pretraining for Foundation Models"**, completed as part of the **Theoretical Foundations of LLMs** course. The project explores both theoretical and empirical aspects of **Layered Pretraining (LP)** strategies in the context of multimodal foundation models, with a particular focus on healthcare applications.

### 🔬 About the Paper

The accompanying paper provides a theoretical analysis of Layered Pretraining—where Masked Language Modeling (MLM) is followed by contrastive multimodal alignment and supervised fine-tuning. We prove that this sequential strategy reduces generalization error through reductions in intraclass variance, and we validate this with empirical experiments on clinical prediction tasks using MIMIC-IV data.

You can find the paper [here](./Baraa_ba2797_Final.pdf).

---

### ⚠️ Important Notes

- **This repository is incomplete** by design. The dataloaders and other project components are based on unpublished code from a separate project and are therefore not included here.
- All code that was **written specifically for the class project** is included in this repository.
- As such, **this repo is not meant to be executed out-of-the-box**. If you are interested in reproducing or building on this work, feel free to reach out to the author.

---

### 📁 Contents

- `model/`: Core model definitions used in the study
- `trainer/`: trainer scripts  defining contrastive vs supervised learning
- `utils/`: Utilities for logging, configuration, etc.
- `Baraa_ba2797_Final.pdf`: Final project report detailing theory and results

---

### 📫 Contact

For questions, collaboration inquiries, or to request access to related resources, please contact **Baraa Al Jorf** at `ba2797@columbia.edu`.
