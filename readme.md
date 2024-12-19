# SIREN: Synthetic Image Recognition and Explanation Network

## Overview
This repository hosts the **SIREN** project, the winning solution for the InterIIT Tech Meet 13.0. The project focuses on detecting AI-generated images and generating interpretable explanations for their classification. SIREN comprises two main components:

1. **SIREN-𝛿**: A robust classifier to distinguish real images from AI-generated ones.
2. **SIREN-𝛼**: An explainability pipeline for artifact identification and explanation generation.

---

## Problem Statement
With the proliferation of advanced generative AI models, ensuring the authenticity of digital content has become crucial. AI-generated images often contain subtle artifacts that differentiate them from real images. SIREN addresses this challenge by:

- Accurately classifying images as real or AI-generated.
- Generating human-interpretable explanations for the classification, including artifact identification and localization.

---

## Dataset
### CIFAKE Dataset
- **Training Data**: CIFAKE, containing 120,000 labeled samples of real and synthetic images, with subjects from CIFAR-10 classes.
- **SIREN-63K Dataset**: Augmented dataset created by the team with 64,086 samples (30,000 real and 34,086 fake) derived from ImageNet-1k and various generative models.

---

## Solution Approach
### SIREN-𝛿: Classification of Real vs. AI-Generated Images
- **Model**: EfficientNet_B2_S with adversarial robustness.
- **Training**: Purification-based techniques to enhance resistance against adversarial attacks.
- **Accuracy**: Achieved 94.23% on the SIREN-63K dataset.

### SIREN-𝛼: Artifact Detection and Explanation Generation
- **Artifact Detection**:
  - Leveraged CLIP-ViT embeddings for artifact relevance.
  - Trained multi-label classifiers for CIFAR class-specific artifacts.
- **Artifact Explanation**:
  - Used GPT-4 to generate descriptors and CLIPSeg for localization maps.
  - Provided explanations with identifiers for detected artifacts.

---

## Repository Structure
```plaintext
├── Submission_Task_1/
│   ├── infer.py           # Inference script for classification
│   ├── model.py           # Core classification model
│   ├── model_gan.py       # GAN-based purification model
│   ├── config.py          # Configuration file
│   ├── saved_weights/     # Pretrained weights for classification
│   └── atop/              # Implementation of Adversarial Training on Purification (ATOP)
├── Submission_Task_2/
│   ├── infer.py           # Inference script for artifact explanation
│   ├── main.py            # GradIO UI for interactive model usage
│   ├── model.py           # Core artifact detection model
│   ├── PEFT_ViT.py        # LoRA-optimized ViT implementation
│   ├── model_CLIP.py      # CLIP-based artifact detection model
│   ├── model_CLIPSeg.py   # Artifact localization with CLIPSeg
│   ├── model_ConvNeXt.py  # ConvNeXt-based feature extractor
│   ├── results/           # Example results and visualizations
│   ├── saved_weights/     # Weights for artifact models
│   ├── idx2label/         # JSON mappings of artifacts to IDs
│   └── Identifiers/       # JSON files for artifact-specific identifiers
```

---

## Installation
### Prerequisites
- Python >= 3.8
- Required dependencies listed in `requirements.txt`.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/username/SIREN.git
   cd SIREN
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
### Task 1: Image Classification
Run the inference script to classify images:
```bash
python Submission_Task_1/infer.py --input <image_folder> --output <output_file>
```

### Task 2: Artifact Explanation
Generate explanations for classified AI-generated images:
```bash
python Submission_Task_2/infer.py --input <classified_images> --output <explanations_json>
```

### Interactive UI with GradIO
Run the GradIO-based interactive UI to use the model:
```bash
python Submission_Task_2/main.py
```

---

## Pretrained Weights
Below are the links to the pretrained weight files for both tasks:

| Task                 | Model Component                  | Download Link                  |
|----------------------|----------------------------------|---------------------------------|
| Task 1: Classification | **SIREN-𝛿**               | [Download](#)                  |
| Task 2: Explanation   | **SIREN-𝛼**                   | [Download](#)                  |

---

## Results
### Task 1: Classification Metrics
| Model             | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| EfficientNet_B2_S | 94.23%   | 94.25%    | 94.23% | 94.23%   |

### Task 2: Artifact Explanation
- Achieved high relevance between predicted artifacts and ground truth.
- Localization maps effectively highlight artifact regions.

---

## Limitations
- Dataset imbalance for certain generative models.
- Limited descriptor granularity due to computational constraints.
- Reduced scalability to novel artifacts and adaptive adversarial attacks.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For questions or contributions:
- **Name**: Owais Mohammad Makroo
- **Email**: makroo.owais@gmail.com
