# Deep Learning-Based Classification of Brain Stroke CT Images Using Transfer Learning with ResNet-50

---

## Abstract

Stroke is a leading cause of death and long-term disability worldwide, and rapid, accurate classification of stroke subtypes from computed tomography (CT) scans is critical for timely clinical intervention. In this study, we propose a deep learning approach utilizing a ResNet-50 convolutional neural network with ImageNet transfer learning for the automated classification of brain CT images into three categories: hemorrhagic stroke (bleeding), ischemic stroke, and normal (non-stroke). Using a dataset of 6,650 brain CT images, we achieved an overall test accuracy of **96.49%**, a macro F1-score of **95.28%**, and a Cohen's kappa of **0.9299**, with per-class AUC values exceeding 0.99 for all three categories. The model demonstrated robust performance despite significant class imbalance (Normal: 4,427; Ischemia: 1,130; Bleeding: 1,093), which was addressed through weighted random sampling and class-weighted cross-entropy loss. Our results demonstrate the feasibility of automated stroke subtype classification from non-contrast CT scans and suggest potential for integration into clinical decision support systems to assist radiologists in emergency triage settings.

**Keywords:** deep learning, stroke classification, computed tomography, transfer learning, ResNet-50, convolutional neural network, medical image analysis

---

## 1. Introduction

### 1.1 Background

Stroke is the second leading cause of death globally and one of the primary causes of long-term disability, affecting approximately 15 million people annually worldwide [1]. Strokes are broadly categorized into two major subtypes: ischemic stroke, caused by a blockage in blood vessels supplying the brain, and hemorrhagic stroke (bleeding), resulting from the rupture of blood vessels. Ischemic strokes account for approximately 87% of all stroke cases, while hemorrhagic strokes, though less common, tend to be more severe and carry higher mortality rates [2].

Non-contrast computed tomography (NCCT) of the brain is the first-line imaging modality in the emergency evaluation of suspected stroke patients due to its widespread availability, rapid acquisition time, and high sensitivity for detecting acute hemorrhage [3]. However, the interpretation of CT scans for stroke classification requires significant radiological expertise, and subtle findings, particularly in the early stages of ischemic stroke, can be challenging to detect [4]. Furthermore, the increasing volume of imaging studies and the time-critical nature of stroke management create an urgent need for automated decision support tools.

### 1.2 Related Work

Deep learning, particularly convolutional neural networks (CNNs), has demonstrated remarkable success in medical image analysis across a wide range of applications, including classification, segmentation, and detection tasks [5]. In the domain of neuroimaging, several studies have explored the application of deep learning for stroke detection and classification. Chilamkurthy et al. [6] developed a deep learning system for detecting critical findings in head CT scans, achieving AUC values of 0.94 for intracranial hemorrhage. Qiu et al. [7] employed deep learning for ischemic stroke lesion segmentation, while Nishi et al. [8] applied CNNs for predicting large vessel occlusion from CT scans.

Transfer learning, the practice of leveraging pre-trained models from large-scale datasets (such as ImageNet) and fine-tuning them for domain-specific tasks, has been particularly effective in medical imaging where labeled data is often limited [9]. ResNet architectures, introduced by He et al. [10], have been widely adopted in medical imaging due to their ability to train very deep networks effectively through residual connections, mitigating the vanishing gradient problem.

### 1.3 Objectives

The primary objectives of this study are:

1. To develop and evaluate a deep learning model for the automated classification of brain CT images into hemorrhagic stroke, ischemic stroke, and normal categories.
2. To assess the effectiveness of transfer learning with ResNet-50 for stroke subtype classification.
3. To address the challenge of class imbalance inherent in stroke CT datasets.
4. To provide comprehensive performance evaluation using multiple metrics including accuracy, F1-score, AUC-ROC, and Cohen's kappa.

---

## 2. Materials and Methods

### 2.1 Dataset

The Brain Stroke CT Dataset used in this study comprises 6,650 brain CT images distributed across three classes:

| Class | Number of Images | Percentage |
|-------|-----------------|------------|
| Bleeding (Hemorrhagic) | 1,093 | 16.4% |
| Ischemia (Ischemic) | 1,130 | 17.0% |
| Normal | 4,427 | 66.6% |
| **Total** | **6,650** | **100%** |

**Table 1.** Distribution of images across classes in the Brain Stroke CT Dataset.

The dataset exhibits notable class imbalance, with the Normal class comprising approximately two-thirds of all images. Each image is available in DICOM format (original clinical format) and PNG format (pre-processed for model input). Bleeding and Ischemia classes additionally include overlay annotations indicating regions of pathological interest. An external test set of 200 images with segmentation masks is also available but was reserved for future segmentation studies.

### 2.2 Data Preprocessing and Splitting

The dataset was divided into training, validation, and test sets using stratified random sampling to maintain class proportions across splits:

| Split | Total | Bleeding | Ischemia | Normal |
|-------|-------|----------|----------|--------|
| Training (70%) | 4,655 | 765 | 791 | 3,099 |
| Validation (15%) | 997 | 164 | 169 | 664 |
| Test (15%) | 998 | 164 | 170 | 664 |

**Table 2.** Dataset split distribution.

All random operations were performed with a fixed random seed (42) to ensure reproducibility. The PNG format images were used as model inputs.

### 2.3 Data Augmentation

To improve model generalization and mitigate overfitting, the following data augmentation transforms were applied exclusively to the training set:

- **Resizing:** All images were resized to 224 x 224 pixels to match the input dimensions of the ResNet-50 architecture.
- **Random Horizontal Flip:** Applied with probability 0.5 to account for bilateral symmetry of brain anatomy.
- **Random Rotation:** Up to +/-15 degrees to simulate variability in patient positioning.
- **Random Affine Translation:** Up to 10% shift in both horizontal and vertical directions.
- **Color Jitter:** Brightness and contrast variations of +/-20% to simulate acquisition variability across different CT scanners and imaging protocols.
- **Normalization:** ImageNet normalization parameters (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) were applied to align with the pre-trained model's expected input distribution.

Validation and test sets received only resizing and normalization transforms to ensure unbiased evaluation.

### 2.4 Class Imbalance Handling

Two complementary strategies were employed to address the significant class imbalance:

1. **Weighted Random Sampling:** During training, a weighted random sampler was used to oversample minority classes (Bleeding and Ischemia), ensuring approximately equal class representation in each training epoch. Sample weights were computed as the inverse of class frequency.

2. **Class-Weighted Cross-Entropy Loss:** The loss function was weighted inversely proportional to class frequency in the training set:

| Class | Weight |
|-------|--------|
| Bleeding | 2.028 |
| Ischemia | 1.962 |
| Normal | 0.501 |

**Table 3.** Class weights used in the loss function.

### 2.5 Model Architecture

We employed a ResNet-50 architecture pre-trained on ImageNet (V2 weights) [10]. ResNet-50 consists of 50 layers organized into four residual blocks with identity shortcuts, enabling effective gradient flow during backpropagation. The architecture comprises approximately 23.5 million parameters.

The final fully connected layer was replaced with a custom classification head consisting of:
- Dropout layer (p = 0.5) for regularization
- Fully connected layer mapping from 2,048 features to 3 output classes

All parameters (23,514,179 total) were set as trainable to allow full fine-tuning of both the pre-trained backbone and the new classification head.

### 2.6 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1 x 10^-4 |
| Weight Decay | 1 x 10^-4 |
| Batch Size | 32 |
| Max Epochs | 30 |
| LR Scheduler | Cosine Annealing |
| Early Stopping Patience | 7 epochs |
| Random Seed | 42 |
| Hardware | Apple M-series GPU (MPS) |

**Table 4.** Training hyperparameters.

The model selection criterion was based on the best macro F1-score achieved on the validation set, as macro F1 provides a balanced measure of performance across all classes regardless of class frequency.

### 2.7 Evaluation Metrics

Model performance was evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified images.
- **Precision, Recall, F1-Score:** Computed per-class and as macro-averaged values.
- **Cohen's Kappa (kappa):** A statistic measuring inter-rater agreement adjusted for chance, providing a more robust evaluation metric than raw accuracy for imbalanced datasets.
- **Area Under the ROC Curve (AUC-ROC):** Computed per-class using one-vs-rest strategy and micro-averaged.
- **Average Precision (AP):** Area under the precision-recall curve, computed per-class.
- **Confusion Matrix:** Detailed breakdown of true positives, false positives, true negatives, and false negatives for each class.

---

## 3. Results

### 3.1 Training Dynamics

The model was trained for 27 epochs before early stopping was triggered (best model at epoch 20, with no improvement for 7 subsequent epochs). Total training time was 72.4 minutes on an Apple M-series GPU.

The training curves (Figure 1) demonstrate characteristic convergence behavior:

- **Loss:** Training loss decreased steadily from 0.617 (epoch 1) to 0.005 (epoch 27), while validation loss decreased from 0.511 to 0.149 (best at epoch 19), indicating effective learning without severe overfitting.
- **Accuracy:** Training accuracy improved from 59.1% to 99.7%, while validation accuracy reached a peak of 97.6% at epoch 20.
- **Macro F1-Score:** Training F1 rose from 0.547 to 0.997, with validation F1 peaking at 0.967 (epoch 20).

The gap between training and validation metrics in later epochs suggests mild overfitting, which was effectively controlled by early stopping and dropout regularization.

### 3.2 Test Set Performance

The best model (selected based on validation macro F1 at epoch 20) achieved the following results on the held-out test set:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **96.49%** |
| **Macro F1-Score** | **95.28%** |
| **Weighted F1-Score** | **96.49%** |
| **Cohen's Kappa** | **0.9299** |
| **Micro-Average AUC** | **0.9959** |
| **Test Loss** | **0.1573** |

**Table 5.** Overall test set performance metrics.

### 3.3 Per-Class Performance

| Class | Precision | Recall | F1-Score | AUC-ROC | Average Precision | Support |
|-------|-----------|--------|----------|---------|-------------------|---------|
| Bleeding | 96.93% | 96.34% | 96.64% | 0.9971 | 0.9919 | 164 |
| Ischemia | 91.72% | 91.18% | 91.45% | 0.9918 | 0.9687 | 170 |
| Normal | 97.60% | 97.89% | 97.74% | 0.9951 | 0.9973 | 664 |

**Table 6.** Per-class test set performance metrics.

The Normal class achieved the highest F1-score (97.74%), followed by Bleeding (96.64%) and Ischemia (91.45%). The relatively lower performance on the Ischemia class is consistent with clinical experience, as ischemic stroke findings on CT can be subtle, particularly in the early hours after onset, and may overlap visually with normal brain parenchyma.

### 3.4 Confusion Matrix Analysis

|  | Pred: Bleeding | Pred: Ischemia | Pred: Normal |
|--|---------------|----------------|--------------|
| **True: Bleeding** | **158** | 3 | 3 |
| **True: Ischemia** | 2 | **155** | 13 |
| **True: Normal** | 3 | 11 | **650** |

**Table 7.** Confusion matrix on the test set.

Key observations from the confusion matrix:

- **Bleeding** was the most reliably classified pathological class, with only 6 misclassifications out of 164 (3 as Ischemia, 3 as Normal). This is expected, as hemorrhagic stroke presents with distinctive hyperdense (bright) regions on CT.
- **Ischemia** had the highest misclassification rate, with 15 of 170 images misclassified (2 as Bleeding, 13 as Normal). The predominant confusion with Normal reflects the subtle nature of early ischemic changes on CT.
- **Normal** images were misclassified in 14 of 664 cases (3 as Bleeding, 11 as Ischemia), representing a false positive rate of only 2.1%.

### 3.5 Validation Set Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.59% |
| Macro F1-Score | 96.71% |

**Table 8.** Validation set performance of the best model.

Validation performance was slightly higher than test performance, which is expected as the model was selected based on validation metrics.

### 3.6 ROC and Precision-Recall Analysis

The ROC curves demonstrate near-perfect discrimination ability across all three classes:

- Bleeding: AUC = 0.9971
- Ischemia: AUC = 0.9918
- Normal: AUC = 0.9951
- Micro-average: AUC = 0.9959

Average Precision scores confirm strong performance across the full precision-recall spectrum:

- Bleeding: AP = 0.9919
- Ischemia: AP = 0.9687
- Normal: AP = 0.9973

---

## 4. Discussion

### 4.1 Key Findings

This study demonstrates that a ResNet-50 model with ImageNet transfer learning can achieve high classification performance (96.49% accuracy, 0.9528 macro F1) for distinguishing between hemorrhagic stroke, ischemic stroke, and normal brain CT images. Several key findings emerge:

**Transfer learning effectiveness.** The pre-trained ImageNet weights provided a strong initialization, enabling the model to converge rapidly and achieve high performance despite the relatively modest dataset size (6,650 images). The model reached 91% validation accuracy by epoch 3, demonstrating the efficiency of transfer learning for medical imaging tasks.

**Class-specific performance patterns.** The performance hierarchy (Normal > Bleeding > Ischemia) aligns with clinical expectations. Hemorrhagic stroke produces highly distinctive hyperdense regions on CT that are relatively straightforward for both human radiologists and automated systems to detect. In contrast, early ischemic changes are notoriously subtle on CT and represent a well-known challenge in neuroradiology [4]. The model's 91.45% F1-score for Ischemia, while lower than the other classes, still represents clinically meaningful performance for a task that challenges even experienced radiologists.

**Effective class imbalance mitigation.** Despite the Normal class being approximately 4 times larger than either pathological class, the combination of weighted random sampling and class-weighted loss successfully prevented the model from developing a bias toward the majority class, as evidenced by the balanced per-class performance metrics.

### 4.2 Clinical Implications

The potential clinical applications of this system include:

1. **Emergency triage support:** Automated prioritization of CT scans showing signs of stroke in busy emergency departments, reducing time-to-treatment for stroke patients.
2. **Quality assurance:** Serving as a second reader to flag potentially missed findings, particularly subtle ischemic changes.
3. **Resource-limited settings:** Providing decision support in facilities that lack 24/7 neuroradiology coverage.

The high negative predictive value for both stroke subtypes (implied by the Normal class recall of 97.89%) suggests the system could be particularly useful as a screening tool to identify scans requiring urgent radiologist review.

### 4.3 Limitations

Several limitations of this study should be acknowledged:

1. **Dataset characteristics:** The dataset may not fully represent the diversity of stroke presentations seen in clinical practice, including varying stroke ages, patient demographics, CT scanner manufacturers, and imaging protocols. External validation on multi-center datasets is needed to assess generalizability.

2. **Binary slice-level classification:** This study performs classification on individual CT slices rather than volumetric analysis. In clinical practice, a single patient exam typically comprises multiple slices, and integrating predictions across an entire exam volume could improve diagnostic accuracy.

3. **Lack of clinical metadata:** Patient demographics, symptom onset time, and clinical history were not available and could potentially improve classification performance, particularly for ischemic stroke.

4. **Simplified class taxonomy:** The three-class scheme does not distinguish between hemorrhage subtypes (e.g., epidural, subdural, subarachnoid, intraparenchymal) or ischemic stroke severity, which may limit clinical utility in certain scenarios.

5. **No comparison with radiologist performance:** Direct comparison with human expert performance would provide important context for evaluating the clinical relevance of these results.

### 4.4 Future Work

Several directions for future research are identified:

1. **Multi-center external validation** using independent datasets to assess model robustness and generalizability.
2. **Hemorrhage subtype classification** to provide more granular diagnostic information.
3. **Volume-level prediction** by aggregating slice-level predictions across entire CT exams.
4. **Explainability analysis** using gradient-weighted class activation mapping (Grad-CAM) to visualize the regions driving model predictions, enhancing clinical trust and interpretability.
5. **Segmentation integration** leveraging the available external test set masks for combined classification and lesion delineation.
6. **Comparison with additional architectures** such as EfficientNet, Vision Transformers (ViT), and DenseNet to establish comprehensive baselines.

---

## 5. Conclusion

This study presents a deep learning approach for automated classification of brain CT images into hemorrhagic stroke, ischemic stroke, and normal categories using a ResNet-50 model with ImageNet transfer learning. The model achieved 96.49% overall accuracy, 95.28% macro F1-score, and a Cohen's kappa of 0.9299 on the held-out test set, with per-class AUC values exceeding 0.99 for all categories. These results demonstrate the viability of deep learning-based automated stroke classification from CT scans and suggest potential for integration into clinical decision support workflows. Future work should focus on multi-center validation, explainability analysis, and clinical integration studies to assess real-world utility.

---

## References

[1] Feigin VL, Stark BA, Johnson CO, et al. Global, regional, and national burden of stroke and its risk factors, 1990-2019: a systematic analysis for the Global Burden of Disease Study 2019. *The Lancet Neurology*. 2021;20(10):795-820.

[2] Virani SS, Alonso A, Aparicio HJ, et al. Heart Disease and Stroke Statistics-2021 Update: A Report From the American Heart Association. *Circulation*. 2021;143(8):e254-e743.

[3] Powers WJ, Rabinstein AA, Ackerson T, et al. Guidelines for the Early Management of Patients With Acute Ischemic Stroke: 2019 Update to the 2018 Guidelines for the Early Management of Acute Ischemic Stroke. *Stroke*. 2019;50(12):e344-e418.

[4] Wardlaw JM, Mielke O. Early signs of brain infarction at CT: observer reliability and outcome after thrombolytic treatment--systematic review. *Radiology*. 2005;235(2):444-453.

[5] Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. *Medical Image Analysis*. 2017;42:60-88.

[6] Chilamkurthy S, Ghosh R, Tanamala S, et al. Deep learning algorithms for detection of critical findings in head CT scans: a retrospective study. *The Lancet*. 2018;392(10162):2388-2396.

[7] Qiu W, Kuang H, Teleg E, et al. Machine Learning for Detecting Early Infarction in Acute Stroke with Non-Contrast-enhanced CT. *Radiology*. 2020;294(3):638-644.

[8] Nishi H, Oishi N, Ishii A, et al. Deep Learning-Derived High-Level Neuroimaging Features Predict Clinical Outcomes for Large Vessel Occlusion. *Stroke*. 2020;51(5):1484-1492.

[9] Tajbakhsh N, Shin JY, Gurudu SR, et al. Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning? *IEEE Transactions on Medical Imaging*. 2016;35(5):1299-1312.

[10] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016:770-778.

---

## Appendix A: Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Model | ResNet-50 (ImageNet-V2 weights) |
| Total Parameters | 23,514,179 |
| Trainable Parameters | 23,514,179 |
| Input Resolution | 224 x 224 pixels |
| Batch Size | 32 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| LR Scheduler | Cosine Annealing (T_max=30) |
| Loss Function | Weighted Cross-Entropy |
| Dropout Rate | 0.5 |
| Early Stopping | Patience=7 (on validation macro F1) |
| Epochs Trained | 27 (best at epoch 20) |
| Training Time | 72.4 minutes |
| Hardware | Apple M-series GPU (MPS backend) |
| Random Seed | 42 |

**Table A1.** Complete experimental configuration.

## Appendix B: Training History Summary

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Best? |
|-------|-----------|-----------|----------|---------|--------|-------|
| 1 | 0.6175 | 59.10% | 0.5111 | 72.52% | 0.7224 | Yes |
| 2 | 0.2894 | 82.90% | 0.4109 | 73.72% | 0.7324 | Yes |
| 3 | 0.2171 | 87.50% | 0.3396 | 90.97% | 0.8773 | Yes |
| 4 | 0.1704 | 90.03% | 0.2121 | 89.97% | 0.8819 | Yes |
| 8 | 0.0633 | 96.26% | 0.2024 | 92.28% | 0.9051 | Yes |
| 9 | 0.0656 | 96.26% | 0.1882 | 93.88% | 0.9228 | Yes |
| 13 | 0.0372 | 98.07% | 0.1673 | 95.49% | 0.9414 | Yes |
| 14 | 0.0291 | 98.13% | 0.1655 | 96.29% | 0.9508 | Yes |
| 19 | 0.0153 | 99.08% | 0.1491 | 97.19% | 0.9626 | Yes |
| 20 | 0.0085 | 99.59% | 0.1605 | 97.59% | 0.9671 | Yes |
| 27 | 0.0046 | 99.70% | 0.1642 | 96.89% | 0.9575 | Stop |

**Table B1.** Training history at key epochs (epochs with best model improvement and final epoch).

## Appendix C: Generated Figures

The following figures were generated during the experiment and are saved in the `results/` directory:

1. **training_curves.png** - Loss, accuracy, and F1-score curves over training epochs
2. **class_distribution.png** - Class distribution across train/validation/test splits
3. **confusion_matrix_test.png** - Confusion matrix on the test set (counts and percentages)
4. **confusion_matrix_val.png** - Confusion matrix on the validation set
5. **roc_curves_test.png** - Per-class and micro-average ROC curves on the test set
6. **pr_curves_test.png** - Precision-recall curves on the test set
7. **sample_predictions.png** - Grid of sample predictions with true/predicted labels
