Avila Dataset Classification using Neural Networks

# 🧠 Project Overview
This project focuses on **neural network–based document classification** using the **Avila dataset**, a real-world handwriting dataset.

The goal was to design, train, and evaluate a **Multi-Layer Perceptron (MLP)** that can accurately classify manuscript samples into distinct writer classes.

The work emphasizes:
- Clean data preparation
- Robust cross-validation
- Neural network architecture tuning
- Interpretable performance evaluation (accuracy, confusion matrix, ROC/AUC)

# 📘 Dataset Description – Avila Dataset
The **Avila dataset** consists of numerical features extracted from **historical handwritten manuscripts**.

# Dataset Characteristics
- **Features:** 10 numerical attributes describing handwriting structure
- **Original Classes:** 12 writer classes
- **Subset Used:** **A, F, G, H**
- **Task Type:** Multiclass classification (4 classes)

# Why this dataset?
This dataset closely mirrors real-world document intelligence problems such as:
- Document categorization
- Handwriting-based author identification
- OCR post-processing validation
- Automated archival and indexing systems

# 🔧 Data Preparation & Preprocessing
Steps implemented in the notebook:
1. Filtered classes to include only `A`, `F`, `G`, and `H`
2. Extracted:
   - `X` → numerical feature matrix
   - `y` → class labels
3. Applied label encoding / one-hot encoding
4. Split data into:
   - Training set (80%)
   - Test set (20%)
5. Used 5-fold cross-validation to estimate generalization performance

This ensured:
- No data leakage
- Stable evaluation across folds
- Reliable inference error estimation

<img width="552" height="487" alt="image" src="https://github.com/user-attachments/assets/ed6fe754-c6f9-4add-971e-d89874a88eeb" />


# 🏗️ Model Architecture
We used **scikit-learn’s MLPClassifier** with the following configuration:
- Hidden layers: `(128, 64, 32)`
- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.003
- Max iterations: 1000
- Random state: Fixed for reproducibility

This architecture was selected after comparing simpler baseline models and observing improved generalization behavior.

# 📊 Model Evaluation Strategy
The notebook evaluates the model using multiple complementary metrics:

- **Accuracy:** overall classification correctness
- **Confusion Matrix:** correct predictions per class + misclassification patterns
- **Classification Report:** precision, recall, and F1-score per class
- **ROC Curves & AUC (Multiclass – One-vs-Rest):**
  - Per-class ROC curves
  - Macro AUC (class-balanced performance)
  - Micro AUC (overall probabilistic quality)

This evaluation setup ensures both statistical rigor and business interpretability.

<img width="404" height="341" alt="image" src="https://github.com/user-attachments/assets/9be07f70-ee07-4e70-be67-b1dc68740cce" />

<img width="404" height="341" alt="image" src="https://github.com/user-attachments/assets/af480c81-4d49-47d3-8246-9744464c5a4e" />


# 📈 Results

# Cross-Validation (CV)
- Accuracy: **95.8%**
- Macro AUC: **0.988**
- Micro AUC: **0.990**

# Test Dataset
- Accuracy: **97.6%**
- Macro AUC: **0.981**
- Micro AUC: **0.984**

✅ Interpretation:
- High test accuracy (>97%) indicates low generalization error
- ROC curves show excellent class separability across A, F, G, H
- Confusion matrices reveal only minor misclassifications


# 🧩 Tech Stack
- Python
- NumPy, pandas
- scikit-learn (MLPClassifier, metrics, KFold/StratifiedKFold)
- Matplotlib
