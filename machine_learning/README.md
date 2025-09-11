# 🤖 Machine Learning – Detailed Notes for Interviews & Practice

This document provides a **structured and detailed overview of Machine Learning concepts**, mathematical foundations, algorithms, and practical implementations. It is organized in a **theory + formula + intuition + use case** manner, ideal for interviews and project prep.

---

## 1. Introduction to Machine Learning and Key Concepts

- **Artificial Intelligence (AI)**  
  - Broadest concept → Systems performing tasks without human intervention.  
  - Examples: Netflix recommendations, Amazon product suggestions, YouTube ads, Tesla self-driving.  

- **Machine Learning (ML)**  
  - Subset of AI.  
  - Uses statistical techniques to **analyze, learn, and predict from data**.  

- **Deep Learning (DL)**  
  - Subset of ML.  
  - Uses **multi-layered neural networks** to mimic brain-like structures.  
  - Effective for images, speech, and NLP.  

- **Data Science (DS)**  
  - Encompasses **data analysis + ML + DL**.  
  - Goal = Solve **business problems** end-to-end.  

### Types of ML
1. **Supervised Learning**  
   - Data: Input + Output labels.  
   - **Regression** → Predict continuous values (e.g., predict weight from age).  
   - **Classification** → Predict discrete categories (e.g., pass/fail).  

2. **Unsupervised Learning**  
   - Data: Only Input (no labels).  
   - **Clustering** → Group similar points (e.g., customer segmentation).  
   - **Dimensionality Reduction** → Reduce feature space (e.g., PCA).  

3. **Reinforcement Learning (RL)**  
   - Agent learns via **rewards/penalties**.  
   - Example: Robotics, games.  

### Algorithms Covered
- **Regression**: Linear, Ridge, Lasso  
- **Classification**: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM  
- **Ensemble**: AdaBoost, Gradient Boosting, XGBoost  
- **Instance-based**: KNN  
- **Unsupervised**: K-Means, Hierarchical Clustering, DBSCAN  
- **Dimensionality Reduction**: PCA, LDA  

---

## 2. Linear Regression: Theory, Mathematics & Performance Metrics
⏱️ *[18:09 → 56:05]*

- **Linear Regression Model**:  
  \[
  y = \theta_0 + \theta_1 x
  \]

- **Cost Function (MSE / Least Squares)**:  
  \[
  J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m \big(h_\theta(x_i) - y_i\big)^2
  \]

- **Gradient Descent Update Rule**:  
  \[
  \theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}
  \]

  - α = Learning rate  
  - Convex cost function → No local minima issues.  

### Performance Metrics
- **R² (Coefficient of Determination)**: Proportion of variance explained.  
- **Adjusted R²**: Penalizes irrelevant features → avoids artificial inflation.  

📌 **Regression vs Classification**  
- Regression → Continuous output.  
- Classification → Categorical output.  

---

## 3. Ridge & Lasso Regression (Regularization)
⏱️ *[56:05 → 01:27:44]*

- **Ridge Regression (L2 penalty)**:  
  \[
  J(\theta) = \text{MSE} + \lambda \sum_j \theta_j^2
  \]  
  Shrinks coefficients, reduces overfitting.  

- **Lasso Regression (L1 penalty)**:  
  \[
  J(\theta) = \text{MSE} + \lambda \sum_j |\theta_j|
  \]  
  Shrinks coefficients + performs **feature selection** (forces some to 0).  

- **Hyperparameter λ (alpha)**:  
  - Large → Higher bias, less variance.  
  - Small → Lower bias, higher variance.  

### Assumptions of Linear Models
- Linearity  
- Normality of errors  
- Feature standardization (zero mean, unit variance)  
- Handle multicollinearity (drop correlated features).  

---

## 4. Logistic Regression & Classification Metrics
⏱️ *[56:05 → 01:27:44]*

- **Sigmoid Function**:  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]  
  Maps output → (0,1).  

- **Cost Function (Log-Loss)**:  
  \[
  J(\theta) = -\frac{1}{m} \sum \big[ y \log(h_\theta(x)) + (1-y)\log(1-h_\theta(x)) \big]
  \]  
  Convex → No local minima.  

### Confusion Matrix Metrics
- **Accuracy** = (TP + TN) / Total  
- **Precision** = TP / (TP + FP)  
- **Recall (Sensitivity)** = TP / (TP + FN)  
- **F1-Score** = 2 * (Precision * Recall) / (Precision + Recall)  

📌 Use cases:  
- Spam detection → prioritize precision.  
- Cancer diagnosis → prioritize recall.  

---

## 5. Practical Implementations
⏱️ *[01:27:44 → 02:14:09]*

- **Linear, Ridge, Lasso Regression**:  
  - Datasets: *Boston Housing*.  
  - Tools: `train_test_split`, `cross_val_score`, `GridSearchCV`.  

- **Logistic Regression**:  
  - Dataset: *Breast Cancer*.  
  - Metrics: Confusion matrix, precision, recall, F1.  

- **Naive Bayes**:  
  - Based on **Bayes’ theorem**:  
    \[
    P(A|B) = \frac{P(B|A)P(A)}{P(B)}
    \]  
  - Assumes feature independence.  
  - Example: Tennis dataset (categorical probabilities).  

- **K-Nearest Neighbors (KNN)**:  
  - Predict label by **majority of k neighbors**.  
  - Distance: Euclidean, Manhattan.  
  - Sensitive to outliers & scaling.  

- **Decision Trees**:  
  - Splits data based on **Entropy or Gini impurity**.  
  - Entropy:  
    \[
    H(p) = -p \log_2 p - (1-p) \log_2 (1-p)
    \]  
  - Gini:  
    \[
    G = 1 - \sum p_i^2
    \]  
  - **Pruning** avoids overfitting.  

- **Ensemble Methods**:  
  - **Bagging** (Bootstrap Aggregation) → Random Forest.  
  - **Boosting** → Sequential, focus on hard examples.  

- **AdaBoost**  
  - Weak learners = decision stumps.  
  - Iteratively reweight misclassified samples.  

- **XGBoost**  
  - Optimized Gradient Boosting.  
  - Uses regularization, residuals, similarity scores.  

- **Support Vector Machines (SVM)**  
  - Separates classes with **maximum-margin hyperplane**.  
  - Controlled by penalty parameter **C**.  
  - **Kernel trick** → Handle non-linear data.  

---

## 6. Clustering Algorithms
⏱️ *[02:14:09 → 02:56:00]*

### K-Means Clustering
- Iterative algorithm: Initialize centroids → Assign points → Recompute → Repeat.  
- **Elbow Method**: Plot WCSS vs k → Find "elbow".  
- **Silhouette Score**:  
  \[
  S = \frac{b - a}{\max(a, b)}
  \]  
  - Near +1 → Good clusters.  

### Hierarchical Clustering
- Builds nested clusters using **dendrogram**.  
- Cut longest vertical line without crossing → Optimal clusters.  
- Expensive → Better for small datasets.  

### DBSCAN
- Density-Based Spatial Clustering.  
- Parameters:  
  - **ε (epsilon)** = neighborhood radius.  
  - **minPts** = minimum neighbors.  
- Finds arbitrarily shaped clusters.  
- Handles noise/outliers better than k-means.  

---

## 7. Bias–Variance Tradeoff

- **Bias**: Error due to simplifying assumptions.  
  - High Bias = Underfitting.  

- **Variance**: Error due to sensitivity to training data.  
  - High Variance = Overfitting.  

📌 **Goal**: Find balance between bias and variance.  

---

## 8. Summary

- Covered **Supervised, Unsupervised, and Ensemble learning**.  
- Regression → Linear, Ridge, Lasso.  
- Classification → Logistic, Naive Bayes, KNN, Decision Trees, SVM.  
- Ensemble → Bagging, Boosting, XGBoost.  
- Clustering → K-Means, Hierarchical, DBSCAN.  
- Key metrics: R², Adjusted R², Precision, Recall, F1, Silhouette Score.  
- Concepts: Bias–Variance, Overfitting, Underfitting.  

📌 These form the **core ML toolkit for interviews & real-world projects**.  

---
