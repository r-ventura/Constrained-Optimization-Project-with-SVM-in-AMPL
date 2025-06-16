## 📐 Constrained Optimization: Support Vector Classifier in AMPL

Implementation and validation of primal and dual quadratic formulations for Support Vector Classifiers (SVC) using AMPL, including an alternative implementation with an RBF kernel.

## 📌 Project Context

Developed as a lab assignment for the **Bachelor's Degree in Data Science & Engineering** at **Universitat Politècnica de Catalunya (UPC)**, this project focuses on the practical application of constrained optimization to machine learning, specifically for Support Vector Classifiers.

## 📝 Project Overview

This report details the development of a Support Vector Classifier in AMPL, aiming to maximize the margin between separating hyperplanes while minimizing classification errors. The project explores both primal and dual quadratic formulations of the SVC problem.

## 🛠️ Implementation Details

### 1. Primal and Dual Quadratic Formulations
- [cite_start]**Objective:** Find two parallel hyperplanes ($w^{T}x+\gamma$) to separate two classes, minimizing errors and maximizing margin[cite: 5].
- [cite_start]**Weighting Parameter:** Parameter $\nu$ (referred to as `c` in AMPL code) balances these objectives[cite: 6].
- **Formulations Implemented:**
    - **Primal Quadratic Formulation:**
        $min_{(w,y,s)\in\mathbb{R}^{s+1+n}}\frac{1}{2}w^{T}w+\nu\sum_{l=1}^{m}s_{l}$
        s.to $y_{i}(w^{T}x_{i}+\gamma)+s_{i}-1\le0$
        [cite_start]$-s_{i}\le0$ [cite: 8, 9]
    - **Dual Quadratic Formulation (with Linear Kernel):**
        $max_{i\in\mathbb{R}^{n}}\sum_{i=1}^{m}\hat{\lambda}_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\hat{\lambda}_{i}y_{i}\hat{\lambda}_{j}y_{j}K_{ij}$
        s.to $\sum_{i=1}^{m}\lambda_{i}y_{i}=0$
        [cite_start]$0\le\lambda_{l}\le v$ [cite: 11]

### 2. Model Validation and Analysis
- **Datasets Used:**
    - [cite_start]**Dataset 1 (Generated):** Custom-generated dataset with 100 training points and 50 testing points[cite: 21].
    - [cite_start]**Dataset 2 (Internet-Found):** Diabetes diagnostics dataset from OpenML, with 700 instances (350 for training, 350 for testing) and 8 numeric features[cite: 25, 26].
- [cite_start]**Validation Process:** Accuracies were computed for both training and testing sets using `svm_primal.run` and `svm_dual.run`[cite: 31].
- [cite_start]**Classification Criteria:** Predictions are based on the sign of $w^{T}\cdot x+\gamma=0$, with positive results classified as 1 and negative as -1[cite: 43].

### 3. Dual Model Implementation with RBF Kernel
- [cite_start]**Kernel Used:** Radial Basis Function (RBF) or Gaussian kernel, defined as $K(x,y)=e^{-\frac{||x-y||^{2}}{2\sigma^{2}}}$[cite: 67].
    - [cite_start]$\sigma$ is related to $\text{gamma}$ (parameter in implementation) where $\text{gamma} = \frac{1}{2\sigma^{2}}$[cite: 69, 70].
- [cite_start]**Dataset:** `sklearn.datasets.make_swiss_roll(m)` generated a linearly non-separable dataset with 350 instances (300 training, 50 testing) and 3 features[cite: 74].

## 📈 Key Results

- **Model Validation (Linear Kernel):**
    - [cite_start]For Dataset 1, primal and dual solutions coincided, yielding 93% training accuracy and 88% testing accuracy[cite: 59, 60].
    - [cite_start]For Dataset 2, both dual solutions also coincided, with 77% training accuracy and 81% testing accuracy[cite: 63, 64].
- **RBF Kernel Implementation:**
    - [cite_start]Achieved the highest precision among all implementations, with 100% training accuracy and 98% testing accuracy on the linearly non-separable dataset[cite: 79].

## 👥 Authors
- Adrián Cerezuela Hernández
- Ramon Ventura Navarro

## 📚 Course
**Constrained Optimization**
Universitat Politècnica de Catalunya (UPC)
