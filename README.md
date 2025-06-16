# 🎯 Support Vector Classifier in AMPL

Implementation of primal/dual SVM formulations and RBF kernel for binary classification. Developed for Constrained Optimization course at UPC.

## 📌 Project Context

Developed for the *Constrained Optimization* course in the **Bachelor's Degree in Data Science and Engineering** at **Universitat Politècnica de Catalunya (UPC)**, this project implements Support Vector Machines (SVM) using AMPL with Gurobi solver.

## 📊 Problem Definition
- **Objective:** Find optimal separating hyperplanes for binary classification
- **Formulations:**
  - Primal quadratic problem (maximize margin + minimize errors)
  - Dual quadratic problem (kernelizable formulation)
  - RBF kernel extension for non-linear separation
- **Key Parameter:** Regularization constant `c ∈ [0,10]` (optimal `c=2` found)

## ⚙️ Methodology

### 1. Core Implementations
- **Primal Formulation**  
  ```AMPL
  minimize obj: 0.5*sum(w^2) + c*sum(s);
  subject to y_i(w·x_i + γ) + s_i ≥ 1
