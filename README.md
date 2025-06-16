# 🎯 Support Vector Classifier in AMPL

Implementation of primal/dual SVM formulations and RBF kernel for binary classification. Developed for Mathematical Optimization course at UPC.

## 📌 Project Context

Developed for the *Mathematical Optimization* course in the **Bachelor's Degree in Data Science and Engineering** at **Universitat Politècnica de Catalunya (UPC)**, this project implements Support Vector Machines (SVM) using AMPL with Gurobi solver.

## 📊 Problem Definition

- **Objective:** Find optimal separating hyperplanes for binary classification by finding two parallel hyperplanes ($w^{T}x+\gamma$) that separate two classes such that classification errors are minimized and the margin between hyperplanes is maximized.
- **Formulations:**
  - Primal quadratic problem (maximize margin + minimize errors)
  - Dual quadratic problem (kernelizable formulation)
  - RBF kernel extension for non-linear separation
- **Key Parameter:** Regularization constant `c` (parameter $\nu$ in the formulation) weights the opposite objectives of maximizing margin and minimizing errors. An optimal `c=2` was decided after some tries.

## 🛠️ Tools & Libraries

- **AMPL:** For model implementation
- **Gurobi:** As the solver for optimization problems
- **Python:** For data generation and formatting (e.g., `sklearn.datasets.make_swiss_roll`)
