# 🎯 Support Vector Classifier in AMPL

Implementation of primal/dual SVM formulations and RBF kernel for binary classification. Developed for Constrained Optimization course at UPC.

## 📌 Project Context

Developed for the *Mathematical Optimization* course in the **Bachelor's Degree in Data Science and Engineering** at **Universitat Politècnica de Catalunya (UPC)**, this project implements Support Vector Machines (SVM) using AMPL with Gurobi solver.

## 📊 Problem Definition

- **Objective:** Find optimal separating hyperplanes for binary classification by finding two parallel hyperplanes ($w^{T}x+\gamma$) that separate two classes such that classification errors are minimized and the margin between hyperplanes is maximized.
- **Formulations:**
  - Primal quadratic problem (maximize margin + minimize errors)
  - Dual quadratic problem (kernelizable formulation)
  - RBF kernel extension for non-linear separation
- **Key Parameter:** Regularization constant `c` (parameter $\nu$ in the formulation) weights the opposite objectives of maximizing margin and minimizing errors. An optimal `c=2` was decided after some tries.

## ⚙️ Methodology

### 1. Core Implementations

- **Primal Formulation**
  ```AMPL
  #parameters
  param m > 0;
  param n > 0;
  param c >= 0;
  param y{1..m};
  param x{{1..m},{1..n}};

  #variables
  var w{1..n};
  var gamma;
  var s{1..m} >= 0;

  #objective function: primal quadratic formulation
  minimize objf1: 0.5*sum{i in {1..n}}(w[i]^2) + c*sum{i in {1..m}}s[i];

  #constraint
  subject to c1{i in {1..m}}: -y[i]*(sum{j in {1..n}}(x[i,j]*w[j])+gamma) -s[i]+1<=0;
  ```
 
- **Dual Formulation (Linear Kernel)**
  ```AMPL
  #parameters
  param m > 0;
  param n > 0;
  param c >= 0;
  param y{1..m};
  param x{{1..m},{1..n}};

  #variables
  var lambda{1..m} >=0 <=C;

  #objective function: primal quadratic formulation
  #linear kernel is used: Kij = sum{k in 1..n} A[i,k]*A[j,k]
  maximize objf2: sum{i in {1..m}}lambda[i] - 0.5*sum{i in {1..m}, j in {1..m}}(lambda[i]*y[i]*lambda[j]*y[j]*(sum{k in {1..n}} (x[i,k]*x[j,k])));

  #constraints
  subject to c2: sum{i in {1..m}}(y[i]*lambda[i]) = 0;
  ```
 
- **Dual Formulation (RBF Kernel)**
  ```AMPL
  #parameters
  param m;
  param n;
  param c;
  param y{1..m};
  param x{1..m,1..n};

  #variables
  var lambda{1..m} >=0, <= c;

  #gaussian kernel is used: Kij -> exp(-(1/(2*(sigma)^2)) * (sum{k in 1..n)(x[i,k]-x[j,k])^2))
  maximize objf3: sum{i in 1..m}lambda[i] - 0.5 * sum{i in 1..m, j in 1..m) lambda[i]*y[i]*lambda[j]*y[j]*exp(-1/n*(sum{k in 1..n}(x[i,k]-x[j,k])^2));

  #constraints
  subject to c3: (sum{i in {1..m}}lambda[i]*y[i]) = 0;
  ```
 

### 2. Model Validation

- **Datasets:**
  - **Dataset 1 (Generated):** Custom-generated with a specific seed, consisting of 100 training points and 50 testing points in $\mathbb{R}^4$.
  - **Dataset 2 (OpenML Diabetes):** Downloaded from OpenML website, related to diabetes diagnostics. Contains 768 instances (700 used: 350 for training, 350 for testing) with 8 numeric features and one symbolic feature for class value.
  - **Dataset 3 (Sklearn Swiss Roll):** Generated using `sklearn.datasets.make_swiss_roll(m)` from the Python `sklearn` library, with 300 training points and 50 testing points in $\mathbb{R}^3$. This dataset is linearly non-separable.
- **Metrics:** Training and testing accuracies are computed.
- **Workflow:** AMPL `.run` files (`svm_primal.run`, `svm_dual.run`) calculate accuracies by comparing predicted values ($w^{T}\cdot x+\gamma$) with true labels. For the dual problem, `w` and `gamma` are first calculated from the dual solution.

## 📈 Key Findings

- **Model Consistency:** Primal and dual solutions coincided for Dataset 1, validating the models.
- **Accuracy Overview:**
  - **Dataset 1 (Linear Kernel):** 93% training accuracy, 88% testing accuracy.
  - **Dataset 2 (Linear Kernel):** 77% training accuracy, 81% testing accuracy.
  - **Dataset 3 (RBF Kernel):** Achieved the most precise results with 100% training accuracy and 98% testing accuracy, demonstrating superior performance on non-linarly separable data.
- **Parameter Analysis:** The regularization parameter `c` was set to 2. For the RBF kernel, `gamma` (related to $\sigma$) was used as $1/n$.

## 🛠️ Tools & Libraries

- **AMPL:** For model implementation
- **Gurobi:** As the solver for optimization problems
- **Python:** For data generation and formatting (e.g., `sklearn.datasets.make_swiss_roll`)

## 👥 Authors

- Adrián Cerezuela Hernández
- Ramon Ventura Navarro

## 📚 Course

**Constrained Optimization**
Universitat Politècnica de Catalunya (UPC)
```
