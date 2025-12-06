# Product Requirements Document (PRD)
## Subject: Support Vector Machine (SVM) Classification Assignment

**Date:** October 26, 2023
**Course:** AI / Machine Learning
**Dataset:** Iris Flower Dataset

---

## 1. Executive Summary
The objective of this assignment is to implement a Support Vector Machine (SVM) to classify the Iris dataset. The assignment is divided into two distinct phases: a standard implementation using existing libraries and an advanced manual implementation that requires solving the optimization problem from scratch.

The core challenge of the advanced phase is adapting the binary nature of SVM to handle the multi-class nature (3 classes) of the Iris dataset using a recursive binary decomposition strategy.

## 2. Dataset Specifications
* **Source:** [Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
* **Classes:** 3 (Setosa, Versicolor, Virginica) - Referred to as A, B, and C in the logic section.
* **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width).

## 3. Functional Requirements

### 3.1. Phase I: Standard Library Implementation
* **Objective:** Establish a performance baseline using industry-standard tools.
* **Tools:** Python, Scikit-Learn (`sklearn`).
* **Requirements:**
    1.  Load the Iris dataset.
    2.  Train an SVM model using `sklearn.svm`.
    3.  Evaluate the accuracy of the model on the 3-class data.

### 3.2. Phase II: Manual Optimization (Advanced)
* **Objective:** Demonstrate deep understanding of the mathematical principles behind SVM by implementing the solver manually without `sklearn`'s black-box methods.
* **Constraint:** The implementation must solve the optimization problem (finding weights $w$ and bias $b$) directly.
* **Multi-Class Strategy:** Since SVM is natively a binary classifier, students must implement a **One-vs-Rest** or **Recursive Split** strategy to handle the 3 classes.

## 4. Technical Logic: Multi-Class Binary Decomposition
*Based on the provided whiteboard sketch and lecture instructions.*

To classify 3 groups (A, B, C) using a binary solver, the data must be split hierarchically.

### Logic Flow
1.  **Split 1 (Global Classification):**
    * Define two super-groups:
        * **Group $\alpha$:** Contains Class A.
        * **Group $\beta$:** Contains Class B + Class C.
    * Train the SVM to separate $\alpha$ from $\beta$.
2.  **Split 2 (Sub-group Classification):**
    * If a data point is classified as $\beta$, a second SVM model is required.
    * Train the second SVM to separate Class B from Class C.

### Decision Tree Visualization

```mermaid
graph TD
    Input[Input Data Point] --> Node1{SVM Model 1}
    Node1 -- Predicts Alpha --> ClassA[Result: Class A]
    Node1 -- Predicts Beta --> Node2{SVM Model 2}
    Node2 -- Predicts Left --> ClassB[Result: Class B]
    Node2 -- Predicts Right --> ClassC[Result: Class C]
. Implementation Scope & Assessment Policy
```
## 5.1. Difficulty Level
Phase II represents a Master's degree level engineering challenge. It requires translating complex optimization math into code.

5.2. Opt-Out Clause (Scope Management)
Assessment: Phase II is designed as a self-assessment tool for the student's data science capabilities.

Policy: Students who find the manual optimization and recursive logic implementation prohibitively complex or time-consuming may opt to submit only Phase I (The Library Implementation).

Penalty: There is no penalty for omitting Phase II if the student determines it is outside their current capacity.

5.3. Success Metrics
P0 Deliverable: A working sklearn script with high accuracy on the Iris dataset.

P1 Deliverable: A manual script where the custom recursive logic yields accuracy comparable to the P0 baseline.

Approvals:

AI Course Professor