# SVM Classification on Iris Dataset

This project implements a Support Vector Machine (SVM) classifier for the Iris dataset. It follows the specifications outlined in the `prd.md` file and is divided into two main phases:

1.  **Phase I:** A baseline implementation using the `scikit-learn` library.
2.  **Phase II:** An advanced manual implementation of a linear SVM from scratch, using a recursive binary decomposition strategy to handle the multi-class problem.

## Project Structure

```
.
├── logs/
│   └── execution.log       # Log file for program output
├── plots/
│   ├── iris_data_visualization.png
│   ├── decision_boundary_phase_one.png
│   └── decision_boundary_phase_two.png
├── src/
│   ├── __init__.py
│   ├── phase_one.py        # Scikit-learn implementation
│   ├── phase_two.py        # Manual SVM implementation
│   └── plotting.py         # Plotting utility functions
├── prd.md                  # Product Requirements Document
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── run.py                  # Main execution script
```

## Setup and Execution

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the project:**
    ```bash
    python run.py
    ```
    This script will execute both Phase I and Phase II, generate the plots in the `/plots` directory, and save all console output to `logs/execution.log`.

## Results Explained (Simply Put)

Imagine you have a basket of mixed Iris flowers, and you want to sort them into three piles based on their species: **Setosa**, **Versicolor**, and **Virginica**. You can't ask the flowers their names, but you *can* measure their petals.

This project is like building a robot to do that sorting for you.

### 1. The Data (The Flowers)
First, we looked at the flowers we already knew. We plotted them on a chart based on their **Petal Length** and **Petal Width**.

![Iris Data Visualization](plots/iris_data_visualization.png)

**What are we looking at?**
*   The **horizontal line (X-axis)** represents the *length* of the flower petal.
*   The **vertical line (Y-axis)** represents the *width* of the flower petal.
*   Each **colored dot** is one specific flower we measured.

**Conclusion:**
*   **The Blue Dots (Setosa)** are clustered in the bottom-left corner (small length and width). They are far apart from the others, meaning they will be very easy for our robot to identify.
*   **The Red (Versicolor) and Gray (Virginica) Dots** are bunched up together in the top-right. This tells us the real challenge is figuring out where the Red group ends and the Gray group begins.

### 2. Phase I: The "Professional" Robot (Scikit-Learn)
In Phase I, we used a famous, pre-built tool called `scikit-learn`. Think of this as hiring a master gardener who already knows exactly how to sort flowers.

![Phase I Decision Boundary](plots/decision_boundary_phase_one.png)

**What are we looking at?**
*   The background has been painted into three zones.
*   If a flower falls in the **Blue Zone**, the robot calls it "Setosa".
*   If it falls in the **Red Zone**, the robot calls it "Versicolor".
*   The straight lines where the colors meet are the "Decision Boundaries" – the strict rules the robot follows.

**Conclusion:**
*   The robot drew a perfect line separating the Blue flowers.
*   Crucially, it found an optimal line between the Red and Gray flowers that separates them with very few mistakes. This shows the power of standard algorithms to handle the "messy" parts of data.

### 3. Phase II: The "Handmade" Robot (Manual Code)
In Phase II, we didn't use the pre-built tool. We built our own sorting logic from scratch using math. We used a strategy called **"Divide and Conquer"**.

**Step A:** The robot asks, "Is it small (Setosa)?" -> *Draws the first line.*
**Step B:** If it's not small, it asks, "Is it medium (Versicolor) or large (Virginica)?" -> *Draws the second line.*

![Phase II Decision Boundary](plots/decision_boundary_phase_two.png)

**What are we looking at?**
*   This map looks very similar to Phase I, which is good!
*   It shows the result of our two-step logic combined into one final map.

**Conclusion:**
*   Our handmade robot achieved **~95.6% accuracy**, which is almost as good as the professional tool (**~97.8%**).
*   **The Big Lesson:** We proved that you don't always need a "black box" super-computer. By breaking a complex problem (3 flower types) into two simple "Yes or No" questions, we built a highly effective classifier from scratch.