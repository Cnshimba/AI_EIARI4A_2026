import json
import os

filepath = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 5 - Supervised Learning Algorithms\Week_5_Lab_1_Linear_Regression.ipynb"

with open(filepath, "r") as f:
    nb = json.load(f)

# Cell 0: Foreword -> Goal, Why this matters, etc.
nb["cells"][0]["source"] = [
    "# Week 5 Lab 1: Linear Regression (Kaggle & Pipelines)\n",
    "\n",
    "**Goal**: To predict **Medical Charges** based on a patient's **Age** using a single-feature Linear Regression model.\n",
    "\n",
    "> **Why this lab matters**:\n",
    "> Before handling 100-feature datasets or complex neural networks, we must understand how a model draws a straight line through data. This establishes the foundation of the mathematical **Adjustment Loop**.\n",
    "\n",
    "> **Structure**:\n",
    "> We will use the **6-Phase Professional Workflow** and introduce **Cross Validation** to ensure our model's performance isn't just a lucky guess.\n",
    "\n",
    "---\n",
    "## Foreword\n",
    "In this lab, we use a professional **Kaggle Dataset**: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichampel/medical-cost-personal-datasets).\n",
    "\n",
    "1. **Phase 1: Splitting**\n",
    "2. **Phase 2: Preprocessing (Standardization)**\n",
    "3. **Phase 3: Assembly (Pipeline Design)**\n",
    "4. **Phase 4: Training**\n",
    "5. **Phase 5: Evaluation (RMSE, MAE & Cross-Validation)**\n",
    "6. **Phase 6: Optimization**"
]

# Formatting Step 1
nb["cells"][1]["source"] = [
    "### 1.1 Import Dependencies & Load Data\n",
    "**Concept**: We fetch the dataset directly from a reliable GitHub mirror so we always have the same starting point.\n"
]

# Update imports in cell 2
nb["cells"][2]["source"] = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
    "\n",
    "# Load Medical Insurance Dataset\n",
    "url = \"https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv\"\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "# Simple Regression: Age (X) vs Charges (y)\n",
    "X = df[['age']]\n",
    "y = df['charges']\n",
    "\n",
    "plt.scatter(X, y, alpha=0.5)\n",
    "plt.xlabel(\"Age\")\n",
    "plt.ylabel(\"Medical Charges\")\n",
    "plt.title(\"Kaggle: Age vs Medical Charges\")\n",
    "plt.show()"
]

# Cell 3: Splitting
nb["cells"][3]["source"] = [
    "---\n",
    "### 1.2 Phase 1: Data Splitting\n",
    "**Concept**: We never train on the final exam. We split the data into 80% training and 20% testing sets.\n"
]

# Cell 5: Pipeline
nb["cells"][5]["source"] = [
    "---\n",
    "#### Theory: Scikit-Learn Pipelines\n",
    "A `Pipeline` links multiple steps together so they act as one single object.\n",
    "\n",
    "| Pipeline Step | Function | ML Use Case |\n",
    "| :--- | :--- | :--- |\n",
    "| `StandardScaler()` | Transformer | Scales numbers to have a mean of 0 and variance of 1. |\n",
    "| `LinearRegression()` | Estimator | Learns the coefficients (weights) and intercept (bias). |\n",
    "\n",
    "### 1.3 Phase 2, 3 & 4: Preprocessing, Assembly & Training\n",
    "**Concept**: We must scale `Age` so it's on a leveled mathematical playing field. But scaling the test set using Test Set averages causes **Data Leakage**.\n",
    "\n",
    "**Solution**: We use a **Pipeline** to tie the scaler and model together seamlessly. This ensures scaling rules from `X_train` are applied to `X_test`.\n"
]

# Cell 7: Evaluation
nb["cells"][7]["source"] = [
    "---\n",
    "### 1.4 Phase 5: Evaluation (Quality Control)\n",
    "**Concept**: We evaluate how well the model predicts charges for the test group.\n",
    "\n",
    "**Solution**: We calculate **RMSE** (penalizes large errors heavily) and **MAE** (gives the plain average error).\n"
]

# Cell 9: Summary -> Convert to Cross Validation, then Summary
nb["cells"][9]["source"] = [
    "---\n",
    "#### Theory: Cross-Validation (K-Fold)\n",
    "How do we know our 80/20 split didn't just get \"lucky\" with easy test data?\n",
    "**Cross-Validation** chops the data into $K$ parts (folds). It trains on $K-1$ parts and tests on the remaining 1 part. It repeats this $K$ times, testing on a different part each time.\n",
    "\n",
    "### 1.5 Phase 5: Cross-Validation\n",
    "**Concept**: We want to prove our model's accuracy is stable.\n",
    "**Solution**: We use `cross_val_score` to run 5-Fold cross validation on our pipeline.\n"
]

# Insert code block for CV
cv_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# We pass the WHOLE dataset (X, y) into cross_val_score because it handles the splitting!\n",
        "# Note: Scikit-learn uses negative MSE by default for scoring, so we convert it back.\n",
        "scores = cross_val_score(workflow, X, y, scoring='neg_mean_squared_error', cv=5)\n",
        "rmse_scores = np.sqrt(-scores)\n",
        "\n",
        "print(\"RMSE Scores across 5 folds:\", np.round(rmse_scores, 2))\n",
        "print(f\"\\nAverage CV RMSE: ${rmse_scores.mean():,.2f}\")\n",
        "print(f\"Standard Deviation: ${rmse_scores.std():,.2f} (How much it varies)\")\n",
        "\n",
        "# Observation: If the Average CV RMSE is close to our Test RMSE from earlier, our model is stable!"
    ]
}

# Add Tasks and Observations
obs_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "> **Observation**: The cross-validation error is closely aligned with our initial test error. This proves our pipeline is robust and hasn't just memorized a \"lucky\" subset of data."
    ]
}

task_cell1 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Task 1**: Change `cv=5` to `cv=10` in the `cross_val_score` function above. Run the cell again. Does the average error change significantly?"
    ]
}

task_cell2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "<details>\n",
        "<summary><strong> Click here for Solution (Try it yourself first!)</strong></summary>\n",
        "\n",
        "```python\n",
        "scores_10 = cross_val_score(workflow, X, y, scoring='neg_mean_squared_error', cv=10)\n",
        "rmse_scores_10 = np.sqrt(-scores_10)\n",
        "print(f\"10-Fold Average CV RMSE: ${rmse_scores_10.mean():,.2f}\")\n",
        "```\n",
        "The average will change slightly, but it should remain in the same ballpark, further proving stability.\n",
        "</details>"
    ]
}


summary_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "### Summary\n",
        "Using a real-world dataset, you've seen that as **Age** increases, **Medical Charges** tend to rise linearly. You completely automated this with a **Pipeline** and verified its stability with **Cross-Validation**.\n",
        "\n",
        "However, notice the wide spread in the scatter plot — there are clearly other factors at play (like smoking or BMI) which we will cover in Lab 2!"
    ]
}

nb["cells"].extend([cv_cell, obs_cell, task_cell1, task_cell2, summary_cell])

with open(filepath, "w") as f:
    json.dump(nb, f, indent=1)
