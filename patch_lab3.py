import json
import os

filepath = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 5 - Supervised Learning Algorithms\Week_5_Lab_3_Logistic_Regression.ipynb"
with open(filepath, "r") as f:
    nb = json.load(f)

# Cell 0: Foreword
nb["cells"][0]["source"] = [
    "# Week 5 Lab 3: Logistic Regression (Titanic Survival & Pipelines)\n",
    "\n",
    "**Goal**: Predict whether a passenger **Survived (1)** or **Perished (0)** based on their social status and demographics using the Titanic dataset.\n",
    "\n",
    "> **Why this lab matters**:\n",
    "> Logistic Regression introduces **Binary Classification**. It teaches models to output probabilities (0 to 100%) rather than infinite continuous numbers. This is the foundation for all decision-making AI.\n",
    "\n",
    "> **Structure**:\n",
    "> We follow the **6-Phase Professional Workflow** and use a `ColumnTransformer` to handle both numerical data and categorical text data simultaneously. We will test the model's reliability using **Cross-Validation**.\n",
    "\n",
    "---\n",
    "## Foreword\n",
    "In this lab, we use the legendary **Kaggle Titanic Dataset**.\n",
    "\n",
    "1. **Phase 1: Splitting**\n",
    "2. **Phase 2: Preprocessing (ColumnTransformers)**\n",
    "3. **Phase 3: Assembly (Pipeline)**\n",
    "4. **Phase 4: Training**\n",
    "5. **Phase 5: Evaluation (Accuracy, Probabilities & Cross-Validation)**\n",
    "6. **Phase 6: Optimization**"
]

# Cell 1
nb["cells"][1]["source"] = [
    "### 1.1 Import Dependencies & Load Data\n",
    "**Concept**: We fetch the Titanic data and focus on key features: Passenger Class (`Pclass`), Gender (`Sex`), and `Age`.\n"
]

# Cell 2: add cv
nb["cells"][2]["source"] = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix\n",
    "\n",
    "# Load Titanic Dataset\n",
    "url = \"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv\"\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "# Preprocess: Keep key features and drop missing values for simplicity\n",
    "df = df[['Survived', 'Pclass', 'Sex', 'Age']].dropna()\n",
    "\n",
    "X = df[['Pclass', 'Sex', 'Age']]\n",
    "y = df['Survived']"
]

# Cell 3
nb["cells"][3]["source"] = [
    "---\n",
    "### 1.2 Phase 1: Data Splitting\n",
    "**Concept**: We secure 20% of the passenger data for our unbiased survival final exam.\n"
]

# Cell 5
nb["cells"][5]["source"] = [
    "---\n",
    "#### Theory: Logistic Regression & Dummy Variables\n",
    "When One-Hot Encoding binary features like `Sex` ('male', 'female'), creating two columns is redundant (if not male, must be female). We drop the first column to prevent the \"Dummy Variable Trap.\"\n",
    "\n",
    "| Component | Target | Function |\n",
    "| :--- | :--- | :--- |\n",
    "| `StandardScaler()` | `['Pclass', 'Age']` | Logistic Regression converges faster when numbers are scaled. |\n",
    "| `OneHotEncoder(drop='first')` | `['Sex']` | Converts 'male'/'female' into a single binary 1/0 column. |\n",
    "\n",
    "### 1.3 Phase 2 & 3: Preprocessing & Assembly\n",
    "**Concept**: We map our operations to the exact columns they belong to.\n",
    "**Solution**: We use `ColumnTransformer` for routing and wrap it in our `Pipeline`.\n"
]

# Cell 7
nb["cells"][7]["source"] = [
    "---\n",
    "### 1.4 Phase 4: Training\n",
    "**Concept**: We train the model to find the optimal 'S-Curve' (Sigmoid) that separates survivors from those who perished.\n",
    "**Solution**: Call `.fit()` on the entire pipeline.\n"
]

# Cell 9
nb["cells"][9]["source"] = [
    "---\n",
    "### 1.5 Phase 5: Evaluation (Accuracy & Probabilities)\n",
    "**Concept**: We check the model's accuracy on the unseen test data. More importantly, we look at the raw probabilities the Sigmoid gives us.\n",
    "**Solution**: We use `.predict()` for the final 0/1 decision, and `.predict_proba()` to see the percentage certainty.\n"
]

# Cell 11 (Summary cell) -> Replace with CV and new Summary cell
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### 1.6 Phase 5: Cross-Validation\n",
            "**Concept**: Does the model perform consistently across all variations of the Titanic passenger lists?\n",
            "**Solution**: We run a 5-fold cross-validation on the pipeline, using standard `accuracy` as the metric.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "scores = cross_val_score(workflow, X, y, scoring='accuracy', cv=5)\n",
            "print(\"Accuracy Scores across 5 folds:\", np.round(scores, 3))\n",
            "print(f\"\\nAverage CV Accuracy: {scores.mean():.2%}\")\n",
            "print(f\"Standard Deviation: {scores.std():.2%} (How much the accuracy fluctuates)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Observation**: The cross-validation score reveals if the model's high accuracy was just luck. A stable accuracy around 78-80% is typical for this subset of features."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Task 1**: In the `ColumnTransformer` in Phase 2, change `drop='first'` to `None` in the `OneHotEncoder`. Rerun the cross-validation. Does creating two dummy variables instead of one crash the model or change the accuracy significantly? Why or why not?"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<details>\n",
            "<summary><strong> Click here for Solution (Try it yourself first!)</strong></summary>\n",
            "\n",
            "If you remove `drop='first'`, `OneHotEncoder` creates two columns: `Sex_female` and `Sex_male`. The accuracy will likely remain identical.\n",
            "\n",
            "Modern Scikit-Learn's `LogisticRegression` includes $L_2$ Regularization by default (the `C=1.0` parameter), which gracefully handles the collinearity (Dummy Variable Trap). However, for mathematical purity and slightly faster training, dropping the first column is best practice.\n",
            "</details>"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Summary\n",
            "You've just built an AI that can predict historical outcomes with high consistency across cross-validation folds. Notice how the model doesn't just guess Yes/No; it calculates a **Probability** before making the final call!"
        ]
    }
]

nb["cells"].pop(11) # drop summary
nb["cells"].extend(new_cells)

with open(filepath, "w") as f:
    json.dump(nb, f, indent=1)
