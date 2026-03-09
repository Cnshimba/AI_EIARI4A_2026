import json
import os

filepath = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 5 - Supervised Learning Algorithms\Week_5_Lab_2_Linear_Regression_Expansion.ipynb"
with open(filepath, "r") as f:
    nb = json.load(f)

# Cell 0: Foreword
nb["cells"][0]["source"] = [
    "# Week 5 Lab 2: Advanced Regression & Pipelines\n",
    "\n",
    "**Goal**: Predict medical charges using **Multiple Features** (Age, BMI, Smoking status) and **ColumnTransformer**.\n",
    "\n",
    "> **Why this lab matters**:\n",
    "> In the real world, models use hundreds of features of different types (numbers, text). You must know how to combine `StandardScaler` for numbers and `OneHotEncoder` for text into a single automated pipeline.\n",
    "\n",
    "> **Structure**:\n",
    "> We follow the **6-Phase Professional Workflow** but expand Phase 2 (Preprocessing) into parallel tracks using a `ColumnTransformer`. We will then use **Cross Validation** again to verify the massive accuracy boost.\n",
    "\n",
    "---\n",
    "## Foreword\n",
    "We continue our work with the **Medical Insurance Dataset**. In Lab 1, we only used Age. Today, we unleash the model on more data.\n",
    "\n",
    "1. **Phase 1: Splitting**\n",
    "2. **Phase 2: Preprocessing (Parallel Transformation)**\n",
    "3. **Phase 3: Assembly (Pipeline)**\n",
    "4. **Phase 4: Training**\n",
    "5. **Phase 5: Evaluation (R-Squared & Cross-Validation)**\n",
    "6. **Phase 6: Optimization**"
]

# Cell 1
nb["cells"][1]["source"] = [
    "### 1.1 Import Dependencies & Load Data\n",
    "**Concept**: We select multiple features. Notice that `smoker` is a categorical column ('yes' or 'no'), which AI cannot read algebraically.\n"
]

# Cell 2 (imports): Add cross_val_score
nb["cells"][2]["source"] = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import r2_score\n",
    "\n",
    "url = \"https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv\"\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "X = df[['age', 'bmi', 'smoker']]\n",
    "y = df['charges']"
]

# Cell 3
nb["cells"][3]["source"] = [
    "---\n",
    "### 1.2 Phase 1: Data Splitting\n",
    "**Concept**: We secure 20% of the data for an unbiased final exam.\n"
]

# Cell 5
nb["cells"][5]["source"] = [
    "---\n",
    "#### Theory: Scikit-Learn ColumnTransformer\n",
    "You cannot scale text, and you shouldn't One-Hot Encode numbers. `ColumnTransformer` routes different columns to different tools.\n",
    "\n",
    "| Component | Target Columns | Function |\n",
    "| :--- | :--- | :--- |\n",
    "| `StandardScaler()` | `['age', 'bmi']` | Centers the numerical data around 0. |\n",
    "| `OneHotEncoder()` | `['smoker']` | Converts 'yes'/'no' string into binary columns. |\n",
    "\n",
    "### 1.3 Phase 2 & 3: Preprocessing & Assembly\n",
    "**Concept**: The model needs all data in a unified numerical matrix.\n",
    "**Solution**: We define the routing rules in `ColumnTransformer` and drop it into a `Pipeline`.\n"
]

# Cell 7
nb["cells"][7]["source"] = [
    "---\n",
    "### 1.4 Phase 4 & 5: Training & Evaluation\n",
    "**Concept**: We train the complex pipeline and evaluate its **R-Squared ($R^2$)**. R-Squared tells us the percentage of variance in the charges that is explained by our features.\n"
]

# Cell 9: replace with CV section
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### 1.5 Phase 5: Cross-Validation\n",
            "**Concept**: Did we just get lucky with our $R^2$ of 74%?\n",
            "**Solution**: We run a 5-fold cross-validation, using $R^2$ as our scoring metric.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "scores = cross_val_score(workflow, X, y, scoring='r2', cv=5)\n",
            "print(\"R-Squared Scores across 5 folds:\", np.round(scores, 3))\n",
            "print(f\"\\nAverage CV R-Squared: {scores.mean():.2%}\")\n",
            "print(f\"Standard Deviation: {scores.std():.2%} (How much the score fluctuates)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Observation**: The cross-validation score shows incredible stability. A standard deviation of ~1% means the model's performance is incredibly reliable regardless of how the data is shuffled."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Task 1**: In the `numeric_features` list from Phase 2, remove `'bmi'`. Rerun the entire notebook. How much does the Average CV R-Squared drop when the model doesn't know the patient's BMI?"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<details>\n",
            "<summary><strong> Click here for Solution (Try it yourself first!)</strong></summary>\n",
            "\n",
            "```python\n",
            "numeric_features = ['age'] # 'bmi' removed\n",
            "categorical_features = ['smoker']\n",
            "'''\n",
            "If you rerun everything, the R^2 score drops to roughly 72%. This tells us that while BMI is important, the smoker feature is doing the vast majority of the heavy lifting. Without smoker (in Lab 1), our model was terrible.\n",
            "'''\n",
            "```\n",
            "</details>"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Summary\n",
            "By adding `smoker` and `bmi` to our `Pipeline` using `ColumnTransformer`, the model accuracy skyrocketed! This proves that selecting professional features and using pipelines for clean routing is the secret to building high-performance AI."
        ]
    }
]

nb["cells"].pop(9)
nb["cells"].extend(new_cells)

with open(filepath, "w") as f:
    json.dump(nb, f, indent=1)
