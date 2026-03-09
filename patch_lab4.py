import json
import os

filepath = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 5 - Supervised Learning Algorithms\Week_5_Lab_4_Decision_Trees.ipynb"
with open(filepath, "r") as f:
    nb = json.load(f)

# Cell 0: Foreword
nb["cells"][0]["source"] = [
    "# Week 5 Lab 4: Decision Trees (Titanic Logic & Pipelines)\n",
    "\n",
    "**Goal**: Predict Titanic survival using a Decision Tree to automatically find the optimal \"questions\" to ask.\n",
    "\n",
    "> **Why this lab matters**:\n",
    "> Unlike Regression models which calculate formulas, Decision Trees build highly interpretable logical flowcharts. They teach us how AI can automatically segment data by reducing **Entropy** (messiness).\n",
    "\n",
    "> **Structure**:\n",
    "> We follow the **6-Phase Professional Workflow** and use `OrdinalEncoder` instead of `OneHotEncoder`. We will visualize \"The Mind of the Machine\" and use **Cross-Validation** to test node stability.\n",
    "\n",
    "---\n",
    "## Foreword\n",
    "Decision Trees are highly interpretable. In this lab, we use the **Titanic Dataset** to see the logic.\n",
    "\n",
    "1. **Phase 1: Splitting**\n",
    "2. **Phase 2: Preprocessing (OrdinalEncoding)**\n",
    "3. **Phase 3: Assembly (Pipeline)**\n",
    "4. **Phase 4: Training**\n",
    "5. **Phase 5: Evaluation (Visualization & Cross-Validation)**\n",
    "6. **Phase 6: Optimization**"
]

# Cell 1
nb["cells"][1]["source"] = [
    "### 1.1 Import Dependencies & Load Data\n",
    "**Concept**: We fetch the Titanic data, extracting `Pclass`, `Sex`, and `Age`.\n"
]

# Cell 2: add cv
nb["cells"][2]["source"] = [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn import tree\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "# Load Titanic Dataset\n",
    "url = \"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv\"\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "df = df[['Survived', 'Pclass', 'Sex', 'Age']].dropna()\n",
    "\n",
    "X = df[['Pclass', 'Sex', 'Age']]\n",
    "y = df['Survived']"
]

# Cell 3
nb["cells"][3]["source"] = [
    "---\n",
    "### 1.2 Phase 1: Data Splitting\n",
    "**Concept**: We secure 20% of the dataset for an unbiased evaluation.\n"
]

# Cell 5
nb["cells"][5]["source"] = [
    "---\n",
    "#### Theory: Ordinal Encoder vs One-Hot Encoder\n",
    "Decision trees don't calculate slopes, they split data based on thresholds (e.g., `Sex <= 0.5`). Because of this, it is perfectly fine to map text to a single column of numbers (0, 1, 2) rather than creating separate columns.\n",
    "\n",
    "| Component | Target | Function |\n",
    "| :--- | :--- | :--- |\n",
    "| `OrdinalEncoder()` | `['Sex']` | Converts 'female' to 0.0 and 'male' to 1.0. Keeps data sparse and efficient for trees. |\n",
    "| `remainder='passthrough'` | `Age, Pclass`| Tells the ColumnTransformer to ignore everything else but KEEP IT in the matrix! |\n",
    "\n",
    "### 1.3 Phase 2 & 3: Preprocessing & Assembly\n",
    "**Concept**: We only need to convert text to numbers; trees don't care about standardization (Scaling).\n",
    "**Solution**: We define the `OrdinalEncoder` in `ColumnTransformer` and build a pipeline with a max depth of 3.\n"
]

# Cell 7
nb["cells"][7]["source"] = [
    "---\n",
    "### 1.4 Phase 4: Training\n",
    "**Concept**: The tree calculates the \"Information Gain\" (reduction in Entropy) for every possible feature and threshold, and creates a split.\n",
    "**Solution**: Call `.fit()` on the pipeline.\n"
]

# Cell 9
nb["cells"][9]["source"] = [
    "---\n",
    "### 1.5 Phase 5: Evaluation (Visualization)\n",
    "**Concept**: One of the biggest strengths of Decision Trees is that we can literally view the mathematical conditions.\n",
    "**Solution**: We use `tree.plot_tree` to render the model's logic.\n"
]

# Replace Summary with CV
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### 1.6 Phase 5: Cross-Validation\n",
            "**Concept**: A major flaw of Decision Trees is **High Variance**. Small changes in the training data can result in a completely different tree structure.\n",
            "**Solution**: We use Cross-Validation to see how much the accuracy jumps around when the data changes.\n"
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
            "print(f\"Standard Deviation: {scores.std():.2%} (The higher this is, the higher the variance!)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Observation**: Notice the standard deviation. Decision trees tend to fluctuate slightly more than Logistic Regression because they overfit to specific subsets of data. This variance is exactly what Random Forests (Lab 5) are designed to fix."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Task 1**: In Phase 3, change `max_depth=3` to `max_depth=None`. Rerun the entire notebook. What happens to the visualization? What happens to the Cross-Validation average accuracy and standard deviation?"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<details>\n",
            "<summary><strong> Click here for Solution (Try it yourself first!)</strong></summary>\n",
            "\n",
            "If you remove the depth limit, the tree becomes immense and highly complex. \n",
            "\n",
            "The Average CV Accuracy will **DROP** and the standard deviation will **INCREASE**, proving that an infinitely deep tree simply **Overfits** to noise rather than learning genuine patterns.\n",
            "</details>"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Summary\n",
            "Look at the first split (The Root Node). Was it `Sex`? On the Titanic, \"Women and children first\" was a strong rule — the Decision Tree discovered this automatically by calculating which question reduced **Entropy** (messiness) the most! We achieved this professionally using pipelines."
        ]
    }
]

nb["cells"].pop(11) # drop summary
nb["cells"].extend(new_cells)

with open(filepath, "w") as f:
    json.dump(nb, f, indent=1)
