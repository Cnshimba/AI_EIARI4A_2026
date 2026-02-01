import json
import os

# Pointing to the SOURCE file
nb_path = r"d:\OneDrive - Vaal University of Technology\WORK\2026\AI_v2\Week 9 - Natural Language Processing (NLP)\Week_9_Lab_1_Text_Classification.ipynb"

# 1. Read Notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Define New Cells for Saving Model/Vocab
new_cells_md = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2: Deployment Preparation\n",
    "\n",
    "> **Step for Edge AI:** Run the cell below to save your trained model and vocabulary. These files will be used to run the model on the Jetson Orin Nano."
   ]
}

new_cells_code = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save Model State Dict\n",
    "torch.save(model.state_dict(), 'text_classifier.pth')\n",
    "\n",
    "# Save Vocabulary (Word to Index Mapping)\n",
    "import json\n",
    "with open('vocab.json', 'w') as f:\n",
    "    json.dump(word_to_ix, f)\n",
    "\n",
    "print(\"Files saved: text_classifier.pth, vocab.json\")"
   ]
}

# 3. Check and Append
last_source = data['cells'][-1]['source']
is_already_there = any("Files saved: text_classifier.pth" in line for line in last_source)

if not is_already_there:
    data['cells'].append(new_cells_md)
    data['cells'].append(new_cells_code)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print("Week 9 Notebook updated successfully.")
else:
    print("Week 9 Notebook already contains save section.")
