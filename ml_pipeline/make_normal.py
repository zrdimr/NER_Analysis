import pandas as pd
import numpy as np
import random
from textblob import TextBlob
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/Vibree_synthetic_stress_dataset.csv")

unique_texts = []
labels_list = []
for text, label in zip(df["text"].dropna(), df["label"].dropna()):
    unique_texts.append(text)
    labels_list.append(label)

new_texts = []
new_labels = []

np.random.seed(42)
random.seed(42)

for i in range(5000):
    k = np.random.randint(4, 9)  # 4 to 8 sentences merged
    indices = np.random.choice(len(unique_texts), size=k, replace=True)
    
    merged_text = " ".join([unique_texts[idx] for idx in indices])
    majority_label = 1 if np.mean([labels_list[idx] for idx in indices]) > 0.5 else 0
    
    # Add random noise to labels so it's not 100% predictable 
    if random.random() < 0.2:
        majority_label = 1 - majority_label
        
    new_texts.append(merged_text)
    new_labels.append(majority_label)

df_new = pd.DataFrame({
    "text": new_texts,
    "label": new_labels
})

# Save new raw dataset
df_new.to_csv("data/raw/Vibree_synthetic_stress_dataset.csv", index=False)
print("Saved modified Vibree dataset with", len(df_new), "rows.")
