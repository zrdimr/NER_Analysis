import pandas as pd
from sklearn.model_selection import train_test_split
import os

from src.data.preprocess import preprocess_text

def process_dataset(input_file, prefix):
    print(f"\n===========================================")
    print(f"Split Processing: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if this data has just 'text' and 'label' columns
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"Error: expected columns 'text' and 'label' in {input_file}")
        return
        
    print(f"Total rows: {len(df)}")
    
    # Clean NaN drops to avoid split stratify errors
    df = df.dropna(subset=['text', 'label'])
    
    # 70% / 15% / 15% Split
    try:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
        test_df, eval_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])
    except Exception as e:
        print(f"Stratification error ({e}), falling back to purely random split for {prefix}")
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
        test_df, eval_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save RAW Splits
    raw_dir = "data/raw_splits"
    os.makedirs(raw_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(raw_dir, f"{prefix}_train.csv"), index=False)
    test_df.to_csv(os.path.join(raw_dir, f"{prefix}_test.csv"), index=False)
    eval_df.to_csv(os.path.join(raw_dir, f"{prefix}_eval.csv"), index=False)
    print(f"Saved Raw Splits for {prefix} in data/raw_splits")
    
    # Save PROCESSED Splits
    print(f"Applying text preprocessing for {prefix}...")
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    eval_df = eval_df.copy()
    
    train_df['text'] = train_df['text'].astype(str).apply(preprocess_text)
    test_df['text'] = test_df['text'].astype(str).apply(preprocess_text)
    eval_df['text'] = eval_df['text'].astype(str).apply(preprocess_text)
    
    train_df.to_csv(os.path.join(out_dir, f"{prefix}_train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"{prefix}_test_processed.csv"), index=False)
    eval_df.to_csv(os.path.join(out_dir, f"{prefix}_eval_processed.csv"), index=False)
    
    print("\n--- Distribution After Split ---")
    print(f"Train Dataset (70%): {len(train_df)} rows")
    print(f"Test Dataset (15%): {len(test_df)} rows")
    print(f"Eval Dataset (15%): {len(eval_df)} rows")
    

if __name__ == "__main__":
    process_dataset("data/raw/Vibree_synthetic_stress_dataset.csv", prefix="Vibree_Synthetic_English")
    process_dataset("data/raw/dreaddit_StressAnalysis.csv", prefix="dreaddit")
