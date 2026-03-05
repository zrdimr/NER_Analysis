import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_file, prefix="Vibree_indonesian"):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if this data has just 'text' and 'label' columns
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"Error: expected columns 'text' and 'label' in {input_file}")
        return
        
    print(f"Total rows: {len(df)}")
    
    # 70% / 15% / 15% Split
    # First split: 70% Train, 30% Temp (Test+Eval)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    
    # Second split: 15% Test, 15% Eval (50% of the 30% Temp)
    test_df, eval_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])
    
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save RAW Splits
    raw_dir = "data/raw_splits"
    os.makedirs(raw_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(raw_dir, f"{prefix}_train.csv"), index=False)
    test_df.to_csv(os.path.join(raw_dir, f"{prefix}_test.csv"), index=False)
    eval_df.to_csv(os.path.join(raw_dir, f"{prefix}_eval.csv"), index=False)
    
    # Save PROCESSED Splits (Same for now, but usually this would be preprocessed)
    from src.data.preprocess import preprocess_text
    
    print("Applying preprocessing to save in processed folder...")
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
    
    print(f"\nFiles successfully saved to 'data/raw_splits' and '{out_dir}'.")

if __name__ == "__main__":
    split_data("data/raw/Vibree_indonesian_stress_dataset.csv", prefix="Vibree_indonesian")
