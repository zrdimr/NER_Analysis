import argparse
import os
import pandas as pd
import torch
import traceback
import sys

# Ensure Python can find the src module relative to this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.train_generic import train_and_eval_generic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--entda", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--architecture", required=True)
    
    args = parser.parse_args()
    
    apply_entda = True if args.entda.lower() == 'true' else False
    
    os.makedirs("results", exist_ok=True)
    
    print(f"RUNNING EXPERIMENT: {args.dataset} | EnTDA={apply_entda} | {args.base_model} | {args.architecture}")
    
    try:
        eval_metrics = train_and_eval_generic(
            dataset_name=args.dataset, 
            model_base=args.base_model, 
            architecture=args.architecture, 
            apply_entda=apply_entda, 
            epochs=3, 
            batch_size=8, 
            lr=2e-5
        )
        
        new_row = {
            "Dataset": args.dataset,
            "Jumlah Dataset": eval_metrics.get("jumlah_dataset", 0),
            "Balancing (EnTDA)": apply_entda,
            "Positif Dataset": eval_metrics.get("positif_dataset", 0),
            "Negatif Dataset": eval_metrics.get("negatif_dataset", 0),
            "Base Model": args.base_model,
            "Architecture": args.architecture,
            "Epoch": 3,
            "Training Time": round(eval_metrics.get("training_time", 0), 2),
            "Model Weight Size (MB)": round(eval_metrics.get("model_size_mb", 0), 2),
            "Accuracy": round(eval_metrics.get("eval_accuracy", 0), 4),
            "Precision": round(eval_metrics.get("eval_precision", 0), 4),
            "Recall": round(eval_metrics.get("eval_recall", 0), 4),
            "F1": round(eval_metrics.get("eval_f1", 0), 4),
            "Status": "SUCCESS"
        }
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        new_row = {
            "Dataset": args.dataset,
            "Jumlah Dataset": 0,
            "Balancing (EnTDA)": apply_entda,
            "Positif Dataset": 0,
            "Negatif Dataset": 0,
            "Base Model": args.base_model,
            "Architecture": args.architecture,
            "Epoch": 3,
            "Training Time": 0,
            "Model Weight Size (MB)": 0,
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1": 0,
            "Status": f"FAILED: {str(e)[:50]}"
        }
        
    df = pd.DataFrame([new_row])
    safe_name = f"{args.dataset}_{apply_entda}_{args.base_model}_{args.architecture}.csv"
    df.to_csv(f"results/{safe_name}", index=False)
    print(f"Saved metric to results/{safe_name}")

if __name__ == "__main__":
    main()
