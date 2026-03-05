import os
import pandas as pd
import torch
from generate_academic_report import generate_full_report
import traceback
from src.models.train_generic import train_and_eval_generic

DATASETS = ["dreaddit", "Vibree_Synthetic_English", "Vibree_indonesian"]
BALANCING = [False, True]
MODELS = ["bert", "mobilebert", "indobert", "mentalbert"]
ARCHITECTURES = ["transformer", "transformer_lstm", "transformer_lstm_crf"]

RESULTS_FILE = "results/research_matrix.csv"

def run_experiment():
    os.makedirs("results", exist_ok=True)
    
    # Load existing matrix if it exists
    if os.path.exists(RESULTS_FILE):
        results_df = pd.read_csv(RESULTS_FILE)
    else:
        results_df = pd.DataFrame(columns=[
            "Dataset", "Balancing (EnTDA)", "Base Model", "Architecture", 
            "Accuracy", "Precision", "Recall", "F1", "Status"
        ])
        
    for ds in DATASETS:
        for bal in BALANCING:
            for base in MODELS:
                for arch in ARCHITECTURES:
                    
                    # Check if already run recently
                    mask = (
                        (results_df["Dataset"] == ds) &
                        (results_df["Balancing (EnTDA)"] == bal) &
                        (results_df["Base Model"] == base) &
                        (results_df["Architecture"] == arch) &
                        (results_df["Status"] == "SUCCESS")
                    )
                    
                    if mask.any():
                        print(f"Skipping {ds} | {bal} | {base} | {arch} (Already Succeeded)")
                        continue
                        
                    print(f"\n=======================================================")
                    print(f"RUNNING EXPERIMENT: {ds} | EnTDA={bal} | {base} | {arch}")
                    print(f"=======================================================\n")
                    
                    try:
                        # Clear cache
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        if torch.backends.mps.is_available(): torch.mps.empty_cache()
                        
                        eval_metrics = train_and_eval_generic(
                            dataset_name=ds, 
                            model_base=base, 
                            architecture=arch, 
                            apply_entda=bal, 
                            epochs=3, 
                            batch_size=8, 
                            lr=2e-5
                        )
                        
                        new_row = {
                            "Dataset": ds,
                            "Balancing (EnTDA)": bal,
                            "Base Model": base,
                            "Architecture": arch,
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
                            "Dataset": ds,
                            "Balancing (EnTDA)": bal,
                            "Base Model": base,
                            "Architecture": arch,
                            "Accuracy": 0,
                            "Precision": 0,
                            "Recall": 0,
                            "F1": 0,
                            "Status": f"FAILED: {str(e)[:50]}"
                        }
                    
                    # Update DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    # Drop duplicates keeping the last run
                    results_df = results_df.drop_duplicates(subset=["Dataset", "Balancing (EnTDA)", "Base Model", "Architecture"], keep="last")
                    
                    results_df.to_csv(RESULTS_FILE, index=False)
                    print(f"Saved checkpoint to {RESULTS_FILE}")
                    
                    # Update report dynamically
                    generate_full_report()

if __name__ == "__main__":
    run_experiment()

