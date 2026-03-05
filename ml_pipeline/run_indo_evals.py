import yaml
import subprocess
import os

def update_config(model_type):
    print(f"Updating config for {model_type}...")
    with open("config/config_indo.yaml") as f:
        config = yaml.safe_load(f)
    config['model']['type'] = model_type
    
    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f)

models = ['mobilebert', 'bert', 'indobert', 'mentalbert']
os.makedirs("indo_results", exist_ok=True)
os.makedirs("models/inference_models", exist_ok=True)

for m in models:
    print(f"\n--- Running evaluation for {m.upper()} on Indonesian EnTDA ---")
    update_config(m)
    
    log_file = f"indo_results/out_{m}.txt"
    with open(log_file, "w") as f:
        process = subprocess.Popen(["python3", "main.py"], stdout=f, stderr=subprocess.STDOUT)
        process.wait()
        
    print(f"Finished {m}. Logs saved to {log_file}")
