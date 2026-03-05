import yaml
import subprocess
import os

def update_config(model_type):
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config['model']['type'] = model_type
    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f)

models = ['mobilebert', 'bert', 'indobert', 'mentalbert']
os.makedirs("vibree_results", exist_ok=True)

for m in models:
    print(f"Running {m}...")
    update_config(m)
    subprocess.run(["python3", "main.py"], stdout=open(f"vibree_results/out_{m}.txt", "w"), stderr=subprocess.STDOUT)
    print(f"Finished {m}.")
