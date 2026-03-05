import yaml
import subprocess

def update_config(model_type):
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config['model']['type'] = model_type
    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f)

models = ['bert', 'indobert']
for m in models:
    print(f"Running {m}...")
    update_config(m)
    subprocess.run(["python3", "main.py"], stdout=open(f"out_{m}.txt", "w"), stderr=subprocess.STDOUT)
    print(f"Finished {m}.")
