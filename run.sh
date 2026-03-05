# Move into the new directory
cd ./ml_pipeline

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the Python dependencies (if lacking)
pip3 install -r requirements.txt

# Run the end-to-end training and inference test!
python3 main.py
