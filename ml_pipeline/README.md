# Stress Analysis NLP Pipeline

This project is a refactor of the `Thesis_BERT.ipynb` notebook into a professional Python machine learning pipeline structure. It maintains identical logic (text preprocessing, TF-IDF + Sentiment scoring, Logistic Regression model) but organizes it systematically for easy production deployment and scaling.

## Directory Structure

```plaintext
ml_pipeline/
├── data/
│   ├── raw/           # Put original dreaddit_StressAnalysis.csv here
│   └── processed/     # Auto-generated processed datasets
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py    # Text cleaning, Tokenization, Stemming
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py# TF-IDF, Polarity Analysis, Feature Combination
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py   # Modeling and Scaling logic
│   │   └── predict_model.py # Inference module
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py      # Metric generation
├── models/            # Stored joblib models, scalers, and vectorizers
├── config/
│   └── config.yaml    # Hyperparameters and Paths definitions
├── requirements.txt   # Dependencies
├── README.md          # Project instructions
└── main.py            # End-to-end pipeline execution script
```

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Supply raw data**:
   Make sure you copy the dataset to the `data/raw/` directory:
   ```bash
   cp ../dreaddit_StressAnalysis.csv data/raw/
   ```

3. **Execute Pipeline**:
   Run the `main.py` entrypoint. This will ingest data, build models, evaluate performance, and run a quick test sentence inference.
   ```bash
   cd ml_pipeline/
   python main.py
   ```
