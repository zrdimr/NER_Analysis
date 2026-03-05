from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluates the model and computes key metrics inline with the notebook.
    """
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred)
    }
    
    return metrics
