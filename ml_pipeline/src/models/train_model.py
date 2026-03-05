from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train(X, y, test_size=0.2, random_state=42):
    """
    Splits the data, scales it, and trains a Logistic Regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test
