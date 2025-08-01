from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.utils import load_data, evaluate_model, save_params

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
save_params(model, "models/model.joblib")

# Evaluate model
evaluate_model(model, X_test, y_test)
