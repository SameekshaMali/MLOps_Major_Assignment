import numpy as np
from src.utils import load_data, load_params, save_params
from sklearn.metrics import r2_score, mean_squared_error

# Load trained model
model = load_params("models/model.joblib")
weights = model.coef_
bias = model.intercept_

# Save original weights
save_params({"weights": weights, "bias": bias}, "models/unquant_params.joblib")

# Simulated quantization: round to 2 decimals
quant_weights = np.round(weights, 2).astype(np.float32)
quant_bias = round(bias, 2)

# Save quantized weights
save_params({"weights": quant_weights, "bias": quant_bias}, "models/quant_params.joblib")

# Evaluate dequantized model
X, y = load_data()
preds = np.dot(X, quant_weights) + quant_bias

r2 = r2_score(y, preds)
mse = mean_squared_error(y, preds)

print(f"R2 after simulated quantization: {r2:.4f}")
print(f"MSE after simulated quantization: {mse:.4f}")
