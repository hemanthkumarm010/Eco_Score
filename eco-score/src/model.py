from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import pandas as pd

# Load your dataset
data = pd.read_csv(r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\data\processed\processed_data.csv')  # Replace with your actual processed CSV path

# Define your feature columns and target variable
X = data.drop('Rating', axis=1)  # Assuming 'Rating' is your target variable
y = data['Rating']

# Numerical features that need to be scaled
numerical_features = ['Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption']

# Scale the numerical features using StandardScaler
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Save the scaler for later use in predictions
joblib.dump(scaler, r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\models\scaler.joblib')

# Train your model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, r'C:\Users\Satya Prakash\OneDrive\Desktop\eco-score\models\random_forest_model.joblib')

print("Model and scaler saved successfully.")

# Checking the feature columns of the trained model
print("Model Feature Columns:", X.columns.tolist())  # Instead of `model.feature_names_in_`, we use `X.columns`
y_pred = model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared (Coefficient of Determination): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")