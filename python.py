import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data.csv")

print("========== DATASET ==========")
print(df)

# Features and Target
X = df[['Experience']]
y = df['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n========== MODEL PERFORMANCE ==========")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict salary for new experience
experience = [[12]]
predicted_salary = model.predict(experience)

print("\nPredicted Salary for 12 Years Experience:")
print("₹", round(predicted_salary[0], 2))

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2,
         label='Regression Line')

plt.title("Salary vs Experience")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.savefig("prediction.png")
plt.show()

print("\nGraph saved as prediction.png")
