import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data.csv")

print("="*50)
print("SALARY PREDICTION PROJECT")
print("="*50)
print(df)

# ----------------------------
# Train Model
# ----------------------------
X = df[['Experience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Predictions
# ----------------------------
predictions = model.predict(X)

# Accuracy
score = r2_score(y, predictions)

print("\nModel Accuracy:", round(score * 100, 2), "%")

# ----------------------------
# Future Prediction
# ----------------------------
future_exp = [[12], [15], [20]]

future_salary = model.predict(future_exp)

print("\nFuture Salary Forecast")
print("-"*30)

for exp, sal in zip([12, 15, 20], future_salary):
    print(f"{exp} Years Experience --> ₹{sal:,.0f}")

# ----------------------------
# Attractive Visualization
# ----------------------------
plt.figure(figsize=(10,6))

plt.scatter(
    df['Experience'],
    df['Salary'],
    s=120,
    label="Actual Salary Data"
)

plt.plot(
    df['Experience'],
    predictions,
    linewidth=3,
    label="Regression Line"
)

plt.scatter(
    [12,15,20],
    future_salary,
    s=180,
    marker='*',
    label="Future Predictions"
)

for exp, sal in zip([12,15,20], future_salary):
    plt.annotate(
        f"₹{int(sal)}",
        (exp, sal),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

plt.title("Employee Salary Prediction Using Machine Learning", fontsize=16)
plt.xlabel("Years of Experience")
plt.ylabel("Salary (₹)")
plt.legend()
plt.grid(True)

plt.savefig("salary_prediction_dashboard.png", dpi=300)
plt.show()

print("\nDashboard saved as salary_prediction_dashboard.png")