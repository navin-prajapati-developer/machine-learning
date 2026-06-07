# Salary Predictor - Machine Learning
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
# [years_experience, skills_score]
X = np.array([
    [1, 60], [2, 65], [3, 70],
    [4, 75], [5, 80], [6, 85],
    [7, 90], [8, 92], [9, 95],
    [10, 98]
])

# Salary in thousands
y = np.array([25, 30, 35, 42, 50, 60, 72, 85, 95, 110])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
print("--- Salary Predictions ---")
candidates = [
    [1, 60, "Fresher"],
    [3, 70, "Junior"],
    [5, 80, "Mid Level"],
    [8, 92, "Senior"]
]

for exp, skill, level in candidates:
    salary = model.predict([[exp, skill]])[0]
    print(f"{level}: ₹{salary:.1f}K per month")
