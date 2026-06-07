# Linear Regression
# Predicting marks based on study hours

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([40, 50, 55, 65, 70, 75, 85, 90])

# Create and train model
model = LinearRegression()
model.fit(study_hours, marks)

# Predictions
print("--- Mark Predictions ---")
hours = [2, 4, 6, 9]
for h in hours:
    pred = model.predict([[h]])[0]
    print(f"{h} hours study = {pred:.1f} marks")

# Model accuracy
score = model.score(study_hours, marks)
print(f"\nModel Accuracy: {score*100:.1f}%")
