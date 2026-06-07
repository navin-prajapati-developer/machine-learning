# Machine Learning Basics
# Predicting if a student passes or fails

from sklearn.tree import DecisionTreeClassifier

# Training data
# [study_hours, attendance]
X = [[2, 60], [5, 80], [1, 50], [8, 90],
     [3, 70], [7, 85], [1, 40], [6, 88]]

# Labels: 1 = Pass, 0 = Fail
y = [0, 1, 0, 1, 0, 1, 0, 1]

# Create and train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict new students
print("--- Predictions ---")
print("2hrs study, 60% attendance:",
      "Pass" if model.predict([[2, 60]])[0] == 1 else "Fail")
print("7hrs study, 85% attendance:",
      "Pass" if model.predict([[7, 85]])[0] == 1 else "Fail")
print("1hr study, 45% attendance:",
      "Pass" if model.predict([[1, 45]])[0] == 1 else "Fail")
