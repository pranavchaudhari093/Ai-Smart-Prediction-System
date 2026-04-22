import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv("datasets/student.csv")

X = data[['study_hours', 'attendance', 'previous_marks']]
y = data['final_score']

model = LinearRegression()
model.fit(X, y)


pickle.dump(model, open("models/student_model.pkl", "wb"))

print("Model trained & saved")
print("Model accuracy:", model.score(X, y))