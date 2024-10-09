from sklearn.tree import DecisionTreeClassifier

# Sample student data: [task_completion_time, mistakes_made]
X = [[30, 5], [25, 3], [40, 7], [15, 1], [20, 2]]  # Mock data
y = ['slow', 'medium', 'slow', 'fast', 'fast']  # Class labels

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Predict the learning speed of a new student
new_student = [[35, 6]]  # Task completion time = 35, Mistakes = 6
prediction = clf.predict(new_student)

print(f"The new student is a {prediction[0]} learner.")
