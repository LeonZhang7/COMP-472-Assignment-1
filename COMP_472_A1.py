import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

abalone_data = pd.read_csv('abalone.csv')
penguins_data = pd.read_csv('penguins.csv')


column_names = penguins_data.columns
print(column_names)

# One-Hot Encoding for 'island' and 'sex' columns
penguins_data = pd.get_dummies(penguins_data, columns=['island', 'sex'], drop_first=True)

# Define features and target for the penguins dataset
penguins_features = penguins_data.drop(columns=['species']) 
penguins_target = penguins_data['species']  

# Define features and target for the abalone dataset
abalone_features = abalone_data.drop(columns=['Rings']) 
abalone_target = abalone_data['Rings'] 


penguins_X_train, penguins_X_test, penguins_y_train, penguins_y_test = train_test_split(penguins_features, penguins_target, test_size=0.2, random_state=42)
abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test = train_test_split(abalone_features, abalone_target, test_size=0.2, random_state=42)

# Base Decision Tree
base_dt = DecisionTreeClassifier()
base_dt.fit(penguins_X_train, penguins_y_train)
y_pred_base_dt = base_dt.predict(penguins_X_test)

# Top Decision Tree with GridSearch
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],  # Add more values
    'min_samples_split': [2, 5, 10]  # Add more values
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(penguins_X_train, penguins_y_train)
top_dt = grid_search.best_estimator_
y_pred_top_dt = top_dt.predict(penguins_X_test)

# Base MLP
base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', max_iter=1000)
base_mlp.fit(penguins_X_train, penguins_y_train)
y_pred_base_mlp = base_mlp.predict(penguins_X_test)

# Top MLP with GridSearch
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  # Add more architectures
    'solver': ['adam', 'sgd']
}
grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)
grid_search.fit(penguins_X_train, penguins_y_train)
top_mlp = grid_search.best_estimator_
y_pred_top_mlp = top_mlp.predict(penguins_X_test)

# Evaluate and print results
def evaluate_classifier(y_true, y_pred, classifier_name):
    print(f"Classifier: {classifier_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")

evaluate_classifier(penguins_y_test, y_pred_base_dt, "Base Decision Tree")
evaluate_classifier(penguins_y_test, y_pred_top_dt, "Top Decision Tree")
evaluate_classifier(penguins_y_test, y_pred_base_mlp, "Base MLP")
evaluate_classifier(penguins_y_test, y_pred_top_mlp, "Top MLP")