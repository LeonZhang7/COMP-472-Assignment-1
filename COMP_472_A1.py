import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
from sklearn.metrics import f1_score

#part 1
#Load the dataset in Python
abalone_data = pd.read_csv('abalone.csv')
penguins_data = pd.read_csv('penguins.csv')


column_names = penguins_data.columns
print("penguin")
print(column_names)

column_names_aba = abalone_data.columns
print("abalone")
print(column_names_aba)

#uncomment either method you want to use
#part 1 a)
# One-Hot Encoding for 'island' and 'sex' columns
#get_dummies is used to convert 'island' and 'sex' into one-hot encoded vectors.
penguins_data = pd.get_dummies(penguins_data, columns=['island', 'sex'], drop_first=True)

#part 1 b)
# Convert 'island' and 'sex' features into categories 
# label_encoder_island = LabelEncoder()
# penguins_data['island'] = label_encoder_island.fit_transform(penguins_data['island'])

# label_encoder_sex = LabelEncoder()
# penguins_data['sex'] = label_encoder_sex.fit_transform(penguins_data['sex'])



# Define features and target for the penguins dataset
penguins_features = penguins_data.drop(columns=['species']) 
penguins_target = penguins_data['species']  

# Define features and target for the abalone dataset
abalone_features = abalone_data.drop(columns=['Type']) 
abalone_target = abalone_data['Type'] 



#part 2
# Calculate class distribution in the penguins target variable
class_distribution = penguins_target.value_counts(normalize=True) * 100

# Plot the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.xlabel('Species')
plt.ylabel('Percentage of Instances')
plt.title('Class Distribution in Penguins Dataset')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('penguin-classes.png')
plt.show()

# Calculate class distribution in the abalone target variable
class_distribution = abalone_target.value_counts(normalize=True) * 100

# Plot the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.xlabel('Type')
plt.ylabel('Percentage of Instances')
plt.title('Class Distribution in Abalone Dataset')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('abalone-classes.png')
plt.show()

# Open the PNG file
png_image = Image.open('abalone-classes.png')

# Convert and save as GIF
png_image.save('abalone-classes.gif')



#part 3
#split both the Penguins and Abalone datasets into training and test sets.
# The code uses a test size of 0.2 (which means 20% of the data will be used for testing, and the remaining 80% for training
penguins_X_train, penguins_X_test, penguins_y_train, penguins_y_test = train_test_split(penguins_features, penguins_target, test_size=0.2, random_state=42)
abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test = train_test_split(abalone_features, abalone_target, test_size=0.2, random_state=42)


#part 4
# a) Base Decision Tree
#penguin
base_dt = DecisionTreeClassifier()
base_dt.fit(penguins_X_train, penguins_y_train)
y_pred_base_dt = base_dt.predict(penguins_X_test)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(base_dt, filled=True, feature_names=penguins_features.columns, class_names=base_dt.classes_)
plt.title('Decision Tree for Penguins Dataset')
plt.savefig('penguins_decision_tree.png')
plt.show()


#abalone
# Fit the classifier without limiting depth
base_dt = DecisionTreeClassifier()
base_dt.fit(abalone_X_train, abalone_y_train)
y_pred_base_dt_aba = base_dt.predict(abalone_X_test)

# # Plot the decision tree with limited depth
plt.figure(figsize=(20,10))
plot_tree(base_dt, filled=True, feature_names=abalone_features.columns, class_names=base_dt.classes_, max_depth=2 )
plt.title('Decision Tree for Abalone Dataset')
plt.savefig('abalone_decision_tree.png')
plt.show()


# b) Top Decision Tree with GridSearch
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],  # Add more values
    'min_samples_split': [2, 5, 10]  # Add more values
}
#penguin
grid_search_DT = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search_DT.fit(penguins_X_train, penguins_y_train)
top_dt = grid_search_DT.best_estimator_
best_dt = grid_search_DT.best_params_
y_pred_top_dt = top_dt.predict(penguins_X_test)

# Plot the decision tree of the best estimator
plt.figure(figsize=(20, 10))
plot_tree(top_dt, filled=True, feature_names=penguins_features.columns, class_names=top_dt.classes_)
plt.title('Top Decision Tree for Penguins Dataset')
plt.savefig('penguins_top_decision_tree.png')
plt.show()

#abalone
grid_search_DT = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search_DT.fit(abalone_X_train, abalone_y_train)
top_dt_aba = grid_search_DT.best_estimator_
best_dt_aba = grid_search_DT.best_params_
y_pred_top_dt_aba = top_dt_aba.predict(abalone_X_test)

# Plot the decision tree of the best estimator found by GridSearchCV
plt.figure(figsize=(20, 10))
plot_tree(top_dt_aba, filled=True, feature_names=abalone_features.columns, class_names=top_dt_aba.classes_, max_depth=2)
plt.title('Top Decision Tree for Abalone Dataset')
plt.savefig('abalone_top_decision_tree.png')
plt.show()


# c) Base MLP
#penguin
base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', max_iter=1000)
base_mlp.fit(penguins_X_train, penguins_y_train)
y_pred_base_mlp = base_mlp.predict(penguins_X_test)

#abalone
base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', max_iter=1000)
base_mlp.fit(abalone_X_train, abalone_y_train)
y_pred_base_mlp_aba = base_mlp.predict(abalone_X_test)

# d) Top MLP with GridSearch
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],  # 'logistic' is another name for 'sigmoid'
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  # Two network architectures of your choice
    'solver': ['adam', 'sgd']  # Both 'adam' and 'stochastic gradient descent'
}
#penguin
grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)
grid_search.fit(penguins_X_train, penguins_y_train)
top_mlp = grid_search.best_estimator_
best_mlp = grid_search.best_params_
y_pred_top_mlp = top_mlp.predict(penguins_X_test)

#abalone
grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)
grid_search.fit(abalone_X_train, abalone_y_train)
top_mlp_aba = grid_search.best_estimator_
best_mlp_aba = grid_search.best_params_
y_pred_top_mlp_aba = top_mlp_aba.predict(abalone_X_test)


#part 5

#Print in output file, we can remove the printing in terminal for just the file after finishing 

def evaluate_classifier(y_true, y_pred, classifier_name, file, data_set):
    file.write('-' * 50 + '\n')
    file.write(f"Data Set: {data_set}"+ '\n')
    file.write("(A)"+ '\n')
    file.write(f"Classifier: {classifier_name}\n")
    if(classifier_name=='Top Decision Tree' and data_set == 'penguin'):
        file.write(f"Best Hyperparameter: { best_dt }" + "\n")
    if(classifier_name=='Top Decision Tree' and data_set == 'abalone'):
        file.write(f"Best Hyperparameter: { best_dt_aba }" + "\n")
    if(classifier_name=='Top MLP' and data_set== 'penguin'):
        file.write(f"Best Hyperparameter: { best_mlp }" + "\n")
    if(classifier_name=='Top MLP' and data_set== 'abalone'):
        file.write(f"Best Hyperparameter: { best_mlp_aba }" + "\n")
    file.write("(B)"+ '\n')
    file.write("Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)) + "\n")
    file.write("(C)"+ "(D)"+  '\n')
    file.write("Classification Report:\n" + str(classification_report(y_true, y_pred)) + "\n")
    file.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")

with open('penguin-performance.txt', 'w') as file:
    evaluate_classifier(penguins_y_test, y_pred_base_dt, "Base Decision Tree", file, "penguin")
    evaluate_classifier(penguins_y_test, y_pred_top_dt, "Top Decision Tree", file, "penguin")
    evaluate_classifier(penguins_y_test, y_pred_base_mlp, "Base MLP", file, "penguin")
    evaluate_classifier(penguins_y_test, y_pred_top_mlp, "Top MLP", file, "penguin")

with open('abalone-performance.txt', 'w') as file_aba:
    evaluate_classifier(abalone_y_test, y_pred_base_dt_aba, "Base Decision Tree", file_aba, "abalone")
    evaluate_classifier(abalone_y_test, y_pred_top_dt_aba, "Top Decision Tree", file_aba, "abalone")
    evaluate_classifier(abalone_y_test, y_pred_base_mlp_aba, "Base MLP", file_aba,"abalone")
    evaluate_classifier(abalone_y_test, y_pred_top_mlp_aba, "Top MLP", file_aba, "abalone")

def repeat_experiment(model, X_train, X_test, y_train, y_test, model_name, file, data_set):
    accuracies = []
    macro_f1s = []
    weighted_f1s = []

    for i in range(5):  # Repeat the experiment 5 times
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Append information to the performance file
        evaluate_classifier(y_test, y_pred, model_name, file, data_set)

        # Append metrics for later analysis
        accuracies.append(accuracy_score(y_test, y_pred))
        macro_f1s.append(f1_score(y_test, y_pred, average='macro'))
        weighted_f1s.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate and append average and variance to the performance file
    file.write(f"A) Average Accuracy: {np.mean(accuracies)}, Variance: {np.var(accuracies)}\n")
    file.write(f"B) Average Macro-average F1: {np.mean(macro_f1s)}, Variance: {np.var(macro_f1s)}\n")
    file.write(f"C) Average Weighted-average F1: {np.mean(weighted_f1s)}, Variance: {np.var(weighted_f1s)}\n")


# Repeat the experiment for each model 5 times
with open('penguin-performance.txt', 'a') as file:
    repeat_experiment(DecisionTreeClassifier(), penguins_X_train, penguins_X_test, penguins_y_train, penguins_y_test,
                      "Base Decision Tree", file, "penguin")
    repeat_experiment(top_dt, penguins_X_train, penguins_X_test, penguins_y_train, penguins_y_test, "Top Decision Tree",
                      file, "penguin")
    repeat_experiment(MLPClassifier(), penguins_X_train, penguins_X_test, penguins_y_train, penguins_y_test, "Base MLP",
                      file, "penguin")
    repeat_experiment(top_mlp, penguins_X_train, penguins_X_test, penguins_y_train, penguins_y_test, "Top MLP", file,
                      "penguin")

with open('abalone-performance.txt', 'a') as file_aba:
    repeat_experiment(DecisionTreeClassifier(), abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test,
                      "Base Decision Tree", file_aba, "abalone")
    repeat_experiment(top_dt_aba, abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test, "Top Decision Tree",
                      file_aba, "abalone")
    repeat_experiment(MLPClassifier(), abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test, "Base MLP",
                      file_aba, "abalone")
    repeat_experiment(top_mlp_aba, abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test, "Top MLP", file_aba,
                      "abalone")
#6 
# # Initialize lists to store metrics for each run
# accuracies = []
# macro_f1s = []
# weighted_f1s = []

# # Loop for 5 iterations
# for i in range(5):

#     # Get the evaluation metrics
#     report = classification_report(abalone_y_test, penguins_y_test, output_dict=True)
#     accuracies.append(report['accuracy'])
#     macro_f1s.append(report['macro avg']['f1-score'])
#     weighted_f1s.append(report['weighted avg']['f1-score'])

# # Calculate averages and variances
# average_accuracy = np.mean(accuracies)
# variance_accuracy = np.var(accuracies)
# average_macro_f1 = np.mean(macro_f1s)
# variance_macro_f1 = np.var(macro_f1s)
# average_weighted_f1 = np.mean(weighted_f1s)
# variance_weighted_f1 = np.var(weighted_f1s)

# # Append results to performance files
# with open('performance.txt', 'a') as file:
#     file.write("Averages and Variances for 5 Runs\n")
#     file.write(f"(A) Average Accuracy: {average_accuracy}, Variance: {variance_accuracy}\n")
#     file.write(f"(B) Average Macro-average F1: {average_macro_f1}, Variance: {variance_macro_f1}\n")
#     file.write(f"(C) Average Weighted-average F1: {average_weighted_f1}, Variance: {variance_weighted_f1}\n")