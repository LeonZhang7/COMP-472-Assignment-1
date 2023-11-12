# README for COMP-472-Assignment-1

## Overview
This project involves analyzing two datasets: Penguins and Abalones. It applies machine learning models such as Decision Trees and Multi-Layer Perceptrons (MLP) to classify these datasets. The script includes data preprocessing, model training, and performance evaluation.

## Dependencies
- pandas
- scikit-learn
- Pillow
- matplotlib
- NumPy

## Running the Script
Execute the script in a Python environment with the necessary libraries installed. 
Namely, type in the terminal: 
- pip install pandas
- pip install scikit-learn
- pip install pillow
- pip install matplotlib

## Dataset
- `abalone.csv`: Dataset containing information about abalones.
- `penguins.csv`: Dataset containing information about penguins.

## Usage

1. **Data Loading and Preprocessing**:
   - Load both datasets.
   - Preprocess data using One-Hot Encoding or Label Encoding (uncomment the preferred method).
   - Split features and targets for both datasets.

2. **Data Analysis**:
   - Calculate and visualize class distribution for both datasets.
   - Save the class distribution plots as images.

3. **Data Splitting**:
   - Split both datasets into training and testing sets.

4. **Model Training and Visualization**:
   - Train basic Decision Tree and MLP classifiers on both datasets.
   - Optimize Decision Trees using GridSearchCV.
   - Plot and save the decision trees.

5. **Model Evaluation**:
   - Evaluate the performance of each model.
   - Write performance metrics to text files.

6. **Experiment Repetition**:
   - Repeat the experiment five times for statistical robustness.
   - Calculate and record average accuracies and F1 scores.

## Notes
- The script uses a fixed `random_state` for reproducibility.
- GridSearchCV is used for hyperparameter tuning of Decision Trees and MLPs.
- Performance metrics include confusion matrix, classification report, accuracy, and F1 scores.
- The script saves various plots and performance metrics as external files.

## File Outputs
- `penguin-classes.png`, `abalone-classes.png`: Class distribution plots.
- `abalone-classes.gif`: Converted class distribution plot of Abalone dataset.
- `penguins_decision_tree.png`, `abalone_decision_tree.png`: Decision tree plots for base models.
- `penguins_top_decision_tree.png`, `abalone_top_decision_tree.png`: Decision tree plots for optimized models.
- `penguin-performance.txt`, `abalone-performance.txt`: Text files containing model performance metrics.

