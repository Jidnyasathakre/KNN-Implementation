# Diamond Price Prediction using KNN (From Scratch)

## Machine Learning Project | Regression | Algorithm Implementation

This project implements the K-Nearest Neighbors (KNN) algorithm from scratch to predict diamond prices using the popular diamonds dataset.
The goal is to understand and build the full ML workflow manually — from preprocessing to modeling — and then compare the scratch implementation with Scikit-Learn’s KNN.

## What This Project Demonstrates 
- Demonstrates how the K-Nearest Neighbors (KNN) algorithm works internally by building it completely from scratch
- Shows how to predict continuous values (regression) using KNN on real-world data
- Implements a full machine learning pipeline including preprocessing, encoding, scaling, training, and evaluation
- Highlights the importance of feature scaling and distance calculation in KNN performance
- Compares custom KNN implementation vs sklearn KNN to validate correctness and performance

## Objective
To design and implement a KNN regression model from scratch for predicting diamond prices, apply appropriate data preprocessing techniques, and compare its performance with Scikit-Learn’s KNN model.

## Dataset Overview
Dataset: diamonds.csv (53,940 records, 10 features)
#### Features:
- carat — weight of the diamond
- cut — quality of the cut
- color — diamond color (graded J–D)
- clarity — level of inclusions
- depth, table — proportions
- x, y, z — dimensions in mm
#### Target:
- price — price in USD

### Project Workflow

1. Load the dataset
2. Separate input (X) and target (y = price)
3. Train–Test split (75% Train / 25% Test)
4. Preprocess the data
  - Encode categorical features
  - Scale numerical features
5. Implement KNN from scratch
  - Compute distances
  - Select k-nearest neighbors
  - Predict using mean of neighbors
6. Evaluate model performance
7. Train sklearn KNN regressor
8. Compare both approaches

## Tech Stack & Tools

- Programming Language: Python
- Environment: Jupyter Notebook
- Libraries Used
   - numpy, pandas
   - matplotlib, seaborn (EDA & visualization)
   - Scikit-learn

## Model Evaluation

Evaluation metrics used to measure regression performance:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Result: 
The scratch KNN model produced results closely aligned with sklearn’s KNN, confirming the correctness of the implementation.

## Key Learnings

- How distance-based algorithms work internally
- Importance of feature scaling in KNN
- Handling categorical and numerical features together
- Performance trade-offs in brute-force KNN
- Writing production-readable ML code

## Conclusion

The scratch implementation of KNN produced results comparable to sklearn’s implementation, validating the correctness of the algorithm and reinforcing core machine learning concepts.
