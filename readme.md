# MAchine Learning Project

This repository contains the code and report for the "Tree Predictors for Mushroom Classification" project. The project implements a decision tree classifier from scratch to determine whether mushrooms are poisonous or edible based on their features.

## Project Overview

- **Custom Decision Tree:** Implements a tree predictor using single-feature binary tests.
- **Impurity Measures:** Supports three impurity measures: Gini index, entropy, and scaled entropy.
- **Stopping Criteria:** Incorporates multiple stopping criteria including node purity, minimum samples per leaf, and maximum tree depth.
- **Hyperparameter Tuning:** Uses 5-fold cross-validation for small datasets (primary) and an 80/20 train/test split for large datasets (secondary), with grid search over key hyperparameters.
- **Experimental Analysis:** Provides detailed evaluation through accuracy, confusion matrices, classification reports, and feature importance analysis.

## Repository Contents

- `main.py`: Contains the complete implementation including data preprocessing, model training, evaluation, and feature importance computation.
- `ML.pdf`: The project report (in PDF format) that details the methodology, experiments, and conclusions.
