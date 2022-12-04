# Credit_Risk_Analysis
Using machine learning to solve a real-world challenge: credit card risk.

# Purpose
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to employ different techniques to train and evaluate models with unbalanced classes. We are using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we’ll oversample the data using the **RandomOverSampler** and **SMOTE** algorithms, and undersample the data using the **ClusterCentroids** algorithm. Then, we’ll use a combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm. Next, we’ll compare two new machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk. 

In the end, we’ll evaluate the performance of these models to make a recommendation on whether they should be used to predict credit risk.

## 1. Resampling Models to Predict Credit Risk
First we check the balance of our target values, in particular, loan_status, which is our target, to see if we hace a **class imbalance**.
Class imbalance refers to a situation in which the existing classes in a dataset aren't equally represented. In this case, we can confirm that high risk loans ae the minority class, by contrast, low risk loans are the majority class.

- low_risk (1)= 68,470 values
- high_risk (0) = 347 values

Once we split the data set for train and test, the balance of values was checked again. Therefore, we can confirm the imbalance in the trainig set:
- low_risk (1)= 51,612 values
- high_risk (0) = 95 values

### 1. Naive Random Oversampling
In **random oversampling**, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. 
So, using the random oversampling model, which counted the classes of the resampled target and verified that the minority class (high risk loans) has been enlarged. Now the balance of the set is as follows:

- low_risk (1)= 51,352 values
- high_risk (0) = 51,352 values

A confusion matrix was built with the following results:

![image](https://user-images.githubusercontent.com/43974872/205478213-2021562d-d3c6-4138-9a12-511715ce52b6.png)

- Out of 87 high risk loans, (Actual high risk), 52 were predicted to be true (Predicted high risk), which we call true positives.
- Out of 87 high risk loans, (Actual high risk), 35 were predicted to be false (Predicted low risk), which are considered false negatives.
- Out of 17118 low risk loans (Actual low risk), 5952 were predicted to be false (Predicted high risk) and are considered false positives.
- Out of 17118 low risk loans (Actual low risk), 11166 were predicted to be true (Predicted low risk) and are considered true negatives.
- Predicted high risk = 6,004
- Predicted low risk =  11,201
- Actual high risk = 87
- Actual low risk = 17,118 
- Total observations= 17,205

Also, a classification report was built. The recisión of the model with a random oversamplig for low risk is **1**, a very good one; but the recall (sensitivity) is **0.65**, what means that is not so good beacause it means thatthere are many of false negatives. Although the F1 score for predicitng low risk loans are **0.79** what means is a ok balance between precision an recall. The **F1 score** is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0. Overall, the **Random oversampling predictive accuracy is 0.625** meaning that the model was correct **62.5%** of the time.

![image](https://user-images.githubusercontent.com/43974872/205478252-afd5fa06-8c33-4716-b632-7eadc5e926a4.png)


## 2. Using SMOTEENN algorithm to Predict Credit Risk
## 3. Using Ensemble Classifiers to Predict Credit Risk

# Analysis Report

## 1. Overview of the analysis: Explain the purpose of this analysis.

## 2. Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models

## 3. Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
