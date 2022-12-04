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

Also, a classification report was built. The precisión of the model with a random oversamplig for low risk is **1**, a very good one; but the recall (sensitivity) is **0.65**,it  what means is not so good beacause there are many false negatives. Although the F1 score for predicitng low risk loans are **0.79** which is an ok balance between precision an recall. The **F1 score** is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0. Overall, the **Random oversampling predictive accuracy is 0.625** meaning that the model was correct **62.5%** of the time.

![image](https://user-images.githubusercontent.com/43974872/205478252-afd5fa06-8c33-4716-b632-7eadc5e926a4.png)

In summary, this model may not be the best one for predict credit risk because the model's accuracy, 0.625, is low, and the precision and recall are not good enough to state that the model will be good at classifying high risk loans. To have a better model, may be we need more data.

### 2. SMOTE Oversampling
The **synthetic minority oversampling technique (SMOTE)**, like random oversampling, the size of the minority is increased, but because it create new instances are interpolated.Random oversampling draws from existing observations, whereas SMOTE generates synthetic observations, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

Now the balance of the set is the same as the random oversampling:
- low_risk (1)= 51,352 values
- high_risk (0) = 51,352 values

The confusion matrix was built with the following results:

![image](https://user-images.githubusercontent.com/43974872/205478903-2f39f870-4df4-4696-9100-5260bf164379.png)

- Out of 87 high risk loans, (Actual high risk), 56 were predicted to be true (Predicted high risk), which we call true positives.
- Out of 87 high risk loans, (Actual high risk), 31 were predicted to be false (Predicted low risk), which are considered false negatives.
- Out of 17118 low risk loans (Actual low risk), 5840 were predicted to be false (Predicted high risk) and are considered false positives.
- Out of 17118 low risk loans (Actual low risk), 11278 were predicted to be true (Predicted low risk) and are considered true negatives.
- Predicted high risk = 5,896
- Predicted low risk =  11,309
- Actual high risk = 87
- Actual low risk = 17,118 
- Total observations= 17,205

In this model, there are less loans predicted as high risk thatn the previous model, and more loans prediected as low risk, which can be harmful if we want to predict credit risk.

The classification report shows that the precision of the model with a SMOTE oversamplig for low risk is still **1**, a very good one; the recall (sensitivity) improved marginally to **0.66**, what it means that, the model still have many false negatives. Although the F1 score for predicitng low risk loans are the same as before **0.79** which is an ok balance between precision an recall. Overall, the **SMOTE oversampling predictive accuracy is 0.651** meaning that the model was correct **65.1%** of the time, a litlle better that the random oversampling model.

In summary, the metrics of the minority class (recall, and F1 score) are slightly improved over those of random oversampling. SMOTE reduces the risk of oversampling, but its vulnerability to outliers: So this model, even if improved, still not the best model to predict credit risk, maybe we have to do more preprocessing the data to check for big outliers, because if there are extreme outliers, the new values creted bay SMOTE will reflect it.

![image](https://user-images.githubusercontent.com/43974872/205479225-90033d0a-02c1-4813-bde6-683372f37cb5.png)

### 3. Undersampling
In **Undersampling** instead of increasing the number of the minority class, the size of the majority class is decreased; so only uses actual data, therefore is practical only when there is enough data in the training set. In particular, the **Cluster centroid undersampling** identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

The balance of the set
- low_risk (1)= 260 values
- high_risk (0) = 260 values

The confusion matrix was built with the following results:

![image](https://user-images.githubusercontent.com/43974872/205479414-b1abf8e2-30ec-442f-810f-516988295084.png)

- Out of 87 high risk loans, (Actual high risk), 51 were predicted to be true (Predicted high risk), which we call true positives.
- Out of 87 high risk loans, (Actual high risk), 36 were predicted to be false (Predicted low risk), which are considered false negatives.
- Out of 17118 low risk loans (Actual low risk), 9679 were predicted to be false (Predicted high risk) and are considered false positives.
- Out of 17118 low risk loans (Actual low risk), 7439 were predicted to be true (Predicted low risk) and are considered true negatives.
- Predicted high risk = 9,730
- Predicted low risk =  7,475
- Actual high risk = 87
- Actual low risk = 17,118 
- Total observations= 17,205

In this model, there are more loans predicted as high risk than the two previous model, and less loans predicted as low risk, which can be more helpful to predict credit risk.

The classification report shows that the precision of the model with a **Cluster centroid undersampling** for low risk is still **1**, a very good one; the recall (sensitivity) dropped to **0.43**, what it means that, the model have a lot of false negatives. Although the F1 score for predicitng low risk loans decrease to **0.60** which means there is no balance between precision an recall. Overall, the **Cluster centroid undersampling accuracy is 0.51** meaning that the model was correct **51%** of the time, worst accuracy than the two previous models.

In summary, the results were worse than those from random undersampling and SMOTE. Therefore, the **Cluster centroid undersampling** attempted to address imbalance, but does not guarantee better results.

![image](https://user-images.githubusercontent.com/43974872/205479804-13471c34-ba4c-4d2c-8d60-91879571d611.png)

## 2. Using SMOTEENN algorithm to Predict Credit Risk
**SMOTEENN** combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process, first, oversample the minority class with SMOTE; secondly, cleans the resulting data with an undersampling strategy, If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

The balance of the set
- low_risk (1)= 62,011 values
- high_risk (0) = 68,460 values

The confusion matrix was built with the following results:

![image](https://user-images.githubusercontent.com/43974872/205479932-5fa15c6e-c4f2-4018-aa46-16a575210d0b.png)

- Out of 87 high risk loans, (Actual high risk), 61 were predicted to be true (Predicted high risk), which we call true positives.
- Out of 87 high risk loans, (Actual high risk), 26 were predicted to be false (Predicted low risk), which are considered false negatives.
- Out of 17118 low risk loans (Actual low risk), 7,294 were predicted to be false (Predicted high risk) and are considered false positives.
- Out of 17118 low risk loans (Actual low risk), 9,824 were predicted to be true (Predicted low risk) and are considered true negatives.
- Predicted high risk = 7,355
- Predicted low risk =  9,850
- Actual high risk = 87
- Actual low risk = 17,118 
- Total observations= 17,205

In this model, we can tell there are less false negatives, false positives and more high risk loans, which can be helpful to predict credit risk.

The classification report shows that the precision of the model with a **SMOTEEN** for low risk is still **1**, a very good one; the recall (sensitivity) is **0.57**, what it means that, the model have a lot of false negatives, although is better that the undersampling model, it not better than the two oversampling models. Although the F1 score for predicitng low risk loans improved to **0.73** which means there is balance between precision an recall for low risk loans, ist also show a marginall improvement in the F1 score for hig risk loans. In particualr, the sensitivy or recal for high risk loans improved. Overall, the **SMOTEEN accuracy is 0.638** meaning that the model was correct **63.8%** of the time, better accuracy than the previous models, but not better accurary than the SMOTE Oversampling accuracy (0.65).

In summary, the results were better than those from random undersampling and undersampling, but not better then SMOTE oversampling. 

![image](https://user-images.githubusercontent.com/43974872/205479966-fd53b562-d953-4117-9815-92588089369c.png)

## 3. Using Ensemble Classifiers to Predict Credit Risk

# Analysis Report

## 1. Overview of the analysis: Explain the purpose of this analysis.

## 2. Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models

## 3. Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
