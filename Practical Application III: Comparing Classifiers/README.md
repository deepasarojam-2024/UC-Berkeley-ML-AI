# Practical Application III: Comparing Classifiers

**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  

### Getting Started

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.

## Data:
Input variables:
### bank client data:
1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')
### related with the last contact of the current campaign:
8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
### other attributes:
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
### social and economic context attributes
16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. cons.price.idx: consumer price index - monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. euribor3m: euribor 3 month rate - daily indicator (numeric)
20. nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):

21. y - has the client subscribed to a term deposit? (binary: 'yes','no')


### Business Objectives:
The Business objective is to The data is predict if the client will subscribe to a term deposit for data related to direct marketing campaigns (phone calls) of a Portuguese banking institution using classification models namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.

## Reference Paper 
According to USING DATA MINING FOR BANK DIRECT MARKETING: AN APPLICATION OF THE CRISP-DM METHODOLOGY, this is an implementation of a DM project based on the CRISP-DM methodology. The business goal is to find a model that can explain the success of a contract, i.e. if the client subscribes to the deposit. Such a model can increase campaign efficiency by identifying the main characteristics that affect success, helping in better management of the available resources (e.g. human effort, phone calls, time), and selecting a high-quality and affordable set of potential buying customers.

![](https://imgur.com/ebY1cOb.png) 

According to that, the best predictive model is **SVM**, which provides a high-quality **AUC value, higher than 0.9**.

![](https://imgur.com/HRjM226.png)

**Call duration is the most relevant feature**, meaning longer calls increase success. Secondly, **the month of contact**. Further analysis can show (Figure 6) that success is most likely to occur in the last month of each trimester (March, June, September, and December). Such knowledge can be used to shift campaigns to happen in those months.

## Data Understanding
First, the dataset was analyzed in detail. The dataset contained 41188 records, with 10 numerical and 11 categorical features. 

## Data Cleaning
There were no missing values in the data. Since there was no impact on the social and economic context attributes, these fields were dropped. 
 			

## Data Preparation
In addition to outlier cleaning from the data understanding process, I've split the data into categorical, numerical, and ordinal features.
- Numerical features: 'age', 'duration', 'campaign', 'pdays', 'previous'
- Categorical features: 'job', 'marital', 'default', 'housing', 'loan'
- Ordinal features: 'month', 'day_of_week'
- Ohe features: 'education', 'poutcome', 'contact'

Treated them with separate scaling, imputing, and encoding techniques.
- **StandardScaler** - Standardize features by removing the mean and scaling to unit variance.
- **Polynomial Features** - Polynomial Features are created by raising existing features to an exponent.
- **IterativeImputer** - A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.
- **RandomSampleImputer** - The RandomSampleImputer() replaces missing data with a random sample extracted from the variable. It works with both numerical and categorical variables.
- **OneHotEncoder** - Encodes categorical features as a one-hot numeric array.
- **JamesSteinEncoder** - This is a target-based encoder used for Categorical Encoding. It dominates the "ordinary" least squares approach, i.e., it has a lower mean squared error.
- **OrdinalEncoder** - Encode categorical features as an integer array.

![image](https://imgur.com/xUJ00l5.png)


## Modeling
The cleaned data was divided into target and feature (X and y) and then split into training and test data. First, a Baseline model (Naive Bayes) was used to evaluate the performance, followed by a Simple model (Logistic Regression). Then, the **RocCurveDisplay** curve was plotted for both models. 

Baseline model 	(Naive Bayes) 	

![](https://imgur.com/YPAtbtl.png) 	

Simple model (logistic Regression) 

![](https://imgur.com/5EWaMcY.png) 

          

Models used:
- **KNN Classifier**: A non-parametric classification algorithm that assigns labels to data points based on the majority class of their k-nearest neighbors.
- **SVM (Support Vector Machine)**: A supervised learning algorithm that finds the optimal hyperplane to classify data points by maximizing the margin between classes.
- **Decision Tree Classifier**: A hierarchical tree-like structure that recursively splits data based on feature attributes to make classification decisions.
- **Logistic Regression**: A statistical method for binary classification that models the probability of a binary outcome using a logistic function.

A GridSearchCV was performed finally along with all models. GridSearchCV is the process of performing hyperparameter tuning to determine the optimal values for a given model.  

	sklearn.model_selection.GridSearchCV(estimator, param_grid,scoring=None,
	          n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, 
	          pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)

## Performance Evaluation of models 
1. **Accuracy**: The ratio of correctly predicted instances to the total number of instances, measuring the overall correctness of the model's predictions.
2. **Precision**: The ratio of correctly predicted positive observations to the total predicted positive observations, measuring the accuracy of positive predictions.
3. **Recall**: The ratio of correctly predicted positive observations to all observations in actual class, measuring the ability of the model to identify all relevant instances.
4.** F1-Score:** The harmonic mean of precision and recall, providing a single metric that balances both precision and recall, is useful for comparing models with imbalanced class distributions.

# Result 
The following variables are the important features of the model:

1.   duration
2.   pdays
3.   previous
4.   month

SVM is the best model with an accuracy of 0.903937 and a Precision of 0.682403. Below are the results of all the models for different evaluation metrics (execution time is higher). 

![](https://imgur.com/cY0kGTh.png) 

# Findings
The data mining analysis in the Portuguese marketing campaign related to bank deposit subscriptions aimed to find a model that could explain the success of a contact, i.e., whether the client subscribes to the deposit. The Support Vector Machine (SVM) model achieved high predictive performances and the insights can be used by managers to enhance campaigns, such as increasing the length of phone calls or scheduling campaigns to specific months. Additionally, using open-source technology in the data mining field can provide high-quality models for real applications, such as *using cellular phones over telephones. 

Call duration is the most relevant feature, meaning longer calls increase success. Secondly, the month of contact. Further analysis shows that success is most likely to occur in the last month of each trimester (March, June, September, and December). Such knowledge can be used to shift campaigns to happen in those months.

Notebook: [link](https://github.com/deepasarojam-2024/UC-Berkeley-ML-AI/blob/main/Practical%20Application%20III%3A%20Comparing%20Classifiers/comparing_classifiers.ipynb)
