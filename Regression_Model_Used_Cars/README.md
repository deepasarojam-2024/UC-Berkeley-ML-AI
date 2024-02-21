# Practical Application 2 - Used Cars Price Prediction using Regression

## Overview
In this application, you will explore a dataset from Kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

To frame the task, throughout these practical applications, we will refer back to a standard process in the industry for data projects called CRISP-DM. This process provides a framework for working through a data problem.  

## Data:
The code used to perform the analysis and create this report can be found here. 
As it was mentioned, our original data holds half a million observations with a few dozen features, most categorical, so accurate feature selection and model selection were extremely important.  
As we had some intuition in the target area as well as some practical experience, we were able to prune our feature list to just 12 most important in our opinion:
	• 10 categorical features:
		○ manufacturer (brand)
		○ condition
		○ cylinders
		○ fuel type
		○ title_status
		○ transmission type
		○ drive type (AWD / FWD / RWD)
		○ car type
		○ paint color
		○ state
	• 2 continuous features:
		○ year
		○ odometer

## Business Understanding

From a business perspective, we are asked to identify the factors that influence used car prices. In CRISP-DM terms, we are asked to convert this business problem into a data problem definition.
### Business Objectives:
  - As a client of a car dealership, we want to identify what features play a key role in a used car's price. 
  - Based on the recommendations, dealerships can stock up on cars with those features that are significant for used cars based on our analysis.  
  - Dealerships can increase their profit if they implement our recommendations and stock used cars with those recommended features. 

### Assumptions: 

- VINs are unique identifications for a car and no two cars can have the same VIN. 
- Used car prices depend on the condition of the car, the better the condition, the higher the price. (condition) 
- The older the car becomes the price drops. (age of car or year of manufacture) 
- The more used the car is, the more the price is. (odometer reading). 


## Data Understanding
First, the dataset was analyzed in detail. The dataset contained 427K records, with 3 numerical and 13 categorical features: id, price, year, manufacturer, model, condition, cylinders, fuel, title_status, transmission, VIN, drive, size, type, paint_color, and state. 

With close consideration, outliers were identified for some of the features that would skew the data. To avoid any skewness, the outliers were removed. Identified duplicate VINs and those were removed first. Later, Nan's, and values like 'other' were removed/imputed. 

Visualized the data using a seaborn plot to understand each feature. 
- Of the Luxury brands, Ferrari was the most pricey with a price > 65K. 
- Of the Economy brands, Volkswagen was the most pricey with a price >12K. 
- The 'Other' transmission type was more pricey followed by `automatic` and then `manual`. I assume that the other category is unknown or data missing. 
- Fuel type `Diesel` was more pricey followed by `Electric`. Hybrid cars were the least. 
- `4WD` drive types were more pricey compared to `RWD` and `FWD`. 
- `PICKUP` cars were more pricey followed by `TRUCK`. 
- Most used cars were `OFFROAD` type followed by `CONVERTIBLE`. 
- There are cars with 0~100K price cars in the dataset. As the odometer reading increases, the price of the car decreases. 

## Data Cleaning
As part of data cleaning, we performed the below.
- Dropped the id column since it's holding a unique identifier that has no predictive meaning
- Dropped the size column since it's missing more than 70% of its data
- Dropped duplicate VINs, with the assumption that no two cars can have the same VINs. 
- Dropped the VIN column after cleaning the duplicates, since it's just holding a unique identifier that has no predictive meaning
- Removed odometer outliers and kept cars with odometer which is less than 500K
- Removed cars with age > 80
- Remove price outliers and keep cars with prices between 100 and 100,000 USD 
- Removed all title_status besides clean as 90% of data had `clean` title status.
- Removed rows that contained other value  
			

## Data Preparation
In addition to outlier cleaning from the data understanding process, I've split the data into categorical, numerical, and ordinal features. 

Treated them with separate scaling, imputing, and encoding techniques.
**StandardScaler** - Standardize features by removing the mean and scaling to unit variance.
Polynomial Features - Polynomial Features are created by raising existing features to an exponent.
**IterativeImputer** - A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.
**RandomSampleImputer** - The RandomSampleImputer() replaces missing data with a random sample extracted from the variable. It works with both numerical and categorical variables.
**OneHotEncoder** - Encodes categorical features as a one-hot numeric array.
**JamesSteinEncoder** - This is a target-based encoder used for Categorical Encoding. It dominates the "ordinary" least squares approach, i.e., it has a lower mean squared error.
**OrdinalEncoder** - Encode categorical features as an integer array.


## Modeling
The cleaned data was divided into target and feature (X and y) and then split into training and test data. I used a GridSearchCV and K-FOLD cross-validation on all models. GridSearchCV is the process of performing hyperparameter tuning to determine the optimal values for a given model. K-FOLD splits the dataset into k consecutive folds (without shuffling by default).

sklearn.model_selection.GridSearchCV(estimator, param_grid,scoring=None,
          n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, 
          pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
          
<img width="443" alt="image" src="https://github.com/deepasarojam-2024/UC-Berkeley-ML-AI/assets/153694311/e0e365bb-4543-4bba-9257-9d03518a00e1">

Models used:
	• Ridge regression
	• Lasso regression 
	• Linear regression
	• Linear regression with TransformedTargetRegressor; 3-degree Polynomial
	• Linear regression with Feature Selection (SFS) and TransformedTargetRegressor 

## Evaluation
The following variables are the top predictors in the model:

1.   Year
2.   Odometer
3.    Fuel type (gas and other)


There are a large number of records with missing and other values. Instead of imputing these values, we decided to use OneHot, Ordinal, and JamesStein encoding after assigning missing and others to different categories, which increased the dimensionality of the dataset. 
We see a better `R-Squared = 0.591952` with Ridge followed by `R-Square=0.591950` for the Lasso Model. 

![image](https://github.com/deepasarojam-2024/UC-Berkeley-ML-AI/assets/153694311/5e50f15a-49f3-4b51-88bf-07613d2f547f)

For the test RMSE, Linear Regression with Feature Selection (SFS) and TransformedTargetRegressor gave a higher price of ~`9943`, followed by Linear Regression with TransformedTargetRegressor with a price value of ~`9537`.

![image](https://github.com/deepasarojam-2024/UC-Berkeley-ML-AI/assets/153694311/b6edcf68-acfe-4e13-b8cc-8763ff88e6af)



## Deployment
From the above results, the data shows that the main features that drive the car price were: year, odometer, fuel type, number of cylinders and type of the car are the most important features for predicting car price.

Year and odometer were the most dominant features, so when looking at buying used cars for the dealership, we'd recommend buying newer cars and cars with low odometer. We'd also stay away from cars with high odometers and old cars.

Data also show that the number of cylinders (more cylinders the higher the price), and type of the car (4wd), and the fuel type (diesel), play a role in driving the price up, so it's worth highlighting these features when selling a used car, and pay attention to these details when pricing used cars.
