# Practical Application 2 - Used Cars Price Prediction using Regression

## Business Understanding
The objective of this project is to identify the factors affecting the resale value of the used cars. A dataset of 426K records was used in the analysis to determine whether there is a relationship between factors such as the age of the car, manufacturer, condition, fuel type, transmission type, mileage, color, etc., and car resale value.

We answered several business questions using the historical data for the used car sales:

Is the price of the car related to the following factors?

Region: Does the resale value of the car change for different regions in the US? This information may be useful for a dealer with multiple locations in the country.
Year: Along with the other factors, how does the age of the car impact its resale value?
Manufacturer: What type of cars do have a higher resale value?
Model: There are hundreds of models. Can we predict which ones have higher resale value?
Condition: How does the condition of the car affect its resale value along with other factors such age and mileage?
Number of cylinders: Do the larger cars have higher resale value?
Fuel type: Does the electric cars have higher resale value?
Odometer: Is the mileage the top factor for resale value?
Title status: Is it a requirement to have a clean title to sell a used car?
Transmission type: Are the cars with automatic transmission more expensive?
Size: How does the size of the used car impact its resale value?
Color: Do people prefer more neutral colors when they buy a used car?
Data Understanding
Several steps were taken to get to know the dataset and identify any data quality issues with the dataset.

Review of the columns in the dataset: -Do we have a good understanding of all the columns in the dataset?
Which columns can be used for the prediction problem?-
Are there any columns not required for the modeling problem?
Review of the rows in the dataset:
Are there sufficient records for the prediction problem?

Are there any duplicate records?
Are there any outliers, e.g., records with very high resale value or mileage?
Are there any records outside our analysis space such as data related with new cars for a prediction problem of used car resale value?
Are there any records with missing values that can be excluded from the analysis sample?
How do you deal with records with missing values that we plan to keep?
Review of the final analysis sample:

Do we have any bias in the data? Does the data represent represent only certain categories of cars?
Do we have sufficient number of rows for model development?
Are all the columns in the dataset potentially related with the resale value? For example, did we remove any match keys, etc.?
Data Preparation
Any issues with the data were resolved in this step:

We confirmed that there are no duplicate records on the data based on "id".
We found a significant number of records with missing or duplicate VIN numbers. Since VIN would not be used for model development, we decided to keep the records.
The small sample of records with missing data for the continous variables were excluded from the analysis sample.
The variable for State was grouped into nine US Census Divisions.
A "missing" category was defined for the categorical variables with missing data.
We excluded any outliers for the car price and applied log transformation. We confirmed that the distribution of log-price looks normal.
We analyzed the relationship between log-price and independent variables to make a decision on the final list of variables to keep in the model.
We hot encoded the categorical variables. We did not do any label encoding since the values were not rank ordering.
We prepared train (70%) and test (30%) samples for model development.
Modeling
We developed five different models:

## Multiple linear regression
Multiple linear regression with 2-degree polynomial features
Ridge regression
Sequential forward with ridge regression with 20 features
Lasso regression using the dataset constructed by the sequential forward model
Models were validated using both train/test and k-fold validation techniques. Mean Squared Error (MSE) was used to compare the error in the prediction for different modelling methodologies. R-sq was used to understand what percentage of the variation in the data could be explained by each model. Grid search was applied to find the optimum alpha values for ridge and lasso regression.

## Evaluation
Below is the summary of our observations:

The multiple linear regression model was able to explain 64% of the variation in the data.
Adding interaction variables (2-Degree polynomial features) did work very well on the train dataset. More research is required to understand why the model failed on the test sample.
We did not get any additional value from ridge regression. The coefficients of the variables are slightly different but there was no change in performance statistics.
We were able to develop a more condensed model with 20 features using sequential forward selection but of course, this came at a cost of a lower R-sq and higher MSE.
Lasso regression failed to generate any meaningful coefficients for the model developed using the dataset created by the 20 features selected in Step 4. We used a dataset with the selected variables from the forward selection, ridge regression model due to the very long time on the original hot-encoded dataset
The top factors that impact the car price are the age of the car, number of cylinders, fuel type, mileage, drive type, and its condition.

The study had limitations that should be revisited in the future:

There are a large number of records with missing and "other" values. Instead of imputing these values, we decided to use hot encoding after assigning missing and other to different categories, which increased the dimensionality of the dataset. We can repeat our steps by trying label encoder for the variables with ordinal values. For example, the variable 'cyclinders' has values 3-12 cyclinders, other and missing. If we didn't have other and missing, we could label encode 3 cylinders with 3, 4 cylinders with 4, etc.
The dataset could be enhanced further by pulling information from public resources. For example, features such as gas mileage, number of doors, luxury/economy indicator might also be good predictors of resale value. This information could be pulled online by the using the column for 'model'.
The interaction between the variables should be revisited. The model with 2-degree polynomial features provided an R-sq of 72% on the training dataset. After reprocessing the dataset (use label encoder instead of one hot encoding, remove other/missing categories), polynomial features could be tested.
Building a set of segmented models instead of one model could also improve the prediction quality. For example, the depreciation of the luxury cars might be different. We can build a separate model for luxury cars and another model for economy cars. We can also build a different model for trucks.

## Deployment
We made the following recommendations to our client:

Newer cars tend to have a much higher resale value.
Cars with more than 50K miles on them have a much lower resale value.
Even though age and mileage are important factors for the resale value, size of the car, fuel and drive type and condition impact the resale value.
We would like to continue our working relationship to further enhance our prediction by enhancing the dataset using other publicly available data such as gas mileage, number of doors, developing separate models for cars and trucks, and testing other modeling techniques.
You can find the Jupyter notebook in this location: https://github.com/SunaHafizogullari/BerkeleyMLCertificate-CarSalePrice/blob/main/prompt_II.ipynb
