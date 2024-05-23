                                                                
# <center> **Data-Driven Insights: Enhancing User Experience through Advanced Recommendation Systems**</center>  
### **A Deep Dive into Collaborative Filtering, Content-Based Filtering, Rank-Based, Popularity-Based Methods, and Sentiment Analysis**

## **Author -** **Deepa Sarojam** 

# <center> ![Capstone Header Image](https://imgur.com/U8CuIiT.gif)</center>

## **Executive Summary**

Recommender systems are algorithms designed to predict user preferences or recommend items to users. These systems have become ubiquitous in today's digital landscape, powering recommendations on platforms like Amazon, Netflix, and Spotify. My fascination with the transformative impact of these technologies on user engagement and satisfaction has inspired me to explore this topic for my Capstone project.

By exploring sentiment analysis on the data, I aimed to understand user preferences and reactions more deeply, although the implementation was focused solely on the recommendation system. This project seeks to enhance the user experience by delivering accurate recommendations based on collaborative and popularity-based filtering techniques.

## **Data Sources**

To construct a dataset suitable for exploring recommendation systems and sentiment analysis, I utilized **Python's Faker** library, with **30, 000 records and 25 columns**.

Drawing inspiration from publicly available datasets such as H&M data from Kaggle and the ShopperSentiments dataset, I synthesized a dataset containing relevant attributes like Transaction ID, Date, Product ID, Product Name, Product Description, Product Category, Sub Category, Color, Rating, Product Review, Quantity, Customer ID, Customer Name, Age, Price, Currency, Discounts, Sales Revenue, Payment Method, Store Type, Season, Latitude, Longitude, and Country. This rich dataset mirrors real-world transactional and review data, enabling comprehensive analysis and experimentation to enhance user experience through advanced recommendation systems and sentiment analysis.

**H&M dataset**- https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data


**ShopperSentiments dataset** - https://www.kaggle.com/datasets/nelgiriyewithana/shoppersentiments

There are 25 columns in the dataset.

* **Transaction ID**: Unique identifier for each transaction.
* **Dat**e: Date of the transaction.
* **Product ID**: Unique identifier for each product.
* **Product Name**: Name of the product.
* **Product Description**: Description of the product (some missing values).
* **Product Category**: Category of the product.
* **Sub Category**: Sub-category of the product.
* **Color**: Color of the product.
* **Rating**: Rating given by customers for the product.
* **Product Review**: Review provided by customers for the product (some missing values).
* **Number of Reviews**: Total number of reviews for the product.
* **Quantity**: Quantity of the product purchased in the transaction.
* **Customer ID**: Unique identifier for each customer.
* **Customer Name**: Name of the customer.
* **Age**: Age of the customer (some missing values).
* **Price**: Price of the product.
* **Currency**: Currency used for the transaction.
* **Discounts**: Boolean indicating whether any discounts were applied.
* **Sales Revenue**: Revenue generated from the transaction.
* **Payment Method**: Method used for payment.
* **Store Type**: Type of store where the transaction occurred.
* **Season**: Season in which the transaction occurred.
* **Latitude**: Latitude of the transaction location.
* **Longitude**: Longitude of the transaction location.
* **Country**: Country where the transaction occurred.


**Dataset**

The dataset for this project was synthesized using Python Faker by taking inspiration from publicly available datasets such as the H&M dataset from Kaggle and the ShopperSentiments dataset. It has **30, 000 records and 25 columns**. It includes various attributes like Transaction ID, Date, Product ID, Product Name, Product Description, Product Category, Sub Category, Color, Rating, Product Review, Quantity, Customer ID, Customer Name, Age, Price, Currency, Discounts, Sales Revenue, Payment Method, Store Type, Season, Latitude, Longitude, and Country. 

**Exploratory Data Analysis (EDA)** 

- **Frequency of Purchases**: The analysis revealed that many users made frequent purchases in specific product categories. For instance, categories like "Womens Everyday Collection" and "Divided Collection" saw higher user interaction.

- **Seasonal Trends**: There were clear seasonal patterns in user purchases, with spikes observed during certain times of the year, such as holiday seasons and sales events. This indicated that users were more active and engaged during these periods.

- **Top Product Categories**: The EDA identified the top 30 product categories based on the number of occurrences in the dataset. The "Womens Everyday Collection" was the most popular, followed by the "Divided Collection" and "Womens Swimwear, beachwear".

- **High-Rated Products**: Some products consistently received higher ratings, suggesting their popularity and user satisfaction. Products with high ratings were often from well-known brands or belonged to trending categories.

- **Global Reach**: The dataset showed a wide geographic distribution of shoppers, indicating that the user base was diverse and spread across various regions. This was visualized using interactive maps highlighting the concentration of purchases in different areas.

- The United States dominates the number of purchases, followed by Canada, the United Kingdom, and Australia.
![](https://imgur.com/uxsrr4j.png)


![](https://imgur.com/wiyWFtB.png)

- **Rating Patterns:** The distribution of ratings indicated that most users tended to give higher ratings (4 or 5 stars), suggesting overall satisfaction with their purchases. There were fewer low ratings (1 or 2 stars), which could indicate a generally positive user experience.

- **Skewness and Kurtosis:** The ratings distribution exhibited positive skewness, with a peak towards the higher end of the rating scale. The kurtosis value suggested that the distribution had lighter tails, meaning fewer extreme ratings.

- **Average Transaction Value**: The EDA revealed the average transaction value across different categories. High-value transactions were more common in categories like electronics and luxury items, while everyday items had lower average transaction values.

- **Discount Impact:** Transactions with applied discounts or promotions saw a higher frequency, indicating that discounts were a significant motivator for purchases. This insight could be used to strategize marketing and promotional campaigns.

## **Recommendation System**

Recommendation systems are algorithmic approaches used to suggest items of interest to users, such as movies, music, products, or articles, based on their preferences and behaviors. These systems leverage various techniques, including collaborative filtering, content-based filtering, and hybrid methods, to analyze user data and generate personalized recommendations.

**Collaborative filtering** methods rely on user-item interactions, identifying similarities between users or items to make predictions.

**Content-based filtering** considers the attributes of items and user preferences to recommend similar items.

**Hybrid** approaches combine the strengths of both methods to enhance recommendation accuracy and overcome their limitations.

Recommendation systems play a crucial role in enhancing user experience, increasing engagement, and driving sales in e-commerce, entertainment, and content platforms.

Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are two commonly used metrics for evaluating the performance of regression models, including recommendation systems. Here's a brief overview of each:

**Mean Squared Error (MSE):**

**Definition**: MSE is the average of the squared differences between the predicted values and the actual values.
![](https://imgur.com/MM5TcyU.png)

**Interpretation**: A lower MSE indicates better predictive accuracy, as it means the predicted values are closer to the actual values. However, MSE can be sensitive to outliers due to the squaring of errors.

**Root Mean Squared Error (RMSE):**

**Definition**: RMSE is the square root of the average of the squared differences between the predicted values and the actual values.
![](https://imgur.com/af3YTSi.png)

**Interpretation**: RMSE is in the same units as the original data, making it more interpretable than MSE. Like MSE, a lower RMSE indicates better model performance. It also provides a sense of the magnitude of typical prediction errors.


Both MSE and RMSE are useful for assessing the accuracy of predictive models, with RMSE often being preferred for its interpretability in the context of the original data.

![](https://imgur.com/yhBNYpe.png)


**Grid search** is a systematic method used to find the optimal hyperparameters for a machine learning model. It works by exhaustively searching through a specified subset of hyperparameter combinations and evaluating each combination using cross-validation.


**The goal is to identify the hyperparameters that result in the best performance metric, such as accuracy, precision, or RMSE.** Grid search iterates over all possible combinations of hyperparameters defined in a grid, hence the name "grid search."

## **Key Findings/Results**

**Model Evaluation Results**

The performance of various recommendation algorithms was evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The algorithms tested included KNNBasic, SVD, NMF, SlopeOne, CoClustering, KNNBaseline, KNNWithMeans, and KNNWithZScore. Key results are:

- **KNNBasic**: MSE = 1.379, RMSE = 1.174

- **SVD**: MSE = 1.398, RMSE = 1.182

- **NMF**: MSE = 1.412, RMSE = 1.188

- **SlopeOne**: MSE = 1.401, RMSE = 1.183

- **CoClustering**: MSE = 1.408, RMSE = 1.187

- **KNNBaseline**: MSE = 1.395, RMSE = 1.181

- **KNNWithMeans**: MSE = 1.408, RMSE = 1.187

- **KNNWithZScore**: MSE = 1.407, RMSE = 1.186


Among these, KNNBasic showed the lowest RMSE and MSE, indicating its superior predictive accuracy​.

**Recommendation Systems**

**Popularity-Based Recommendation**: Items with the highest overall ratings were recommended to all users. This approach was particularly useful for new users with no prior interaction history.

**Rank-Based Recommendation**: This approach used the frequency and recency of user interactions to rank items, providing recommendations based on the most interacted-with items.
**Collaborative Filtering**: Utilized user-item interactions to identify similarities between users or items for making predictions.

**Content-Based Filtering**: Focused on the attributes of items and user preferences to recommend similar items.
  
  - **Cosine Similarity:** Utilized to measure the similarity between items or users. This technique was particularly useful in the implementation of a Streamlit app for real-time recommendations.

**Customer Sentiment Analysis**
Sentiment analysis was performed on user reviews to gain deeper insights into user preferences and reactions:

  - **Text Analysi**s: User reviews were analyzed to determine the sentiment polarity (positive, negative, neutral).
  - **Correlation with Ratings**: The sentiment scores were correlated with user ratings to validate the accuracy of the recommendation system.
  - **Improvement Insights**: Sentiment analysis provided actionable insights into product improvement and user satisfaction, allowing for more personalized recommendations.

A peak at the Product Review
![](https://imgur.com/5sYgUTZ.png)

When evaluating the performance of machine learning models, especially recommender systems, several criteria beyond accuracy score are considered to provide a comprehensive assessment. These criteria include:

- **Precision**: Measures the proportion of true positive recommendations among all the recommendations made. It indicates the quality of the recommendations.

- **Recall (Sensitivity)**: Measures the proportion of true positive recommendations identified among all relevant items. It assesses the ability of the model to find all relevant items.

- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two. It is particularly useful when dealing with imbalanced datasets.

- **Mean Absolute Error (MAE)**: Calculates the average absolute difference between predicted and actual ratings, providing an indication of the prediction accuracy.

- **Root Mean Squared Error (RMSE)**: Similar to MAE but penalizes larger errors more significantly. It is the square root of the average of squared differences between predicted and actual ratings.

and more.

The evaluation criteria I'm considering here is the Accuracy Score.

- **Accuracy Score** is a metric used to evaluate the performance of a classification model. It measures the ratio of correctly predicted instances to the total instances in the dataset. In other words, it indicates how often the classifier is correct.


![](https://imgur.com/RZLlAzi.png)

Accuracy Score is: 0.8405

![](https://imgur.com/eA2pwb2.png)

**Cosine Similarity Based Recommendation System**  

A cosine similarity-based recommendation system is a technique used in recommendation systems to suggest items to users based on their similarity to other items or users. It leverages the concept of cosine similarity, which measures the cosine of the angle between two vectors, to determine how similar they are.

![](https://imgur.com/zOY56Qa.png)

This approach is particularly effective in scenarios where the dataset is sparse (i.e., many items/users have missing data) and when the dimensionality of the dataset is high. It's commonly used in various recommendation systems such as movie recommendations, music recommendations, e-commerce product recommendations, etc.

We've used Cosine Similarity in the Streamlit app as the recommendation technique.  

## **Streamlit**

Streamlit is an open-source Python library that simplifies the process of creating and sharing custom web applications for data science and machine learning projects. Streamlit helps render maps, plots, and other visualizations interactively and easily. It allows you to integrate various data visualization libraries and tools to create engaging and informative web applications.

Streamlit turns data scripts into shareable web apps in comparatively less time, so I chose to use Streamlit as the UI for the recommender bot.


## <center>**Eva - the recommender bot!** 

**1. Landing Page**

The landing page has Header, product search bar, footer. Sidebar has the Eva app logo, and dropdown menu. 

![](https://imgur.com/6gJRn3P.png)

**2. Landing Page - Search for recommendations**

You can search the product and Eva will suggest products accordingly. 

![](https://imgur.com/WDwD7oA.png)

**3. SideBar - Top Ranked Products**

You can view the top ranked products for the dataset under consideraton. 

![](https://imgur.com/KO9nEcj.png)

**4. SideBar - Popular Products (Based on Sales)**

You can see the Popular products for the dataset under consideraton. 

![](https://imgur.com/bfuRxha.png)

**5. SideBar - WordCloud for Product Reveiws**

You can see the WordCloud for the dataset under consideraton. 

![](https://imgur.com/QNRU1uG.png)

**6. SideBar - Folium Map**

You can see the location of shoppers across the world. 

![](https://imgur.com/J8ABssl.png) 


**Conclusion**
The capstone project demonstrated the efficacy of various recommendation algorithms using a synthesized retail dataset. KNNBasic emerged as the most accurate model based on RMSE and MSE metrics. The project also highlighted the importance of understanding user preferences through exploratory data analysis and the use of multiple recommendation techniques to improve user engagement and satisfaction on digital platforms.


**Files Submitted for Review**
The Capstone project consists of

- Python Notebook - Recommender_System_using_Streamlit.ipynb
- Dataset file -
  - retail_data.csv - https://drive.google.com/file/d/1bXXuGvYf2v5siYy1cIGeEy-O6WRc4Hwb/view?usp=drive_link

- Python File for Streamlit - eva_recommender.py

- Datasets for Streamlit app

  - retail_data.csv - https://drive.google.com/file/d/1bXXuGvYf2v5siYy1cIGeEy-O6WRc4Hwb/view?usp=drive_link

  - country_counts.csv - https://drive.google.com/file/d/1cWLIl8gsEuw0HqxdXxAddsGsqWijA7rG/view?usp=drive_link
 
**Libraries Used**
Was trained using a Jupyter Notebook and the following Python 3 libraries:

NumPy  
matplotlib  
Plotly
Folim
Seaborn
Scikit-Learn  
Pandas  
NLTK

