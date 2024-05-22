Machine Learning Capstone
 

## **Key Findings/Results**

**Dataset**

The dataset for this project was synthesized using Python Faker by taking inspiration from publicly available datasets such as the H&M dataset from Kaggle and the ShopperSentiments dataset. It includes various attributes like Transaction ID, Date, Product ID, Product Name, Product Description, Product Category, Sub Category, Color, Rating, Product Review, Quantity, Customer ID, Customer Name, Age, Price, Currency, Discounts, Sales Revenue, Payment Method, Store Type, Season, Latitude, Longitude, and Country. 

**Exploratory Data Analysis (EDA)** 

- **Frequency of Purchases**: The analysis revealed that a significant proportion of users made frequent purchases in specific product categories. For instance, categories like "Womens Everyday Collection" and "Divided Collection" saw higher user interaction compared to others.

- **Seasonal Trends**: There were clear seasonal patterns in user purchases, with spikes observed during certain times of the year, such as holiday seasons and sales events. This indicated that users were more active and engaged during these periods.

- **Top Product Categories**: The EDA identified the top 30 product categories based on the number of occurrences in the dataset. The "Womens Everyday Collection" was the most popular, followed by the "Divided Collection" and "Womens Swimwear, beachwear".

- **High-Rated Products**: Some products consistently received higher ratings, suggesting their popularity and user satisfaction. Products with high ratings were often from well-known brands or belonged to trending categories.

- **Global Reach**: The dataset showed a wide geographic distribution of shoppers, indicating that the user base was diverse and spread across various regions. This was visualized using interactive maps that highlighted the concentration of purchases in different areas.

- **Rating Patterns:** The distribution of ratings indicated that most users tended to give higher ratings (4 or 5 stars), suggesting overall satisfaction with their purchases. There were fewer low ratings (1 or 2 stars), which could indicate a generally positive user experience.

- **Skewness and Kurtosis:** The ratings distribution exhibited positive skewness, with a peak towards the higher end of the rating scale. The kurtosis value suggested that the distribution had lighter tails, meaning fewer extreme ratings.

- **Average Transaction Value**: The EDA revealed the average transaction value across different categories. High-value transactions were more common in categories like electronics and luxury items, while everyday items had lower average transaction values.

- **Discount Impact:** Transactions with applied discounts or promotions saw a higher frequency, indicating that discounts were a significant motivator for purchases. This insight could be used to strategize marketing and promotional campaigns.

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


Among these, KNNBasic showed the lowest RMSE and MSE, indicating its superior predictive accuracy​​.

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
  

**Conclusion**
The capstone project demonstrated the efficacy of various recommendation algorithms using a synthesized retail dataset. KNNBasic emerged as the most accurate model based on RMSE and MSE metrics. The project also highlighted the importance of understanding user preferences through exploratory data analysis and the use of multiple recommendation techniques to improve user engagement and satisfaction on digital platforms.


**Files Submitted for Review**
The Capstone Project proposal proposal.pdf.
The Capstone Project Report capstone_project_report.pdf.
The implemented code, in the form of an ipython notebook Capstone Project.ipynb
The datasets used for the project  
 
**Libraries Used**
Was trained using a Jupyter Notebook and the following Python 3 libraries:

NumPy  
MatPlotLib  
Plotly
Folim
Seaborn
Scikit-Learn  
Pandas  
NLTK

