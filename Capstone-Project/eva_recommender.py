'''
Author: Deepa Sarojam
Date: May 22, 2024

'''

import streamlit as st
import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(layout="centered")
                   
def add_logo():
    # Display the logo image in the sidebar
    # st.sidebar.image("https://imgur.com/43jcSya.png", width=180
    st.logo(
        "https://imgur.com/43jcSya.png"
    )
add_logo()

# Function to set background image and font color
def sidebar_css():
    '''
    A function to unpack an image from url and set as bg.
    '''
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://imgur.com/3Z0VYOW.jpg");
            background-size: cover;
            color: white;
        }
        .stTextInput > div > div {
            background: black;
            color: white;
        }
        .stTextInput > div > div > input {
            background: black;
            color: white;
        }
        div.st-emotion-cache-6qob1r.eczjsme3 {
            color: white;
            background-color: #005f73;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background
sidebar_css()

def about():
    st.markdown('**About**')
    st.markdown('''
                Eva uses 
                ''')
    about()

# Define the footer content
def footer():
    st.divider()
    st.write("""Created by Deepa Sarojam
             | Follow me on [LinkedIn](https://www.linkedin.com/in/deepa-sarojam/)
             | [Medium](https://medium.com/@deepa-sarojam)
             | [GitHub](https://github.com/deepasarojam-2024)
             """)
    st.markdown('''*Eva, a Product Recommendation Bot, was developed exclusively for educational purposes as a part of the 
                Capstone project for the UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence, May 2024. 
                Your feedback is greatly appreciated!*
                ''')


# Setup NLTK data path
nltk_data_path = nltk.data.path[0]

# Download NLTK data if not already downloaded
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_path)

# Load the dataset
products_df = pd.read_csv("https://github.com/deepasarojam-2024/UC-Berkeley-ML-AI/blob/main/Capstone-Project/retail_data.csv")   
country_counts_df = pd.read_csv("https://github.com/deepasarojam-2024/UC-Berkeley-ML-AI/blob/main/Capstone-Project/country_counts.csv")

# Preprocessing setup
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Tokenizer, stemmer, and lemmatizer function
def tokenize_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Apply tokenization to the dataset
products_df['lemma_tokens'] = products_df.apply(lambda row: tokenize_text(f"{row['Product Name']} {row['Product Category']}"), axis=1)

# TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(products_df['lemma_tokens'])

# Recommend function
def recommend_products(query):
    query_vec = tfidf.transform([tokenize_text(query)])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Set a threshold to filter out irrelevant recommendations
    threshold = 0.1  # Adjust this value as needed
    products_df['similarity'] = similarity
    filtered_products = products_df[products_df['similarity'] > threshold]

    if filtered_products.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no products meet the threshold

    top_products = filtered_products.nlargest(25, 'similarity')[['Product Name', 'Product Category']]
    unique_products = top_products.drop_duplicates(subset='Product Name')

    return unique_products[['Product Name', 'Product Category']]

# Define the get_top_ranked_products function
def get_top_ranked_products(products_df):
    # Rank the items based on rating (higher rating gets a higher rank)
    products_df['Rank'] = products_df['Rating'].rank(ascending=False)

    # Sort the DataFrame by rank
    rank_df = products_df.sort_values(by='Rank')

    # Reset the index and drop the old index column
    rank_df.reset_index(drop=True, inplace=True)

    # Display top 10 ranked products without the index column
    top_10_ranked = rank_df[['Product Name', 'Product Category']].head(10)
    
    return top_10_ranked

def display_top_10_popular(df):
    # Sort products by Sales Revenue (Popularity)
    popular_products = df.sort_values(by='Sales Revenue', ascending=False)

     # Reset the index and drop the old index column
    popular_products.reset_index(drop=True, inplace=True)
    
    # Display top 10 popular products
    top_10_popular = popular_products.head(10)[['Product ID', 'Product Name']]
    return top_10_popular

def generate_wordcloud(cloud):

    # Combine all reviews into a single string
    text = ' '.join(cloud['Product Review'].tolist())

    # Remove special characters and numbers using regular expressions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers

    # Tokenize the text into individual words
    words = text.split()

    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stopwords]

    # Join the remaining words back into a single string
    filtered_text = ' '.join(filtered_words)

    # Create the WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

    return wordcloud

# Plotly map function 
def plotly_map():

    # Initialize map
    m = folium.Map(location=[30, 0], zoom_start=2, width='100%', height='100%')

    # Initialize MarkerCluster
    marker_cluster = MarkerCluster().add_to(m)

    # Define colormap
    colormap = linear.YlGnBu_09.scale(country_counts_df['Count'].min(), country_counts_df['Count'].max())

    # Iterate over each row in the DataFrame
    for index, row in country_counts_df.iterrows():
        # Extract country and count from the row
        country = row['Country']
        count = row['Count']
        
        # Get marker color based on count
        colormap(count)
        
        # Construct label
        label = 'Country: {}<br>Count: {}'.format(country, count)
        
        # Add marker to MarkerCluster with shopping icon and colored based on count
        folium.Marker(location=[row['Latitude'], row['Longitude']], popup=label, icon=folium.Icon(color='blue', icon_color='white', icon='shopping-cart', angle=0, prefix='fa')).add_to(marker_cluster)
        
    # Add colormap to the map
    colormap.caption = 'Count of Shopping'
    m.add_child(colormap)
    
    return m

def padding(lines):
    for _ in range(lines):
        st.write('&nbsp;')

# Sidebar with dropdown list
option = st.sidebar.selectbox(
    'Select an option:',
    ('Cosine Similarity Based Recommendation System', 'Top Ranked Products', 'Popular Products', 'WordCloud', 'Folium Map')
)

# Main content
if option == 'Cosine Similarity Based Recommendation System':
    st.markdown('<h1 style="text-align: center; background-color: white; color: black;"> Recommender Bot</h1>', unsafe_allow_html=True)

    def product_recommendation_chatbot():
        padding(1)
        user_input = st.text_input("💬 👩 Type 'products' to view your personalized recommendations: ", "")
        if st.button('🔍 Recommend'):
            if user_input:
                recommendations = recommend_products(user_input)
                if recommendations.empty:
                    st.markdown('<div style="background-color: white; color: black;">🤖 Sorry, I couldn\'t find any matching products.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background-color: white; color: black;">🤖 Here are some products you might like: </div><br>', unsafe_allow_html=True)
                    for idx, row in recommendations.iterrows():
                        st.markdown(f'<div style="background-color: white; color: black;">&#8226; {row["Product Name"]} ({row["Product Category"]})</div>', unsafe_allow_html=True)

    product_recommendation_chatbot()
    # Display the footer
    footer()
    
elif option == 'Top Ranked Products':
    st.markdown('<h1 style="text-align: center; background-color: white; color: black;"> Top Ranked Products</h1>', unsafe_allow_html=True)
    top_ranked_products = get_top_ranked_products(products_df)
    st.table(top_ranked_products)

elif option == 'Popular Products':
    st.markdown('<h1 style="text-align: center; background-color: white; color: black;"> Popular Products (Based on Sales)</h1>', unsafe_allow_html=True)
    top_ranked_products = display_top_10_popular(products_df)
    st.table(top_ranked_products)

elif option == 'WordCloud':
    # Define the header HTML
    st.markdown('<h1 style="text-align: center; background-color: white; color: black;">WordCloud for Product Reviews</h1>', unsafe_allow_html=True)
    stopwords = set(STOPWORDS)
    # Filter the DataFrame for reviews with a rating greater than the threshold
    rating_threshold = 4
    cloud = products_df[products_df['Rating'] > rating_threshold]
    
    # Ensure all reviews are strings
    cloud['Product Review'] = cloud['Product Review'].astype(str)
    
    wordcloud = generate_wordcloud(cloud)
    
    # Display the word cloud
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    # ax.set_title('Word Cloud for Product Reviews')
    ax.axis('off')  # Hide the axes
    st.pyplot(fig)
    
elif option == 'Folium Map':
    # Define the header HTML
    st.markdown('<h1 style="text-align: center; background-color: white; color: black;">We\'re across the World!</h1>', unsafe_allow_html=True)

    # Display the map
    folium_static(plotly_map())
