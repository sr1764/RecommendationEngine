from flask import Flask, jsonify, request, render_template
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import pymysql
from sqlalchemy import create_engine

app = Flask(__name__)

# Replace 'C://Users//patha//archive//index.csv' with your actual CSV file path
# Load data into pandas DataFrame
data = pd.read_csv('C://Users//patha//archive//fashion_products.csv')

# Display basic info about the dataset
print(data.head())
print(data.info())

# MySQL connection details
DB_USERNAME = 'root'
DB_PASSWORD = 'rootpass123'
DB_HOST = 'localhost'
DB_NAME = 'rec_engine'

# Create MySQL connection
engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')

# Store DataFrame in MySQL
data.to_sql('retail_data', con=engine, if_exists='replace', index=False)

data.dropna(inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Example data transformation steps
# Convert data types
data['User ID'] = data['User ID'].astype(str)
data['Product ID'] = data['Product ID'].astype(str)

# Create new features if necessary
# Example: Convert price to categorical
data['Price'] = pd.cut(data['Price'], bins=[0, 50, 100, 200, float('inf')], labels=['0-50', '51-100', '101-200', '200+'])

top_10_items = data['Product ID'].value_counts().head(10)
sns.barplot(x=top_10_items.index, y=top_10_items.values)
plt.title('Top 10 Most Popular Items')
plt.xlabel('User ID')
plt.ylabel('Number of Interactions')
plt.xticks(rotation=45)
plt.show()

# User behavior patterns
user_interaction_counts = data['User ID'].value_counts()
sns.histplot(user_interaction_counts, bins=50)
plt.title('Distribution of User Interactions')
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Users')
plt.show()


top_10_items = data['Product ID'].value_counts().head(10)
sns.barplot(x=top_10_items.index, y=top_10_items.values)
plt.title('Top 10 Most Popular Items')
plt.xlabel('Product ID')
plt.ylabel('Number of Interactions')
plt.xticks(rotation=45)
plt.show()

# User behavior patterns
user_interaction_counts = data['User ID'].value_counts()
sns.histplot(user_interaction_counts, bins=50)
plt.title('Distribution of User Interactions')
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Users')
plt.show()

# Load Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(data[['User ID', 'Product ID', 'Rating']], reader)

# Build and train SVD algorithm
trainset = data_surprise.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Function to get top-N recommendations for a user
def get_top_n_recommendations(user_id, n=10):
    # Get a list of all product IDs
    all_product_ids = set(data['Product ID'].unique())
    
    # Get product IDs the user has already interacted with
    interacted_product_ids = set(data[data['User ID'] == user_id]['Product ID'])
    
    # Products that the user has not interacted with
    products_to_recommend = list(all_product_ids - interacted_product_ids)
    
    # Predict ratings for all products that the user has not interacted with
    predictions = [algo.predict(user_id, prod_id) for prod_id in products_to_recommend]
    
    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top-N recommendations
    top_n_recommendations = predictions[:n]
    
    return top_n_recommendations

# Example endpoint for real-time recommendations
@app.route('/recommendations', methods=['GET'])
def recommend_products():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID parameter is required'}), 400
    
    try:
        user_id = str(int(user_id))  # Convert user_id to string if not already
    
        # Get top-N recommendations for the user
        recommendations = get_top_n_recommendations(user_id)
        
        recommended_products = []
        # Format recommendations as JSON response
        for rec in recommendations:
            product_details = data[data['Product ID'] == rec.iid].iloc[0]  # Fetch product details
            recommendation = {
                'user_id': rec.uid,
                'product_id': rec.iid,
                'rating': rec.est,
                'product_name': product_details['Product Name'],
                'brand': product_details['Brand'],
                'category': product_details['Category'],
                'color': product_details['Color'],
                'size': product_details['Size']
            }
            recommended_products.append(recommendation)
        
        return jsonify(recommended_products)
    
    except ValueError as ve:
        return jsonify({'error': 'Invalid user_id format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
