import os
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from flask import Flask, request, jsonify
from surprise import SVD, Dataset, Reader

app = Flask(__name__)

df_products = pd.read_csv("egyptian_handmade_crafts.csv")

model_path = "trained_svd_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Run /train endpoint first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

def get_popular_products(n=5, min_ratings=5):
    popular_df = df_products[df_products['rating_count'] >= min_ratings]
    top = popular_df.sort_values(by='avg_rating', ascending=False).head(n)[
        ['prod_id', 'prod_name', 'price', 'avg_rating']
    ]
    return top.to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
    
    if str(user_id) not in model.trainset._raw2inner_id_users:
        return jsonify({
            "message": f"User {user_id} is new. Showing popular items instead.",
            "recommendations": get_popular_products()
            })
    
    product_ids = df_products['prod_id'].unique()
    predictions = [(pid, model.predict(str(user_id), str(pid)).est) for pid in product_ids]
    top = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    recs = df_products[df_products['prod_id'].isin([pid for pid, _ in top])][
        ['prod_id', 'prod_name', 'price', 'avg_rating']
    ].drop_duplicates('prod_id')
    return jsonify(recs.to_dict(orient='records'))

@app.route('/popular', methods=['GET'])
def popular():
    return jsonify(get_popular_products())

@app.route('/train', methods=['POST'])
def train():
    try:
        MONGO_URI = os.environ.get("MONGO_URI")
        client = MongoClient(MONGO_URI)
        db = client["handMade"]
        collection = db["ratings"]

        data = list(collection.find({}, {"user_id": 1, "prod_id": 1, "rating": 1, "_id": 0}))
        ratings_df = pd.DataFrame(data)

        if ratings_df.empty:
            return jsonify({"error": "No data found in MongoDB to train on."}), 400

        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(ratings_df[['user_id', 'prod_id', 'rating']], reader)
        trainset = dataset.build_full_trainset()

        global model
        model = SVD()
        model.fit(trainset)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return jsonify({"message": "Model successfully re-trained and updated."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)