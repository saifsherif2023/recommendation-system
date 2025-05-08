import os
import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from pymongo import MongoClient

def fetch_real_ratings():

    MONGO_URI = os.environ.get("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("Environment variable MONGO_URI is not set.")
    
    client = MongoClient(MONGO_URI)
    db = client["handMade"]
    collection = db["ratings"]

    data = list(collection.find({}, {"user_id": 1, "prod_id": 1, "rating": 1, "_id": 0}))
    return pd.DataFrame(data)

ratings_df = fetch_real_ratings()
print("Fetched ratings:\n", ratings_df.head())

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'prod_id', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

print("\nEvaluating model with 3-fold cross-validation...")
cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

with open("trained_svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as trained_svd_model.pkl")