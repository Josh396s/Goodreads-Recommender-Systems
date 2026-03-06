import gzip
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split

# Helper Functions
def readGz(path):
    """Reads a gzipped file and yields each line as a Python object"""
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    """Reads a gzipped CSV file and yields each line as a list of values"""
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')

def popular_books(data, threshold):
    """Returns a set of popular books that account for at least the given threshold of total reads."""
    totalRead = 0
    bookCount = defaultdict(int)
    for u, b, _ in data:
        bookCount[b] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort(reverse=True)

    pop_books = set()
    count = 0
    for book_count, book in mostPopular:
        count += book_count
        pop_books.add(book)
        if count > (totalRead) * threshold:
            break
    return pop_books

def jaccard(s1, s2):
    """Computes the Jaccard similarity between two sets."""
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom if denom > 0 else 0

def pred_sim(u, b, threshold, ratingsPerUser, ratingsPerItem):
    """Predicts interaction based on Jaccard similarity of item interaction sets."""
    similarities = []
    for b1, _ in ratingsPerUser[u]:
        if b == b1:
            continue
        sim = jaccard(set(u1 for u1, _ in ratingsPerItem[b]), 
                      set(u1 for u1, _ in ratingsPerItem[b1]))
        similarities.append(sim)

    max_similarity = max(similarities) if similarities else 0
    return max_similarity >= threshold

def combine_pred(u, b, pop_books, threshold, pop_w, jac_w, ratingsPerUser, ratingsPerItem):
    """Combines popularity and Jaccard predictions using weighted averaging."""
    pop_pred = 1 if b in pop_books else 0
    jac_pred = 1 if pred_sim(u, b, threshold, ratingsPerUser, ratingsPerItem) else 0

    combined_score = (pop_w * pop_pred + jac_w * jac_pred)
    return 1 if combined_score >= (pop_w + jac_w) / 2 else 0

def main():
    data_path = "data/"
    
    # Load Data
    print("Loading data...")
    allRatings = []
    for l in readCSV(data_path + "train_Interactions.csv.gz"):
        allRatings.append(l)
    
    # Create Validation Set
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))

    # User Book Recommendation (Jaccard + Popularity)
    print("Running User Book Recommendation...")
    sim_threshold = 0.001
    pop_weight = 0.75
    jac_weight = 0.65
    pop_books = popular_books(allRatings, 0.7)

    with open(data_path + "predictions_Read.csv", 'w') as f:
        for l in open(data_path + "pairs_Read.csv"):
            if l.startswith("userID"):
                f.write(l)
                continue
            u, b = l.strip().split(',')
            pred = combine_pred(u, b, pop_books, sim_threshold, pop_weight, jac_weight, ratingsPerUser, ratingsPerItem)
            f.write(f"{u},{b},{pred}\n")

    # User Rating Prediction (SVD)
    print("Running User Rating Prediction (SVD)...")
    df = pd.read_csv(data_path + "train_Interactions.csv.gz", compression='gzip', 
                     header=0, names=['userID', 'itemID', 'rating'])

    # Feature Engineering & Normalization
    df['user_rating_count'] = df.groupby('userID')['rating'].transform('count')
    df['item_rating_count'] = df.groupby('itemID')['rating'].transform('count')
    df['user_avg_rating'] = df.groupby('userID')['rating'].transform('mean')
    df['item_avg_rating'] = df.groupby('itemID')['rating'].transform('mean')

    scaler = MinMaxScaler()
    df[['user_rating_count', 'item_rating_count', 'user_avg_rating', 'item_avg_rating']] = \
        scaler.fit_transform(df[['user_rating_count', 'item_rating_count', 'user_avg_rating', 'item_avg_rating']])

    reader = Reader()
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.15) # Example split

    model = SVD(n_factors=5, lr_all=0.01, reg_all=0.3, n_epochs=20)
    model.fit(trainset)

    with open(data_path + "predictions_Rating.csv", 'w') as f:
        for l in open(data_path + "pairs_Rating.csv"):
            if l.startswith("userID"):
                f.write(l)
                continue
            u, b = l.strip().split(',')
            est_rating = model.predict(u, b).est
            f.write(f"{u},{b},{est_rating}\n")
    
    print("Process complete. Predictions saved to CSV.")

if __name__ == "__main__":
    main()
