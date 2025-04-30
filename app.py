import zipfile
import os
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from fuzzywuzzy import process

# Extract ZIP if not already extracted
if not os.path.exists("tmdb_5000_credits.csv") or not os.path.exists("tmdb_5000_movies.csv"):
    with zipfile.ZipFile("tmdb_5000_credits.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge and preprocess
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    titles = new_df['title'].str.lower().tolist()
    match = process.extractOne(movie, titles)
    if match[1] < 60:
        return []
    index = new_df[new_df['title'].str.lower() == match[0]].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movie_list]

# Streamlit UI
st.title("Movie Recommendation System")
user_input = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if user_input:
        recommendations = recommend(user_input)
        if recommendations:
            st.write("Top 5 Recommendations:")
            for movie in recommendations:
                st.write(movie)
        else:
            st.write("Movie not found. Please try another.")