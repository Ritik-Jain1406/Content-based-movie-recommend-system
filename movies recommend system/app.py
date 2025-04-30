import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function
from fuzzywuzzy import process

def recommend(movie):
    movie = movie.lower()
    titles = movies['title'].str.lower().tolist()

    closest_match = process.extractOne(movie, titles)

    if closest_match[1] < 60:  # similarity threshold
        return []

    index = movies[movies['title'].str.lower() == closest_match[0]].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    return [movies.iloc[i[0]].title for i in movie_list]



# Streamlit UI
st.title('Movie Recommender System')

movie_name = st.text_input("Enter movie name")

if st.button('Recommend'):
    recommendations = recommend(movie_name)
    if recommendations:
        st.write("Top 5 Recommendations:")
        for name in recommendations:
            st.write(name)
    else:
        st.write("Movie not found.")
