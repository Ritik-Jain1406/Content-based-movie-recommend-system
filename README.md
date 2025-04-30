# Movie Recommendation System

This is a content-based movie recommendation system built using Python, Pandas, and Scikit-Learn. It suggests similar movies based on the content (overview, cast, genres, keywords, and crew) of the selected movie using Natural Language Processing techniques.

## Features

- Recommends top 5 similar movies based on the selected movie
- Uses TF-IDF vectorization and cosine similarity for comparison
- Cleaned and processed data using pandas and `ast` module
- Fuzzy matching support for better search
- Simple web interface using Streamlit

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit (for frontend)
- Natural Language Processing (TF-IDF)
- Cosine Similarity
- FuzzyWuzzy (optional for better search handling)

## How It Works

1. The system loads movie and credits datasets.
2. Merges important features such as overview, genres, keywords, cast, and crew.
3. Cleans the data and combines it into a single `tags` column.
4. Vectorizes the `tags` using `TfidfVectorizer`.
5. Computes cosine similarity between movies.
6. Given a movie name, returns top 5 most similar movies.

## Setup Instructions

1. Clone the repository
2. Install dependencies:
