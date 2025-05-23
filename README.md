## Demo 
[Check out the Content-Based Movie Recommendation System](https://content-based-movie-recommend-system-ijcvulpvjgix5etr5kwwuw.streamlit.app/)


## Sample Output
![Screenshot_2025_0502_113238](https://github.com/user-attachments/assets/036b1a4e-0b44-4273-9a91-31b2df866597)
![Screenshot_2025_0502_113329](https://github.com/user-attachments/assets/ad1751c6-67d5-458c-8e62-00d964504e86)

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


## Alert
🔴 If you got page shows "there is no activity for longer time" then it shows a `backup` button just click on it and website's start again.

