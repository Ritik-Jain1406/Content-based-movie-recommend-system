import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Data Loading
movies = pd.read_csv(r'C:\Users\Hp\Downloads\archive (1)\tmdb_5000_credits.csv')
credits = pd.read_csv(r'C:\Users\Hp\Downloads\archive (1)\tmdb_5000_movies.csv')
# Step 2: Merge
movies = movies.merge(credits, on='title')

# Step 3: Select Important Columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Step 4: Functions to clean data
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

# Step 5: Apply cleaning functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
# Step 6: Create 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Step 7: Create new dataframe
new_df = movies[['movie_id', 'title', 'tags']]

# Step 8: Join list into string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Step 9: Vectorize the text
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Step 10: Calculate Similarity
similarity = cosine_similarity(vectors)

# Step 11: Recommendation Function
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return []
    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Example usage
recommend('iron man')
import pickle

# Save the cleaned dataframe and similarity matrix
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
