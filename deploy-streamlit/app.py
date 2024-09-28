import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('bert-base-nli-mean-tokens')  # Replace with your model

# Load the DataFrame data
df = pd.read_csv('./deploy-streamlit/Xtest.csv')

# Title of the application
st.title("Movie Recommendation System")

# User input for the synopsis
new_synopsis = st.text_area("Enter the new movie synopsis:", "american car designer carroll shelby driver")

if st.button("Recommend Movies"):
    # Encode the new synopsis
    new_embedding = model.encode(new_synopsis)

    # Encode existing synopses
    df['embeddings'] = df['synopsis'].apply(lambda x: model.encode(x))
    # Convert existing embeddings to a NumPy matrix
    Xtest = np.vstack(df['embeddings'].values)  # Ensure df['embeddings'] is a list of arrays

    # Calculate cosine similarity
    similarities = cosine_similarity([new_embedding], Xtest)[0]  # Returns an array of similarities

    # Get the index of the most similar movie
    most_similar_index = similarities.argmax()  # Index of the highest similarity value

    # Get the cluster of the most similar movie
    target_cluster = df.iloc[most_similar_index]['cluster']

    # Filter movies that belong to the same cluster
    cluster_movies = df[df['cluster'] == target_cluster]

    if not cluster_movies.empty:  # Check if there are movies in the cluster
        # Calculate cosine similarity only for movies in the same cluster
        cluster_embeddings = np.vstack(cluster_movies['embeddings'].values)
        cluster_similarities = cosine_similarity([new_embedding], cluster_embeddings)[0]

        # Get the indices of the 5 most similar movies in the cluster
        top_indices_in_cluster = cluster_similarities.argsort()[-5:][::-1]  # Indices of the top 5 highest values, in descending order

        # Get the recommended movies
        recommended_movies = cluster_movies.iloc[top_indices_in_cluster]

        # Display results
        st.subheader("Recommended Movies")
        for _, row in recommended_movies.iterrows():
            st.write(f"**{row['title_en']}** - {row['synopsis']}")
    else:
        st.warning("There are no movies in the same cluster.")