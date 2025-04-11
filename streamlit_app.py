import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
@st.cache
def load_data():
    df = pd.read_csv('lastfm_data.csv')  # Замени на свой путь к файлу
    return df

df = load_data()

# Предобработка данных
def preprocess_data(df):
    # Убедимся, что все песни имеют уникальное имя
    df = df.dropna(subset=['song'])
    return df

df = preprocess_data(df)

# Рекомендации на основе похожести песен
def get_similar_songs(song_name, n=10):
    song_index = df[df['song'] == song_name].index[0]
    song_features = df.iloc[:, 2:].values  # Допустим, после второго столбца идут признаки
    similarity_scores = cosine_similarity([song_features[song_index]], song_features)
    similar_songs = similarity_scores.argsort()[0][::-1][1:n+1]
    return df.iloc[similar_songs]['song'].tolist()

# Интерфейс Streamlit
st.title("🎶 Music Recommendation System")

song_name = st.text_input("Enter a song name:")
if st.button("Recommend"):
    if song_name:
        recommendations = get_similar_songs(song_name)
        st.write(f"Recommendations based on '{song_name}':")
        for song in recommendations:
            st.write(song)
    else:
        st.write("Please enter a song name.")
