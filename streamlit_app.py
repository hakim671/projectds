import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview']].dropna()
    return df

# Построение матрицы TF-IDF и косинусного сходства
@st.cache_resource
def create_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Рекомендательная функция
def recommend(title, data, cosine_sim):
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return ["Фильм не найден в базе. Попробуйте другой."]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices].tolist()

# Streamlit интерфейс
def main():
    st.set_page_config(page_title="🎬 Рекомендательная система фильмов", layout="centered")
    st.title("🎥 Рекомендательная система фильмов")
    
    data = load_data()
    cosine_sim = create_similarity_matrix(data)
    
    selected_movie = st.selectbox("Выберите фильм:", data['title'].sort_values())
    
    if st.button("Показать рекомендации"):
        recommendations = recommend(selected_movie, data, cosine_sim)
        st.subheader("🎯 Рекомендуемые фильмы:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

if __name__ == '__main__':
    main()
