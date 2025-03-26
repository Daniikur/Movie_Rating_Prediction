import numpy as np
import pickle
import streamlit as st

xgb_model = pickle.load(open("movie_rating_model.pkl", "rb"))
sentence_model = pickle.load(open("sentence_model.pkl", "rb"))
genres_columns = pickle.load(open("genre_columns.pkl", "rb"))
avg_actor_rating = pickle.load(open("avg_actor_rating.pkl", "rb"))
avg_director_rating = pickle.load(open("avg_director_rating.pkl", "rb"))

st.title("Movie Rating Predictor")
st.markdown("Enter movie details below to predict the rating!")

actors = st.text_input("Enter Actors ")
directors = st.text_input("Enter Director")
genres = st.text_input("Enter Genres ")
synopsis = st.text_area("Enter Movie Synopsis")

if st.button("Predict Rating"):
    with st.spinner("Predicting..."):
        try:

            input_actor_avg = np.mean([avg_actor_rating.get(actor.strip(), np.nan) for actor in actors.split(",")])
            input_director_avg = np.mean([avg_director_rating.get(director.strip(), np.nan) for director in directors.split(",")])

            overall_mean_rating = np.mean(list(avg_actor_rating.values()) + list(avg_director_rating.values()))
            input_actor_avg = input_actor_avg if not np.isnan(input_actor_avg) else overall_mean_rating
            input_director_avg = input_director_avg if not np.isnan(input_director_avg) else overall_mean_rating

            synopsis_embedding = sentence_model.encode([synopsis])

            input_genres = genres.split(", ")
            genres_vector = np.zeros((1, len(genres_columns)))
            for genre in input_genres:
                if genre in genres_columns:
                    genres_vector[0, genres_columns.index(genre)] = 1

            features = np.hstack((synopsis_embedding, genres_vector, [[input_actor_avg, input_director_avg]]))

            predicted_rating = xgb_model.predict(features)[0]
            st.success(f"Predicted Rating: {predicted_rating:.2f} / 10")
        except Exception as e:
            st.error(f"Error: {str(e)}")