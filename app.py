from fastapi import FastAPI

from models import MovieOut, Query
from Recommender import Recommender

recommender = Recommender('SVD++_20_20_0.007_0.02.pkl')

app = FastAPI()


@app.post("/recommend/", response_model=dict[int, MovieOut])
async def recommend_movies(query: Query):

    user_id = query.user_id
    n_recommendations = query.n_recommendations

    response = recommender.recommend(user_id, n_recommendations)

    responses = {}
    for i, (raw_user_id, _) in enumerate(response):
        out_data = recommender.get_movie_metadata(raw_user_id)
        print(out_data)
        responses[i+1] = MovieOut(**out_data)
    
    return responses

