from pydantic import BaseModel


class Query(BaseModel):
    user_id: int = 1
    n_recommendations: int = 10

class MovieOut(BaseModel):
    movie_id: int
    title: str
    genre: list[str]
