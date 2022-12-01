import os
from operator import itemgetter

import pandas as pd
from surprise import AlgoBase, Dataset, Reader
from surprise.dump import load


class Recommender():

    def __init__(self, model_name:str) -> None:
        """

        Args:
            model_name (str): Name of the model to be used for inference
        """
        self.model_name = model_name
        self.algo = self._load_model()
        self.trainset = self._load_trainset()
        self.movie_df = self._load_movie_df()

    def _load_trainset(self) -> Dataset:
        """Loads dataset used for training to get recommendations

        Returns:
            Dataset: dataset object used for training
        """
        loo_trainset_df = pd.read_csv(os.path.join(os.getcwd(), 'Algorithm', 'oneleftout.csv'))

        reader = Reader(line_format="user item rating timestamp", sep= ",")
        
        loo_trainset_dataset = Dataset.load_from_df(loo_trainset_df[["user_id", "movie_id", "rating"]], reader)
        loo_trainset = loo_trainset_dataset.build_full_trainset()

        return loo_trainset

    def _load_movie_df(self) -> pd.DataFrame:
        """Loads the movie dataframe to get movie metadata

        Returns:
            pd.DataFrame: Dataframe of the movie dataset
        """
        movie_path = os.path.join(os.getcwd(), 'dataset', 'movies.dat')
        movie_df = pd.read_csv(movie_path, sep='::', names=['movie_id', 'title', 'genre'], encoding='ISO-8859-1', engine="python")
        return movie_df

    def _load_model(self) -> AlgoBase:
        """Loads the model into memory

        Returns:
            AlgoBase: Instance of the model
        """
        algo_path = os.path.join(os.getcwd(), 'Algorithm', 'models', self.model_name)
        _, algo = load(algo_path)
        return algo

    def get_movie_metadata(self, movie_id: int) -> dict:
        """Gets movie metedata from the movie dataframe

        Args:
            movie_id (int): Movie id to get metadata for

        Returns:
            dict: Dict containing movie metadata
        """
        mov_data = self.movie_df[self.movie_df['movie_id'] == movie_id]
        dict_ = {
            "movie_id": movie_id,
            "title": mov_data['title'].values[0],
            "genre": mov_data['genre'].values[0].split("|")

        }
        return dict_

    def recommend(self, raw_user_id:int, n_recommendations:int) -> list(tuple((int, float))):
        """Gets movie recommendations for a user

        Args:
            raw_user_id (int): User ID to get recommendations for
            n_recommendations (int): Number of recommendations to get

        Returns:
            list(tuple(int,float)): list of tuples containing movie id and estimated rating
        """

        watched = []

        inner_user_id = self.trainset.to_inner_uid(raw_user_id)

        for inner_item_id, _ in self.trainset.ur[inner_user_id]:
            watched.append(inner_item_id)

        
        predictions = {}
        for inner_item_id in self.trainset.all_items():
            if inner_item_id not in watched:
                raw_item_id = self.trainset.to_raw_iid(inner_item_id)
                prediction = self.algo.predict(str(raw_user_id), str(raw_item_id))
                predictions[raw_item_id] = prediction.est

        
        cnt = 0 
        recommendations = []

        for raw_item_id, est_rating in sorted(predictions.items(), key=itemgetter(1), reverse=True):
            if cnt == n_recommendations:
                break
                
            recommendations.append((raw_item_id, est_rating))
            cnt += 1

        return recommendations

if __name__ == "__main__":
    r = Recommender('SVD++_20_20_0.007_0.02.pkl')
    print(r.recommend(1, 10))