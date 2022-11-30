import os
from operator import itemgetter

import pandas as pd
from surprise import AlgoBase, Dataset, Reader
from surprise.accuracy import mae, rmse
from surprise.dump import dump
from surprise.model_selection import train_test_split


class AlgoEvaluator():
    def __init__(self, algo: AlgoBase, algo_name:str , n_factors:int, n_epochs:int, lr_all:float = None, reg_all: float = None, data_file: str = None, loo_train_file: str = None, loo_test_file: str = None, verbose: bool = True) -> None:
        """

        Args:
            algo (AlgoBase): Instance of the algorithm to be evaluated
            algo_name (str): Name of the algorithm
            n_factors (int): Number of factors (passed on to the algorithm)
            n_epochs (int): Number of epochs to train (passed on to the algorithm)
            lr_all (float): Learning rate all (passed on to the algorithm)
            reg_all (float): Regularization all (passed on to the algorithm)
            data_file (str): Data file containing ratings
            loo_train_file (str): Leave One Out Train File
            loo_test_file (str): Leave One Out Test File
            verbose (bool, optional): Show progress or not. Defaults to True.
        """
        print(f"Working for {algo_name}")

        root_wd = os.getcwd()
        self.cwd = os.path.join(root_wd, 'Algorithm')
        self.data_wd = os.path.join(root_wd, 'dataset')
        self.algo = algo
        self.algo_name = algo_name
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.loo_train_file = loo_train_file
        self.loo_test_file = loo_test_file
        self.verbose = verbose
        
        self.data = self._create_surprise_dataset(data_file)
        self.trainset, self.testset, self.loo_trainset, self.loo_testset = self._create_modular_datasets()
        
    def _create_surprise_dataset(self, data_file:str):
        """_summary_

        Args:
            data_file (str): _description_

        Returns:
            _type_: _description_
        """
        if self.verbose: print("Creating surprise dataset")
        reader = Reader(line_format="user item rating", sep="::", rating_scale=(1, 5))
        data = Dataset.load_from_file(data_file, reader=reader)
        if self.verbose: print("Done")
        
        return data

    def _create_modular_datasets(self):

        if self.verbose: print("Seperating dataset")

        # For RMSE and MAE
        trainset, testset = train_test_split(self.data, test_size = 0.2)

        # For Top 1 Hitrate
        loo_trainset_df = pd.read_csv(os.path.join(self.cwd, self.loo_train_file))
        loo_testset_df = pd.read_csv(os.path.join(self.cwd, self.loo_test_file))

        reader = Reader(line_format="user item rating timestamp", sep= "::")
        
        loo_trainset_dataset = Dataset.load_from_df(loo_trainset_df[["user_id", "movie_id", "rating"]], reader)
        loo_trainset = loo_trainset_dataset.build_full_trainset()

        loo_testset = loo_testset_df.to_numpy()

        if self.verbose: print("Done")
        return trainset, testset, loo_trainset, loo_testset

    def _generate_recommendations_one_users(self, algo:AlgoBase, raw_user_id: int, n_recommendations:int):

        watched = []

        inner_user_id = self.loo_trainset.to_inner_uid(raw_user_id)

        for inner_item_id, _ in self.loo_trainset.ur[inner_user_id]:
            watched.append(inner_item_id)

        
        predictions = {}
        for inner_item_id in self.loo_trainset.all_items():
            if inner_item_id not in watched:
                raw_item_id = self.loo_trainset.to_raw_iid(inner_item_id)
                prediction = algo.predict(str(raw_user_id), str(raw_item_id))
                predictions[raw_item_id] = prediction.est

        
        cnt = 0 
        recommendations = []

        for raw_item_id, est_rating in sorted(predictions.items(), key=itemgetter(1), reverse=True):
            if cnt == n_recommendations:
                break
                
            recommendations.append((raw_item_id, est_rating))
            cnt += 1

        return recommendations

    
    def _generate_recommendations_all_users(self, algo: AlgoBase, n_recommendations: int):
        recommendations_all = {}

        ten_percent = self.loo_trainset.n_users // 10

        for inner_user_id in self.loo_trainset.all_users():

            if self.verbose and inner_user_id % ten_percent == 0:
                print(f"Processed {inner_user_id} users.")
            raw_user_id = self.loo_trainset.to_raw_uid(inner_user_id)
            recommendations_all[raw_user_id] = self._generate_recommendations_one_users(algo, raw_user_id, n_recommendations)

        return recommendations_all

    def _calculate_hitrate(self, recommendations_all: dict):
        hits = 0
        total = 0
        for eval_data in self.loo_testset:
            user_id = eval_data[0]
            left_out_movie_id = eval_data[1]

            
            user_top_n_movies = recommendations_all[user_id]
            predicted_movies = [int(movie_id) for (movie_id, _) in user_top_n_movies]
            if int(left_out_movie_id) in predicted_movies: hits +=1
            total += 1


        return hits/total
    
    def evaluate(self):


        # For HitRate
        if self.verbose: print("Training for HitRate")

        model1 = self.algo.fit(self.loo_trainset)
        if self.verbose: print("Done")

        if self.verbose: print("Generating recommendations for HitRate")
        recommendations_all = self._generate_recommendations_all_users(model1, 10)
        if self.verbose: print("Done")

        if self.verbose: print("Calculating HitRate")
        hitrate = self._calculate_hitrate(recommendations_all)
        if self.verbose: print("Done")

        # For RMSE, MAE

        if self.verbose: print("Training for RMSE, MAE")
        model2 = self.algo.fit(self.trainset)
        if self.verbose: print("Done")

        if self.verbose: print("Calculating RMSE, MAE")
        predictions = model2.test(self.testset)
        rmse_score = rmse(predictions, verbose=False)
        mae_score = mae(predictions, verbose=False)
        if self.verbose: print("Done")


        print(f"HitRate: {hitrate}")
        print(f"RMSE: {rmse_score}")
        print(f"MAE: {mae_score}")

        self._save_results_to_csv(hitrate, rmse_score, mae_score)
        self._save_model(model2)

    def _save_results_to_csv(self, hitrate: float, rmse: float, mae:float):
        if self.verbose: print("Saving results to csv")
        result_path = os.path.join(self.cwd, "results.csv")
        if not os.path.exists(result_path):
            df = pd.DataFrame(columns=["Name", "n_factors", "n_epochs", "lr_all", "reg_all","HitRate", "RMSE", "MAE"])
            
        else:
            df = pd.read_csv(result_path)

        if self.lr_all is None:
            self.lr_all = "None"
       
        if self.reg_all is None:
            self.reg_all = "None"
        
        # Add data to the dataframe
        row_dict = {
            "Name": self.algo_name,
            "n_factors": self.n_factors, "n_epochs": self.n_epochs,
            "lr_all": self.lr_all,
            "reg_all": self.reg_all,
            "HitRate": hitrate,
            "RMSE": rmse,
            "MAE": mae
        }
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
        
        df.to_csv(result_path, index=False)
        if self.verbose: print("Done")
    
    def _save_model(self, algo: AlgoBase):
        if self.verbose: print("Saving model")
        model_path = os.path.join(self.cwd, "models")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_file_name = f"{self.algo_name}_{self.n_factors}_{self.n_epochs}_{self.lr_all}_{self.reg_all}.pkl"
        dump(file_name=os.path.join(model_path, model_file_name), algo=algo)
        if self.verbose: print("Done")

if __name__ == "__main__":
    print(os.getcwd())
    from surprise import NMF, SVD, SVDpp

    svd = SVD()
    svdpp = SVDpp()
    nmf = NMF()

    model_config = {
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
    }
    dataset_config = {
        "data_file": "dataset/ratings.dat",
        "loo_train_file": "oneleftout.csv",
        "loo_test_file": "test_set.csv",
        "verbose": True
    }

    svd_eval = AlgoEvaluator(svd, "SVD", **model_config, **dataset_config)
    svd_eval.evaluate()

    # svdpp_eval = AlgoEvaluator(svdpp, "SVD++", **dataset_config)
    # svdpp_eval.evaluate()

    # nmf_eval = AlgoEvaluator(nmf, "NMF", **dataset_config)
    # nmf_eval.evaluate()

