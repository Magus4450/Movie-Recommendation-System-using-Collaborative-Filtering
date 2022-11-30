import os

import pandas as pd


class DataSeperator():

    def __init__(self, data_path: str, output_train_file: str, output_test_file: str, verbose_:bool = True) -> None:
        self.verbose_ = verbose_
        self.df = self._load_dataframe(data_path)
        self._sort_dataframe()
        self.output_train_file = output_train_file
        self.output_test_file = output_test_file
        self.cwd = os.path.join(os.getcwd(), 'Algorithm')

    

    def _load_dataframe(self, data_path: str) -> pd.DataFrame:
        if self.verbose_: print("Loading dataframe")
        return pd.read_csv(data_path, 
        sep="::", names=["user_id", "movie_id", "rating", "timestamp"])

    def _sort_dataframe(self):
        if self.verbose_: print("Sorting dataframe")
        self.df.sort_values(['user_id', 'rating'], ascending=[True, False], inplace=True)
        if self.verbose_: print("Done")

    
    def seperate_data(self):
        if self.verbose_: print("Seperating data")

        in_test_set = []
        test_set = []

        len_df = len(self.df)
        # Iterate over dataframe
        for index, rows  in self.df.iterrows():

            if self.verbose_ and index % 100000 == 0:
                print(f"Seperation progress: {index}/{len_df}")

            user_id = rows.user_id
            movie_id = rows.movie_id
            rating = rows.rating

            if user_id not in in_test_set:
                in_test_set.append(user_id)
                test_set.append((user_id, movie_id, rating))
            
                drop_index = self.df[((self.df["user_id"] == user_id) & (self.df["movie_id"] == movie_id))].index
                self.df.drop(drop_index, inplace=True)

        if self.verbose_: print("Done")



        if self.verbose_: print("Saving data")
        self.df.to_csv(os.path.join(self.cwd, self.output_train_file), index=False)
        pd.DataFrame(test_set, columns=["user_id", "movie_id", "rating"]).to_csv(os.path.join(self.cwd, self.output_test_file), index=False)
        if self.verbose_: print("Done")


if __name__ == "__main__":
    data_seperator = DataSeperator("dataset/ratings.dat", "oneleftout.csv", "test_set.csv")
    data_seperator.seperate_data()

            
    

