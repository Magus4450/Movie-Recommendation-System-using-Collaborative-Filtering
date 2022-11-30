from AlgoEvaluator import AlgoEvaluator
from DataSeperator import DataSeperator

if __name__ == "__main__":

    

    # data_seperator = DataSeperator("dataset/ratings.dat", "oneleftout.csv", "test_set.csv")
    # data_seperator.seperate_data()
    from surprise import NMF, SVD, SVDpp

    base_svd_config = {
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
    }
    base_svdpp_config = {
        "n_factors": 20,
        "n_epochs": 20,
        "lr_all": 0.007,
        "reg_all": 0.02,
    }
    base_nmf_config = {
        "n_factors": 15,
        "n_epochs": 50,
    }

    dataset_config = {
        "data_file": "dataset/ratings.dat",
        "loo_train_file": "oneleftout.csv",
        "loo_test_file": "test_set.csv",
        "verbose": True
    }


    svd = SVD(**base_svd_config)
    svdpp = SVDpp(**base_svdpp_config)
    nmf = NMF(**base_nmf_config)

    svd_eval = AlgoEvaluator(svd, "SVD", **base_svd_config, **dataset_config)
    svd_eval.evaluate()

    svdpp_eval = AlgoEvaluator(svdpp, "SVD++", **base_svdpp_config, **dataset_config)
    svdpp_eval.evaluate()

    nmf_eval = AlgoEvaluator(nmf, "NMF", **base_nmf_config, **dataset_config)
    nmf_eval.evaluate()

    base_svd_config["n_factors"] = 50
    base_svd_config["n_epochs"] = 50

    svd1 = SVD(**base_svd_config)
    svd_eval1 = AlgoEvaluator(svd1, "SVD", **base_svd_config, **dataset_config)
    svd_eval1.evaluate()

    base_svdpp_config["n_factors"] = 10
    base_svdpp_config["n_epochs"] = 40

    svdpp1 = SVDpp(**base_svdpp_config)
    svdpp_eval1 = AlgoEvaluator(svdpp1, "SVD++", **base_svdpp_config, **dataset_config)
    svdpp_eval1.evaluate()

    base_nmf_config["n_factors"] = 10

    nmf1 = NMF(**base_nmf_config)
    nmf_eval1 = AlgoEvaluator(nmf1, "NMF", **base_nmf_config, **dataset_config)
    nmf_eval1.evaluate()
    
