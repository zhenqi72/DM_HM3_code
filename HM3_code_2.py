from surprise import Dataset,KNNBasic,Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD


reader = Reader(line_format='userId movieId rating timestamp', sep=',', skip_lines=1)
dataset = Dataset.load_from_file("./recsys_data/ratings_small.csv",reader)
#user based 
sim_options_cosine_user = {
    'name': 'cosine',
    'user_based': True,
}

sim_options_msd_user = {
    'name': 'msd',
    'user_based': True,
}

sim_options_pearson_user = {
    'name': 'pearson',
    'user_based': True,
}
# item based
sim_options_cosine_item = {
    'name': 'cosine',
    'user_based': False,
}

sim_options_msd_item = {
    'name': 'msd',
    'user_based': False,
}

sim_options_pearson_item = {
    'name': 'pearson',
    'user_based': False,
}
algo_list = ["User CF","Item CF"]
valid_list_user = [sim_options_cosine_user,sim_options_msd_user,sim_options_pearson_user]
valid_list_item = [sim_options_cosine_item,sim_options_msd_item,sim_options_pearson_item]
result_user_cf = []
result_item_cf = []
result_pmf =    []
#for number_Neighbors in range(20,50):
number_Neighbors = 10
for algorithm in algo_list:
    if algorithm == "PMF":
        algo = SVD(biased=False)
        result = cross_validate(algo=algo,data=dataset,measures=["rmse", "mae"],cv=5,verbose=True)
    elif algorithm == "User CF":
        for _,valid_type in enumerate(valid_list_user):
            algo = KNNBasic(k=number_Neighbors,sim_options=valid_type,verbose=True)
            result = cross_validate(algo=algo,data=dataset,measures=["rmse", "mae"],cv=5,verbose=True)
            avg_mae = result['test_mae'].mean()
            avg_rmse = result['test_rmse'].mean()
            result_user_cf.append(result)
    elif algorithm == "Item CF":
        for _,valid_type in enumerate(valid_list_item):
            algo = KNNBasic(k=number_Neighbors,sim_options=valid_type,verbose=True)
            result = cross_validate(algo=algo,data=dataset,measures=["rmse", "mae"],cv=5,verbose=True)
            avg_mae = result['test_mae'].mean()
            avg_rmse = result['test_rmse'].mean()
            result.append(result)
                