import pickle
import torch
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import json
import random 
random.seed(2022)


tensor_pkl='/data/home/hujie/workspace/hujie_project/Rubisco/paper_data/mut_one_points.pkl'
json_folder = '/data/home/hujie/workspace/hujie_project/Rubisco/paper_data'
gp_pkl = '/data/home/hujie/workspace/hujie_project/Rubisco/paper_data/gp_rubisco_spec.pkl'




def main():
    with open(tensor_pkl,'rb') as f:
        datas = pickle.load(f)
    all_jsons = [file for file in os.listdir(json_folder) if file[-5:]=='.json']
    value_dict = {}
    for data in all_jsons:
        with open(os.path.join(json_folder,data),'r') as f:
            information = json.load(f)
            for mut in information['mut']:
                value_dict[information['name']+'_'+mut[0]] = float(mut[1])
    y_value = []
    for seq in datas['seqs_ids']:
        assert seq in value_dict,'Pls check value dict!'
        y_value.append(value_dict[seq])
    y=np.array(y_value)
    x=datas['tensor'].numpy()
    kernel = DotProduct() + WhiteKernel()
    Y_train = []
    X_train = []
    for _ in range(5):
        choice_list = random.sample([i for i in range(len(datas['seqs_ids']))],max(1,min(len(datas['seqs_ids'])-5,15)))
        X_train.append(x[choice_list])
        Y_train.append(y[choice_list])
    X_sample = np.vstack(X_train)
    Y_sample = np.hstack(Y_train)
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_sample, Y_sample)
    print('Model score for GP:',gpr.score(X_sample, Y_sample))
    with open(gp_pkl,'wb') as f:
        pickle.dump(gpr,f)
    



if __name__=='__main__':
    main()
    
    