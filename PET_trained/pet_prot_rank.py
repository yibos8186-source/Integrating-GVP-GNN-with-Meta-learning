
import torch
# import os
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from datetime import datetime 
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np

import pickle
from tqdm import tqdm
import argparse
import json 
parser = argparse.ArgumentParser()


# tasks_per_batch = 10,num_sample_per_task = 12 ,test_lift_list=[19,20],num_batch=1000,adapt_lr=0.01,meta_lr=0.001,adapt_steps=5,hidden_dim=2000,test_random_select_num=1

parser.add_argument('--model_path',  default='/data/home/hujie/workspace/hujie_project/PET/learn2learn/models/no_fintune_tensor/2022-11-07_09:20:38/mlp3_listmleloss_19252_2.263.pt',type=str,
                    help='model_path')
parser.add_argument('-s','--seq_file',  type=str,
                   help='seq_file')
parser.add_argument('-o','--out_json', type=str, 
                    help='out_json')
parser.add_argument('-b','--batch', type=int, 
                    help='batch')

args = parser.parse_args()

class SimpleModel(nn.Module):
    def __init__(self,input,dim):
        super().__init__()
        self.hidden1 = nn.Linear(input, dim)
        
        self.hidden2 = nn.Linear(dim,dim )
        # self.batchnorm = nn.BatchNorm1d(dim)
        # self.hidden3 = nn.Linear(dim, dim)
        self.hidden4 = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        # x = nn.functional.relu(self.hidden3(x))
        # x = self.batchnorm(x)
        x = self.hidden4(x)
        return x


# model_p

def main():
    
    model = SimpleModel(1282,2000)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()    
    model.to('cuda:4')
    with open(args.seq_file,'rb') as f:
        data = pickle.load(f)
    inputs = data['tensor']
    predict_list = []
    with torch.no_grad():
        if len(inputs)>args.batch:
            for i in range(0,len(inputs),args.batch):
                input = inputs[i:min(len(inputs),i+args.batch)]
                tep = torch.tensor([0.6]*len(input)).reshape(len(input),-1)
                hou = torch.tensor([0.6]*len(input)).reshape(len(input),-1)
                input = torch.hstack([tep,hou,input]).to('cuda:4')
                
                predict_list.append(model(input))
        else:
            tep = torch.tensor([0.6]*len(inputs)).reshape(len(inputs),-1)
            hou = torch.tensor([0.6]*len(inputs)).reshape(len(inputs),-1)
            input = torch.hstack([tep,hou,inputs]).to('cuda:4')
            
            predict_list(model(input))
        predicts = torch.vstack(predict_list).cpu().reshape(len(inputs))
        argsort = torch.argsort(predicts,descending=True)
        value_sort = [data['seqs_ids'][i] for i in  argsort.numpy().tolist()]
    out = {'value_rank':value_sort,'argsort':predicts[argsort].numpy().tolist()}
    with open(args.out_json,'w') as f:
        json.dump(out,f,indent=2)


if __name__=='__main__':
    main()






    