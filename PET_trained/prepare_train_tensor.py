import torch
from transformers import EsmTokenizer,EsmModel

# import os
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
# import torch.distributed as dist
from collections import OrderedDict

# import torch.nn.functional as F
# from datetime import datetime 
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np
from pyfaidx import Fasta
import pickle
from tqdm import tqdm
import argparse
import re
import os


parser = argparse.ArgumentParser()
# tasks_per_batch = 10,num_sample_per_task = 12 ,test_lift_list=[19,20],num_batch=1000,adapt_lr=0.01,meta_lr=0.001,adapt_steps=5,hidden_dim=2000,test_random_select_num=1

parser.add_argument('--model_path',  default='/data/home/hujie/workspace/hujie_project/PET/PET_pretrained/results/checkpoint-2860',type=str,
                    help='model_path')
parser.add_argument('-s','--seq_file',  type=str,default='/data/home/hujie/workspace/hujie_project/DiffDock/data/pet/pot_changed_points.fasta',
                   help='seq_file')
parser.add_argument('-o','--out_pickle', type=str, 
                    help='out_pickle')
parser.add_argument('-g','--gpu',type=str,)
args = parser.parse_args()






def main():
    output={}
    seq_tensor_list=[]
    tokenizer = EsmTokenizer.from_pretrained(args.model_path)
    model = EsmModel.from_pretrained(args.model_path)
    # model.to('cuda:'+args.gpu)
    fa = Fasta(args.seq_file)
    seqs_ids = []
    seqs = {}
    for name,seq in tqdm(fa.items()):
        seqs_ids.append(name)
        seqs[name] = str(seq)
        # print(str(seq))
        inputs = tokenizer(str(seq),return_tensors='pt')
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.mean(1).squeeze().cpu()
        print(outputs.last_hidden_state.size())
        seq_tensor_list.append(last_hidden_states)
    all_tensor = torch.stack(seq_tensor_list)
    output['seqs_ids']=seqs_ids
    output['seqs'] = seqs
    output['tensor'] = all_tensor
    with open(args.out_pickle,'wb') as f:
        pickle.dump(output,f)


if __name__=='__main__':
    main()