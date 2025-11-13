from transformers import EsmTokenizer,EsmModel
import torch
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
import numpy as np
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
parser.add_argument('--simple_model_path',  default='/data/home/hujie/workspace/hujie_project/PET/learn2learn/models/no_fintune_tensor/2022-11-07_09:20:38/mlp3_listmleloss_19252_2.263.pt',type=str,
                    help='simple_model_path')                 
parser.add_argument('-s','--seq_file',  type=str,default='/data/home/hujie/workspace/hujie_project/DiffDock/data/pet/pot_changed_points.fasta',
                   help='seq_file')
parser.add_argument('-o','--out_file', type=str, 
                    help='out_file')
parser.add_argument('-b','--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('-g','--gpu',nargs='+',type=int,)
args = parser.parse_args()



path =args.model_path
seq_file = args.seq_file
# out_pickle = args.out_pickle
gpu = args.gpu


os.environ['CUDA_VISIBLE_DEVICE']=','.join([str(i) for i in args.gpu])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiFastaDataset(Dataset):
    def __init__(self, tokenizer, seqs, seqs_ids):
        self.tokenizer = tokenizer
        self.seqs_ids = seqs_ids
        self.seqs = seqs
    def __getitem__(self, idx):
        seq = self.seqs[self.seqs_ids[idx]]
        seq = re.sub(r"[\n\*]", '', seq)
        seq = re.sub(r"[UZOB]", "X", seq)
        seq = seq.replace("", " ").strip()
        sample  = self.tokenizer(seq,return_tensors='pt')
        sample['name'] = self.seqs_ids[idx]
        return sample
    def __len__(self):
        return len(self.seqs_ids)


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



class PredictModel(nn.Module):
    def __init__(self,path,model_path):
        super().__init__()
        self.esmmodel =  EsmModel.from_pretrained(path)
        SM = torch.load(model_path)
        new_state_dict = OrderedDict()
        for key,value in SM.items():
            name = key.replace("module.","")
            new_state_dict[name]=value
        SimModel = SimpleModel(1282,2000)
        SimModel.load_state_dict(new_state_dict,strict=False)
        self.SimModel = SimModel
        # self.SimpleModel = self.SimpleModel.eval()
    def forward(self,x):
        # print(x)
        # print(x['input_ids'].size(),x['attention_mask'].size())
        s = self.esmmodel(input_ids=x['input_ids'].squeeze(),attention_mask=x['attention_mask'].squeeze()).last_hidden_state.mean(1).squeeze()
        # print(s.size())
        y = torch.tensor([[0.6,0.6]]*s.size()[0]).to(device)
        s = torch.hstack([y,s])
        s = self.SimModel(s)
        return s






def main():
    seqs_ids = []
    seqs = {}
    fa = Fasta(args.seq_file)
    tokenizer = EsmTokenizer.from_pretrained(path)
    for seq in fa:
        name = seq.name
        seqs_ids.append(name)
        seqs[name] = str(seq)
    
    test_dataset = MultiFastaDataset(tokenizer, seqs, seqs_ids)
    loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=20)
    model = PredictModel(args.model_path,args.simple_model_path)
    model = nn.DataParallel(model,device_ids=args.gpu)
    model.to(device)
    model.eval()
    all_name = []
    all_tensor = []
    with torch.no_grad():
        for data in tqdm(loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            names = data['name']
            input = {'input_ids':input_ids,'attention_mask':attention_mask}
            output = model(input)
            out = output.squeeze().cpu().numpy()
            all_name.extend(names)
            all_tensor.extend(out)
    all_value = np.hstack(all_tensor)
    argsort = np.argsort(all_value)
    tensor_sort = all_value[argsort][::-1].tolist()
    argsort = argsort[::-1].tolist()
    name_sort = [all_name[i] for i in argsort]
    with open(args.out_file,'w') as f:
        rank=1
        for name,value in zip(name_sort,tensor_sort):
            f.write(name+','+str(round(value,5))+str(rank)+'\r\n')
            rank+=1


if __name__=='__main__':
    main()