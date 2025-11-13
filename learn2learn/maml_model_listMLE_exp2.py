
# from torch import nn, optim

import argparse
parser = argparse.ArgumentParser()


# tasks_per_batch = 10,num_sample_per_task = 12 ,test_lift_list=[19,20],num_batch=1000,adapt_lr=0.01,meta_lr=0.001,adapt_steps=5,hidden_dim=2000,test_random_select_num=1

parser.add_argument('--tasks_per_batch',  default=10,type=int,
                    help='tasks_per_batch, default=10')
parser.add_argument('--num_sample_per_task', metavar='N', type=int, default=10,
                   help='num_sample_per_task, default=10')
parser.add_argument('--test_lift_list', nargs='+',type=int,required=True,
                    help='test_lift_list')
parser.add_argument('--epochs', metavar='N', type=int, default=1,
                    help='training epochs, default=10')
parser.add_argument('--num_batch', metavar='N', type=int, default=1000,
                    help='training epochs, default=100')
parser.add_argument('--adapt_lr',  type=float, default=0.01,
                    help='adapt_lr, default=0.01')
parser.add_argument('--meta_lr',  type=float, default=0.001,
                    help='meta_lr, default=0.001')
parser.add_argument('--adapt_steps', metavar='N', type=int, default=5,
                    help='adapt_steps, default=5')
parser.add_argument('--hidden_dim', metavar='N', type=int, default=2000,
                    help='hidden_dim')
parser.add_argument('--test_random_select_num', metavar='N', type=int, default=1,
                    help='test_random_select_num, default=1')
parser.add_argument('--gpu', metavar='N', type=int, default=2,
                    help='gpu, default=1')
parser.add_argument('--topn', metavar='N', type=int, default=1,
                    help='topn, default=1')
args = parser.parse_args()



import os
import random
from datetime import datetime
import json
import learn2learn as l2l
import numpy as np
import torch
from torch import nn, optim
# import learn2learn as l2l
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle

random.seed(2022)

random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
np.random.seed(2022)



# device = "cuda:3" if torch.cuda.is_available() else "cpu"
# os.environ['CUDA_VISIBLE_DEVICE']=str(args.gpu)

topN = args.topn

class ProteinMeta(data.Dataset):        #[task_per_batch,shots,embedding_dim]
    def __init__(self,mode,tasks_per_batch = 8,num_sample_per_task = 10,test_lift_list=[],test_random_select_num=1, num_batch=1000):
        super().__init__()
        # data_path = '/data/home/hujie/workspace/hujie_project/DiffDock/data/pet/esm2_outputv3'
        datas = []
        points = []
        self.mode = mode
        # with open('/data/home/hujie/workspace/hujie_project/PET/learn2learn/pet_test1_esm2_emb.pkl','rb') as f:
        #     datas = pickle.load(f)
        # with open('/data/home/hujie/workspace/hujie_project/PET/learn2learn/pet_test1_esm2_emb_label.pkl','rb') as f:
        #     points = pickle.load(f)

        #Load first change tensor.
        with open('/data/home/hujie/workspace/hujie_project/PET/PET_pretrained/output_tensor/pet_first_change_esm2_f2.pkl','rb') as f:
            all_datas = pickle.load(f)
        points =  [ids.split('_')[-1] for ids in  all_datas['seqs_ids']]
        
        datas = all_datas['tensor']
        #Load second change tensor.
        with open('/data/home/hujie/workspace/hujie_project/PET/PET_pretrained/output_tensorv2/pet_second_change_ems2_f2.pkl','rb') as f:
            all_datas1 = pickle.load(f)
        points1 = [ids.split('_')[-1] for ids in all_datas1['seqs_ids']]
        datas1 = all_datas1['tensor']
        datas = torch.concat([datas,datas1])
        # print('datas',datas.size())
        points.extend(points1)
        # print('points',points)
        with open('/data/home/hujie/workspace/hujie_project/PET/learn2learn/pet_tpa_exp_results_v1_v2.txt') as f:
            readlines = f.readlines()
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        test_line = test_lift_list
        total_task = []
        train_task = []
        train_task_dict = {}
        test_task = []
        test_task_dict = {}
        test_random_select_line = random.sample([i for i in range(len(readlines)) if i not in test_lift_list],test_random_select_num)
        print('test_random_select_line',test_random_select_line)
        # print(len(readlines))
        for line_index in tqdm(range(len(readlines))):
            content = readlines[line_index].split(',')
            for i in range(1,len(content),3):
                total_task.append(content[i]+':'+content[i+1])
                if content[i]+':'+content[i+1] not in train_task_dict.keys():
                    train_task_dict[content[i]+':'+content[i+1]] = []
                elif content[i]+':'+content[i+1] not in test_task_dict.keys():
                    test_task_dict[content[i]+':'+content[i+1]] = []
                x = torch.cat([torch.tensor([float(content[i]),float(content[1+i])]),datas[points.index(content[0])]])
                # print('x',x.size())
                y = torch.tensor([float(content[2+i])])
                if line_index not in test_line:
                    self.train_X.append(x)
                    self.train_Y.append(y)
                    train_task.append(content[i]+':'+content[i+1])
                    train_task_dict[content[i]+':'+content[i+1]].append([x,y])
                    if line_index in test_random_select_line:
                        self.test_X.append(x)
                        self.test_Y.append(y)
                        test_task.append(content[i]+':'+content[i+1])
                        if content[i]+':'+content[i+1] not in test_task_dict.keys():
                            test_task_dict[content[i]+':'+content[i+1]] = []
                        test_task_dict[content[i]+':'+content[i+1]].append([x,y])

                else:
                    print('this is line_index',line_index)
                    self.test_X.append(x)
                    self.test_Y.append(y)
                    test_task.append(content[i]+':'+content[i+1])
                    test_task_dict[content[i]+':'+content[i+1]].append([x,y])
        print('X train length',len(self.train_X),'Y train length',len(self.train_Y)+len(self.test_Y))
        print('X test length',len(self.test_X),'Y test length',len(self.test_Y))
        total_task = list(set(total_task))
        train_task = list(set(train_task))
        print('There are',len(total_task),'tasks!')
        print('test_task_dict',test_task_dict)
        self.all_train_X = []
        self.all_train_Y = []
        
        for batch in tqdm(range(num_batch)):
            batch_list = []
            batch_X = []
            batch_Y = []

            batch_select = random.sample(['0.3:0.1','0.3:0.3','0.3:0.6','0.4:0.1','0.4:0.3','0.4:0.6','0.5:0.3','0.5:0.6','0.6:0.1','0.6:0.3'],tasks_per_batch-1)   #平均筛选task，第二次对0.6:0.6 给与高权重
            batch_select.append('0.6:0.6')
            for sel in batch_select:
                batch_list.extend(random.sample(train_task_dict[sel],num_sample_per_task))
            for i in range(len(batch_list)):
                batch_X.append(batch_list[i][0])
                batch_Y.append(batch_list[i][1])
            batch_X = torch.vstack(batch_X).view(tasks_per_batch,num_sample_per_task,x.shape[0])
            batch_Y = torch.vstack(batch_Y).view(tasks_per_batch,num_sample_per_task,1)
            self.all_train_X.append(batch_X)
            self.all_train_Y.append(batch_Y)
        
        
        if self.mode == 'test':
            self.test_X = []
            self.test_Y = []
            self.all_test_X = []
            self.all_test_Y = []
            for task,value in test_task_dict.items():
                if len(value)>0:
                    for i in range(len(value)):
                        self.test_X.append(value[i][0])
                        self.test_Y.append(value[i][1])
                    self.all_test_X.append(torch.vstack(self.test_X)[None,:,:])
                    self.all_test_Y.append(torch.vstack(self.test_Y)[None,:,:])

            # print('len self test_x',len(self.all_test_X[0].size()))
            # print('test_x',self.all_test_X[0])
            # self.all_test_X = [self.test_X]
            # self.all_test_Y = [self.test_Y]
    def __getitem__(self, index):
        if self.mode =='train':
            return self.all_train_X[index],self.all_train_Y[index]
        else:
            return self.all_test_X[index],self.all_test_Y[index]
    
    def __len__(self):
        if self.mode=='train':
            return len(self.all_train_X)
        else:
            return len(self.all_test_Y)

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



def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))




def main(tasks_per_batch = args.tasks_per_batch,num_sample_per_task = args.num_sample_per_task ,test_lift_list=args.test_lift_list,num_batch=args.num_batch,adapt_lr=args.adapt_lr,meta_lr=args.meta_lr,adapt_steps=args.adapt_steps,hidden_dim=args.hidden_dim,test_random_select_num=args.test_random_select_num):
    
    dataloader = ProteinMeta(mode = 'train',tasks_per_batch = tasks_per_batch,num_sample_per_task = num_sample_per_task ,test_lift_list=test_lift_list,num_batch=num_batch,test_random_select_num=test_random_select_num)
    test_dataloader = ProteinMeta(mode = 'test',tasks_per_batch = tasks_per_batch,num_sample_per_task = num_sample_per_task ,test_lift_list=test_lift_list,num_batch=num_batch,test_random_select_num=test_random_select_num)
    date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if not os.path.exists(os.path.join('./models_v2',date_str)):
        os.mkdir(os.path.join('./models_v2',date_str))
        os.mkdir(os.path.join('./models_v2',date_str,'train_log'))
    
    writer = SummaryWriter(os.path.join('./models_v2',date_str,'train_log'))

    # create the model
    model = SimpleModel(1282,dim=hidden_dim)
    # model = SimpleModel(1280,dim=hidden_dim)
    model.to('cuda:2')
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    maml.to('cuda:2')
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')
    
    
    save_model_loss = torch.tensor(9999999.0).to('cuda:2')
    # for each iteration
    step = 0

    meta_loss_list = []
    

    for epoch in range(args.epochs):
        for batch_num, batch in tqdm(enumerate(dataloader)): # num_tasks/batch_size
            meta_train_loss = torch.tensor(0.0).to('cuda:2')
            meta_trainloss = meta_train_loss.clone()
    
            meta_support_loss = torch.tensor(0.0).to('cuda:2')
            meta_supportloss = meta_support_loss.clone()
            # for each task in the batch
            effective_batch_size = batch[0].shape[0]
    
            for i in range(effective_batch_size):
                learner = maml.clone()
                # divide the data into support and query sets
                train_inputs, train_targets = batch[0][i].to('cuda:2'), batch[1][i].to('cuda:2')
                x_support, y_support = train_inputs[::2], train_targets[::2]
                x_query, y_query = train_inputs[1::2], train_targets[1::2]
                meta_support_adapt_loss = torch.tensor(0.0).to('cuda:2')
                meta_support_adaptloss = meta_support_adapt_loss.clone()

                for _ in range(adapt_steps): # adaptation_steps
                    support_preds = learner(x_support)           #support:[sample,dim]   preds:[sample,1]
                    # support_mseloss = lossfn(support_preds, y_support)
                    support_preds = support_preds.view(1,-1)
                    y_support = y_support.view(1,-1)
                    support_loss=listMLE(support_preds, y_support)  
                    learner.adapt(support_loss)
                    meta_support_adaptloss+=support_loss

                meta_supportloss += (meta_support_adaptloss / adapt_steps)
                query_preds = learner(x_query)
                # query_mseloss = lossfn(query_preds, y_query)
                query_preds = query_preds.view(1,-1)
                y_query = y_query.view(1,-1)
                query_loss = listMLE(query_preds, y_query) 
                meta_trainloss += query_loss
    
            meta_trainloss = meta_trainloss / effective_batch_size
            meta_supportloss = meta_supportloss /effective_batch_size
    
            writer.add_scalar('loss/query_loss',meta_trainloss,step)
            writer.add_scalar('loss/support_loss',meta_supportloss,step)

            meta_dict = {}
            if meta_supportloss<save_model_loss:
                torch.save(model.state_dict(),os.path.join('./models_v2',date_str,'mlp3_listmleloss_{}_{}.pt'.format(str(step),str(round(meta_supportloss.cpu().item(),3)))))
                save_model_loss = meta_supportloss
                with torch.no_grad():
                    model.eval()
                    topright = 0
                    alltop = 0
                    print('Iteration:', batch_num, 'Meta Train Loss', meta_trainloss.item())
                    for test_iter,test_batch in enumerate(test_dataloader):  # type: ignore
                        # print('test_batch:',test_batch)
                        effective_test_batch_size = len(test_batch[0])
                        print('effective_test_batch_size',effective_test_batch_size)
                        for i in range(effective_test_batch_size):
                            test_inputs,test_targets = test_batch[0][i].to('cuda:2'),test_batch[1][i].to('cuda:2')
                            print('test_inputs',test_inputs)
                            print('test_targets',test_targets)
                            predict = model(test_inputs).data.cpu().numpy().squeeze()
                            real = test_targets.data.cpu().numpy().squeeze()
                            predict_argmax5 = np.argpartition(predict,-topN)[-topN:]
                            predict_argmax5 = predict_argmax5.tolist()
                            # print('predict_argmax{}'.format(str(topN)),predict_argmax5)
                            real_argmax5 = np.argpartition(real,-topN)[-topN:]
                            real_argmax5 = real_argmax5.tolist()
                            # all_arg = len(set(predict_argmax5 + real_argmax5))
                            top5_right = len([i for i in predict_argmax5 if i in real_argmax5])
                            # print('real_argmax{}'.format(str(topN)),real_argmax5)
                            print('Top {} right number:{}'.format(str(topN),top5_right))
                            # print('Predict:Real',np.vstack([predict,real]))
                            print('Predict:Real',predict,real)
                            topright+=top5_right
                            alltop+=len(predict)
                    print('Total have {} / {} = {} right.'.format(topright,alltop,float(topright/alltop)))
                    meta_dict['all_right'] = topright
            with torch.no_grad():
                meta_dict['step'] = step
                meta_dict['meta_query_loss'] = meta_trainloss.cpu().numpy().squeeze().tolist()
                meta_dict['meta_support_loss'] = meta_supportloss.cpu().numpy().squeeze().tolist()
            meta_loss_list.append(meta_dict)
            opt.zero_grad()
            meta_trainloss.backward()
            opt.step() 
            if batch_num % 50 == 0:
                print('Iteration:', batch_num, 'Meta Train Loss', meta_trainloss.item())
            step+=1   
    save_json = args.__dict__
    save_json['step_loss'] = meta_loss_list

    with open(os.path.join('./models_v2',date_str,'traing_parms.json'),'w') as f:
        json.dump(save_json,f)


if __name__ == '__main__':
    main()