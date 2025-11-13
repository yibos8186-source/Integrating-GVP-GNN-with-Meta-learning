import argparse
# from genericpath import isfile

parser = argparse.ArgumentParser()
parser.add_argument('--modelsdir', metavar='PATH', default='./models/',
                    help='directory to save trained models, default=./models/')
parser.add_argument('--num-workers', metavar='N', type=int, default=30,
                   help='number of threads for loading data, default=4')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='max number of nodes per batch, default=3000')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='training epochs, default=100')
parser.add_argument('--train_path', metavar='PATH', default='/data/home/hujie/workspace/gvp/0pet_src/data/final_train_data.json',
                    help='location of CATH dataset, default=/data/home/hujie/workspace/gvp/0pet_src/data/final_train_data.json')
parser.add_argument('--test_path', metavar='PATH', default='/data/home/hujie/workspace/gvp/0pet_src/data/final_validate_data.json',
                    help='location of CATH split file, default=/data/home/hujie/workspace/gvp/0pet_src/data/final_validate_data.json')
parser.add_argument('--ts50', metavar='PATH', default='./data/ts50.json',
                    help='location of TS50 dataset, default=./data/ts50.json')
parser.add_argument('--train', action="store_true", help="train a model")
parser.add_argument('--test-r', metavar='PATH', default=None,
                    help='evaluate a trained model on recovery (without training)')
parser.add_argument('--base_dir', metavar='PATH', default=None,
                    help='base dir.')
parser.add_argument('--test-p', metavar='PATH', default=None,
                    help='evaluate a trained model on perplexity (without training)')
parser.add_argument('--n-samples', metavar='N', default=100,type=int,
                    help='number of sequences to sample (if testing recovery), default=100')

args = parser.parse_args()
assert sum(map(bool, [args.train, args.test_p, args.test_r])) == 1, \
    "Specify exactly one of --train, --test_r, --test_p"

import torch
import torch.nn as nn
import gvp.data, gvp.models
from datetime import datetime
import tqdm, os, json
import numpy as np
from sklearn.metrics import confusion_matrix
import torch_geometric
from functools import partial
from torch.utils.tensorboard import SummaryWriter 
import random 

random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
np.random.seed(2023)

char_num_dict = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
num_char_dict = {4: 'C', 3: 'D', 15: 'S', 5: 'Q', 11: 'K', 9: 'I', 14: 'P', 16: 'T', 13: 'F', 0: 'A', 7: 'G', 8: 'H', 6: 'E', 10: 'L', 1: 'R', 17: 'W', 19: 'V', 2: 'N', 18: 'Y', 12: 'M'}

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
if not os.path.exists(os.path.join(args.modelsdir,date_str)):
    os.mkdir(os.path.join(args.modelsdir,date_str))
    os.mkdir(os.path.join(args.modelsdir,date_str,'train_log'))
    


models_dir = os.path.join(args.modelsdir,date_str)

print = partial(print, flush=True)
node_dim = (100, 16)
edge_dim = (32, 1)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# os.environ['CUDA_VISIBLE_DEVICE']='7,6,5,4,2'

# if not os.path.exists(args.models_dir): os.makedirs(args.models_dir)
model_id = date_str
dataloader = lambda x: torch_geometric.data.DataLoader(x, 
                        num_workers=args.num_workers,
                        batch_sampler=gvp.data.BatchSampler(
                            x.node_counts, max_nodes=args.max_nodes))

def main():
    
    model = gvp.models.CPDModel((7, 3), node_dim, (32, 1), edge_dim)
    # model = gvp.models.CPDModel((6, 3), node_dim, (32, 1), edge_dim)
    # model = gvp.models.CPDModel((6+30, 3), node_dim, (32, 1), edge_dim)
    # model = gvp.models.CPDModel((6+20, 3), node_dim, (32, 1), edge_dim)
    model.to(device)
    
    
    
    if args.test_r or args.test_p:
        ts50set = gvp.data.ProteinGraphDataset(json.load(open(args.ts50)))
        model.load_state_dict(torch.load(args.test_r or args.test_p,map_location=torch.device(device)))
        
    
    if args.test_r:
        # print("Testing on CATH testset"); test_recovery(model, testset)
        print("Testing on TS50 set"); test_recovery(model, ts50set,args.base_dir)
    
    elif args.test_p:
        # print("Testing on CATH testset"); test_perplexity(model, testset)
        print("Testing on TS50 set"); test_perplexity(model, ts50set)
    
    elif args.train:
        print("Loading CATH dataset")
        
        cath = gvp.data.CATHDataset(path=args.train_path,
                                test_path=args.test_path)
        trainset, valset, testset = map(gvp.data.ProteinGraphDataset,
                                    (cath.train, cath.val, cath.test))  
        train(model, trainset, valset, testset)
        print("Testing on CATH testset"); test_recovery(model, testset)
    
    
def train(model, trainset, valset, testset):
    train_loader, val_loader, test_loader = map(dataloader,
                    (trainset, valset, testset))
    optimizer = torch.optim.Adam(model.parameters())
    best_path, best_val = None, np.inf
    lookup = train_loader.dataset.num_to_letter
    train_step=1
    test_step=1
    for epoch in range(args.epochs):
        model.train()
        loss, acc, confusion = loop(model, train_loader, optimizer=optimizer,if_train=True,step=train_step)
        path = f"{models_dir}/{model_id}_{epoch}_{acc}.pt"
        print(f'EPOCH {epoch} TRAIN loss: {loss:.4f} acc: {acc:.4f}')
        print_confusion(confusion, lookup=lookup)
        model.eval()
        with torch.no_grad():
            loss, acc, confusion = loop(model, val_loader,if_train=False,step=test_step)    
        print(f'EPOCH {epoch} VAL loss: {loss:.4f} acc: {acc:.4f}')
        print_confusion(confusion, lookup=lookup)
        
        if loss < best_val:
            best_path, best_val = path, loss
            torch.save(model.state_dict(), best_path)
        print(f'\r\nBEST {best_path} VAL loss: {best_val:.4f}')
        
    print(f"TESTING: loading from {best_path}")
    model.load_state_dict(torch.load(best_path))
    
    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, test_loader)
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')
    print_confusion(confusion,lookup=lookup)

def test_perplexity(model, dataset):
    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, dataloader(dataset))
    print(f'TEST perplexity: {np.exp(loss):.4f}')
    print_confusion(confusion, lookup=dataset.num_to_letter)

def test_recovery(model, dataset,base_dir):
    recovery = []
    
    for protein in tqdm.tqdm(dataset):
        protein = protein.to(device)
        h_V = (protein.node_s, protein.node_v)
        h_E = (protein.edge_s, protein.edge_v) 
        sample,value_dict = model.sample(h_V, protein.edge_index, 
                              h_E, n_samples=args.n_samples)
        
        recovery_ = sample.eq(protein.seq).float().mean().cpu().numpy()
        recovery.append(recovery_)
        print(protein.name, recovery_, flush=True)   #TODO
            
            # print(sample.eq(protein.seq).float().cpu().numpy())
            # print('Actual:',protein.seq.cpu().numpy())
            # print('Predict:',sample.cpu().numpy())
            # print(value_dict)
        with open(os.path.join(base_dir,protein.name+'_'+str(recovery_)+'.csv'),'w') as f:
            for key,value in value_dict.items():
                all_value = [str(v) for v in  value.tolist()]
                cotent = [key,num_char_dict[protein.seq.cpu().numpy()[key]],protein.seq.cpu().numpy()[key],num_char_dict[sample.cpu().numpy()[0][key]],','.join(all_value)]
                cotent = [str(i) for i in cotent]
                f.write(','.join(cotent)+'\r\n')

    recovery = np.median(recovery)
    print(f'TEST recovery: {recovery:.4f}')
    
def loop(model, dataloader, optimizer=None,if_train=True,step=1):
    writer = SummaryWriter(os.path.join(args.modelsdir,date_str,'train_log'))
    confusion = np.zeros((20, 20))
    t = tqdm.tqdm(dataloader)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0, 0, 0
    for batch in t:
        if optimizer: optimizer.zero_grad()
        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        logits = model(h_V, batch.edge_index, h_E, seq=batch.seq)
        logits, seq = logits[batch.mask], batch.seq[batch.mask]
        loss_value = loss_fn(logits, seq)

        if optimizer:
            loss_value.backward()
            optimizer.step()

        num_nodes = int(batch.mask.sum())
        total_loss += float(loss_value) * num_nodes
        total_count += num_nodes
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true = seq.detach().cpu().numpy()
        total_correct += (pred == true).sum()
        confusion += confusion_matrix(true, pred, labels=range(20))
        t.set_description("%.5f" % float(total_loss/total_count))
        if if_train==True:
            writer.add_scalar('loss/train_loss',float(total_loss/total_count),step)
            writer.add_scalar('acc/train_acc',float(total_correct / total_count),step)
        else:
            writer.add_scalar('loss/test_loss',float(total_loss/total_count),step)
            writer.add_scalar('acc/test_acc',float(total_correct / total_count),step)
        step+=1
        torch.cuda.empty_cache()
        
    return total_loss / total_count, total_correct / total_count, confusion
    
def print_confusion(mat, lookup):
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = '\n'
    for i in range(20):
        res += '\t{}'.format(lookup[i])
    res += '\tCount\n'
    for i in range(20):
        res += '{}\t'.format(lookup[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)
    
if __name__== "__main__":
    main()