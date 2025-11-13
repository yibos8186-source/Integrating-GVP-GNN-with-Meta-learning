from torch.utils.data import Dataset,DataLoader
import re
from transformers import EsmTokenizer, EsmForMaskedLM
import torch
import random
import copy
import os
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import logging
from datetime import datetime 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os 
import wandb 

# wandb.init(mode="disabled")

# os.environ['CUDA_VISIBLE_DEVICE']='7,6,5,4'

char_num_dict = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}

seqs_dir = '/data/home/hujie/workspace/hujie_project/PET/PET_database/related_fasta'
# seqs_dir = '/data/home/hujie/workspace/hujie_project/Rubisco/database/related_fasta'

train_ratio = 0.8
test_ratio =1 - train_ratio

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

log_file = './logs2' + "/" + date_str + ".log"




logging.basicConfig(filename=log_file, level=logging.INFO)


tokenizer = EsmTokenizer.from_pretrained("/data/home/hujie/workspace/hujie_project/PET/PET_pretrained/models/esm2_t33_650M_UR50D",truncation=True, padding='max_length', max_length=1023)

class FastaDataset(Dataset):
    """
    The details of the masking procedure for each sequence follow Devlin et al. 2019:
    10% of the amino acids are masked.
    In 80% of the cases, the masked amino acids are replaced by <mask>.
    In 10% of the cases, the masked amino acids are replaced by a random amino acid (different) from the one they replace.
    In the 10% remaining cases, the masked amino acids are left as is.
    """
   
    def __init__(self, tokenizer, seqs_dir,seqs_labels):
        self.tokenizer = tokenizer
        self.seqs_dir = seqs_dir
        self.seqs_ids = [file for file in os.listdir(seqs_dir) if file[-5:]=='fasta']
        self.seqs_labels = seqs_labels
    def __getitem__(self, idx):
        filename = self.seqs_ids[self.seqs_labels[idx]]
        filename = os.path.join(self.seqs_dir, filename)
        lines = open(filename, 'r').readlines()
        # print('filename:',filename)
        if lines == None or len(lines) == 0:
            logging.info(f"Error: {filename} is empty")
            seq = ''
            print('This is error!')
            return None
        else:
            try:
                if (lines[0][0] == '>'):
                    lines = lines[1:]
                seq = ''.join(lines)
                seq = re.sub(r"[\n\*]", '', seq)
                seq = re.sub(r"[UZOB]", "X", seq).strip().upper()
                length = len(seq)
                seq = [i for i in seq]
                replace_index = random.sample(range(length),int(length*0.10))
                # print('len(seq):',len(seq))
                if len(seq)>1023:
                    logging.info(f"Error:{filename} is empty.")
                replace_dict = {}
                real_seq = copy.copy(seq)
                for index in replace_index:
                    rand_num = random.random()
                    if rand_num >= 0.2:
                        replace_dict[index]='<mask>'
                        # labels = self.tokenizer(seq,return_tensors="pt")["input_ids"]
                        seq[index] = '<mask>'
                        # inputs = self.tokenizer(seq,return_tensors="pt")
                    elif rand_num >= 0.1:
                        real_char = seq[index]
                        choice_char = [x for x in char_num_dict.keys() if x!=real_char]
                        replace_dict[index]=random.choice(choice_char)
                        seq[index] = replace_dict[index]
                        real_seq[index] = replace_dict[index]
                    else:
                        seq[index]='-'
                        replace_dict[index]='-'
                        real_seq[index]='-'
                # print(replace_dict)
                # print(len(real_seq))
                real_seq = ''.join(real_seq).replace('-','')
                need_add = 1021-len(real_seq)
                real_seq = real_seq+'<pad>'*need_add
                seq = ''.join(seq).replace('-','')
                seq = seq + '<pad>'*need_add
    
                real_inputs = self.tokenizer(real_seq,return_tensors='pt')['input_ids']
                mask_inputs = self.tokenizer(seq,return_tensors='pt')
            # print(real_inputs)
            # print(mask_inputs)
            
                labels = torch.where(mask_inputs.input_ids == self.tokenizer.mask_token_id,real_inputs,-100)
                mask_inputs['labels'] = labels
                sample = {key: val.squeeze() for key, val in mask_inputs.items()}
                logging.info(f"OK: {filename}.")
                return sample 
            except:
                logging.info(f"Error: file {filename} is empty!")
            
    def __len__(self):
        return len(self.seqs_labels)


all_fasta = [file for file in os.listdir(seqs_dir) if file[-5:]=='fasta']
all_seqs_labels = len(all_fasta)
validate_seqs_labels = random.sample(range(all_seqs_labels),int(all_seqs_labels*test_ratio))
train_seqs_labels = [i for i in range(all_seqs_labels) if i not in validate_seqs_labels]
random.shuffle(train_seqs_labels)


train_dataset = FastaDataset(tokenizer, seqs_dir, train_seqs_labels)
validate_dataset = FastaDataset(tokenizer, seqs_dir, validate_seqs_labels)
# test_dataset = FastaDataset(tokenizer, seqs_dir, test_seqs_labels, max_length=max_length)


training_args = TrainingArguments(
    output_dir='./results2',
    logging_dir = './logs2',
    # num_train_epochs=5,              # total number of training epochs
    # per_device_train_batch_size=2,   # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    auto_find_batch_size=True,
    prediction_loss_only=False,
    include_inputs_for_metrics=True,
    max_steps=5000,
    dataloader_num_workers=20,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=30,
    log_level='info',
    # gradient_checkpointing=True,    #ESM不支持
    optim="adafactor",
    #warmup_steps=500,               # number of warmup steps for learning rate scheduler
    #weight_decay=0.01,               # strength of weight decay
    learning_rate=1e-4,
    logging_steps=1,                 # How often to print logs
    save_total_limit=5,
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="steps",     # evalute after eachh epoch
    eval_steps=30,
    gradient_accumulation_steps=32,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="PETFinetue-1",       # experiment name
    dataloader_drop_last=True,
    seed=42                         # Seed for experiment reproducibility 3x3
)








model = EsmForMaskedLM.from_pretrained("/data/home/hujie/workspace/hujie_project/PET/PET_pretrained/models/esm2_t33_650M_UR50D")




def compute_metrics(pred):
    labels = pred.label_ids
    prediction = pred.predictions
    labels_index = (labels!=-100).nonzero()
    predict_label = prediction.argmax(-1)[labels_index]
    real_label = labels[labels_index]
    precision, recall, f1, _ = precision_recall_fscore_support(real_label, predict_label,  average='micro')
    acc = accuracy_score(real_label, predict_label)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall}




trainer = Trainer(
    model=model,                
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('./results2')
