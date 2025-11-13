import json 




# path_train1 = '/data/home/hujie/workspace/gvp/gvp-pytorch/data/chain_set.jsonl'
path_train1 = '/data/home/hujie/workspace/gvp/neurips19-graph-protein-design/data/cath/cath_data/cath_all_S40.json'
path_train2 = '/data/home/hujie/workspace/gvp/0pet_src/data/train_protein_all_atoms.json'
path_train3 = '/data/home/hujie/workspace/gvp/0pet_src/data/validate_protein_all_atoms.json'


splits_path ='/data/home/hujie/workspace/gvp/gvp-pytorch/data/chain_set_splits.json'



out_json_train = '/data/home/hujie/workspace/gvp/0pet_src/data/final_train_data.json'
out_json_val = '/data/home/hujie/workspace/gvp/0pet_src/data/final_validate_data.json'
out_json_test = '/data/home/hujie/workspace/gvp/0pet_src/data/final_test_data.json'


with open(path_train2,'r') as f:
    train_protein = json.load(f)

with open(path_train3,'r') as f:
    valid_protein = json.load(f)

with open(splits_path) as f:
    dataset_splits = json.load(f)




chain_set_lines = open(path_train1,'r').readlines()

test_protein = []
for line in chain_set_lines:
    entry = json.loads(line)
    name = entry['name']
    coords = entry['coords']
    entry['coords'] = list(zip(
        coords['N'], coords['CA'], coords['C'], coords['O']
    ))
    if name in dataset_splits['train']:
        train_protein.append(entry)
    elif name in dataset_splits['validation']:
        valid_protein.append(entry)
    elif name in dataset_splits['test']:
        test_protein.append(entry)


with open(out_json_train,'w') as f:
    json.dump(train_protein,f)
print('train_protein:',len(train_protein))

with open(out_json_val,'w') as f:
    json.dump(valid_protein,f)
print('validation_protein:',len(valid_protein))

with open(out_json_test,'w') as f:
    json.dump(test_protein)
print('test_protein',len(test_protein))
        



    
    


