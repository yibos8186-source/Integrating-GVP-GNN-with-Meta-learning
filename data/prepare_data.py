
from tqdm import tqdm
import json
from Bio import SeqIO
import os
import collections
# from atom_res_dict import *
import multiprocess as mp
from tqdm import tqdm 
from Bio.PDB.PDBParser import PDBParser
import numpy as np


# path = '/data/home/hujie/workspace/datasets/pet_data/pet_fasta/validate'   #train
# path = '/data/home/hujie/workspace/datasets/pet_data/pet_fasta/train'   #train
# path = '/data/home/hujie/workspace/datasets/pet_data/test_pet_6'    #pet6
# path = '/data/home/hujie/workspace/datasets/pet_data/high_pet' #valid paper pet
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/valid_paper_cas'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_at_pg'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_new'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_add'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/pet_top'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/LMDH'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/C4'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/PC'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/ALAS'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/Levodopa'
# path = '/data/home/hujie/workspace/gvp/0pet_src/data/Levodopa/4data'
path = '/data/home/hujie/workspace/gvp/0pet_src/data/MeiLiu'
# path = '/data/home/hujie/workspace/gvp/src'
num_thread = 15
# out_json = '/data/home/hujie/workspace/gvp/src/data/train_protein.json'  #train
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/validate_protein_all_atoms.json'  #test
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/train_protein_all_atoms.json'  #test
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/test_pet6/testpet6.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/valid_paper_pet/valid_paper_pet.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/valid_paper_cas/valid_paper_cas.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/rubisco1.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_at_pg/rubisco_at_pg.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_new/rubisco_new.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_add/rubisco_add.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/C4/C4.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/PC/PC.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/ALAS/ALAS.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/Levodopa/Levodopa.json'
# out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/Levodopa/4data/4Levodopa.json'
out_json = '/data/home/hujie/workspace/gvp/0pet_src/data/MeiLiu/MeiLiu.json'



# get_data_path = '/data/home/hujie/workspace/datasets/pet_data/pet.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/test_pet6/used_chain.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/valid_paper_cas/used_chain_20220907.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/used_chain_rbcl.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_at_pg/used_chain_at_pg.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/Rubisco/Rubisco_add/used_chain_at_add.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/pet_top/used_chain.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/LMDH/used_chain.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/PC/used_chain.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/Levodopa/used_chain.txt'
# get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/Levodopa/4data/used_chain.txt'
get_data_path = '/data/home/hujie/workspace/gvp/0pet_src/data/MeiLiu/used_chain.txt'

# fasta_path='/data/home/hujie/workspace/datasets/pet_data/pet_fasta/pet.fasta'




def get_allpdb(folder):
    all_pdb = [file for file in os.listdir(folder) if file[-3:]=='pdb']
    return all_pdb


def get_protein_chain(path):
    all_line = open(path,'r').readlines()
    content_dict={}
    for line in all_line:
        content_dict[line.split('|')[1]]=line[9]
    return content_dict


def get_all_fasta(fasta_path):
    all_lines = open(fasta_path,'r').readlines()
    fasta_dict = {}
    for index in range(0,len(all_lines),2):
        line = all_lines[index]
        if line[0]=='>':
            content = line[1:].split('|')
            fasta_dict[content[0][:4]+'|'+content[1][6:].replace(' ','').replace(',','_')] = all_lines[index+1][:-1]
    return fasta_dict
        
            


content_dict = get_protein_chain(get_data_path)
# fasta_dict = get_all_fasta(fasta_path)
error_list=[]


res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}

abrev={'HIS':'H','LYS':'K','ARG':'R','ASP':'D','GLU':'E','SER':'S','THR':'T','ASN':'N','GLN':'Q','ALA':'A','VAL':'V','LEU':'L','ILE':'I','MET':'M','PHE':'F','TYR':'Y','TRP': 'W','PRO': 'P','GLY': 'G', 'CYS': 'C'}

letter1_3_dict={'H':'HIS','K':'LYS','R':'ARG','D':'ASP','E':'GLU','S':'SER','T':'THR','N':'ASN','Q':'GLN','A':'ALA','V':'VAL','L':'LEU','I':'ILE','M':'MET','F':'PHE','Y':'TYR','W':'TRP','P':'PRO','G':'GLY','C':'CYS'}


def parse_pdb_file(pdb_file,content_dict=content_dict):
    try:
        p = PDBParser()
        structure = p.get_structure(pdb_file[:-4], os.path.join(path,pdb_file))
        sequence = next(SeqIO.parse(os.path.join(path,pdb_file),"pdb-atom"))
        pos_list_out = []
        structure = next(structure.get_models())           #only think about first structure.
        chain = structure[content_dict[pdb_file[:-4]]]
        
        for residue in chain:
            try:
                pos_list_out.append(ca_coord_and_orientation(residue))
            except Exception as e:
                print(residue.id[1])
                if e=='CA':
                    pass
            # print(len(chain))
            # print(chain.id,residue.id)
            # print(pos_list_out)
        out_dict ={}
        out_dict['name']=pdb_file[:-4]+'.'+content_dict[pdb_file[:-4]]
        seqs =''
        # for seq in sequence:
        #     seqs+=seq
        # for file in fasta_dict.keys():
        #     if pdb_file[:-4] == file[:4] and content_dict[pdb_file[:-4]] in file[5:]:
        #         seqs = fasta_dict[file]
        #         break
        
        # print(pos_list_out)
        # if  seqs[0]=='M' and letter1_3_dict[seqs[0]] != pos_list_out[0][-1]:
        #     seqs=seqs[1:]
        
        # 查看RCSB下载的一些pdb是否和fasta对应一致
        # for i in range(len(seqs)):
        #         # print(letter1_3_dict[seqs[i]],pos_list_out[i][-1])
        #     assert letter1_3_dict[seqs[i]] == pos_list_out[i][-1],pdb_file+',{} not equal.'.format(i)
        
            
        # assert len(out_dict['seq']) == len(out_dict['coords']),'seq {} and coords {} do not have same length.'.format(len(out_dict['seq']),len(out_dict['coords']))
        
        # 不去关注fasta的文件，只计算有三维结构的
        
        for file in pos_list_out:
            if file[-1] in abrev.keys():
                seqs+=abrev[file[-1]]
            else:
                print(file[-1],'not have.')
        out_dict['seq']= seqs
        # print(seqs,len(seqs))
        out_dict['coords']=[file[:-2] for file in pos_list_out]
        out_dict['res_id']=[file[-2] for file in pos_list_out]
        assert len(seqs)==len(pos_list_out),'{}!={},pdb name: {}'.format(len(seqs),len(pos_list_out),pdb_file[:-4]+'.'+content_dict[pdb_file[:-4]])
        
        return out_dict
    except Exception as e:
        print(e)
        print(pdb_file,'need_fix.')
        error_list.append(pdb_file)
        return None
        


def ca_coord_and_orientation(residue):
    """
    Coordinates of a residue's carbon alpha and direction CA->CB.
    If the residue is GLY, the atom CB is "virtual".
    """
    
    ca = np.round_(residue["CA"].get_vector().get_array(),3).tolist()
    n  = np.round_(residue["N"].get_vector().get_array(),3).tolist()
    o  = np.round_(residue["O"].get_vector().get_array(),3).tolist()
    # try:
    #     c= np.round_(residue["CB"].get_vector().get_array(),3).tolist()
    # except:
    #     c = np.round_(residue["C"].get_vector().get_array(),3).tolist()
    c = np.round_(residue["C"].get_vector().get_array(),3).tolist()
    # print([n,ca,c,o] )
    return [n,ca,c,o,residue.id[1],residue.get_resname()]   


def main():
    all_pdbs = get_allpdb(path)
    pool = mp.Pool(min(num_thread,len(all_pdbs)))
    error_list=[]
    all_data=list(tqdm(pool.imap(parse_pdb_file,all_pdbs),total=len(all_pdbs)))
    pool.close()
    # print(all_data)
    for data in all_data:
        if data is None:
            all_data.remove(data)
    with open(out_json,'w') as f:
        json.dump(all_data,f)
    print('Finished!')
    print('Succ:',len([file for file in all_data if file!=None]),'Total:',len(all_pdbs))
    # print('Error pdb:')
    # print(error_list)
    
  
       



if __name__=='__main__':
    error_list=[]
    main()
               
            
            
        
    
    
