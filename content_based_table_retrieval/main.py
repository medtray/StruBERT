import sys
from pathlib import Path
import os
cwd = os.getcwd()
parent_path=str(Path(cwd).parent)
print(parent_path)

sys.path.append(parent_path)
import torch.nn.functional as F
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from data_reader import DataAndQueryReader
from table_matching_model import strubert

import numpy as np
import pandas as pd
import subprocess
import random
import argparse
import json
import math
from tqdm import tqdm
import shutil
from utils import pad_table_query,train_val_dataset,save_checkpoint,load_checkpoint_for_eval,read_file_for_nfcg,qrel_for_data,calculate_metrics
from transformers import get_linear_schedule_with_warmup,AdamW

parser = argparse.ArgumentParser(description='Content-based table retrieval', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=int, default=3)
parser.add_argument("--table_folder", type=str, default='/path/to/wikitables/folder')
parser.add_argument("--tabert_path", type=str, default='/path/to/tabert/model.bin')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00002)
parser.add_argument("--balance_data", dest="balance", action="store_true")
args = parser.parse_args()
#print(torch.cuda.current_device())
#torch.cuda.set_device(args.device)
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(args.device))
#print(torch.cuda.is_available())
args.device='cuda:'+str(args.device)
#args.device='cpu'

out_str = str(args)
print(out_str)
data_folder= args.table_folder
base='data'

loss_function=nn.MSELoss()
m = nn.Sigmoid()
loss_bce = nn.BCELoss()

batch_size=args.batch_size
NUM_EPOCH=args.epochs
start_epoch=0
test_every=1
del_threshold = 0.5

kfolds = np.load('data/folds.npy', allow_pickle=True)
text_file = open(os.path.join(base,"qrels.tsv"), "r")
lines = text_file.readlines()

queries_id_qrels = []
list_lines_qrels = []

for line in lines:
    # print(line)
    line = line[0:len(line) - 1]
    aa = line.split('\t')
    queries_id_qrels += [aa[0]]
    list_lines_qrels.append(aa)

data=read_file_for_nfcg(os.path.join(base,"all.txt"))

queries_file='data/query.txt'
queries = open(queries_file, "r")
lines = queries.readlines()
id_to_queries={}

for line in lines:
    # print(line)
    line = line[0:len(line) - 1]
    aa = line.split('\t')
    id_to_queries[aa[0]]=aa[1]



all_folds_ndcg = []
all_folds_map = []
all_folds_mrr = []

for fold_index,(train, test) in enumerate(kfolds):

    print('>>>> fold {}'.format(str(fold_index + 1)))

    loss_train = []

    best_valid_loss = float('inf')
    save_path = 'wiki_tabert_' + str(fold_index + 1) + '.pt'
    save_path_best = 'best_wiki_tabert_' + str(fold_index + 1) + '.pt'

    output_qrels_test = 'qrels_test'+str(fold_index+1)+'.txt'
    qrel_for_data(data[test], list_lines_qrels, output_qrels_test)

    train_file_name = './train_'+str(fold_index+1)+'.txt'
    np.savetxt(train_file_name, data[train], fmt="%s",delimiter='\t')
    test_file_name = './test_'+str(fold_index+1)+'.txt'
    np.savetxt(test_file_name, data[test], fmt="%s",delimiter='\t')
    output_train_ndcg = './train_ndcg_'+str(fold_index+1)+'.txt'
    output_test_ndcg = './test_ndcg_'+str(fold_index+1)+'.txt'
    test_scores = './scores_' + str(fold_index + 1) + '.txt'

    train_dataset = DataAndQueryReader(train_file_name,output_train_ndcg,data_folder,id_to_queries)
    train_dataset, val_set = train_val_dataset(train_dataset, val_split=0.2)
    train_indices = train_dataset.indices[:]
    print('number of training samples = {}'.format(len(train_dataset)))
    print('number of validation samples = {}'.format(len(val_set)))
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_table_query)
    valid_iter = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_table_query)

    test_dataset = DataAndQueryReader(test_file_name, output_test_ndcg,data_folder,id_to_queries)
    print('number of testing samples = {}'.format(len(test_dataset)))
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_table_query)

    inter_test = read_file_for_nfcg(output_test_ndcg)

    model = strubert(tabert_path=args.tabert_path, device=args.device)  # .to(args.device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_batches = int(math.ceil(len(train_dataset) / batch_size))
    num_batches_test = int(math.ceil(len(test_dataset) / batch_size))
    num_batches_valid = int(math.ceil(len(val_set) / batch_size))

    if args.balance_data:
        nb_not_relevant=len([j for j in train_dataset.indices if train_dataset.dataset.labels[j]==0])
        approximate_nb_not_relevant = int((1-del_threshold)*nb_not_relevant)
        approximate_train_batches = int(math.ceil((approximate_nb_not_relevant+ len(train_dataset)-nb_not_relevant) / batch_size ))
        num_steps = approximate_train_batches * NUM_EPOCH
    else:
        num_steps = num_batches * NUM_EPOCH

    # learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_steps // 10,
                                                num_training_steps=num_steps)
    ## training
    print('>>>> training phase ')

    for epoch in range(start_epoch, NUM_EPOCH + start_epoch):

        if args.balance_data:

            tran_indices_copy = train_indices[:]

            random.shuffle(tran_indices_copy)
            for i in range(len(tran_indices_copy)-1,-1,-1):

                if train_dataset.dataset.labels[tran_indices_copy[i]] == 0:
                    prob = random.random()
                    if prob<del_threshold:
                        del tran_indices_copy[i]

            train_dataset.indices = tran_indices_copy
            train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_table_query)
            num_batches = int(math.ceil(len(train_dataset) / batch_size))

        epoch_loss = 0
        all_outputs = []
        all_labels = []

        for i, batch in enumerate(train_iter):
            tables, queries, all_tables_meta, all_query_meta, labels = batch
            labels = torch.FloatTensor(labels).to(args.device)

            outputs = model(tables, all_tables_meta, queries, all_query_meta)  # .to(args.device)

            loss = loss_function(outputs, labels)

            epoch_loss += loss.item()

            all_outputs += outputs.tolist()
            all_labels += labels.tolist()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


        losslogger = epoch_loss / num_batches
        print('done with epoch {}'.format(epoch + 1))

        loss_train.append(losslogger)
        print('training loss')
        print(loss_train)

        ## validation
        validation_loss = 0.0
        all_outputs_valid = []
        all_labels_valid = []

        for i, batch in enumerate(valid_iter):
            tables, queries, all_tables_meta, all_query_meta, labels = batch
            labels = torch.FloatTensor(labels).to(args.device)

            outputs = model(tables, all_tables_meta, queries, all_query_meta)  # .to(args.device)

            loss = loss_function(outputs, labels)
            validation_loss += loss.item()
            all_labels_valid += labels.tolist()

        lossvalid = validation_loss / num_batches_valid


        print('validation results')
        print('validation loss = {}'.format(lossvalid))

        is_best = False
        if lossvalid < best_valid_loss:
            is_best = True
            best_valid_loss = lossvalid

        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'losslogger': lossvalid, 'best_valid_loss': best_valid_loss}

        torch.save(state, save_path)
        save_checkpoint(save_path, is_best, save_path_best)

    ## testing
    testing_loss = 0.0
    all_labels_test = []
    all_outputs_test = []
    model = load_checkpoint_for_eval(model, save_path_best)

    for i, batch in enumerate(test_iter):
        tables, queries, all_tables_meta, all_query_meta, labels = batch
        labels = torch.FloatTensor(labels).to(args.device)

        outputs = model(tables, all_tables_meta, queries, all_query_meta)  # .to(args.device)

        loss = loss_function(outputs, labels)
        testing_loss += loss.item()

        all_outputs_test += outputs.tolist()
        all_labels_test += labels.tolist()

    losstest = testing_loss / num_batches_test

    test_ndcg, test_map, test_mrr = calculate_metrics(inter_test, test_scores, all_outputs_test,output_qrels_test)

    print('testing results')
    print('NDCG@5: {} \n MAP:{} \n MRR:{} \n '.format(test_ndcg, test_map, test_mrr))

    all_folds_ndcg.append(test_ndcg)
    all_folds_map.append(test_map)
    all_folds_mrr.append(test_mrr)

print('final results \n')
print('NDCG@5 results')
print(all_folds_ndcg)
print('mean NDCG@5={}'.format(np.mean(all_folds_ndcg)))
print('std NDCG@5={}'.format(np.std(all_folds_ndcg)))
print('MAP results')
print(all_folds_map)
print('mean MAP={}'.format(np.mean(all_folds_map)))
print('std MAP={}'.format(np.std(all_folds_map)))
print('MRR results')
print(all_folds_mrr)
print('mean MRR={}'.format(np.mean(all_folds_mrr)))
print('std MRR={}'.format(np.std(all_folds_mrr)))

