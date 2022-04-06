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
from table_matching_model import strubert
from utils import pad_table_query,train_val_dataset,save_checkpoint,load_checkpoint_for_eval
from data_reader import DataAndQueryReader
from sklearn.model_selection import KFold
import os
import numpy as np
import torch.nn.functional as F
import pandas as pd
import subprocess
import random
import argparse
import json
import sys
from transformers import get_linear_schedule_with_warmup,AdamW

from pathlib import Path
import math
from sklearn.metrics import precision_recall_fscore_support
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Wikitables similarity', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=int, default=3)
parser.add_argument("--table_folder", type=str, default='/home/mohamedt/tables_redi2_1')
#parser.add_argument("--table_folder", type=str, default='/home/mohamed/PycharmProjects/Data-Search-Project/tables_redi2_1')
parser.add_argument("--tabert_path", type=str, default='/home/mohamedt/tabert_base_k3/model.bin')
#parser.add_argument("--tabert_path", type=str, default='/home/mohamed/PycharmProjects/tabert_base_k3/model.bin')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument("--lr", type=float, default=0.00002)
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

loss_function=nn.MSELoss()
m = nn.Sigmoid()
loss_bce = nn.BCEWithLogitsLoss()

batch_size=args.batch_size
NUM_EPOCH=args.epochs
start_epoch=0

kfolds = np.load('wikitables_similarity_folds.npy', allow_pickle=True)
kfolds = kfolds[()]

all_folds_accuracy =[]
all_folds_p =[]
all_folds_r =[]
all_folds_f =[]

for fold_index,(train, test) in enumerate(kfolds):

    print('>>>> fold {}'.format(str(fold_index+1)))

    train_dataset = DataAndQueryReader(train,data_folder)
    train_dataset,val_set = train_val_dataset(train_dataset, val_split=0.2)
    print(len(train_dataset))
    print(len(val_set))
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_table_query)
    valid_iter = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_table_query)

    test_dataset = DataAndQueryReader(test,data_folder)
    print(len(test_dataset))
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_table_query)

    model = strubert(tabert_path=args.tabert_path, device=args.device)  # .to(args.device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_batches = int(math.ceil(len(train_dataset) / batch_size))
    num_batches_test = int(math.ceil(len(test_dataset) / batch_size))
    num_batches_valid = int(math.ceil(len(val_set) / batch_size))

    # learning rate scheduler
    num_steps = num_batches * NUM_EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_steps // 10,
                                                num_training_steps=num_steps)

    accuracy_train = []
    best_valid_accuracy = 0
    save_path = 'wiki_tabert_'+str(fold_index)+'.pt'
    save_path_best = 'best_wiki_tabert_'+str(fold_index)+'.pt'

    ## training
    with tqdm(total=num_steps) as pbar:

        for epoch in range(NUM_EPOCH):

            epoch_loss = 0
            all_outputs = []
            all_labels = []

            for i, batch in enumerate(train_iter):
                tables,queries,all_tables_meta,all_query_meta,labels = batch
                labels = torch.FloatTensor(labels).to(args.device)
                #print(tables)

                outputs = model(tables,all_tables_meta,queries,all_query_meta)#.to(args.device)

                outputs_prob = m(outputs)
                loss = loss_bce(outputs, labels)
                # print(loss.item())

                epoch_loss += loss.item()

                outputs_prob[outputs_prob < 0.5] = 0
                outputs_prob[outputs_prob >= 0.5] = 1

                all_outputs += outputs_prob.tolist()
                all_labels += labels.tolist()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(1)

            losslogger = epoch_loss / num_batches
            print('done with epoch {}'.format(epoch + 1))

            train_accuracy = sum(1 for x, y in zip(all_outputs, all_labels) if x == y) / len(all_labels)
            all_labels_train = np.array(all_labels)
            all_outputs_train = np.array(all_outputs)
            macro = precision_recall_fscore_support(all_labels_train, all_outputs_train, average='macro')
            micro = precision_recall_fscore_support(all_labels_train, all_outputs_train, average='micro')
            print('training results')
            print('macro: {0} \n micro:{1} \n '.format(macro, micro))

            accuracy_train.append(train_accuracy)
            #print(accuracy_train)

            ## validation
            validation_loss = 0.0
            all_outputs_valid = []
            all_labels_valid = []

            for i, batch in enumerate(valid_iter):
                tables,queries,all_tables_meta,all_query_meta,labels = batch
                labels = torch.FloatTensor(labels).to(args.device)

                outputs = model(tables,all_tables_meta,queries,all_query_meta)#.to(args.device)

                outputs_prob = m(outputs)
                loss = loss_bce(outputs, labels)
                validation_loss += loss.item()
                outputs_prob[outputs_prob < 0.5] = 0
                outputs_prob[outputs_prob >= 0.5] = 1

                all_outputs_valid += outputs_prob.tolist()
                all_labels_valid += labels.tolist()

            lossvalid = validation_loss / num_batches_valid

            valid_accuracy = sum(1 for x, y in zip(all_outputs_valid, all_labels_valid) if x == y) / len(all_labels_valid)
            macro = precision_recall_fscore_support(all_labels_valid, all_outputs_valid, average='macro')
            micro = precision_recall_fscore_support(all_labels_valid, all_outputs_valid, average='micro')

            print('validation results')
            print('macro: {0} \n micro:{1} \n '.format(macro, micro))

            is_best = False
            if valid_accuracy > best_valid_accuracy:
                is_best = True
                best_valid_accuracy = valid_accuracy

            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'losslogger': lossvalid, 'testing_accuracy': best_valid_accuracy}

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

        outputs_prob = m(outputs)
        loss = loss_bce(outputs, labels)
        testing_loss += loss.item()
        outputs_prob[outputs_prob < 0.5] = 0
        outputs_prob[outputs_prob >= 0.5] = 1

        all_outputs_test += outputs_prob.tolist()
        all_labels_test += labels.tolist()

    losstest = testing_loss / num_batches_test

    test_accuracy = sum(1 for x, y in zip(all_outputs_test, all_labels_test) if x == y) / len(all_labels_test)

    all_labels_test = np.array(all_labels_test)
    all_outputs_test = np.array(all_outputs_test)

    macro = precision_recall_fscore_support(all_labels_test, all_outputs_test, average='macro')
    micro = precision_recall_fscore_support(all_labels_test, all_outputs_test, average='micro')
    print('testing results')
    print('macro: {0} \n micro:{1} \n '.format(macro, micro))

    all_folds_accuracy.append(test_accuracy)
    all_folds_p.append(macro[0])
    all_folds_r.append(macro[1])
    all_folds_f.append(macro[2])


print('final results \n')
print('mean accuracy={}'.format(np.mean(all_folds_accuracy)))
print('std accuracy={}'.format(np.std(all_folds_accuracy)))
print('mean macro precision={}'.format(np.mean(all_folds_p)))
print('std macro precision={}'.format(np.std(all_folds_p)))
print('mean macro recall={}'.format(np.mean(all_folds_r)))
print('std macro recall={}'.format(np.std(all_folds_r)))
print('mean macro F={}'.format(np.mean(all_folds_f)))
print('std macro F={}'.format(np.std(all_folds_f)))
