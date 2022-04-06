from collections import Counter
import torch
from torch.utils.data import Dataset
from table_bert import VerticalAttentionTableBert,Table, Column
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocess import data_utils
import numpy as np
import torch.nn.functional as F
import pandas as pd
import subprocess
import random
import argparse
import json
import sys
import os
from pathlib import Path
import math
from tqdm import tqdm

import argparse
import random
import re
from collections import OrderedDict

import spacy
import shutil

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, val_idx)

    return train_set,valid_set

def pad_table_query(batch):

    f = lambda x: [sample[x] for sample in batch]

    tables = f(0)
    queries = f(1)
    all_tables_meta = f(2)
    all_query_meta = f(3)
    labels = f(4)
    labels = torch.FloatTensor(labels)

    return tables,queries,all_tables_meta,all_query_meta,labels


def pad_table_search(batch):

    f = lambda x: [sample[x] for sample in batch]

    tables = f(0)
    queries = f(1)
    labels = f(2)
    labels = torch.FloatTensor(labels)

    return tables,queries,labels

def annotate_schema(data_values, attributes, nlp_model=None):
    # assume the first row is table header
    types = []
    content_rows = data_values
    for col_ids, col_name in enumerate(attributes):
        sample_value = None
        for row in content_rows:
            cell_val = row[col_ids]
            if cell_val=='empty_cell':
                continue
            if len(cell_val.strip()) > 0:
                sample_value = cell_val
                break

        if sample_value is None:
            sample_value='empty_cell'
        sample_value_entry = {
            'value': sample_value,
        }

        assert sample_value is not None
        if nlp_model and sample_value:
            annotation = nlp_model(sample_value)
            tokenized_value = [token.text for token in annotation]
            ner_tags = [token.ent_type_ for token in annotation]
            pos_tags = [token.pos_ for token in annotation]

            sample_value_entry.update({
                'tokens': tokenized_value,
                'ner_tags': ner_tags,
                'pos_tags': pos_tags
            })

        col_type = data_utils.infer_column_type_from_sampled_value(sample_value_entry)

        types.append(col_type)

    return types

def split_token(input):
    input = input.split('|')
    if len(input) > 1:
        if len(input[0]) >= len(input[1]):
            res=input[0][1:]
            res=res.replace('_',' ')
            res = re.sub(r'[^\x00-\x7F]+','-', res)
            return res
        else:
            res=input[1][:-1]
            res = res.replace('_', ' ')
            res = re.sub(r'[^\x00-\x7F]+','-', res)
            return res
    else:
        return re.sub(r'[^\x00-\x7F]+','-', input[0])


def process_table(table_input,data_folder,model,max_tokens,nlp_model):
    inter = table_input.split("-")
    file_number = inter[1]
    table_number = inter[2]

    file_name = 're_tables-' + file_number + '.json'

    table_name = 'table-' + file_number + '-' + table_number

    path = os.path.join(data_folder, file_name)

    with open(path) as f:
        tab_dt = json.load(f)

    test_table = tab_dt[table_name]
    attributes = test_table['title']
    headers = []
    # for att in attributes:
    #    headers.append(Column(att,'text'))

    pgTitle = test_table['pgTitle']
    pgTitle = model.tokenize(pgTitle)[:max_tokens]

    secondTitle = test_table['secondTitle']
    secondTitle = model.tokenize(secondTitle)[:max_tokens]

    caption = test_table['caption']
    caption = model.tokenize(caption)[:max_tokens]

    data = test_table['data']
    data_rc = pd.DataFrame(data, columns=attributes)
    data_rc = data_rc.applymap(split_token)

    data_rc = data_rc.replace(r'^\s*$', np.nan, regex=True)
    data_rc.fillna('empty_cell', inplace=True)

    values_struct = data_rc.values.tolist()
    if len(attributes) == 0:
        headers.append(Column('header0,', 'text'))
        # headers=['header0']
        values_struct = [['empty_cell']]

    else:
        attributes = [split_token(att) for att in attributes]
        if len(values_struct) == 0:
            values_struct = [['empty_cell' for _ in range(len(attributes))]]
            headers = [Column(att, 'text') for att in attributes]

        else:
            types = annotate_schema(values_struct, attributes, nlp_model)
            for col_index, att in enumerate(attributes):
                headers.append(Column(att, types[col_index]))

    table = Table(id=table_input, header=headers, data=values_struct).tokenize(model)

    attributes = model.tokenize(' '.join(attributes))[:max_tokens]

    vector_query_w = pgTitle + ['[SEP]'] + secondTitle + ['[SEP]'] + caption + ['[SEP]'] + attributes
    ql=len(pgTitle)
    table.ql=ql

    return table,vector_query_w


def process_table_pmc(table_input,meta,model,max_tokens,nlp_model):


    attributes = table_input[0]
    headers = []

    caption = model.tokenize(meta)[:max_tokens]

    data = table_input[1:]
    data_rc = pd.DataFrame(data, columns=attributes)

    data_rc = data_rc.replace(r'^\s*$', np.nan, regex=True)
    data_rc.fillna('empty_cell', inplace=True)

    values_struct = data_rc.values.tolist()
    if len(attributes) == 0:
        headers.append(Column('header0,', 'text'))
        # headers=['header0']
        values_struct = [['empty_cell']]

    else:

        if len(values_struct) == 0:
            values_struct = [['empty_cell' for _ in range(len(attributes))]]
            headers = [Column(att, 'text') for att in attributes]

        else:
            types = annotate_schema(values_struct, attributes, nlp_model)
            for col_index, att in enumerate(attributes):
                headers.append(Column(att, types[col_index]))

    table = Table(id='tab', header=headers, data=values_struct).tokenize(model)

    attributes = model.tokenize(' '.join(attributes))[:max_tokens]

    vector_query_w = caption + ['[SEP]'] + attributes
    table.ql=len(caption)

    return table,vector_query_w

def read_file_for_nfcg(file):
    text_file = open(file, "r")
    lines = text_file.readlines()

    queries_id = []
    list_lines = []

    for line in lines:
        # print(line)
        line = line[0:len(line) - 1]
        aa = line.split('\t')
        queries_id += [aa[0]]
        list_lines.append(aa)
    inter = np.array(list_lines)

    return inter


def calculate_metrics(inter, output_file,all_outputs,ndcg_file):
    inter2 = []

    for jj, item in enumerate(inter):
        item_inter = [i for i in item]
        item_inter[4] = str(all_outputs[jj])

        inter2.append(item_inter)

    inter3 = np.array(inter2)

    np.savetxt(output_file, inter3, fmt="%s")

    #batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    batcmd = "./trec_eval -m map " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    map = float(res[2])

    batcmd = "./trec_eval -m recip_rank " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    mrr = float(res[2])

    batcmd = "./trec_eval -m ndcg_cut.5 " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg,map,mrr




def calculate_ndcg(inter, output_file,all_outputs,ndcg_file):
    inter2 = []

    for jj, item in enumerate(inter):
        item_inter = [i for i in item]
        item_inter[4] = str(all_outputs[jj])

        inter2.append(item_inter)

    inter3 = np.array(inter2)

    np.savetxt(output_file, inter3, fmt="%s")

    batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg

def qrel_for_data(data,list_lines_qrels,output_file):
    #list_lines_qrels=np.array(list_lines_qrels)
    df = pd.DataFrame(list_lines_qrels)
    qrel_inter=[]
    for i in range(len(data)):
        row=data[i]
        ii=df[((df[0] == row[0]) & (df[2] == row[2]))]
        qrel_inter+=ii.values.tolist()

    qrel_inter=np.array(qrel_inter)

    np.savetxt(output_file, qrel_inter, fmt="%s",delimiter='\t')

def listnet_loss(y_i, z_i):
    """
    y_i: (n_i, 1)
    z_i: (n_i, 1)
    """

    P_y_i = F.softmax(y_i, dim=0)
    P_z_i = F.softmax(z_i, dim=0)
    return - torch.sum(y_i * torch.log(P_z_i))


def save_checkpoint(checkpoint, is_best, bestmodel):
    if is_best:
        shutil.copyfile(checkpoint, bestmodel)

def load_checkpoint(model, optimizer, losslogger, filename,testing_accuracy,start_epoch):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        testing_accuracy = checkpoint['testing_accuracy']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger,testing_accuracy

def kernal_mus(n_kernels):

    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):

    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

def load_checkpoint_for_eval(model, filename):

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        model=model.eval()

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model