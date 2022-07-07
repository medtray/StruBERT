from collections import Counter
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from table_bert import VerticalAttentionTableBert,Table, Column
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocess import data_utils

import argparse
import random
import re
from collections import OrderedDict
import spacy
from utils import process_table


class DataAndQueryReader(Dataset):
    def __init__(self,file_name,output_file,data_folder,id_to_queries):

        model=BertTokenizer.from_pretrained('bert-base-uncased')
        nlp_model=spacy.load('en_core_web_sm')

        max_tokens=50
        labels = []

        all_tables = []
        all_tables_meta=[]
        all_query = []
        all_query_meta=[]


        text_file = open(file_name, "r")
        # text_file = open("ranking_results/train.txt", "r")
        lines = text_file.readlines()

        queries_id = []
        list_lines = []
        to_save=[]

        processed_tables={}

        for line in lines:
            # print(line)
            line = line[0:len(line) - 1]
            aa = line.split('\t')
            queries_id += [aa[0]]
            list_lines.append(aa)

        for index,line in enumerate(list_lines):

            query=line[0]
            query=id_to_queries[query]
            tab=line[2]
            rel=int(line[4])

            if query in processed_tables:
                proc_query_tab=processed_tables[query]['table']
                proc_query_meta=processed_tables[query]['meta']

            else:
                proc_query_tab,proc_query_meta=process_table(query, data_folder, model, max_tokens, nlp_model)
                processed_tables[query]={}
                processed_tables[query]['table']=proc_query_tab
                processed_tables[query]['meta']=proc_query_meta

            if tab in processed_tables:
                proc_tab = processed_tables[tab]['table']
                proc_meta = processed_tables[tab]['meta']

            else:
                proc_tab, proc_meta = process_table(tab, data_folder, model, max_tokens, nlp_model)
                processed_tables[tab] = {}
                processed_tables[tab]['table'] = proc_tab
                processed_tables[tab]['meta'] = proc_meta



            labels.append(rel)
            all_tables.append(proc_tab)
            all_tables_meta.append(proc_meta)
            all_query.append(proc_query_tab)
            all_query_meta.append(proc_query_meta)
            to_save.append(line)


        self.all_tables=all_tables
        self.all_tables_meta = all_tables_meta
        self.all_query=all_query
        self.all_query_meta = all_query_meta

        self.labels = labels
        inter = np.array(to_save)
        np.savetxt(output_file, inter, fmt="%s", delimiter='\t')

    def __getitem__(self, t):

        return self.all_tables[t], self.all_query[t], self.all_tables_meta[t], self.all_query_meta[t], self.labels[t]

    def __len__(self):
        
        return len(self.all_tables)


