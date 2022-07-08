import sys
from collections import Counter
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from table_bert import Table, Column
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer

import nltk
import spacy
from utils import split_token,annotate_schema


class DataAndQueryReader(Dataset):
    def __init__(self,file_name,output_file,base,data_folder):

        model=BertTokenizer.from_pretrained('bert-base-uncased')
        nlp_model=spacy.load('en_core_web_sm')
        max_tokens=50


        labels = []

        all_tables = []
        all_queries = []

        data_csv = pd.read_csv(os.path.join(base, 'features2.csv'))

        test_data = data_csv['table_id']
        query = data_csv['query']

        text_file = open(file_name, "r")
        lines = text_file.readlines()

        queries_id = []
        list_lines = []

        for line in lines:
            # print(line)
            line = line[0:len(line) - 1]
            aa = line.split('\t')
            queries_id += [aa[0]]
            list_lines.append(aa)

        queries_id = [int(i) for i in queries_id]

        qq = np.sort(list(set(queries_id)))

        test_data = list(test_data)

        to_save = []
        all_query_labels = []

        for q in qq:
            # print(q)
            #if q>2:
            #    break
            indexes = [i for i, x in enumerate(queries_id) if x == q]
            indices = data_csv[data_csv['query_id'] == q].index.tolist()
            # print(indexes)

            inter = np.array(list_lines)[indexes]

            test_query = list(query[indices])[0]

            test_query=test_query.lower()
            vector_query_only_w=model.tokenize(test_query)
            ql=len(vector_query_only_w)

            for item in inter:
                if item[2] in test_data:
                    all_query_labels.append(q)
                    rel = float(
                        data_csv[((data_csv['query_id'] == q) & (data_csv['table_id'] == item[2]))].iloc[0]['rel'])

                    inter = item[2].split("-")
                    file_number = inter[1]
                    table_number = inter[2]

                    file_name = 're_tables-' + file_number + '.json'

                    table_name = 'table-' + file_number + '-' + table_number

                    path = os.path.join(data_folder, file_name)

                    with open(path) as f:
                        tab_dt = json.load(f)

                    test_table = tab_dt[table_name]
                    attributes = test_table['title']
                    headers=[]

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

                    values_struct=data_rc.values.tolist()

                    if len(attributes)==0:
                        headers.append(Column('header0,','text'))
                        #headers=['header0']
                        values_struct=[['empty_cell']]

                    else:
                        attributes = [split_token(att) for att in attributes]
                        if len(values_struct)==0:
                            values_struct=[['empty_cell' for _ in range(len(attributes))]]
                            headers=[Column(att,'text') for att in attributes]

                        else:
                            types=annotate_schema(values_struct,attributes,nlp_model)
                            for col_index,att in enumerate(attributes):
                                headers.append(Column(att,types[col_index]))

                    table = Table(id=item[2], header=headers, data=values_struct).tokenize(model)
                    table.ql=ql
                    attributes = model.tokenize(' '.join(attributes))[:max_tokens]
                    #vector_query_w = vector_query_only_w + ['[SEP]'] + pgTitle + ['[SEP]'] + secondTitle + ['[SEP]'] + caption + ['[SEP]'] + attributes
                    vector_query_w = vector_query_only_w + ['[SEP]'] + pgTitle + ['[SEP]'] + secondTitle + ['[SEP]'] + caption
                    #vector_query_w = vector_query_only_w + vector_query_only_w + ['[SEP]'] +  caption

                    all_tables.append(table)
                    all_queries.append(vector_query_w)
                    labels.append(rel)
                    to_save.append(item)



        self.all_tables=all_tables
        self.all_queries=all_queries

        self.all_query_labels = all_query_labels
        self.labels = labels
        inter = np.array(to_save)
        np.savetxt(output_file, inter, fmt="%s", delimiter='\t')

    def __getitem__(self, t):

        return self.all_tables[t], self.all_queries[t], self.labels[t]

    def __len__(self):

        return len(self.all_tables)


