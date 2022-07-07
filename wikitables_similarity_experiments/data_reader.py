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
    def __init__(self,data,data_folder):

        model=BertTokenizer.from_pretrained('bert-base-uncased')
        nlp_model=spacy.load('en_core_web_sm')

        max_tokens=50
        labels = []

        tables1 = []
        tables2 = []

        metas1 = []
        metas2 = []


        with tqdm(total=len(data)) as pbar:
            for index,row in enumerate(data):

                tab1=row[0]
                tab2=row[1]
                label=int(row[2])
                table1,meta1=process_table(tab1, data_folder, model, max_tokens, nlp_model)
                table2, meta2 = process_table(tab2, data_folder, model, max_tokens, nlp_model)

                tables1.append(table1)
                tables2.append(table2)
                metas1.append(meta1)
                metas2.append(meta2)
                labels.append(label)
                
                pbar.update(1)
                    
        self.tables1=tables1
        self.tables2=tables2
        self.metas1 = metas1
        self.metas2 = metas2
        self.labels = labels


    def __getitem__(self, t):

        return self.tables1[t], self.tables2[t], self.metas1[t], self.metas2[t],  self.labels[t]

    def __len__(self):

        return len(self.tables1)


