# StruBERT: Structure-aware BERT for Table Search and Matching 

This repository contains source code for the [`StruBERT` model](https://arxiv.org/abs/2203.14278), a new structure-aware BERT model that is propsoed to solve three table-related downstream tasks: keyword- and content-based table retrieval, and table similarity. `StruBERT` fuses the textual and structural information of a data table to produce four context-aware representations for both textual and tabular content of a data table. Two fine-grained features represent the context-aware embeddings of rows and columns, where both horizontal and vertical attentions are applied over the columnand row-based sequences, respectively. Two coarse-grained features capture the textual information from both row- and columnbased views of a data table. These features are incorporated into a new end-to-end ranking model, called miniBERT, that is formed of one layer of Transformer blocks, and operates directly on the embedding-level sequences formed from StruBERT features to capture the cross-matching signals of rows and columns.

## Installation

First, install the conda environment `strubert` with supporting libraries.

### First installation method

```bash
conda env create --file scripts/env.yml
```

### Second installation method (manually)

```bash
conda create --name strubert python=3.6
conda activate strubert
pip install torch==1.3.1 torchvision -f https://download.pytorch.org/whl/cu100/torch_stable.html
pip install torch-scatter==1.3.2
pip install fairseq==0.8.0
cd scripts
pip install "--editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert" --no-cache-dir
pip install -r requirements.txt
```

## Pre-trained model from TaBERT

download TaBERT_Base_(K=3) from [the TaBERT Google Drive shared folder](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg?usp=sharing). Please uncompress the tarball files before usage.

## Table Similarity experiments

This part is related to using StruBERT for table similarity.

### Data

2 datasets are used for table similarity:

- [`WikiTables` corpus](http://iai.group/downloads/smart_table/WP_tables.zip) contains over 1.6𝑀 tables that are extracted from Wikipedia. Each table has five indexable fields: table caption, attributes (column headings), data rows, page title, and section title. Download and uncrompress the `WikiTables` corpus. We use the same queries that were used by [Zhang and Balog](https://github.com/iai-group/www2018-table), where every query-table pair is evaluated using three numbers: 0 means “irrelevant”, 1 means “partially relevant” and 2 means “relevant”. We iterate over all the queries of WikiTables, and if two tables are relevant to a query, the table pair is given a label 1. On the other hand, an irrelevant table to a query is considered not similar to all tables that are relevant to the query, and therefore the table pair is given a label 0. We provide the 5 fold cross-validation splits with table pairs and binary labels in the source code.
- [`PMC` corpus](https://github.com/Marhabibi/TabSim) is formed from PubMed Central (PMC) Open Access subset, and used for evaluation on the table similarity task. This collection is related to biomedicine and life sciences. Each table contains a caption and data values. We provide the 5 fold cross-validation splits with table pairs and binary labels in the source code.

### WikiTables similarity experiment

To evaluate `StruBERT` on table similarity task for `WikiTables`:

```bash
cd wikitables_similarity_experiments/
python main.py \
 --table_folder path/to/wikitables_corpus
 --tabert_path path/to/pretrained/model/checkpoint.bin
 --device 0
 --epochs 5
 --batch_size 4
 --lr 3e-5
```

### PMC similarity experiment

To evaluate `StruBERT` on table similarity task for `PMC`:

```bash
cd pmc_similarity_experiments/
python main.py \
 --tabert_path path/to/pretrained/model/checkpoint.bin
 --device 0
 --epochs 5
 --batch_size 4
 --lr 3e-5
```

## Content-based table retrieval experiments

This part is related to using StruBERT for content-based table retrieval.

### Data

- `Query by Example Data`: this dataset is composed of 50 Wikipedia tables used as input queries. The query tables are related to multiple topics, and each table has at least five rows and three columns. For the ground truth relevance scores of table pairs, each pair is evaluated using three numbers: 2 means highly relevant and it indicates that the queried table is about the same topic of the query table with additional content, 1 means relevant and it indicates that the queried table contains a content that largely overlaps with the query table, and 0 means irrelevant.

### Query by Example experiment

```bash
cd content_based_table_retrieval/
python main.py \
 --table_folder path/to/wikitables_corpus
 --tabert_path path/to/pretrained/model/checkpoint.bin
 --device 0
 --epochs 5
 --batch_size 4
 --lr 3e-5
 --balance_data
```


## Keyword-based table retrieval experiments

This part is related to using StruBERT for keyword-based table retrieval.

### Data

- `WikiTables` corpus is used for keyword-based table retrieval.

### WikiTables keyword-based table retrieval experiment

```bash
cd keyword_based_table_retrieval/
python main.py \
 --table_folder path/to/wikitables_corpus
 --tabert_path path/to/pretrained/model/checkpoint.bin
 --device 0
 --epochs 5
 --batch_size 4
 --lr 3e-5
```

## Reference

If you plan to use `StruBERT` in your project, please consider citing [our paper](https://arxiv.org/abs/2203.14278):

```bash
@inproceedings{trabelsi22www,
author = {Trabelsi, Mohamed and Chen, Zhiyu and Zhang, Shuo and Davison, Brian D and Heflin, Jeff},
title = {Stru{BERT}: Structure-aware BERT for Table Search and Matching},
year = {2022},
booktitle = {Proceedings of the ACM Web Conference},
numpages = {10},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}
```
 ## Contact
  
 if you have any questions, please contact Mohamed Trabelsi at mot218@lehigh.edu
 
## Acknowledgements

TaBERT folders (preprocess, table_bert) are important parts of StruBERT, and are initially downloaded from [`TaBERT`](https://github.com/facebookresearch/TaBERT).
