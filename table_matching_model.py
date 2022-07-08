import torch
import torch.nn as nn
from table_bert import VerticalAttentionTableBert, Table, Column
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import sys
import copy

import logging

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, words_embeddings, token_type_ids=None):
        input_ids = torch.zeros([words_embeddings.shape[0], words_embeddings.shape[1]])
        seq_length = words_embeddings.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=words_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # print(words_embeddings.shape)
        # print(position_embeddings.shape)
        # print(token_type_embeddings.shape)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs.detach().cpu()


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_wights = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_wights


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_weights = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_weights


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_attention_weights = []
        for layer_module in self.layer:
            hidden_states, attention_weights = layer_module(hidden_states, attention_mask)
            all_attention_weights.append(attention_weights)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_attention_weights


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class strubert(nn.Module):

    def __init__(self, tabert_path='/path/to/tabert/model.bin',device='cuda:0'):
        """"Constructor of the class."""
        super(strubert, self).__init__()

        self.embedding = VerticalAttentionTableBert.from_pretrained(
            tabert_path,
        ).to(device)

        config2 = self.embedding.config
        config2.num_hidden_layers = 1
        config2.num_attention_heads = 4
        config2.max_position_embeddings = 50

        self.dense_f = nn.Linear(3072, 1, True).to(device)
        # self.dense_map_text = nn.Linear(768, 100, True).to('cpu')
        # self.dense_map_struct = nn.Linear(768, 100, True).to('cpu')

        self.ranking_embedding = BertEmbeddings(config2).to(device)
        self.cls_c = torch.nn.Parameter(torch.randn(1, 768), requires_grad=True).to(device)
        self.cls_r = torch.nn.Parameter(torch.randn(1, 768), requires_grad=True).to(device)
        self.sep = torch.nn.Parameter(torch.randn(1, 768), requires_grad=True).to(device)

        self.encoder = BertEncoder(config2).to(device)
        self.pooler = BertPooler(config2).to(device)

        self.device = device

    def forward(self, tables1, metas1, tables2, metas2):
        context_encoding1, column_encoding1, _, context_encoding_row1, row_encoding1, _ = self.embedding.encode(
            contexts=metas1,
            tables=tables1
        )

        context_encoding2, column_encoding2, _, context_encoding_row2, row_encoding2, _ = self.embedding.encode(
            contexts=metas2,
            tables=tables2
        )

        output_all_encoded_layers = False

        column_guided_cls1 = context_encoding1[:, 0]
        row_guided_cls1 = context_encoding_row1[:, 0]
        column_guided_cls2 = context_encoding2[:, 0]
        row_guided_cls2 = context_encoding_row2[:, 0]

        column_guided_cls1_column_guided_cls2_feat = column_guided_cls1 * column_guided_cls2
        row_guided_cls1_row_guided_cls2_feat = row_guided_cls1 * row_guided_cls2

        batch_size = row_encoding1.shape[0]
        # print(batch_size)
        num_rows1, num_rows2 = row_encoding1.shape[1], row_encoding2.shape[1]
        num_cols1, num_cols2 = column_encoding1.shape[1], column_encoding2.shape[1]

        extra_info = [num_rows1, num_cols1, num_rows2, num_cols2]

        # print('num of cols in table 1 {}'.format(num_cols1))
        # print('num of cols in table 2 {}'.format(num_cols2))
        # num_cols1, num_cols2 = column_encoding1.shape[1], column_encoding2.shape[1]
        cls_to_add = torch.stack([self.cls_c] * batch_size)
        cls_r_to_add = torch.stack([self.cls_r] * batch_size)
        sep_to_add = torch.stack([self.sep] * batch_size)

        seqa = torch.zeros([batch_size, num_cols1 + 2], dtype=torch.long).to(self.device)
        seqb = torch.ones([batch_size, num_cols2], dtype=torch.long).to(self.device)
        col_seq = torch.cat([cls_to_add, column_encoding1, sep_to_add, column_encoding2], dim=1)
        col_types = torch.cat([seqa, seqb], dim=1)
        col_emb = self.ranking_embedding(col_seq, col_types)
        attention_mask = torch.ones_like(col_types)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers, all_attention_weights_col = self.encoder(col_emb,
                                                                 extended_attention_mask,
                                                                 output_all_encoded_layers=output_all_encoded_layers)
        attention_cols = torch.stack(all_attention_weights_col, dim=1)
        sequence_output = encoded_layers[-1]
        pooled_output_col = self.pooler(sequence_output)

        # print('num of rows in table 1 {}'.format(num_rows1))
        # print('num of rows in table 2 {}'.format(num_rows2))

        seqa = torch.zeros([batch_size, num_rows1 + 2], dtype=torch.long).to(self.device)
        seqb = torch.ones([batch_size, num_rows2], dtype=torch.long).to(self.device)
        row_seq = torch.cat([cls_r_to_add, row_encoding1, sep_to_add, row_encoding2], dim=1)
        # print('size of row_sea {}'.format(row_seq.shape))
        row_types = torch.cat([seqa, seqb], dim=1)
        # print('size of row_types={}'.format(row_types.shape))
        row_emb = self.ranking_embedding(row_seq, row_types)
        attention_mask = torch.ones_like(row_types)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers, all_attention_weights_row = self.encoder(row_emb,
                                                                 extended_attention_mask,
                                                                 output_all_encoded_layers=output_all_encoded_layers)
        attention_rows = torch.stack(all_attention_weights_row, dim=1)
        sequence_output = encoded_layers[-1]
        pooled_output_row = self.pooler(sequence_output)

        final_feat=torch.cat([pooled_output_col,pooled_output_row,column_guided_cls1_column_guided_cls2_feat,row_guided_cls1_row_guided_cls2_feat],dim=1)


        outputs = self.dense_f(final_feat).squeeze(1)

        return outputs
    
class strubert_keyword(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, tabert_path='/path/to/tabert/model.bin', device='cuda:0'):
        """"Constructor of the class."""
        super(strubert_keyword, self).__init__()

        self.embedding = VerticalAttentionTableBert.from_pretrained(
            tabert_path,
        ).to(device)

        config2=self.embedding.config
        config2.num_hidden_layers=1
        config2.num_attention_heads=4
        config2.max_position_embeddings=50


        self.dense_f = nn.Linear(3072, 1, True).to(device)

        self.ranking_embedding = BertEmbeddings(config2).to(device)
        self.cls_c=torch.nn.Parameter(torch.randn(1,768),requires_grad=True).to(device)
        self.cls_r=torch.nn.Parameter(torch.randn(1,768),requires_grad=True).to(device)
        self.sep = torch.nn.Parameter(torch.randn(1, 768), requires_grad=True).to(device)

        self.encoder = BertEncoder(config2).to(device)
        self.pooler = BertPooler(config2).to(device)

        self.device = device




    def forward(self,batch_input,batch_queries):

        context_encoding, column_encoding, _, context_encoding_row, row_encoding, _ = self.embedding.encode(
            contexts=batch_queries,
            tables=batch_input
        )

        output_all_encoded_layers=False

        column_guided_cls = context_encoding[:, 0]
        row_guided_cls=context_encoding_row[:, 0]

        batch_size=row_encoding.shape[0]
        num_rows1 = row_encoding.shape[1]
        num_cols1 = column_encoding.shape[1]

        cls_to_add=torch.stack([self.cls_c]*batch_size)
        cls_r_to_add=torch.stack([self.cls_r]*batch_size)
        sep_to_add = torch.stack([self.sep] * batch_size)

        seqa=torch.zeros([batch_size,num_cols1+2],dtype=torch.long).to(self.device)
        col_seq=torch.cat([cls_to_add,column_encoding,sep_to_add],dim=1)
        col_types=seqa
        col_emb=self.ranking_embedding(col_seq,col_types)
        attention_mask = torch.ones_like(col_types)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers,all_attention_weights_col = self.encoder(col_emb,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output_col = self.pooler(sequence_output)

        seqa = torch.zeros([batch_size, num_rows1 + 2], dtype=torch.long)
        seqa=seqa.to(self.device)
        row_seq = torch.cat([cls_r_to_add, row_encoding, sep_to_add], dim=1)
        row_types = seqa

        row_emb = self.ranking_embedding(row_seq, row_types)
        attention_mask = torch.ones_like(row_types)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers,all_attention_weights_row = self.encoder(row_emb,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output_row = self.pooler(sequence_output)

        log_pooling_sum=torch.cat([pooled_output_col,pooled_output_row,column_guided_cls,row_guided_cls],dim=1)



        outputs=self.dense_f(log_pooling_sum).squeeze(1)


        return outputs

