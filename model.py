import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_pretrained_bert.modeling import BertModel

import config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """
    def __init__(self):
        super(Net, self).__init__()
        vision_features = config.output_features
        question_features = config.question_features
        # self.text = BertTextProcessor()
        
        self.text = TextProcessor(
            embedding_tokens=30522,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        
        self.attention = Attention2D(
            dim1=question_features,
            dim2=vision_features,
            att_dim=512
        )
        
        # self.attention = Attention1D(
        #     ques_dim=question_features,
        #     vis_dim=vision_features,
        #     att_dim=512,
        #     drop=0.5
        # )

        self.classifier = Classifier(
            in_features=vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q_inputs, q_masks):
        q = self.text(q_inputs, q_masks)
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        attention_mask = ((1.0 - q_masks) * -10000).float()
        v = v.view(v.shape[0], v.shape[1], v.shape[2] * v.shape[3]).transpose(1,2)
        # v_attended = self.attention(q, v)
        # combined = torch.cat([q, v_attended], dim=1)
        q_attended, v_attended = self.attention(q, v, attention_mask)
        combined = torch.cat([q_attended, v_attended], dim=1)
        answer = self.classifier(combined)
        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1,
                            batch_first=True)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_mask):
        q_len = q_mask.sum(dim=1)
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        # packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        h, (_, c) = self.lstm(tanhed)
        # unpacked, unpacked_len = pad_packed_sequence(h, batch_first=True)
        #return c.squeeze(0)
        return h.contiguous()

class BertTextProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bertModel = BertModel.from_pretrained(config.bert_model)
    def forward(self, q_input_ids, q_input_mask):
        all_encoder_layers, pooled_output = self.bertModel(q_input_ids, token_type_ids=None, attention_mask=q_input_mask, output_all_encoded_layers=False)
        return all_encoder_layers

class Attention1D(nn.Module):
    def __init__(self, ques_dim, vis_dim, att_dim, drop=0.0):
        super().__init__()
        self.ques_dim = ques_dim
        self.vis_dim = vis_dim
        self.att_dim = att_dim
        self.vis2att = nn.Linear(vis_dim, att_dim, bias=False)
        self.ques2att = nn.Linear(ques_dim, att_dim, bias=True)
        self.att2map = nn.Linear(att_dim, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, X, Y):
        """
        Input:
        X (param, queries) : batch_size x ques_dim
        Y (param, contexts) : batch_size x n x vis_dim
        attention_mask (for X): batch_size x m
        Returns:
        Y_attended: batch_size x dim2
        """
        bs, dim1 = X.shape
        bs, n, dim2 = Y.shape
        X2att = self.ques2att(self.drop(X))
        Y2att = self.vis2att(self.drop(Y.contiguous().view(bs * n, self.vis_dim)))
        Y2att = Y2att.view(bs, n, self.att_dim)
        att = self.relu(X2att.unsqueeze(1).expand_as(Y2att) + Y2att)
        Y_att_logits = self.att2map(self.drop(att).view(bs * n, self.att_dim)).view(bs, n)
        Y_attention_weights = self.softmax(Y_att_logits)
        Y_attended = (Y_attention_weights.unsqueeze(2).expand_as(Y) * Y).sum(dim=1)
        return Y_attended

#let's make a simplistic 2D attention mechanism first, uses max activation
class Attention2D(nn.Module):
    def __init__(self, dim1, dim2, att_dim):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.att_dim = att_dim
        self.W_question = nn.Linear(dim1, att_dim, bias=True)
        self.W_image = nn.Linear(dim2, att_dim, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, X, Y, attention_mask):
        """
        Input:
        X (param, queries) : batch_size x m x dim1 
        Y (param, contexts) : batch_size x n x dim2
        attention_mask (for X): batch_size x m
        Returns:
        X_attended: batch_size x dim1
        Y_attended: batch_size x dim2
        """
        bs, m, dim1 = X.shape
        bs, n, dim2 = Y.shape
        X_att = self.W_question(X.view(bs * m, dim1))
        X_att = X_att.view(bs, m, self.att_dim)
        Y_att = self.W_image(Y.contiguous().view(bs * n, dim2))
        Y_att = Y_att.view(bs, n, self.att_dim).transpose(1,2)
        scores = torch.bmm(X_att, Y_att)/np.sqrt(self.att_dim)

        pool_X, _ = scores.max(dim=2)
        masked_pool_X = pool_X + attention_mask
        pool_Y, _ = scores.max(dim=1)
        X_attention_weights = self.softmax(masked_pool_X)
        Y_attention_weights = self.softmax(pool_Y)
        X_attended = (X_attention_weights.unsqueeze(2).expand_as(X) * X).sum(dim=1)
        Y_attended = (Y_attention_weights.unsqueeze(2).expand_as(Y) * Y).sum(dim=1)
        return X_attended, Y_attended


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
