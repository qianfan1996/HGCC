import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from models.gat import GAT
from models.StoG import CapsuleSequenceToGraph
from models.aggregator import MeanAggregator, MaxAggregator, AttentionAggregator, MultimodalGraphReadout


def get_padding_mask(lengths, max_len=None):
    bsz = len(lengths)
    if not max_len:
        max_len = lengths.max()
    mask = torch.zeros((bsz, max_len))
    for i in range(bsz):
        index = torch.arange(int(lengths[i].item()), max_len)
        mask[i] = mask[i].index_fill_(0, index, -1e9)

    return mask


class MultimodalGraphFusionNetwork(nn.Module):
    def __init__(self, config):
        super(MultimodalGraphFusionNetwork, self).__init__()
        dt, da, dv = config["t_size"], config["a_size"], config["v_size"]
        h = config["multi_dim"]

        self.config = config

        self.encoder_t = self._get_encoder(modality='t')
        self.encoder_v = self._get_encoder(modality='v')
        self.encoder_a = self._get_encoder(modality='a')

        self.stog = CapsuleSequenceToGraph(text_dim=dt,
                                           audio_dim=2*da,
                                           video_dim=2*dv,
                                           capsule_dim=config['capsule_dim'],
                                           capsule_num=config['capsule_num'],
                                           routing=config['routing'],
                                           multi_dim=config['multi_dim'],
                                           T_t=config['T_t'],
                                           T_a=config['T_a'],
                                           T_v=config['T_v'])

        self.gat_t = GAT(input_dim=config['capsule_dim'],
                         gnn_dim=config['capsule_dim'] // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                         leaky_alpha=0.2)
        self.gat_v = GAT(input_dim=config['capsule_dim'],
                         gnn_dim=config['capsule_dim'] // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                         leaky_alpha=0.2)
        self.gat_a = GAT(input_dim=config['capsule_dim'],
                         gnn_dim=config['capsule_dim'] // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                         leaky_alpha=0.2)
        self.gat_m = GAT(input_dim=h,
                         gnn_dim=h // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                         leaky_alpha=0.2)

        self.readout_t = AttentionAggregator(config['capsule_dim'])
        self.readout_v = AttentionAggregator(config['capsule_dim'])
        self.readout_a = AttentionAggregator(config['capsule_dim'])
        self.readout_m = AttentionAggregator(h)
        # self.readout_m = MultimodalGraphReadout(m_dim, self.readout_t, self.readout_v, self.readout_a)

        self.fc_out = nn.Linear(h+3*config['capsule_dim'], 1)
        self.dropout_m = nn.Dropout(config["dropout"])
        self.dropout_t = nn.Dropout(config["dropout_t"])
        self.dropout_v = nn.Dropout(config["dropout_v"])
        self.dropout_a = nn.Dropout(config["dropout_a"])

    def _get_encoder(self, modality='t'):
        if modality == 't':
            return BertModel.from_pretrained(self.config["bert_path"])
        elif modality == 'v':
            return nn.LSTM(self.config["v_size"], self.config["v_size"], batch_first=True, bidirectional=True)
        elif modality == 'a':
            return nn.LSTM(self.config["a_size"], self.config["a_size"], batch_first=True, bidirectional=True)
        else:
            raise ValueError('modality should be t or v or a!')

    def _delete_edge(self, adj, ratio=0.1):
        bsz = adj.size(0)
        adj_del = adj.clone().contiguous().view(bsz, -1)
        for i in range(bsz):
            edges = torch.where(adj_del[i] > 0)[0]
            del_num = math.ceil(len(edges) * ratio)
            del_edges = random.sample(edges.cpu().numpy().tolist(), del_num)
            adj_del[i, del_edges] = 0
        adj_del = adj_del.reshape_as(adj)
        return adj_del

    def _add_edge(self, adj, ratio=0.1):
        bsz = adj.size(0)
        adj_add = adj.clone().contiguous().view(bsz, -1)
        for i in range(bsz):
            non_edges = torch.where(adj_add[i] >= 0)[0]
            add_num = math.ceil(len(non_edges) * ratio)
            add_edges = random.sample(non_edges.cpu().numpy().tolist(), add_num)
            adj_add[i, add_edges] = 1
        adj_add = adj_add.reshape_as(adj)
        return adj_add

    def forward(self, text, video, audio, bert_sent_mask):
        bsz, max_len, _ = video.size()

        # encode
        bert_output = self.encoder_t(input_ids=text, attention_mask=bert_sent_mask)
        hs_t = bert_output[0]
        # hs_t = bert_output[0][:, 1:-1]

        hs_a, _ = self.encoder_a(audio)

        hs_v, _ = self.encoder_v(video)

        text_vertex, audio_vertex, video_vertex, multi_vertex, adj_t, adj_a, adj_v, adj_m = self.stog(hs_t.transpose(0, 1), hs_a.transpose(0, 1), hs_v.transpose(0, 1), bsz)

        # multimodal graph
        hs_gnn = self.gat_m(multi_vertex, adj_m)
        hs_gnn = F.relu(hs_gnn + multi_vertex)
        reps_m = self.readout_m(hs_gnn)

        # unimodal graphs
        hs_t_gnn = self.gat_t(text_vertex, adj_t)
        hs_v_gnn = self.gat_v(video_vertex, adj_v)
        hs_a_gnn = self.gat_a(audio_vertex, adj_a)
        hs_t_gnn = F.relu(hs_t_gnn + text_vertex)
        hs_v_gnn = F.relu(hs_v_gnn + video_vertex)
        hs_a_gnn = F.relu(hs_a_gnn + audio_vertex)
        reps_t = self.readout_t(hs_t_gnn)
        reps_v = self.readout_v(hs_v_gnn)
        reps_a = self.readout_a(hs_a_gnn)

        output = self.fc_out(self.dropout_m(torch.cat((reps_m, reps_t, reps_a, reps_v), dim=-1)))

        # augmentation
        adj_matrix_aug1 = self._delete_edge(adj_m, self.config["aug_ratio"])
        adj_matrix_aug1 = self._add_edge(adj_matrix_aug1, self.config["aug_ratio"])
        hs_gnn_aug1 = self.gat_m(multi_vertex, adj_matrix_aug1)
        hs_gnn_aug1 = F.relu(hs_gnn_aug1 + multi_vertex)
        reps_m_aug1 = self.readout_m(hs_gnn_aug1)

        adj_matrix_t_aug1 = self._delete_edge(adj_t, self.config["aug_ratio"])
        adj_matrix_v_aug1 = self._delete_edge(adj_v, self.config["aug_ratio"])
        adj_matrix_a_aug1 = self._delete_edge(adj_a, self.config["aug_ratio"])
        adj_matrix_t_aug1 = self._add_edge(adj_matrix_t_aug1, self.config["aug_ratio"])
        adj_matrix_v_aug1 = self._add_edge(adj_matrix_v_aug1, self.config["aug_ratio"])
        adj_matrix_a_aug1 = self._add_edge(adj_matrix_a_aug1, self.config["aug_ratio"])
        hs_t_gnn_aug1 = self.gat_t(text_vertex, adj_matrix_t_aug1)
        hs_v_gnn_aug1 = self.gat_v(video_vertex, adj_matrix_v_aug1)
        hs_a_gnn_aug1 = self.gat_a(audio_vertex, adj_matrix_a_aug1)
        hs_t_gnn_aug1 = F.relu(hs_t_gnn_aug1 + text_vertex)
        hs_v_gnn_aug1 = F.relu(hs_v_gnn_aug1 + video_vertex)
        hs_a_gnn_aug1 = F.relu(hs_a_gnn_aug1 + audio_vertex)
        reps_t_aug1 = self.readout_t(hs_t_gnn_aug1)
        reps_v_aug1 = self.readout_v(hs_v_gnn_aug1)
        reps_a_aug1 = self.readout_a(hs_a_gnn_aug1)

        reps_t_aug = torch.stack([reps_t_aug1, reps_t], dim=1)
        reps_v_aug = torch.stack([reps_v_aug1, reps_v], dim=1)
        reps_a_aug = torch.stack([reps_a_aug1, reps_a], dim=1)
        reps_m_aug = torch.stack([reps_m_aug1, reps_m], dim=1)

        return output.view(-1), F.normalize(reps_m.unsqueeze(1), dim=-1), F.normalize(reps_t.unsqueeze(1), dim=-1), \
               F.normalize(reps_v.unsqueeze(1), dim=-1), F.normalize(reps_a.unsqueeze(1), dim=-1),\
               F.normalize(reps_m_aug, dim=-1), F.normalize(reps_t_aug, dim=-1), F.normalize(reps_v_aug, dim=-1),\
               F.normalize(reps_a_aug, dim=-1)