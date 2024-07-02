import torch
import torch.nn as nn
from models.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .transformer import TransformerBlock
import torch.nn.functional as F
import dgl
import numpy as np
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout
import matplotlib.pyplot as plt
# device = torch.device("cuda")
def create_model(cfgs,device0):
    """ 构建模型 """
    model = PointTransformerCls(

        npoints=cfgs['STG']['npoints'],  # node_dim (feat is an integer)
        nblocks=cfgs['STG']['nblocks'],
        nneighbor=cfgs['STG']['nneighbor'],
        n_c=cfgs['STG']['n_c'],
        d_points=cfgs['STG']['d_points'],
        out_dim = cfgs['STG']['out_dim'],
        n_classes = cfgs['STG']['num_class'],
        in_dim_node=cfgs['STG']['in_dim_node'],  # node_dim (feat is an integer)
        hidden_dim=cfgs['STG']['hidden_dim'],
        out_dim_g=cfgs['STG']['out_dim_g'],
        num_heads=cfgs['STG']['num_heads'],
        in_feat_dropout=cfgs['STG']['in_feat_dropout'],
        dropout=cfgs['STG']['dropout'],
        n_layers=cfgs['STG']['n_layers'],
        layer_norm=cfgs['STG']['layer_norm'],
        batch_norm=cfgs['STG']['batch_norm'],
        lap_pos_enc=cfgs['STG']['lap_pos_enc'],
        wl_pos_enc=cfgs['STG']['wl_pos_enc'],
        residual=cfgs['STG']['residual'],
        readout=cfgs['STG']['readout'],
        sts_dim=cfgs['STG']['sts_dim'],
        stg_dim=cfgs['STG']['stg_dim'],
        causal_k=cfgs['STG']['causal_k'],
        device = device0

    )
    return model


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)

    # def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class Backbone(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points):
        super().__init__()
        # npoints, nblocks, nneighbor, n_c, d_points = 1024, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )
        self.transformer1 = TransformerBlock(8, 512, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 8 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 2 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, 512, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points, out_dim, in_dim_node, sts_dim,
                 stg_dim, hidden_dim, in_feat_dropout, num_heads, dropout, n_layers, out_dim_g, n_classes,
                 layer_norm, batch_norm, residual, readout, lap_pos_enc, wl_pos_enc, causal_k, device):
        super().__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, n_c, d_points)
        # npoints, nblocks, nneighbor, n_c, d_points = 1024, 4, 16, 40, 6
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
        self.readout = readout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = device
        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        self.causal_k = causal_k
        max_wl_role_index = 100

        # if self.lap_pos_enc:
        #     pos_enc_dim = net_params['pos_enc_dim']
        #     self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                                           dropout, self.layer_norm, self.batch_norm, self.residual) for
                                     _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim_g, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.weight_matrix1 = nn.Linear(out_dim, stg_dim)
        self.weight_matrix2 = nn.Linear(out_dim_g, sts_dim)

        self.weight_matrix1c = nn.Linear(stg_dim + sts_dim, self.causal_k)
        self.weight_matrix2c = nn.Linear(stg_dim + sts_dim, self.causal_k)

        self.MLP_layer = MLPReadout((stg_dim + sts_dim) * 2, n_classes)
        self.bn = nn.BatchNorm1d((stg_dim + sts_dim) * 2)
    def forward(self, x, g, h, cf, cn, h_lap_pos_enc=None, h_wl_pos_enc=None):
        points, _ = self.backbone(x)
        points = torch.squeeze(points, dim=1)
        h = h.unsqueeze(1)
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        h = h.to(self.device)
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        out = torch.cat((self.weight_matrix1(points), self.weight_matrix2(hg)), dim=1)
        # out = torch.cat((points, hg), dim=1)
        sumc = 0
        wi = torch.matmul(self.weight_matrix1c(out), self.weight_matrix2c(cf).T) / np.sqrt(self.causal_k)
        ai = F.log_softmax(wi, dim=1)
        for i in range(self.causal_k):
            sumc += ai[:, i].unsqueeze(1) * cf[i].unsqueeze(0) * cn[i]

        feature1_min = out.min()
        feature1_max = out.max()
        out = (out - feature1_min) / (feature1_max - feature1_min)

        feature2_min = sumc.min()
        feature2_max = sumc.max()
        sumc = (sumc - feature2_min) / (feature2_max - feature2_min)

        outc0 = torch.cat((out, sumc), dim=1)
        outc0 = self.bn(outc0)
        outc = self.MLP_layer(outc0)


        # outc = outc.abs()
        outc = F.log_softmax(outc, dim=1)
        # res = self.fc2(points.mean(1))
        return outc

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def accuracy_MNIST_CIFAR(self, scores, targets):
        scores = scores.detach().argmax(dim=1)
        acc = (scores == targets).float().sum().item()
        return acc




