import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.utils.data
from scipy import sparse as sp
import time
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
class PointDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        """
            Loading Superpixels datasets
        """
        start = time.time()

        self.data_path = path

        with open(self.data_path, "rb") as f:  ##################0000000000000000
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        train_labels = [self.train.graph_labels[i].item() for i in range(len(self.train.graph_labels))]
        # ori_count = Counter(train_labels)
        training_set_size = 0.9

        train, val, _, __ = train_test_split(self.train,
                                             range(len(self.train.graph_lists)),
                                             test_size=1 - training_set_size,
                                             stratify=train_labels)

        self.train = self.format_dataset(train)
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        # print('train, test :', len(self.train),  len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, adjs  = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e