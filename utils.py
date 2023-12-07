import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import scipy.io as scio
import scipy.io as sio
from preprocesss import read_csv

def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    dat = read_csv(dataset, True)    
    data = dat['data']
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)

    return adj,adj_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):

        if dataset == 'cite' or dataset =='hhar' or dataset =='reut' \
            or dataset =='dblp_for_np' or dataset =='acm_for_np'or dataset =='usps_for_np'or dataset =='reut_for_np':
            data_cite = sio.loadmat('./data/{}.mat'.format(dataset))
            self.x = np.array(data_cite['fea'])
            self.x.astype(np.float64)
            self.y = np.array(data_cite['gnd'])
            self.y.astype(np.int64)
            self.y = self.y[:,-1]
        else:
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


