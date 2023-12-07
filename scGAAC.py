from __future__ import print_function, division
import argparse
import random
import numpy as np
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from datetime import datetime
import time
import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from preprocesss import read_csv
from layer import GATLayer
from sklearn.preprocessing import normalize


tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by AE
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))

        z = self.z_layer(enc_z4)

        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z


# ADF
class ADF_L(nn.Module):
    def __init__(self, n_adf):
        super(ADF_L, self).__init__()
        self.wl = Linear(n_adf, 5)

    def forward(self, adf_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(adf_in)), dim=1)

        return weight_output


class ADF_1(nn.Module):
    def __init__(self, n_adf):
        super(ADF_1, self).__init__()
        self.w1 = Linear(n_adf, 2)

    def forward(self, adf_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(adf_in)), dim=1)

        return weight_output


class ADF_2(nn.Module):
    def __init__(self, n_adf):
        super(ADF_2, self).__init__()
        self.w2 = Linear(n_adf, 2)

    def forward(self, adf_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(adf_in)), dim=1)

        return weight_output


class ADF_3(nn.Module):
    def __init__(self, n_adf):
        super(ADF_3, self).__init__()
        self.w3 = Linear(n_adf, 2)

    def forward(self, adf_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(adf_in)), dim=1)

        return weight_output


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class scGAAC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, v=1, alpha=0.2):
        super(scGAAC, self).__init__()

        # AE
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        #GATE
        self.gate_0 = GATLayer(n_input, n_enc_1, alpha)
        self.gate_1 = GATLayer(n_enc_1, n_enc_2, alpha)
        self.gate_2 = GATLayer(n_enc_2, n_enc_3, alpha)
        self.gate_3 = GATLayer(n_enc_3, n_z, alpha)

        self.gate_z = GNNLayer(3020, n_clusters)  # GNN

        self.dec_1 = GATLayer(n_z, n_enc_3, alpha)
        self.dec_2 = GATLayer(n_enc_3, n_enc_2, alpha)
        self.dec_3 = GATLayer(n_enc_2, n_input, alpha)

        self.z = GNNLayer(n_z, n_clusters)

        self.FL = ADF_L(3020)

        # attention on [Z_i || H_i]
        self.F1 = ADF_1(2 * n_enc_1)
        self.F2 = ADF_2(2 * n_enc_2)
        self.F3 = ADF_3(2 * n_enc_3)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

        self.n_clusters = n_clusters


    def forward(self, x, adj, M):
        # AE Module
        x_bar, h1, h2, h3, z = self.ae(x)

        x_array = list(np.shape(x))
        n_x = x_array[0]

        z1 = self.gate_0(x, adj, M)

        # F2
        m1 = self.F1(torch.cat((h1, z1), 1))
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:, 0], [n_x, 1])
        m12 = torch.reshape(m1[:, 1], [n_x, 1])
        m11_broadcast = m11.repeat(1, 500)
        m12_broadcast = m12.repeat(1, 500)
        net1 = m11_broadcast.mul(z1) + m12_broadcast.mul(h1)
        z2 = self.gate_1(net1, adj, M)

        # F3
        m2 = self.F2(torch.cat((h2, z2), 1))
        m2 = F.normalize(m2, p=2)
        m21 = torch.reshape(m2[:, 0], [n_x, 1])
        m22 = torch.reshape(m2[:, 1], [n_x, 1])
        m21_broadcast = m21.repeat(1, 500)
        m22_broadcast = m22.repeat(1, 500)
        net2 = m21_broadcast.mul(h2) + m22_broadcast.mul(z2)
        z3 = self.gate_2(net2, adj, M)

        # F4
        m3 = self.F3(torch.cat((h3, z3), 1))
        m3 = F.normalize(m3, p=2)
        m31 = torch.reshape(m3[:, 0], [n_x, 1])
        m32 = torch.reshape(m3[:, 1], [n_x, 1])
        m31_broadcast = m31.repeat(1, 2000)
        m32_broadcast = m32.repeat(1, 2000)
        net3 = m31_broadcast.mul(h3) + m32_broadcast.mul(z3)
        z4 = self.gate_3(net3, adj, M)

        indicator = update_indicator(z4, self.n_clusters)

        A_pred = dot_product_decode(z4)

        z_gat_1 = self.dec_1(z4, adj, M)
        z_gat_2 = self.dec_2(z_gat_1, adj, M)
        z_gat = self.dec_3(z_gat_2, adj, M)

        A_gat = dot_product_decode(z_gat)

        u = self.FL(torch.cat((z1, z2, z3, z4, z), 1))
        u = F.normalize(u, p=2)

        u0 = torch.reshape(u[:, 0], [n_x, 1])
        u1 = torch.reshape(u[:, 1], [n_x, 1])
        u2 = torch.reshape(u[:, 2], [n_x, 1])
        u3 = torch.reshape(u[:, 3], [n_x, 1])
        u4 = torch.reshape(u[:, 4], [n_x, 1])

        tile_u0 = u0.repeat(1, 500)
        tile_u1 = u1.repeat(1, 500)
        tile_u2 = u2.repeat(1, 2000)
        tile_u3 = u3.repeat(1, 10)
        tile_u4 = u4.repeat(1, 10)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1)
        net_output = self.gate_z(net_output, adj, active=False)
        predict = F.softmax(net_output, dim=1)

        # Self-supervised Module for AE
        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_ae = q_ae.pow((self.v + 1.0) / 2.0)
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        # Self-supervised Module for GAT
        q_gat = 1.0 / (1.0 + torch.sum(torch.pow(z4.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q_gat = q_gat.pow((self.v + 1.0) / 2.0)
        q_gat = (q_gat.t() / torch.sum(q_gat, 1)).t()

        return x_bar, q_ae, q_gat, predict, z, net_output, A_pred, z_gat, A_gat


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def update_indicator(features, n_clusters):
    if features.requires_grad:
        features = features.detach()
    try:
        U, _, __ = torch.svd(features)
    except:
        print('SVD Not Convergence')
    indicator = U[:, :n_clusters]  # c-top
    indicator = indicator.detach()
    return indicator


def train_scGAAC(data, label, name):
    dataname = name
    eprm_state = 'result'

    file_out = open('./output/' + dataname + '_' + eprm_state + '.out', 'a')
    print("The experimental results", file=file_out)

    # hyper parameters
    lambda_1 = [100]  # [0.001,0.01,0.1,1,10,100,1000]
    lambda_2 = [1000]  # [0.001,0.01,0.1,1,10,100,1000]
    for ld1 in lambda_1:
        for ld2 in lambda_2:
            print("lambda_1: ", ld1, "lambda_2: ", ld2, file=file_out)
            model = scGAAC(500, 500, 2000, 2000, 500, 500,
                         n_input=args.n_input,
                         n_z=args.n_z,
                         n_clusters=args.n_clusters,
                         v=1.0).cuda()

            optimizer = Adam(model.parameters(), lr=args.lr)

            # adjacent matrix
            adj, adj_label = load_graph(args.name, args.k)
            adj_dense = adj.to_dense()
            adj_numpy = adj_dense.data.cpu().numpy()
            t = 10
            tran_prob = normalize(adj_numpy, norm="l1", axis=0)
            M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
            M = torch.Tensor(M_numpy).cuda()

            adj = adj_dense.cuda()
            adj_label = adj_label.cuda()

            # cluster parameter initiate
            data = torch.Tensor(data).cuda()
            y = label
            with torch.no_grad():
                _, _, _, _, z = model.ae(data)

            iters10_kmeans_iter_Q_ae = []
            iters10_NMI_iter_Q_ae = []
            iters10_ARI_iter_Q_ae = []
            iters10_F1_iter_Q_ae = []

            iters10_kmeans_iter_Z = []
            iters10_NMI_iter_Z = []
            iters10_ARI_iter_Z = []
            iters10_F1_iter_Z = []

            iters10_kmeans_iter_P_ae = []
            iters10_NMI_iter_P_ae = []
            iters10_ARI_iter_P_ae = []
            iters10_F1_iter_P_ae = []

            z_1st = z

            for i in range(1):

                kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
                y_pred_last = y_pred
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
                acc, nmi, ari, f1 = eva(y, y_pred, 'pae')

                # get the value
                kmeans_iter_Q_ae = []
                NMI_iter_Q_ae = []
                ARI_iter_Q_ae = []
                F1_iter_Q_ae = []

                kmeans_iter_Q_gat = []
                NMI_iter_Q_gat = []
                ARI_iter_Q_gat = []
                F1_iter_Q_gat = []

                kmeans_iter_Z = []
                NMI_iter_Z = []
                ARI_iter_Z = []
                F1_iter_Z = []

                kmeans_iter_P_ae = []
                NMI_iter_P_ae = []
                ARI_iter_P_ae = []
                F1_iter_P_ae = []

                kmeans_iter_P_gat = []
                NMI_iter_P_gat = []
                ARI_iter_P_gat = []
                F1_iter_P_gat = []

                for epoch in range(200):

                    if epoch % 1 == 0:
                        _, tmp_qae, tmp_qgat, pred, _, _, A_pred, z_gat, A_gat = model(data, adj, M)
                        tmp_qae = tmp_qae.data
                        p_ae = target_distribution(tmp_qae)
                        tmp_qgat = tmp_qgat.data
                        p_gat = target_distribution(tmp_qgat)

                        res1 = tmp_qae.cpu().numpy().argmax(1)  # Q_ae
                        res3 = p_ae.data.cpu().numpy().argmax(1)  # P_ae

                        res4 = tmp_qgat.cpu().numpy().argmax(1)  # Q_gat
                        res5 = p_gat.data.cpu().numpy().argmax(1)  # P_gat

                        res2 = pred.data.cpu().numpy().argmax(1)  # Z

                        acc, nmi, ari, f1 = eva(y, res1, str(epoch) + 'Q_ae')
                        kmeans_iter_Q_ae.append(acc)
                        NMI_iter_Q_ae.append(nmi)
                        ARI_iter_Q_ae.append(ari)
                        F1_iter_Q_ae.append(f1)

                        acc, nmi, ari, f1 = eva(y, res4, str(epoch) + 'Q_gat')
                        kmeans_iter_Q_gat.append(acc)
                        NMI_iter_Q_gat.append(nmi)
                        ARI_iter_Q_gat.append(ari)
                        F1_iter_Q_gat.append(f1)

                        acc, nmi, ari, f1 = eva(y, res2, str(epoch) + 'Z')

                        kmeans_iter_Z.append(acc)
                        NMI_iter_Z.append(nmi)
                        ARI_iter_Z.append(ari)
                        F1_iter_Z.append(f1)


                        acc, nmi, ari, f1 = eva(y, res3, str(epoch) + 'P_ae')
                        kmeans_iter_P_ae.append(acc)
                        NMI_iter_P_ae.append(nmi)
                        ARI_iter_P_ae.append(ari)
                        F1_iter_P_ae.append(f1)

                        acc, nmi, ari, f1 = eva(y, res5, str(epoch) + 'P_gat')

                        kmeans_iter_P_gat.append(acc)
                        NMI_iter_P_gat.append(nmi)
                        ARI_iter_P_gat.append(ari)
                        F1_iter_P_gat.append(f1)

                    x_bar, q_ae, q_gat, pred, _, _, A_pred, z_gat, A_gat = model(data, adj, M)

                    klae_loss = F.kl_div(q_ae.log(), p_ae, reduction='batchmean')
                    klgat_loss = F.kl_div(q_gat.log(), p_gat, reduction='batchmean')
                    kl_loss = F.kl_div(q_gat.log(), q_ae, reduction='batchmean')
                    ce_loss = F.kl_div(pred.log(), p_gat, reduction='batchmean')
                    re_loss = F.mse_loss(x_bar, data)
                    gat_loss = F.mse_loss(data, z_gat)
                    re_gat = F.mse_loss(A_pred, A_gat)
                    re_loss_gat = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))

                    loss = ld1 * klae_loss + ld1 * klgat_loss + 1000 * kl_loss + ld2 * ce_loss + re_loss + 500 * re_loss_gat + 1000 * gat_loss + 1000 * re_gat

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # _Q_ae
                kmeans_max = np.max(kmeans_iter_Q_ae)
                nmi_max = np.max(NMI_iter_Q_ae)
                ari_max = np.max(ARI_iter_Q_ae)
                F1_max = np.max(F1_iter_Q_ae)
                iters10_kmeans_iter_Q_ae.append(round(kmeans_max, 5))
                iters10_NMI_iter_Q_ae.append(round(nmi_max, 5))
                iters10_ARI_iter_Q_ae.append(round(ari_max, 5))
                iters10_F1_iter_Q_ae.append(round(F1_max, 5))

                # _Z
                kmeans_max = np.max(kmeans_iter_Z)
                nmi_max = np.max(NMI_iter_Z)
                ari_max = np.max(ARI_iter_Z)
                F1_max = np.max(F1_iter_Z)
                iters10_kmeans_iter_Z.append(round(kmeans_max, 5))
                iters10_NMI_iter_Z.append(round(nmi_max, 5))
                iters10_ARI_iter_Z.append(round(ari_max, 5))
                iters10_F1_iter_Z.append(round(F1_max, 5))

                # _P_ae
                kmeans_max = np.max(kmeans_iter_P_ae)
                nmi_max = np.max(NMI_iter_P_ae)
                ari_max = np.max(ARI_iter_P_ae)
                F1_max = np.max(F1_iter_P_ae)
                iters10_kmeans_iter_P_ae.append(round(kmeans_max, 5))
                iters10_NMI_iter_P_ae.append(round(nmi_max, 5))
                iters10_ARI_iter_P_ae.append(round(ari_max, 5))
                iters10_F1_iter_P_ae.append(round(F1_max, 5))

            print("#####################################", file=file_out)
            print("kmeans Z mean", round(np.mean(iters10_kmeans_iter_Z), 5), "max", np.max(iters10_kmeans_iter_Z), "\n",
                  iters10_kmeans_iter_Z, file=file_out)
            print("NMI mean", round(np.mean(iters10_NMI_iter_Z), 5), "max", np.max(iters10_NMI_iter_Z), "\n",
                  iters10_NMI_iter_Z, file=file_out)
            print("ARI mean", round(np.mean(iters10_ARI_iter_Z), 5), "max", np.max(iters10_ARI_iter_Z), "\n",
                  iters10_ARI_iter_Z, file=file_out)
            print("F1  mean", round(np.mean(iters10_F1_iter_Z), 5), "max", np.max(iters10_F1_iter_Z), "\n",
                  iters10_F1_iter_Z, file=file_out)
            print(
                ':acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_kmeans_iter_Z), 5),
                                                                              round(np.mean(iters10_NMI_iter_Z), 5),
                                                                              round(np.mean(iters10_ARI_iter_Z), 5),
                                                                              round(np.mean(iters10_F1_iter_Z), 5)),
                file=file_out)


    file_out.close()


if __name__ == "__main__":
    # iters
    iters = 10

    for iter_num in range(iters):
        print(iter_num)
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='Baron')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--n_input', type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)

        dat = read_csv(args.name, True)
        x = dat['data']
        y = dat['label']
        print("Data_name:{}".format(args.name))
        print("Data_shape: {}".format(x.shape))
        print("Label_shape: {}".format(y.shape))

        if args.name == 'Klein':
            args.lr = 1e-4
            args.k = 5
            args.n_clusters = 4
            args.n_input = 2000

        if args.name == 'Deng':
            args.lr = 1e-4
            args.k = 5
            args.n_clusters = 6
            args.n_input = 2000

        if args.name == 'Goolam':
            args.lr = 1e-4
            args.k = 5
            args.n_clusters = 5
            args.n_input = 2000

        if args.name == 'Baron':
            args.lr = 1e-4
            args.k = 5
            args.n_clusters = 13
            args.n_input = 2000

        print(args)
        train_scGAAC(data=x, label=y, name = args.name)

    toc = time.time()
    print("Time:", (toc - tic))

