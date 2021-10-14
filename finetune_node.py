import os
import argparse
import numpy as np
from utils.data_loader import DataLoader
import glob
from tqdm import trange
from net.sgcn_model import SparseGCNModel
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.autograd import Variable

from SRC_swig.LKH import getNodeDegree
from SRC_swig.LKH import featureGenerate

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_epoch', type=int, default=10000, help='')
parser.add_argument('--eval_interval', type=int, default=1, help='')
parser.add_argument('--eval_batch_size', type=int, default=20, help='')
parser.add_argument('--n_hidden', type=int, default=128, help='')
parser.add_argument('--n_gcn_layers', type=int, default=30, help='')
parser.add_argument('--n_mlp_layers', type=int, default=3, help='')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--load_pt', type=str, default="pretrained/neurolkh.pt", help='')
args = parser.parse_args()

n_edges = 20

assert args.load_pt

for n_nodes in [1000, 2000, 5000]:
    net = SparseGCNModel()
    net.cuda()

    net.train()

    edge_cw = None
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    saved = torch.load(args.load_pt)
    net.load_state_dict(saved["model"])

    for batch in trange(100):

        batch_size = max(20 * 200 // n_nodes, 1)
        x = np.random.uniform(size=[batch_size, n_nodes, 2])
        node_feat = x

        edge_index = []
        edge_feat = []
        inverse_edge_index = []
        for i in range(batch_size):
            invec = np.concatenate([x[i].reshape(-1) * 1000000, np.zeros([n_nodes * (3 * n_edges - 2)])], -1)
            featureGenerate(1234, invec)
            edge_index.append(invec[:n_nodes * n_edges])
            edge_feat.append(invec[n_nodes * n_edges:n_nodes * n_edges * 2])
            inverse_edge_index.append(invec[n_nodes * n_edges * 2:n_nodes * n_edges * 3])
        edge_index = np.concatenate(edge_index, 0).reshape(batch_size, n_nodes, n_edges).astype("int")
        edge_feat = np.concatenate(edge_feat, 0).reshape(batch_size, n_nodes, n_edges) / 100000000
        inverse_edge_index = np.concatenate(inverse_edge_index, 0).reshape(batch_size, n_nodes, n_edges).astype("int")

        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N

        y_nodes = net.forward_finetune(node_feat, edge_feat, edge_index, inverse_edge_index, n_edges)

        Vs = []
        Norms = []
        for i in range(batch_size):
            result1 = np.concatenate([node_feat[i].view(-1).detach().cpu().numpy() * 1000000, y_nodes[i].view(-1).detach().cpu().numpy() * 1000000], 0).astype("double")
            getNodeDegree(1234, result1)
            Norms.append(result1[n_nodes + 1])
            Vs.append(result1[:n_nodes].reshape(1, -1))
        Vs = np.concatenate(Vs, 0)

        Vs = Variable(torch.FloatTensor(Vs).type(torch.cuda.FloatTensor), requires_grad=False)

        loss_nodes = -y_nodes.view(batch_size, n_nodes) * Vs
        loss_nodes = loss_nodes.mean()
        loss = loss_nodes

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    Norms = []
    for batch in range(10):
        batch_size = 10
        x = np.random.uniform(size=[batch_size, n_nodes, 2])
        node_feat = x

        edge_index = []
        edge_feat = []
        inverse_edge_index = []
        for i in range(batch_size):
            invec = np.concatenate([x[i].reshape(-1) * 1000000, np.zeros([n_nodes * (3 * n_edges - 2)])], -1)
            featureGenerate(1234, invec)
            edge_index.append(invec[:n_nodes * n_edges])
            edge_feat.append(invec[n_nodes * n_edges:n_nodes * n_edges * 2])
            inverse_edge_index.append(invec[n_nodes * n_edges * 2:n_nodes * n_edges * 3])

        edge_index = np.concatenate(edge_index, 0).reshape(batch_size, n_nodes, n_edges).astype("int")
        edge_feat = np.concatenate(edge_feat, 0).reshape(batch_size, n_nodes, n_edges) / 100000000
        inverse_edge_index = np.concatenate(inverse_edge_index, 0).reshape(batch_size, n_nodes, n_edges).astype("int")
        
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 20N

        with torch.no_grad():
            y_nodes = net.forward_finetune(node_feat, edge_feat, edge_index, inverse_edge_index, n_edges)


        for i in range(batch_size):
            result1 = np.concatenate([node_feat[i].view(-1).detach().cpu().numpy() * 1000000, y_nodes[i].view(-1).detach().cpu().numpy() * 1000000], 0).astype("double")
            getNodeDegree(1234, result1)
            Norms.append(result1[n_nodes + 1])

    print ("Nodes {} Norms:".format(n_nodes), str(np.mean(Norms))[:5])
    torch.save({"model": net.state_dict()}, "pretrained/finetuned_" + str(n_nodes) + ".pt")
