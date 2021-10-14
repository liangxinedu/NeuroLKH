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
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--file_path', default='CVRPTW_train', help='')
parser.add_argument('--eval_file_path', default='CVRPTW_val', help='')
parser.add_argument('--n_epoch', type=int, default=10000, help='')
parser.add_argument('--eval_interval', type=int, default=1, help='')
parser.add_argument('--eval_batch_size', type=int, default=20, help='')
parser.add_argument('--n_hidden', type=int, default=128, help='')
parser.add_argument('--n_gcn_layers', type=int, default=30, help='')
parser.add_argument('--n_mlp_layers', type=int, default=3, help='')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--save_interval', type=int, default=5, help='')
parser.add_argument('--save_dir', type=str, default="saved/exp1/", help='')
parser.add_argument('--load_pt', type=str, default="", help='')
args = parser.parse_args()

n_edges = 20
net = SparseGCNModel(problem="cvrptw")
net.cuda()
dataLoader = DataLoader(file_path=args.file_path,
                        batch_size=None, problem="cvrptw")
os.makedirs(args.save_dir, exist_ok=True)
edge_cw = None
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

epoch = 0
if args.load_pt:
    saved = torch.load(args.load_pt)
    epoch = saved["epoch"]
    net.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
while epoch < args.n_epoch:
    statistics = {"loss_train": [],
                  "loss_test": []}

    rank_train1 = [[] for _ in range(16)]
    rank_train2 = [[] for _ in range(16)]

    net.train()
    dataset_index = epoch % 10
    dataLoader.load_data(dataset_index)
    for batch in trange(60 * 16):
        node_feat, edge_feat, label1, label2, edge_index, inverse_edge_index = dataLoader.next_batch()
        batch_size = node_feat.shape[0]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1) # B x 20N x 1
        label1 = Variable(torch.LongTensor(label1).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 1000
        label2 = Variable(torch.LongTensor(label2).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 1000
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 1000
        inverse_edge_index = Variable(torch.LongTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1) # B x 1000
        if type(edge_cw) != torch.Tensor:
            edge_labels = label1.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            edge_cw = torch.Tensor(edge_cw).type(torch.cuda.FloatTensor)

        y_edges1, y_edges2, loss_edges1, loss_edges2, _ = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, label1, label2, edge_cw, n_edges)
        loss_edges = loss_edges1.mean() + loss_edges2.mean()

        n_nodes = node_feat.size(1)
        loss = loss_edges
        loss.backward()
        statistics["loss_train"].append(loss.detach().cpu().numpy())
        optimizer.step()
        optimizer.zero_grad()

        y_edges1 = y_edges1.detach().cpu().numpy()
        y_edges2 = y_edges2.detach().cpu().numpy()
        label1 = label1.cpu().numpy()
        label2 = label2.cpu().numpy()

        rank_batch1 = np.zeros((batch_size * n_nodes, n_edges))
        rank_batch1[np.arange(batch_size * n_nodes).reshape(-1, 1), np.argsort(-y_edges1[:, :, 1].reshape(-1, n_edges))] = np.tile(np.arange(n_edges), (batch_size * n_nodes, 1))
        rank_train1[batch % 16].append((rank_batch1.reshape(-1) * label1.reshape(-1)).sum() / label1.sum())

        rank_batch2 = np.zeros((batch_size * n_nodes, n_edges))
        rank_batch2[np.arange(batch_size * n_nodes).reshape(-1, 1), np.argsort(-y_edges2[:, :, 1].reshape(-1, n_edges))] = np.tile(np.arange(n_edges), (batch_size * n_nodes, 1))
        rank_train2[batch % 16].append((rank_batch2.reshape(-1) * label2.reshape(-1)).sum() / label2.sum())
    print ("Epoch {} loss {:.7f} rank:".format(epoch, np.mean(statistics["loss_train"])), ",".join([str(np.mean(rank_train1[_]) + 1)[:5] + " " + str(np.mean(rank_train2[_]) + 1)[:5]  for _ in range(16)]))

    if epoch % args.eval_interval == 0:
        eval_results = []
        for n_nodes in [40, 100, 200]:
            dataset = pickle.load(open(args.eval_file_path + "/" + str(n_nodes) + ".pkl", "rb"))
            rank_test1 = []
            rank_test2 = []
            for eval_batch in trange(1000 // args.eval_batch_size):
                node_feat = dataset["node_feat"][args.eval_batch_size * eval_batch:args.eval_batch_size * (eval_batch + 1)]
                edge_feat = dataset["edge_feat"][args.eval_batch_size * eval_batch:args.eval_batch_size * (eval_batch + 1)]
                edge_index = dataset["edge_index"][args.eval_batch_size * eval_batch:args.eval_batch_size * (eval_batch + 1)]
                inverse_edge_index = dataset["inverse_edge_index"][args.eval_batch_size * eval_batch:args.eval_batch_size * (eval_batch + 1)]
                label1 = dataset["label1"][args.eval_batch_size * eval_batch:args.eval_batch_size * (eval_batch + 1)]
                label2 = dataset["label2"][args.eval_batch_size * eval_batch:args.eval_batch_size * (eval_batch + 1)]
                with torch.no_grad():
                    node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x 100 x 2
                    edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1, 1) # B x 1000 x 2
                    label1 = Variable(torch.LongTensor(label1).type(torch.cuda.LongTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                    label2 = Variable(torch.LongTensor(label2).type(torch.cuda.LongTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                    edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                    inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000

                    n_nodes = node_feat.size(1)
                    y_edges1, y_edges2, loss_edges1, loss_edges2, y_nodes = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, label1, label2, edge_cw, n_edges)
                    loss_edges = loss_edges1.mean() + loss_edges2.mean()

                y_edges1 = y_edges1.detach().cpu().numpy()
                label1 = label1.cpu().numpy()
                y_edges2 = y_edges2.detach().cpu().numpy()
                label2 = label2.cpu().numpy()

                rank_batch1 = np.zeros((args.eval_batch_size * n_nodes, n_edges))
                rank_batch1[np.arange(args.eval_batch_size * n_nodes).reshape(-1, 1), np.argsort(-y_edges1[:, :, 1].reshape(-1, n_edges))] = np.tile(np.arange(n_edges), (args.eval_batch_size * n_nodes, 1))
                rank_test1.append((rank_batch1.reshape(-1) * label1.reshape(-1)).sum() / label1.sum())
                rank_batch2 = np.zeros((args.eval_batch_size * n_nodes, n_edges))
                rank_batch2[np.arange(args.eval_batch_size * n_nodes).reshape(-1, 1), np.argsort(-y_edges2[:, :, 1].reshape(-1, n_edges))] = np.tile(np.arange(n_edges), (args.eval_batch_size * n_nodes, 1))
                rank_test2.append((rank_batch2.reshape(-1) * label2.reshape(-1)).sum() / label2.sum())
            eval_results.append(np.mean(rank_test1) + 1)
            eval_results.append(np.mean(rank_test2) + 1)
        print ("n=40 %.3f %.3f, n=100 %.3f %.3f, n=200 %.3f %.3f" % (tuple(eval_results)))

    epoch += 1
    if epoch % args.save_interval == 0:
        torch.save({"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict()}, args.save_dir + "/" + str(epoch) + ".pt")
