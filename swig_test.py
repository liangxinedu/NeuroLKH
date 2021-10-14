import os
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle
from net.sgcn_model import SparseGCNModel
from SRC_swig.LKH import main as LKH
from SRC_swig.LKH import featureGenerate
import torch
from torch.autograd import Variable
from tqdm import trange
import argparse
import time
import tempfile

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(data, n_nodes, max_trials=1000):
    invec = data.copy()
    seed = 1234
    result = LKH(0, max_trials, seed, n_nodes, invec)
    return invec

def generate_feat(data, n_nodes):
    n_edges = 20
    data = np.array(data)
    invec = np.concatenate([data.reshape(-1) * 1000000, np.zeros([n_nodes * (3 * n_edges - 2)])], -1)
    feat_runtime = featureGenerate(1234, invec)
    edge_index = invec[:n_nodes * n_edges].reshape(1, -1, 20)
    edge_feat = invec[n_nodes * n_edges:n_nodes * n_edges * 2].reshape(1, -1, 20)
    inverse_edge_index = invec[n_nodes * n_edges * 2:n_nodes * n_edges * 3].reshape(1, -1, 20)
    return edge_index, edge_feat / 100000000, inverse_edge_index, feat_runtime / 1000000

def infer_SGN(net, dataset_node_feat, dataset_edge_index, dataset_edge_feat, dataset_inverse_edge_index, batch_size=100):
    candidate = []
    pi = []
    for i in trange(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20)
        pi.append(y_nodes.cpu().numpy())
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        candidate.append(candidate_index[:, :, :5])
    candidate = np.concatenate(candidate, 0)
    pi = np.concatenate(pi, 0)
    candidate_Pi = np.concatenate([candidate.reshape(dataset_edge_index.shape[0], -1), 1000000 * pi.reshape(dataset_edge_index.shape[0], -1)], -1)
    return candidate_Pi

def solve_NeuroLKH(data, n_nodes, max_trials=1000):
    invec = data.copy()
    seed = 1234
    result = LKH(1, max_trials, seed, n_nodes, invec)
    return invec

def eval_dataset(dataset_filename, method, args, rerun=True, max_trials=1000):
    dataset_name = dataset_filename.strip(".pkl").split("/")[-1]
    with open(dataset_filename, "rb") as f:
        dataset = pickle.load(f)[:args.n_samples]
    if method == "NeuroLKH":
        with Pool(os.cpu_count()) as pool:
            feats = list(tqdm.tqdm(pool.imap(method_wrapper, [("FeatGen", dataset[i], len(dataset[0])) for i in range(len(dataset))]), total=len(dataset)))
        feats = list(zip(*feats))
        edge_index, edge_feat, inverse_edge_index, feat_runtime = feats
        feat_runtime = np.sum(feat_runtime)
        edge_index = np.concatenate(edge_index)
        edge_feat = np.concatenate(edge_feat)
        inverse_edge_index = np.concatenate(inverse_edge_index)

        net = SparseGCNModel()
        net.cuda()
        saved = torch.load(args.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            candidate_Pi = infer_SGN(net, np.array(dataset), edge_index, edge_feat, inverse_edge_index, batch_size=args.batch_size)
        sgn_runtime = time.time() - sgn_start_time
        # with open("candidate_Pi/" + dataset_name + ".pkl", "rb") as f:
        #     candidate_Pi = pickle.load(f)
        invec = np.concatenate([np.array(dataset).reshape(len(dataset), -1) * 1000000, candidate_Pi[:args.n_samples]], 1)

        if invec.shape[1] < max_trials * 2:
            invec = np.concatenate([invec, np.zeros([invec.shape[0], max_trials * 2 - invec.shape[1]])], 1)
        else:
            invec = invec.copy()
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("NeuroLKH", invec[i], len(dataset[0]), max_trials) for i in range(len(dataset))]), total=len(dataset)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        invec = np.array(dataset).reshape(len(dataset), -1) * 1000000
        if invec.shape[1] < max_trials * 2:
            invec = np.concatenate([invec, np.zeros([invec.shape[0], max_trials * 2 - invec.shape[1]])], 1)
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", invec[i], len(dataset[0]), max_trials) for i in range(len(dataset))]), total=len(dataset)))
    results = np.array(results).reshape(len(dataset), -1, 2)[:, :max_trials, :]
    dataset_objs = results[:, :, 0].mean(0)
    dataset_runtimes = results[:, :, 1].sum(0)
    return dataset_objs, dataset_runtimes, feat_runtime, sgn_runtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='test/100.pkl', help='')
    parser.add_argument('--model_path', type=str, default='pretrained/neurolkh.pt', help='')
    parser.add_argument('--n_samples', type=int, default=1000, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--lkh_trials', type=int, default=1000, help='')
    parser.add_argument('--neurolkh_trials', type=int, default=1000, help='')
    args = parser.parse_args()
    
    lkh_objs, lkh_runtimes, _, _ = eval_dataset(args.dataset, "LKH", args=args, rerun=True, max_trials=args.lkh_trials)
    neurolkh_objs, neurolkh_runtimes, feat_runtime, sgn_runtime = eval_dataset(args.dataset, "NeuroLKH", args=args, rerun=True, max_trials=args.neurolkh_trials)

    print ("generating features runtime: %.1fs SGN inferring runtime: %.1fs" % (feat_runtime, sgn_runtime))
    trials = 1
    while trials <= lkh_objs.shape[0]:
        print ("------experiments of trials: %d ------" % (trials))
        print ("LKH      %d %ds" % (lkh_objs[trials - 1], lkh_runtimes[trials - 1]))
        if trials > neurolkh_objs.shape[0]:
            print ("NeuroLKH %d %ds (%d trials)" % (neurolkh_objs[-1], neurolkh_runtimes[-1] + feat_runtime + sgn_runtime, neurolkh_objs.shape[0]))
        else:
            print ("NeuroLKH %d %ds" % (neurolkh_objs[trials - 1], neurolkh_runtimes[trials - 1] + feat_runtime + sgn_runtime))
        trials *= 10
    print ("------comparison with same time limit------")
    trials = 1
    while trials <= lkh_objs.shape[0]:
        print ("------experiments of trials: %d ------" % (trials))
        print ("LKH      %d %ds" % (lkh_objs[trials - 1], lkh_runtimes[trials - 1]))
        neurolkh_trials = 1
        while neurolkh_trials < neurolkh_runtimes.shape[0] and neurolkh_runtimes[neurolkh_trials - 1] + feat_runtime + sgn_runtime < lkh_runtimes[trials - 1]:
            neurolkh_trials += 1
        print ("NeuroLKH %d %ds (%d trials)" % (neurolkh_objs[neurolkh_trials - 1], neurolkh_runtimes[neurolkh_trials - 1] + feat_runtime + sgn_runtime, neurolkh_trials))
        trials *= 10


