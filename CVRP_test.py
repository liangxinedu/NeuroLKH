import os
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle
from net.sgcn_model import SparseGCNModel
import torch
from torch.autograd import Variable
from tqdm import trange
import argparse
import time
import tempfile

def write_instance(instance, instance_name, instance_filename):
    with open(instance_filename, "w") as f:
        n_nodes = len(instance[0]) - 1
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        f.write("DIMENSION : " + str(len(instance[0])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("CAPACITY : " + str(instance[2]) + "\n")
        f.write("NODE_COORD_SECTION\n")
        s = 1000000
        for i in range(n_nodes + 1):
            f.write(" " + str(i + 1) + " " + str(instance[0][i][0] * s)[:15] + " " + str(instance[0][i][1] * s)[:15] + "\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(n_nodes):
            f.write(str(i + 2)+" "+str(instance[1][i])+"\n")
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")

def write_para(dataset_name, instance_name, instance_filename, method, para_filename, max_trials=1000, seed=1234):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("SEED = " + str(seed) + "\n")
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/candidate/" + instance_name + ".txt\n")
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("CANDIDATE_FILE = result/" + dataset_name + "/feat/" + instance_name + ".txt\n")
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MAX_CANDIDATES = 20\n")
        else:
            assert method == "LKH"

def read_feat(feat_filename, max_nodes):
    n_neighbours = 20
    edge_index = np.zeros([1, max_nodes, n_neighbours], dtype="int")
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        n_nodes_extend = int(lines[0].strip())
        for j in range(n_nodes_extend):
            line = lines[j + 1].strip().split(" ")
            line = [int(_) for _ in line]
            assert len(line) == 43
            assert line[0] == j + 1
            for _ in range(n_neighbours):
                edge_index[0, j, _] = line[3 + _ * 2] - 1
    feat_runtime = float(lines[-2].strip())
    return edge_index, n_nodes_extend, feat_runtime

def write_candidate(dataset_name, instance_name, candidate, n_nodes_extend):
    n_node = candidate.shape[0]
    with open("result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        f.write(str(n_nodes_extend) + "\n")
        for j in range(n_nodes_extend):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(dataset_name, instance, instance_name, rerun=False, max_trials=1000):
    para_filename = "result/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

def generate_feat(dataset_name, instance, instance_name, max_nodes):
    para_filename = "result/" + dataset_name + "/featgen_para/" + instance_name + ".para"
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    feat_filename = "result/" + dataset_name + "/feat/" + instance_name + ".txt"
    write_instance(instance, instance_name, instance_filename)
    write_para(dataset_name, instance_name, instance_filename, "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call(["./LKH", para_filename], stdout=f)
    return read_feat(feat_filename, max_nodes)

def infer_SGN(net, dataset_node_feat, dataset_edge_index, dataset_edge_feat, dataset_inverse_edge_index, batch_size=100):
    candidate = []
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
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        candidate.append(candidate_index[:, :, :5])
    candidate = np.concatenate(candidate, 0)
    return candidate

def solve_NeuroLKH(dataset_name, instance, instance_name, candidate, n_nodes_extend, rerun=False, max_trials=1000):
    para_filename = "result/" + dataset_name + "/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/NeuroLKH_log/" + instance_name + ".log" 
    instance_filename = "result/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    if rerun or not os.path.isfile(log_filename):
        # write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "NeuroLKH", para_filename, max_trials=max_trials)
        write_candidate(dataset_name, instance_name, candidate, n_nodes_extend)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

def read_results(log_filename, max_trials):
    objs = []
    penalties = []
    runtimes = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines: # read the obj and runtime for each trial
            if line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-4])
                objs.append(int(line[-2]))
                penalties.append(int(line[-3]))
                runtimes.append(float(line[-1]))
        final_obj = int(lines[-11].split(",")[0].split(" ")[-1])
        assert objs[-1] == final_obj
        return objs, penalties, runtimes



def eval_dataset(dataset_filename, method, args, rerun=True, max_trials=1000):
    dataset_name = dataset_filename.strip(".pkl").split("/")[-1]
    os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True) 
    os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True) 
    os.makedirs("result/" + dataset_name + "/cvrp", exist_ok=True)
    with open(dataset_filename, "rb") as f:
        dataset = pickle.load(f)
        x = dataset["x"][:args.n_samples]
        demand = dataset["demand"][:args.n_samples]
        dataset = [[dataset["x"][i].tolist(), dataset["demand"][i].tolist(), dataset["capacity"]] for i in range(args.n_samples)]
    if method == "NeuroLKH":
        os.makedirs("result/" + dataset_name + "/featgen_para", exist_ok=True) 
        os.makedirs("result/" + dataset_name + "/feat", exist_ok=True)
        n_nodes = len(dataset[0][0]) - 1
        max_nodes = int(n_nodes * 1.15)
        n_samples = args.n_samples
        n_neighbours = 20
        with Pool(os.cpu_count()) as pool:
            feats = list(tqdm.tqdm(pool.imap(method_wrapper, [("FeatGen", dataset_name, dataset[i], str(i), max_nodes) for i in range(len(dataset))]), total=len(dataset)))
        edge_index, n_nodes_extend, feat_runtime = list(zip(*feats))
        feat_runtime = np.sum(feat_runtime)
        feat_start_time = time.time()
        edge_index = np.concatenate(edge_index, 0)
        demand = np.concatenate([np.zeros([n_samples, 1]), demand, np.zeros([n_samples, max_nodes - n_nodes - 1])], -1)
        demand = demand / dataset[0][2]
        capacity = np.zeros([n_samples, max_nodes])
        capacity[:, 0] = 1
        capacity[:, n_nodes + 1:] = 1
        x = np.concatenate([x] + [x[:, 0:1, :] for _ in range(max_nodes - n_nodes - 1)], 1)
        node_feat = np.concatenate([x, demand.reshape([n_samples, max_nodes, 1]), capacity.reshape([n_samples, max_nodes, 1])], -1)
        dist = node_feat[:, :, :2].reshape(n_samples, max_nodes, 1, 2) - node_feat[:, :, :2].reshape(n_samples, 1, max_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1))
        edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype="int")
        inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(max_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(max_nodes).reshape(1, -1, 1) * n_neighbours
        inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        feat_runtime += time.time() - feat_start_time
        net = SparseGCNModel(problem="cvrp")
        net.cuda()
        saved = torch.load(args.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            candidate = infer_SGN(net, node_feat, edge_index, edge_feat, inverse_edge_index, batch_size=args.batch_size)
        sgn_runtime = time.time() - sgn_start_time
        os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("NeuroLKH", dataset_name, dataset[i], str(i), candidate[i], n_nodes_extend[i], rerun, max_trials) for i in range(len(dataset))]), total=len(dataset)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", dataset_name, dataset[i], str(i), rerun, max_trials) for i in range(len(dataset))]), total=len(dataset)))
    results = np.array(results)
    dataset_objs = results[:, 0, :].mean(0)
    dataset_penalties = results[:, 1, :].mean(0)
    dataset_runtimes = results[:, 2, :].sum(0)
    return dataset_objs, dataset_penalties, dataset_runtimes, feat_runtime, sgn_runtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='CVRP_test/cvrp_100.pkl', help='')
    parser.add_argument('--model_path', type=str, default='pretrained/cvrp_neurolkh.pt', help='')
    parser.add_argument('--n_samples', type=int, default=1000, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--lkh_trials', type=int, default=1000, help='')
    parser.add_argument('--neurolkh_trials', type=int, default=1000, help='')
    args = parser.parse_args()
    
    neurolkh_objs, neurolkh_penalties, neurolkh_runtimes, feat_runtime, sgn_runtime = eval_dataset(args.dataset, "NeuroLKH", args=args, rerun=True, max_trials=args.neurolkh_trials) 
    lkh_objs, lkh_penalties, lkh_runtimes, _, _ = eval_dataset(args.dataset, "LKH", args=args, rerun=True, max_trials=args.lkh_trials)

    print ("generating features runtime: %.1fs SGN inferring runtime: %.1fs" % (feat_runtime, sgn_runtime))
    print ("method obj penalties runtime")
    trials = 1
    while trials <= lkh_objs.shape[0]:
        print ("------experiments of trials: %d ------" % (trials))
        print ("LKH      %d %d %ds" % (lkh_objs[trials - 1], lkh_penalties[trials - 1], lkh_runtimes[trials - 1]))
        print ("NeuroLKH %d %d %ds" % (neurolkh_objs[trials - 1], neurolkh_penalties[trials - 1], neurolkh_runtimes[trials - 1] + feat_runtime + sgn_runtime))
        trials *= 10
    print ("------comparison with same time limit------")
    trials = 1
    while trials <= lkh_objs.shape[0]:
        print ("------experiments of trials: %d ------" % (trials))
        print ("LKH      %d %d %ds" % (lkh_objs[trials - 1], lkh_penalties[trials - 1], lkh_runtimes[trials - 1]))
        neurolkh_trials = 1
        while neurolkh_trials < neurolkh_runtimes.shape[0] and neurolkh_runtimes[neurolkh_trials - 1] + feat_runtime + sgn_runtime < lkh_runtimes[trials - 1]:
            neurolkh_trials += 1
        print ("NeuroLKH %d %d %ds (%d trials)" % (neurolkh_objs[neurolkh_trials - 1], neurolkh_penalties[trials - 1], neurolkh_runtimes[neurolkh_trials - 1] + feat_runtime + sgn_runtime, neurolkh_trials))
        trials *= 10


