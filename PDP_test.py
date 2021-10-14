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
        n_nodes = len(instance[1])
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : PDTSP\n")
        f.write("DIMENSION : " + str(len(instance[1]) + 1) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        s = 1000000
        f.write(" " + str(1) + " " + str(instance[0][0] * s)[:15] + " " + str(instance[0][1] * s)[:15] + "\n")
        for i in range(n_nodes):
            f.write(" " + str(i + 2) + " " + str(instance[1][i][0] * s)[:15] + " " + str(instance[1][i][1] * s)[:15] + "\n")
        f.write("PICKUP_AND_DELIVERY_SECTION\n")
        f.write("1 0 0 0 0 0 0\n")
        for i in range(n_nodes):
            if i < n_nodes // 2:
                f.write(str(i+2)+" 0 0 0 0 0 "+str(i+2+n_nodes//2)+"\n")
            else:
                f.write(str(i+2)+" 0 0 0 0 "+str(i+2-n_nodes//2)+" 0\n")
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

def write_candidate(dataset_name, instance_name, candidate1, candidate2):
    n_node = candidate1.shape[0] - 1
    candidate1 = candidate1.astype("int")
    candidate2 = candidate2.astype("int")
    with open("result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        i = 0
        f.write(str((n_node + 1) * 2) + "\n")
        line1 = "1 0 "
        n = 1
        line2 = " " + str(1 + n_node + 1) + " 1"
        for _ in range(20):
            if candidate2[0, _] < n_node // 2 + 1:
                line2 += " " + str(candidate2[0, _] + 1 + n_node + 1) + " 1"
                n += 1
                if n == 5:
                    break
        f.write(line1 + str(n) + line2 + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1) + " 0 5 " + str(j + 1 + n_node + 1) + " 1"
            n = 1
            for _ in range(20):
                if j >= n_node // 2 + 1 and candidate2[j, _] == j - n_node // 20:
                    continue
                if j < n_node // 2 + 1 and candidate2[j, _] == 0:
                    continue
                line += " " + str(candidate2[j, _] + 1 + n_node + 1) + " 1"
                n += 1
                if n == 5:
                    break
            f.write(line + "\n")
    
        line1 = str(1 + n_node + 1) + " 0 "
        n = 1
        line2 = " " + str(1) + " 1"
        for _ in range(20):
            if candidate1[0, _] >= n_node // 2 + 1:
                line2 += " " + str(candidate1[0, _] + 1) + " 1"
                n += 1
                if n == 5:
                    break
        f.write(line1 + str(n) + line2 + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1 + n_node + 1) + " 0 5 " + str(j + 1) + " 1"
            n = 1
            for _ in range(20):
                if j < n_node // 2 + 1 and candidate1[j, _] == j - n_node // 20:
                    continue
                if j >= n_node // 2 + 1 and candidate1[j, _] == 0:
                    continue
                line += " " + str(candidate1[j, _] + 1) + " 1"
                n += 1
                if n == 5:
                    break
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
    instance_filename = "result/" + dataset_name + "/pdp/" + instance_name + ".pdp"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

def infer_SGN(net, dataset_node_feat, dataset_edge_index, dataset_edge_feat, dataset_inverse_edge_index, batch_size=100):
    candidate1 = []
    candidate2 = []
    for i in trange(dataset_edge_index.shape[0] // batch_size):
        node_feat = dataset_node_feat[i * batch_size:(i + 1) * batch_size]
        edge_index = dataset_edge_index[i * batch_size:(i + 1) * batch_size]
        edge_feat = dataset_edge_feat[i * batch_size:(i + 1) * batch_size]
        inverse_edge_index = dataset_inverse_edge_index[i * batch_size:(i + 1) * batch_size]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1)
        edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        y_edges1, y_edges2,  _, _, y_nodes = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, None, 20)
        
        y_edges1 = y_edges1.detach().cpu().numpy()
        y_edges1 = y_edges1[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges1 = np.argsort(-y_edges1, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges1.shape[1], 20)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges1.shape[1]).reshape(1, -1, 1), y_edges1]
        candidate1.append(candidate_index[:, :, :20])

        y_edges2 = y_edges2.detach().cpu().numpy()
        y_edges2 = y_edges2[:, :, 1].reshape(batch_size, dataset_node_feat.shape[1], 20)
        y_edges2 = np.argsort(-y_edges2, -1)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges2.shape[1]).reshape(1, -1, 1), y_edges2]
        candidate2.append(candidate_index[:, :, :20])
    candidate1 = np.concatenate(candidate1, 0)
    candidate2 = np.concatenate(candidate2, 0)
    return candidate1, candidate2

def solve_NeuroLKH(dataset_name, instance, instance_name, candidate1, candidate2, rerun=False, max_trials=1000):
    para_filename = "result/" + dataset_name + "/NeuroLKH_para/" + instance_name + ".para"
    log_filename = "result/" + dataset_name + "/NeuroLKH_log/" + instance_name + ".log" 
    instance_filename = "result/" + dataset_name + "/pdp/" + instance_name + ".pdp"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "NeuroLKH", para_filename, max_trials=max_trials)
        write_candidate(dataset_name, instance_name, candidate1, candidate2)
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
        final_obj = int(lines[-9].split(",")[0].split(" ")[-1])
        assert objs[-1] == final_obj
        return objs, penalties, runtimes

def eval_dataset(dataset_filename, method, args, rerun=True, max_trials=1000):
    dataset_name = dataset_filename.strip(".pkl").split("/")[-1]
    os.makedirs("result/" + dataset_name + "/" + method + "_para", exist_ok=True) 
    os.makedirs("result/" + dataset_name + "/" + method + "_log", exist_ok=True) 
    os.makedirs("result/" + dataset_name + "/pdp", exist_ok=True)
    with open(dataset_filename, "rb") as f:
        dataset = pickle.load(f)[:args.n_samples]
        x = dataset
    if method == "NeuroLKH":
        feat_start_time = time.time()
        x = np.array([[_x[0]] + _x[1] for _x in x])
        n_nodes = x.shape[1]
        n_samples = x.shape[0]
        n_neighbours = 20
        node_feat = x
        dist = x.reshape(n_samples, n_nodes, 1, 2) - x.reshape(n_samples, 1, n_nodes, 2)
        dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
        edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
        edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[n_samples, n_nodes, n_nodes], dtype="int")
        inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_neighbours
        inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        feat_runtime = time.time() - feat_start_time
        net = SparseGCNModel(problem="pdp")
        net.cuda()
        net.train()
        saved = torch.load(args.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            candidate1, candidate2 = infer_SGN(net, node_feat, edge_index, edge_feat, inverse_edge_index, batch_size=args.batch_size)
        sgn_runtime = time.time() - sgn_start_time
        os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(method_wrapper, [("NeuroLKH", dataset_name, dataset[i], str(i), candidate1[i], candidate2[i], rerun, max_trials) for i in range(len(dataset))]), total=len(dataset)))
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
    parser.add_argument('--dataset', type=str, default='PDP_test/pdp_40.pkl', help='')
    parser.add_argument('--model_path', type=str, default='pretrained/pdp_neurolkh.pt', help='')
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
