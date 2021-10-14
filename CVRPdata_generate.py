import os
from subprocess import check_call
from multiprocessing import Pool
import tqdm
import numpy as np
import pickle
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
        if method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write("CANDIDATE_FILE = tmp/" + dataset_name + "/feat/" + instance_name + ".txt\n")
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
    return edge_index, n_nodes_extend

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(dataset_name, instance, instance_name, rerun=False, max_trials=1000):
    para_filename = "tmp/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "tmp/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "tmp/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

def generate_feat(dataset_name, instance, instance_name, max_nodes):
    para_filename = "tmp/" + dataset_name + "/featgen_para/" + instance_name + ".para"
    instance_filename = "tmp/" + dataset_name + "/cvrp/" + instance_name + ".cvrp"
    feat_filename = "tmp/" + dataset_name + "/feat/" + instance_name + ".txt"
    write_instance(instance, instance_name, instance_filename)
    write_para(dataset_name, instance_name, instance_filename, "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call(["./LKH", para_filename], stdout=f)
    return read_feat(feat_filename, max_nodes)

def read_results(log_filename, max_trials):
    with open(log_filename, "r") as f:
        line = f.readlines()[-1]
        line = line.strip().split(" ")
        result = [int(_) for _ in line]
    return result

def generate_dataset(n_samples, n_nodes, save_dir):
    capacity = int((n_nodes - 100) * 0.1 + 50)
    x = np.random.uniform(size=[n_samples, n_nodes + 1, 2])
    demand = np.random.randint(1, 10, size=(n_samples, n_nodes))
    dataset = {"x":x, "demand":demand, "capacity":capacity}
    if save_dir == "CVRP_test":
        with open(save_dir + "/cvrp_" + str(n_nodes) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
        return
    os.makedirs("tmp/" + str(n_nodes) + "/cvrp", exist_ok=True)

    dataset = [[dataset["x"][i].tolist(), dataset["demand"][i].tolist(), dataset["capacity"]] for i in range(n_samples)]

    os.makedirs("tmp/" + str(n_nodes) + "/featgen_para", exist_ok=True) 
    os.makedirs("tmp/" + str(n_nodes) + "/feat", exist_ok=True)
    os.makedirs("tmp/" + str(n_nodes) + "/LKH_para", exist_ok=True) 
    os.makedirs("tmp/" + str(n_nodes) + "/LKH_log", exist_ok=True)
    n_nodes = len(dataset[0][0]) - 1
    max_nodes = int(n_nodes * 1.15)
    n_neighbours = 20
    with Pool(os.cpu_count()) as pool:
        feats = list(tqdm.tqdm(pool.imap(method_wrapper, [("FeatGen", str(n_nodes), dataset[i], str(i), max_nodes) for i in range(len(dataset))]), total=len(dataset)))
    edge_index, n_nodes_extend = list(zip(*feats))
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
    with Pool(os.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", str(n_nodes), dataset[i], str(i), True, 10000) for i in range(len(dataset))]), total=len(dataset)))
    label = np.zeros([n_samples, max_nodes, max_nodes], dtype="bool")
    for i in range(n_samples):
        result = results[i]
        for _ in range(len(result) - 1):
            n_from, n_to = result[_] - 1, result[_ + 1] - 1
            label[i, n_from, n_to] = True
            label[i, n_to, n_from] = True
    label = label[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
    feat = {"node_feat":node_feat,
            "edge_feat":edge_feat,
            "edge_index":edge_index,
            "inverse_edge_index":inverse_edge_index,
            "label":label}
    with open(save_dir + "/" + str(n_nodes) + ".pkl", "wb") as f:
        pickle.dump(feat, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-train", action='store_true', help="Generate training and validation datasets")
    parser.add_argument("-test", action='store_true', help="Generate test datasets")
    args = parser.parse_args()
    if args.train:
        os.makedirs("CVRP_train", exist_ok=True) 
        os.makedirs("CVRP_val", exist_ok=True)
        for n_nodes in range(101, 501):
            n_samples = 30 * (40 * 100 // n_nodes)
            generate_dataset(n_samples, n_nodes, "CVRP_train") 
        for n_nodes in [100, 200, 500]:
            n_samples = 1000
            generate_dataset(n_samples, n_nodes, "CVRP_val") 
    if args.test:
        os.makedirs("CVRP_test", exist_ok=True)
        np.random.seed(1234)
        n_samples = 1000
        for n_nodes in [100, 500, 1000]:
            generate_dataset(n_samples, n_nodes, "CVRP_test") 
