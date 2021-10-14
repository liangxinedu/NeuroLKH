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
        n_nodes = instance.shape[0]
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : PDTSP\n")
        f.write("DIMENSION : " + str(instance.shape[0]) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        s = 1000000
        f.write(" " + str(1) + " " + str(instance[0][0] * s)[:15] + " " + str(instance[0][1] * s)[:15] + "\n")
        for i in range(n_nodes - 1):
            f.write(" " + str(i + 2) + " " + str(instance[i + 1][0] * s)[:15] + " " + str(instance[i + 1][1] * s)[:15] + "\n")
        f.write("PICKUP_AND_DELIVERY_SECTION\n")
        f.write("1 0 0 0 0 0 0\n")
        for i in range(n_nodes - 1):
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

def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "FeatGen":
        return generate_feat(*args[1:])

def solve_LKH(dataset_name, instance, instance_name, rerun=False, max_trials=1000):
    para_filename = "tmp/" + dataset_name + "/LKH_para/" + instance_name + ".para"
    log_filename = "tmp/" + dataset_name + "/LKH_log/" + instance_name + ".log"
    instance_filename = "tmp/" + dataset_name + "/pdp/" + instance_name + ".pdp"
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename)
        write_para(dataset_name, instance_name, instance_filename, "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

def read_results(log_filename, max_trials):
    with open(log_filename, "r") as f:
        line = f.readlines()[-1]
        result = [int(_line) - 1 for _line in line.split(" ")[:-2]]
    return result

def generate_dataset(n_samples, n_nodes, save_dir):
    n_neighbours = 20
    x = np.random.uniform(size=[n_samples, n_nodes + 1, 2])

    os.makedirs("tmp/" + str(n_nodes) + "/pdp", exist_ok=True)
    os.makedirs("tmp/" + str(n_nodes) + "/LKH_para", exist_ok=True) 
    os.makedirs("tmp/" + str(n_nodes) + "/LKH_log", exist_ok=True)
    dist = x.reshape(n_samples, n_nodes + 1, 1, 2) - x.reshape(n_samples, 1, n_nodes + 1, 2)
    dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
    edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]
    inverse_edge_index = -np.ones(shape=[n_samples, n_nodes + 1, n_nodes + 1], dtype="int")
    inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(n_nodes + 1).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(n_nodes + 1).reshape(1, -1, 1) * n_neighbours
    inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]
    with Pool(os.cpu_count()) as pool:
        result = list(tqdm.tqdm(pool.imap(method_wrapper, [("LKH", str(n_nodes), x[i], str(i), True, 10000) for i in range(x.shape[0])]), total=x.shape[0]))

    result = np.array(result) # n_samples x n_nodes
    label1 = np.zeros(shape=[n_samples, n_nodes + 1, n_nodes + 1], dtype="bool")
    label2 = np.zeros(shape=[n_samples, n_nodes + 1, n_nodes + 1], dtype="bool")
    label1[np.arange(n_samples).reshape(-1, 1), result, np.roll(result, 1, -1)] = True
    label2[np.arange(n_samples).reshape(-1, 1), np.roll(result, 1, -1), result] = True
    label1 = label1[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]
    label2 = label2[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes + 1).reshape(1, -1, 1), edge_index]

    feat = {"node_feat":x,
            "edge_feat":edge_feat,
            "edge_index":edge_index,
            "inverse_edge_index":inverse_edge_index,
            "label1":label1,
            "label2":label2}
    with open(save_dir + "/" + str(n_nodes) + ".pkl", "wb") as f:
        pickle.dump(feat, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-train", action='store_true', help="Generate training and validation datasets")
    parser.add_argument("-test", action='store_true', help="Generate test datasets")
    args = parser.parse_args()
    if args.train:
        os.makedirs("PDP_train", exist_ok=True)
        os.makedirs("PDP_val", exist_ok=True)
        for n_nodes in range(40, 201, 2):
            n_samples = 2 * 120000 // n_nodes
            generate_dataset(n_samples, n_nodes, "PDP_train") 
        for n_nodes in [40, 80, 200]:
            n_samples = 1000
        generate_dataset(n_samples, n_nodes, "PDP_val")
    if args.test:
        np.random.seed(1235)
        os.makedirs("PDP_test", exist_ok=True)
        for n_nodes in [40, 200, 300]:
            n_samples = 1000
            data = list(zip(np.random.uniform(size=(n_samples, 2)).tolist(),  # Depot location
                            np.random.uniform(size=(n_samples, n_nodes, 2)).tolist()))
            with open("PDP_test/pdp_" + str(n_nodes) + ".pkl", "wb") as f:
                pickle.dump(data, f)
