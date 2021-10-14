import numpy as np
from concorde.tsp import TSPSolver
import pickle
from multiprocessing import Pool
import os
import tqdm
import argparse

def run_wrapper(args):
    return run(*args)

def run(n_nodes, file_dir="train"):
    os.makedirs("concorde_tmpfiles/" + file_dir + "_" + str(n_nodes), exist_ok=True)
    os.chdir("concorde_tmpfiles/" + file_dir + "_" + str(n_nodes))

    if file_dir == "val":
        n_samples = 1000
    else:
        assert file_dir == "train"
        n_samples = (20 * 200 // n_nodes) * 125
    n_neighbours = 20

    x = np.random.uniform(size=[n_samples, n_nodes, 2])

    f = 10000000
    result=[]
    for i in range(n_samples):
        solver = TSPSolver.from_data(x[i,:,0]*f, x[i,:,1]*f, norm='EUC_2D')
        solution=solver.solve()
        q=solution.tour
        q=[int(p) for p in q]
        result.append(q)
     
    dist = x.reshape(n_samples, n_nodes, 1, 2) - x.reshape(n_samples, 1, n_nodes, 2)
    dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
    edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]

    result = np.array(result) # n_samples x n_nodes
    label = np.zeros(shape=[n_samples, n_nodes, n_nodes], dtype="bool")
    label[np.arange(n_samples).reshape(-1, 1), result, np.roll(result, 1, -1)] = True
    label[np.arange(n_samples).reshape(-1, 1), np.roll(result, 1, -1), result] = True
    label = label[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]

    feat = {"node_feat":x, # n_samples x n_nodes x 2
            "edge_feat":edge_feat, # n_samples x n_nodes x n_neighbours
            "edge_index":edge_index, # n_samples x n_nodes x n_neighbours
            "label":label} # n_samples x n_nodes x n_neighbours

    x = feat["node_feat"]
    assert x.shape[0] == n_samples
    dist = x.reshape(n_samples, n_nodes, 1, 2) - x.reshape(n_samples, 1, n_nodes, 2)
    dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
    inverse_edge_index = -np.ones(shape=[n_samples, n_nodes, n_nodes], dtype="int")
    inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_neighbours
    inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
    assert (np.array_equal(edge_index, feat["edge_index"]))
    feat["inverse_edge_index"] = inverse_edge_index

    os.chdir("../..")
    with open(file_dir + "/" + str(n_nodes) + ".pkl", "wb") as f:
        pickle.dump(feat, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-train", action='store_true', help="Generate training and validation datasets")
    parser.add_argument("-test", action='store_true', help="Generate test datasets")
    args = parser.parse_args()
    if args.train:
        os.makedirs("train", exist_ok=True)
        os.makedirs("val", exist_ok=True)
        pool = Pool()
        n_nodes = 101
        args = []
        for node in range(101, 501):
            args.append((node, "train"))
        args += [(100, "val"), (200, "val"), (500, "val")]
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(run_wrapper, [args[i] for i in range(len(args))]), total=len(args)))
    if args.test:
        os.makedirs("test", exist_ok=True)
        n_samples = 1000
        for nodes in [100, 200, 500, 1000, 2000, 5000]:
            np.random.seed(1234)
            x = np.random.uniform(size=[n_samples, nodes, 2])
            with open("test/" + str(nodes) + ".pkl", "wb") as f:
                pickle.dump(x.tolist(), f)
