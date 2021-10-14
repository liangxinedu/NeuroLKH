import torch
import torch.nn.functional as F
import torch.nn as nn

from net.sgcn_layers import SparseGCNLayer, MLP

def loss_edges(y_pred_edges, y_edges, edge_cw):
    y_pred_edges = y_pred_edges.permute(0, 2, 1)  # batch_size x 2 x n_node * n_edge
    loss_edges = nn.NLLLoss(edge_cw, reduction="none")(y_pred_edges, y_edges)
    return loss_edges

class SparseGCNModel(nn.Module):
    def __init__(self, hidden_dim=128, n_gcn_layers=30, n_mlp_layers=3, problem="tsp"):
        super(SparseGCNModel, self).__init__()
        if problem == "tsp":
            self.node_dim = 2 # x, y
        elif problem == "cvrp":
            self.node_dim = 4 # x, y, demand, capacity
        elif problem == "pdp":
            self.node_dim = 5 # x, y, depot, pickup, delivery
        elif problem == "cvrptw":
            self.node_dim = 6 # x, y, demand, start_time, end_time, capacity
        else:
            assert False
        self.edge_dim = 1
        self.hidden_dim = hidden_dim
        self.n_gcn_layers = n_gcn_layers
        self.n_mlp_layers = n_mlp_layers
        self.aggregation = "mean"
        self.problem = problem

        self.nodes_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        gcn_layers = []
        for layer in range(self.n_gcn_layers):
            gcn_layers.append(SparseGCNLayer(self.hidden_dim, self.aggregation, is_pdp=problem=="pdp"))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        if problem == "pdp" or problem == "cvrptw":
            self.mlp_edges = MLP(self.hidden_dim, 2, self.n_mlp_layers)
        else:
            self.mlp_edges = MLP(self.hidden_dim, 1, self.n_mlp_layers)
        self.mlp_nodes = MLP(self.hidden_dim, 1, self.n_mlp_layers)

    def forward(self, x_nodes, x_edges, edge_index, inverse_edge_index, y_edges, edge_cw, n_edges):
        batch_size, num_nodes, _ = x_nodes.size()
        x = self.nodes_embedding(x_nodes)  # batch_size x n_node x hidden_dimension
        e = self.edges_embedding(x_edges)  # batch_size x n_node * n_edge x hidden_dimension
        loss_mask = (edge_index.view(batch_size, num_nodes, n_edges).sum(-1) != 0)

        for layer in range(self.n_gcn_layers):
            x, e = self.gcn_layers[layer](x, e, edge_index, inverse_edge_index, n_edges)
        y_pred_edges = self.mlp_edges(e).view(batch_size, num_nodes, n_edges)
        y_pred_edges = torch.exp(y_pred_edges)
        y_pred_edges = y_pred_edges / (y_pred_edges.sum(2).view(batch_size, num_nodes, 1) + 1e-5)
        
        y_pred_edges = y_pred_edges.view(batch_size, num_nodes * n_edges, 1)
        y_pred_edges = torch.cat([1 - y_pred_edges, y_pred_edges], dim = 2)
        y_pred_edges = torch.log(y_pred_edges)
        if y_edges != None:
            loss = loss_edges(y_pred_edges, y_edges, edge_cw)
            loss = loss.view(batch_size, num_nodes, n_edges)[loss_mask]
        else:
            loss = None

        y_pred_nodes = self.mlp_nodes(x)
        y_pred_nodes = 10 * torch.tanh(y_pred_nodes)
        return y_pred_edges, loss, y_pred_nodes

    def forward_finetune(self, x_nodes, x_edges, edge_index, inverse_edge_index, n_edges):
        with torch.no_grad():
            batch_size, num_nodes, _ = x_nodes.size()
            x = self.nodes_embedding(x_nodes)
            e = self.edges_embedding(x_edges)

            for layer in range(self.n_gcn_layers):
                x, e = self.gcn_layers[layer](x, e, edge_index, inverse_edge_index, n_edges)

        y_pred_nodes = self.mlp_nodes(x.detach())
        y_pred_nodes = 10 * torch.tanh(y_pred_nodes)

        return y_pred_nodes

    def directed_forward(self, x_nodes, x_edges, edge_index, inverse_edge_index, y_edges1, y_edges2, edge_cw, n_edges):
        batch_size, num_nodes, _ = x_nodes.size()
        if self.problem == "pdp":
            node_state = torch.zeros((batch_size, num_nodes, 3), device=x_nodes.device)
            node_state[:, 0, 0] = 1
            node_state[:, 1 : num_nodes // 2 + 1, 1] = 1
            node_state[:, num_nodes // 2 + 1:, 2] = 1
            x_nodes = torch.cat([x_nodes, node_state], 2)

        x = self.nodes_embedding(x_nodes)
        e = self.edges_embedding(x_edges)
        loss_mask = (edge_index.view(batch_size, num_nodes, n_edges).sum(-1) != 0)

        for layer in range(self.n_gcn_layers):
            x, e = self.gcn_layers[layer](x, e, edge_index, inverse_edge_index, n_edges)
        y_pred_edges = self.mlp_edges(e).view(batch_size, num_nodes, n_edges, 2)

        y_pred_edges1 = torch.exp(y_pred_edges[:, :, :, 0])
        y_pred_edges1 = y_pred_edges1 / (y_pred_edges1.sum(2).view(batch_size, num_nodes, 1) + 1e-5)
        y_pred_edges1 = y_pred_edges1.view(batch_size, num_nodes * n_edges, 1)
        y_pred_edges1 = torch.cat([1 - y_pred_edges1, y_pred_edges1], dim = 2)
        y_pred_edges1 = torch.log(y_pred_edges1)
        y_pred_edges2 = torch.exp(y_pred_edges[:, :, :, 1])
        y_pred_edges2 = y_pred_edges2 / (y_pred_edges2.sum(2).view(batch_size, num_nodes, 1) + 1e-5)
        y_pred_edges2 = y_pred_edges2.view(batch_size, num_nodes * n_edges, 1)
        y_pred_edges2 = torch.cat([1 - y_pred_edges2, y_pred_edges2], dim = 2)
        y_pred_edges2 = torch.log(y_pred_edges2)
        if y_edges1 is None:
            return y_pred_edges1, y_pred_edges2, None, None, None
        loss1 = loss_edges(y_pred_edges1, y_edges1, edge_cw)
        loss1 = loss1.view(batch_size, num_nodes, n_edges)[loss_mask]
        loss2 = loss_edges(y_pred_edges2, y_edges2, edge_cw)
        loss2 = loss2.view(batch_size, num_nodes, n_edges)[loss_mask]
        return y_pred_edges1, y_pred_edges2, loss1, loss2, None
