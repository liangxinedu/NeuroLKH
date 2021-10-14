import torch
import torch.nn.functional as F
import torch.nn as nn


class BatchNormNode(nn.Module):

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        return x_bn

class NodeFeatures(nn.Module):
    
    def __init__(self, hidden_dim, aggregation="mean", is_pdp=False):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.node_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.to_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.edge_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.is_pdp = is_pdp
        if self.is_pdp:
            self.pickup_embedding = nn.Linear(hidden_dim, hidden_dim, True)
            self.deliver_embedding = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e, edge_index, n_edges):
        batch_size, num_nodes, hidden_dim = x.size()
        Ux = self.node_embedding(x)  # batch_size x n_node x hidden_dimension
        Vx = self.to_embedding(x)  # batch_size x n_node x hidden_dimension
        if self.is_pdp:
            Px = self.pickup_embedding(x[:, 1:num_nodes // 2 + 1, :])
            Dx = self.deliver_embedding(x[:, num_nodes // 2 + 1:, :])
        Ve = self.edge_embedding(e) # batch_size x n_node * n_edge x hidden_dimension
        Ve = F.softmax(Ve.view(batch_size, num_nodes, n_edges, hidden_dim), dim=2)
        Ve = Ve.view(batch_size, num_nodes * n_edges, hidden_dim)

        Vx = Vx[torch.arange(batch_size).view(-1, 1), edge_index] # batch_size x n_node * n_edge x hidden_dimension
        # Vx = torch.matmul(Vx.permute(0, 2, 1), adj_to).permute(0, 2, 1)
        to = Ve * Vx
        to = to.view(batch_size, num_nodes, n_edges, hidden_dim).sum(2) # batch_size x n_node x hidden_dimension
        x_new = Ux + to
        if self.is_pdp:
            x_new[:, 1:num_nodes // 2 + 1, :] += Dx
            x_new[:, num_nodes // 2 + 1:, :] += Px
        return x_new


class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_from = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_to = nn.Linear(hidden_dim, hidden_dim, True)
        self.inverse_U = nn.Linear(hidden_dim, hidden_dim, True)

        self.W_placeholder = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def forward(self, x, e, edge_index, inverse_edge_index, n_edges):
        batch_size, graph_size, hidden_dim = x.size()
        Ue = self.U(e) # batch_size x n_node * n_edge x hidden_dimension
        inverse_Ue = self.inverse_U(e) # batch_size x n_node * n_edge x hidden_dimension
        inverse_Ue = torch.cat((inverse_Ue, self.W_placeholder.view(1, 1, hidden_dim).repeat(batch_size, 1, 1)), 1) # batch_size x (n_node * n_edge + 1) x hidden_dimension
        inverse_node_embedding = inverse_Ue[torch.arange(batch_size).view(batch_size, 1), inverse_edge_index]

        Vx_from = self.V_from(x) # batch_size x n_node x hidden_dimension
        Vx_to = self.V_to(x) # batch_size x n_node x hidden_dimension
        Vx = Vx_to[torch.arange(batch_size).view(-1, 1), edge_index] # batch_size x n_node * n_edge x hidden_dimension
        # Vx = torch.matmul(Vx_to.permute(0, 2, 1), adj_to).permute(0, 2, 1) # batch_size x n_node * n_edge x hidden_dimension
        Vx = Vx.view(batch_size, -1, n_edges, 128) + Vx_from.view(batch_size, -1, 1, 128)
        Vx = Vx.view(batch_size, -1, 128)
        # torch.matmul(Vx_from.permute(0, 2, 1), adj - adj_to).permute(0, 2, 1) # batch_size x n_node * n_edge x hidden_dimension
        e_new = Ue + Vx + inverse_node_embedding
        return e_new


class SparseGCNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="mean", is_pdp=False):
        super(SparseGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation, is_pdp)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormNode(hidden_dim)

    def forward(self, x, e, edge_index, inverse_edge_index, n_edges):
        e_in = e # batch_size x n_node * n_edge x hidden_dimension
        x_in = x # batch_size x n_node x hidden_dimension

        x_tmp = self.node_feat(x_in, e_in, edge_index.long(), n_edges)
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x

        e_tmp = self.edge_feat(x_new, e_in, edge_index.long(), inverse_edge_index.long(), n_edges)  # batch_size x n_node * n_edge x hidden_dimension
        e_tmp = self.bn_edge(e_tmp)
        e = F.relu(e_tmp)
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)
            Ux = F.relu(Ux)
        y = self.V(Ux)
        return y
