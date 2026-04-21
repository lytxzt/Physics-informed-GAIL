from copy import deepcopy
from torch.optim import Adam
import Utils
from Configurations import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm

best_hidden_dimension = 500
best_dropout = 0.1
best_lr = 0.0001

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

class GDR_TS:
    """
    GDR-TS Baseline Implementation (Online Optimization Version)
    Based on WCNC 2025: UAV Swarm Network Topology Self-Healing via Graph-Based DRL
    """
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout
        self.gcn_network = GATN_fixed_structure(
            nfeat=config_dimension, nhid=self.hidden_dimension, nclass=config_dimension, 
            dropout=self.dropout_value, if_dropout=True, bias=True
        )
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.gcn_network.cuda()

        self.optimizer = Adam(self.gcn_network.parameters(), lr=best_lr)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor 

    def gdr_ts(self, global_positions, remain_list):
        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        remain_positions = np.array(remain_positions)
        num_remain = len(remain_list)

        # Step 1: build a heuristic target adjacency and contractive map.
        d_min = Utils.smallest_d_algorithm(deepcopy(remain_positions), num_remain, config_communication_range)
        d_max = Utils.calculate_d_max(deepcopy(remain_positions))
        d = d_min + (d_max - d_min) * 0.25

        edge_set = []
        A_mat = np.zeros((num_remain, num_remain))
        for i in range(len(remain_list)):
            for j in range(i+1, len(remain_list)):
                if np.linalg.norm(remain_positions[i]-remain_positions[j]) <= d:
                    edge_set.append([i, j])
                    edge_set.append([j, i])
                    A_mat[i, j] = 1
                    A_mat[j, i] = 1

        A_hat = np.array(edge_set).T
        
        D_mat = np.diag(np.sum(A_mat, axis=1))
        L_mat = D_mat - A_mat
        
        max_degree = np.max(np.sum(A_mat, axis=1))
        theta = 1.0 / (max_degree + 1e-5) if max_degree > 0 else 0
        
        P_hat = deepcopy(remain_positions)
        I = np.eye(num_remain)
        CMO_op = I - theta * L_mat
        
        q_iterations = 5 
        for _ in range(q_iterations):
            P_hat = np.matmul(CMO_op, P_hat)

        P_hat_tensor = torch.FloatTensor(P_hat).type(self.FloatTensor)
        remain_positions_tensor = torch.FloatTensor(remain_positions).type(self.FloatTensor)
        A_hat_tensor = torch.LongTensor(A_hat).type(self.LongTensor)

        best_final_positions = 0
        best_loss = 1000000000000
        
        # Step 2: fine-tune positions with the GAT module.
        for train_step in range(1000):
            final_positions = self.gcn_network(P_hat_tensor, A_hat_tensor)
            
            final_positions = 0.5 * torch.Tensor(config_space_range).type(self.FloatTensor) * final_positions

            final_positions_ = final_positions.cpu().data.numpy()
            A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
            D = Utils.make_D_matrix(A, len(A))
            L = D - A
            flag, num = Utils.check_number_of_clusters(L, len(L))

            mu = 1000.0
            max_moving_distance = torch.max(torch.norm(final_positions - remain_positions_tensor, dim=1))
            loss = mu * (num - 1) + max_moving_distance
            
            if loss.cpu().data.numpy() < best_loss:
                best_loss = deepcopy(loss.cpu().data.numpy())
                best_final_positions = deepcopy(final_positions_)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_ = loss.cpu().data.numpy()
            print("    GDR-TS fine-tuning step %d, loss %f, best loss %f" % (train_step, loss_, best_loss), end='\r')
        
        # Step 3: extract directions and the estimated completion time.
        speed = np.zeros((config_num_of_agents, config_dimension))
        temp_max_distance = 0

        for i in range(num_remain):
            dist = np.linalg.norm(best_final_positions[i] - remain_positions[i])
            if dist > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions[i]) / dist
            if dist > temp_max_distance:
                temp_max_distance = dist

        max_time = temp_max_distance / config_constant_speed
        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)


class GATN_fixed_structure(nn.Module):
    """GATN Module from GDR-TS paper"""
    def __init__(self, nfeat=3, nhid=500, nclass=3, n_heads=4, dropout=0.5, alpha=0.2, if_dropout=True, bias=True):
        super(GATN_fixed_structure, self).__init__()
        self.gat1 = GATv2Conv(in_channels=nfeat, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias, add_self_loops=True)
        self.gat2 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat3 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat4 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat5 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat6 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nclass, heads=n_heads, concat=False, negative_slope=alpha, dropout=0, bias=bias)
                
        self.norm1 = LayerNorm(nhid * n_heads)
        self.norm2 = LayerNorm(nhid * n_heads)
        self.norm3 = LayerNorm(nhid * n_heads)
        self.norm4 = LayerNorm(nhid * n_heads)
        self.norm5 = LayerNorm(nhid * n_heads)
        self.norm6 = LayerNorm(nclass)

        self.dropout = dropout
        self.training = if_dropout
    
    def forward(self, x, adj):
        x = self.norm1(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm2(self.gat2(x, adj))
        x = self.norm3(self.gat3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm4(self.gat4(x, adj))
        x = self.norm5(self.gat5(x, adj))
        x = self.norm6(self.gat6(x, adj))
        return torch.tanh(x) + 1
