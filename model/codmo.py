import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 4396
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_embedding(embFile):
    m = np.load(embFile)
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


class Leaf_attention(nn.Module):
    def __init__(self, embDimSize, attentionDimSize, device):
        super(Leaf_attention, self).__init__()
        # attention parameters
        self.Leaf_W_attention = nn.Parameter(torch.rand(embDimSize * 2, attentionDimSize, requires_grad=True))
        self.Leaf_b_attention = nn.Parameter(torch.zeros(attentionDimSize, requires_grad=True))
        self.Leaf_v_attention = nn.Parameter(torch.rand(attentionDimSize, requires_grad=True))

        self.device = device

    def forward(self, Leaf_emb, nodes, neighbors, masks, weights):
        W_tmp = Leaf_emb.clone()
        for i, (node, neighbor, mask, weight) in enumerate(zip(nodes, neighbors, masks, weights)):
            mask = torch.Tensor(mask).to(self.device)
            weight = torch.Tensor(weight).to(self.device)
            if len(node.shape) == 2:
                node = node[np.newaxis, :, :]
                neighbor = neighbor[np.newaxis, :, :]
            node_tmp = Leaf_emb[node]
            neighbor_tmp = W_tmp[neighbor]
            attentionInput = torch.cat((node_tmp, neighbor_tmp), 2)
            tmp = torch.matmul(attentionInput, self.Leaf_W_attention)
            tmp = tmp + self.Leaf_b_attention
            mlpOutput = nn.LeakyReLU()(tmp)
            preAttention = torch.matmul(mlpOutput, self.Leaf_v_attention)
            preAttention_mask = preAttention + mask
            tempAttention = F.softmax(preAttention_mask, dim=1)
            tempWeight = F.softmax(weight, dim=1)
            tempAttention = tempAttention*tempWeight
            tempEmb = (neighbor_tmp * tempAttention[:, :, None]).sum(axis=1)
            W_tmp[node[:, :, 0]] = tempEmb

        return W_tmp


class Attention(nn.Module):
    def __init__(self, embDimSize, attentionDimSize, device):
        super(Attention, self).__init__()
        # attention parameters
        self.W_attention = nn.Parameter(torch.rand(embDimSize * 2, attentionDimSize, requires_grad=True))
        self.b_attention = nn.Parameter(torch.zeros(attentionDimSize, requires_grad=True))
        self.v_attention = nn.Parameter(torch.rand(attentionDimSize, requires_grad=True))
        self.gru = torch.nn.GRU(input_size=embDimSize, hidden_size=embDimSize)

        self.device = device

    def forward(self, W_emb, p2c_parent, p2c_children, p2c_mask, c2p_parents, c2p_child, c2p_mask):
        W_tmp = W_emb
        for i, (parents, children, mask) in enumerate(zip(p2c_parent, p2c_children, p2c_mask)):
            mask = torch.Tensor(mask).to(self.device)
            if len(parents.shape) == 2:
                parents = parents[np.newaxis, :, :]
                children = children[np.newaxis, :, :]
            parents_tmp = W_emb[parents]
            if i == 0:
                children_tmp = W_emb[children]
            else:
                children_tmp = W_tmp[children]
            attentionInput = torch.cat((parents_tmp, children_tmp), 2)
            tmp = torch.matmul(attentionInput, self.W_attention)
            tmp = tmp + self.b_attention
            mlpOutput = nn.LeakyReLU()(tmp)
            preAttention = torch.matmul(mlpOutput, self.v_attention)
            preAttention_mask = preAttention + mask
            tempAttention = F.softmax(preAttention_mask, dim=1)
            tempEmb = (children_tmp * tempAttention[:, :, None]).sum(axis=1)
            tempEmb = (tempEmb + parents_tmp[:, 0, :])/2
            W_tmp[parents[:, :, 0]] = tempEmb

        for i, (parents, children, mask) in enumerate(zip(c2p_parents, c2p_child, c2p_mask)):
            if len(children.shape) == 2:
                parents = parents[np.newaxis, :, :]
                children = children[np.newaxis, :, :]
            parents_tmp = W_tmp[parents]
            children_tmp = W_tmp[children]
            parents_tmp = torch.transpose(parents_tmp, 0, 1)
            children_tmp = torch.transpose(children_tmp, 0, 1)
            output, hn = self.gru(parents_tmp)
            out = torch.transpose(output, 0, 1)[:, -1, :]
            out = (out + children_tmp[0, :, :])/2
            W_tmp[children[:, :, 0]] = out

        return W_tmp


class CoDMO(nn.Module):
    def __init__(self, inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc,
                 embDimSize, hiddenDimSize, attentionDimSize, dropout_rate, embFile_diag, embFile_proc, dim_y, device,
                 p2c_parent_diag, p2c_children_diag, p2c_mask_diag, c2p_parents_diag, c2p_child_diag, c2p_mask_diag, nodes_diag, neighbors_diag, masks_diag, weights_diag,
                 p2c_parent_proc, p2c_children_proc, p2c_mask_proc, c2p_parents_proc, c2p_child_proc, c2p_mask_proc, nodes_proc, neighbors_proc, masks_proc, weights_proc):
        super(CoDMO, self).__init__()

        self.inputDimSize_diag = inputDimSize_diag
        self.inputDimSize_proc = inputDimSize_proc
        self.device = device

        # Initial embedding
        if len(embFile_diag) > 0:
            self.W_emb_diag = nn.Parameter(torch.from_numpy(np.float32(load_embedding(embFile_diag))), requires_grad=True)
        else:
            self.W_emb_diag = nn.Parameter(torch.rand((inputDimSize_diag + numAncestors_diag), embDimSize), requires_grad=True)

        if len(embFile_proc) > 0:
            self.W_emb_proc = nn.Parameter(torch.from_numpy(np.float32(load_embedding(embFile_proc))), requires_grad=True)
        else:
            self.W_emb_proc = nn.Parameter(torch.rand((inputDimSize_proc + numAncestors_proc), embDimSize), requires_grad=True)

        # Attention layer
        self.leafAttention = Leaf_attention(embDimSize, attentionDimSize, device)
        self.leafAttention2 = Leaf_attention(embDimSize, attentionDimSize, device)
        self.attentionLayer = Attention(embDimSize, attentionDimSize, device)

        # GRU parameters
        self.gru_diag = torch.nn.GRU(input_size=embDimSize, hidden_size=hiddenDimSize, bidirectional=True)
        self.gru_proc = torch.nn.GRU(input_size=embDimSize, hidden_size=hiddenDimSize, bidirectional=True)
        self.gru_cat = torch.nn.GRU(input_size=2*embDimSize, hidden_size=hiddenDimSize, bidirectional=True)

        self.dropout_rate = dropout_rate

        self.W_output_diag = nn.Parameter(torch.rand(hiddenDimSize*2, numClass_diag), requires_grad=True)
        self.b_output_diag = nn.Parameter(torch.zeros(numClass_diag), requires_grad=True)
        self.W_output_proc = nn.Parameter(torch.rand(hiddenDimSize*2 + dim_y, numClass_proc), requires_grad=True)
        self.b_output_proc = nn.Parameter(torch.zeros(numClass_proc), requires_grad=True)

        self.W_y = nn.Parameter(torch.rand(numClass_diag, dim_y), requires_grad=True)
        self.b_y = nn.Parameter(torch.zeros(dim_y), requires_grad=True)

        self.W_attention = nn.Parameter(torch.rand(embDimSize+hiddenDimSize*2, attentionDimSize, requires_grad=True))
        self.b_attention = nn.Parameter(torch.zeros(attentionDimSize, requires_grad=True))
        self.v_attention = nn.Parameter(torch.rand(attentionDimSize, requires_grad=True))

        self.p2c_parent_diag = p2c_parent_diag
        self.p2c_children_diag = p2c_children_diag
        self.p2c_mask_diag = p2c_mask_diag
        self.c2p_parents_diag = c2p_parents_diag
        self.c2p_child_diag = c2p_child_diag
        self.c2p_mask_diag = c2p_mask_diag
        self.nodes_diag = nodes_diag
        self.neighbors_diag = neighbors_diag
        self.masks_diag = masks_diag
        self.weights_diag = weights_diag
        self.p2c_parent_proc = p2c_parent_proc
        self.p2c_children_proc = p2c_children_proc
        self.p2c_mask_proc = p2c_mask_proc
        self.c2p_parents_proc = c2p_parents_proc
        self.c2p_child_proc = c2p_child_proc
        self.c2p_mask_proc = c2p_mask_proc
        self.nodes_proc = nodes_proc
        self.neighbors_proc = neighbors_proc
        self.masks_proc = masks_proc
        self.weights_proc = weights_proc

    def forward(self, x_diag, mask_diag, code_mask_diag, x_proc, mask_proc, code_mask_proc):

        emb_diag = self.leafAttention(self.W_emb_diag, self.nodes_diag, self.neighbors_diag, self.masks_diag, self.weights_diag)
        emb_proc = self.leafAttention(self.W_emb_proc, self.nodes_proc, self.neighbors_proc, self.masks_proc, self.weights_proc)

        emb_diag2 = self.attentionLayer(emb_diag, self.p2c_parent_diag, self.p2c_children_diag, self.p2c_mask_diag, self.c2p_parents_diag, self.c2p_child_diag, self.c2p_mask_diag)
        emb_proc2 = self.attentionLayer(emb_proc, self.p2c_parent_proc, self.p2c_children_proc, self.p2c_mask_proc, self.c2p_parents_proc, self.c2p_child_proc, self.c2p_mask_proc)

        emb_diag3 = self.leafAttention2(emb_diag2, self.nodes_diag, self.neighbors_diag, self.masks_diag, self.weights_diag)
        emb_proc3 = self.leafAttention2(emb_proc2, self.nodes_proc, self.neighbors_proc, self.masks_proc, self.weights_proc)

        embList_diag = (emb_diag + emb_diag2 + emb_diag3)
        embList_proc = (emb_proc + emb_proc2 + emb_proc3)

        # input layer and bi-gru of diagnosis
        x_diag = torch.transpose(x_diag, 0, 1)
        code_mask_diag = torch.transpose(code_mask_diag, 0, 1)
        x_emb_diag = embList_diag[:self.inputDimSize_diag][x_diag.long()]
        x_emb_diag_new = torch.matmul(code_mask_diag[:, :, None], x_emb_diag).squeeze(2)
        x_emb_diag = torch.transpose(x_emb_diag_new, 0, 1)
        x_emb_diag = torch.tanh(x_emb_diag)

        hidden_diag = torch.cat((x_emb_diag, x_emb_diag), dim=2)
        for i in range(x_emb_diag.shape[0]):
            hd, hn = self.gru_diag(x_emb_diag[0:i+1, :, :])
            hd = F.dropout(hd, p=self.dropout_rate)
            hidden_diag[i, :, :] = hd[i, :, :]

        # input layer and bi-gru of procedure
        x_proc = torch.transpose(x_proc, 0, 1)
        code_mask_proc = torch.transpose(code_mask_proc, 0, 1)
        x_emb_proc = embList_proc[:self.inputDimSize_proc][x_proc.long()]
        x_emb_proc_new = torch.matmul(code_mask_proc[:, :, None], x_emb_proc).squeeze(2)
        x_emb_proc = torch.transpose(x_emb_proc_new, 0, 1)
        x_emb_proc = torch.tanh(x_emb_proc)

        hidden_proc = torch.cat((x_emb_proc, x_emb_proc), dim=2)
        for i in range(x_emb_proc.shape[0]):
            hd, hn = self.gru_proc(x_emb_proc[0:i+1, :, :])
            hd = F.dropout(hd, p=self.dropout_rate)
            hidden_proc[i, :, :] = hd[i, :, :]

        # concatenate of diagnosis and procedure
        x_emb_cat = torch.cat((x_emb_diag, x_emb_proc), 2)
        hidden_cat = torch.cat((x_emb_diag, x_emb_proc), 2)
        for i in range(x_emb_cat.shape[0]):
            hd, hn = self.gru_cat(x_emb_cat[0:i+1, :, :])
            hd = F.dropout(hd, p=self.dropout_rate)
            hidden_cat[i, :, :] = hd[i, :, :]

        # diag-visit-attention
        diag_cat1 = torch.cat((x_emb_diag, hidden_diag), 2)
        diag_cat2 = torch.cat((x_emb_diag, hidden_cat), 2)
        tmp_diag = torch.matmul(diag_cat1, self.W_attention)
        tmp_diag = tmp_diag + self.b_attention
        mlpOutput_diag = torch.tanh(tmp_diag)
        preAttention_diag = torch.matmul(mlpOutput_diag, self.v_attention)
        preAttention_diag = torch.unsqueeze(preAttention_diag, 2)

        tmp_diag_cat = torch.matmul(diag_cat2, self.W_attention)
        tmp_diag_cat = tmp_diag_cat + self.b_attention
        mlpOutput_diag_cat = torch.tanh(tmp_diag_cat)
        preAttention_diag_cat = torch.matmul(mlpOutput_diag_cat, self.v_attention)
        preAttention_diag_cat = torch.unsqueeze(preAttention_diag_cat, 2)

        tempAttention_diag = torch.cat((preAttention_diag, preAttention_diag_cat), 2)
        tempAttention_diag = F.softmax(tempAttention_diag, dim=2)
        final_hidden_diag = torch.unsqueeze(tempAttention_diag[:, :, 0], 2) * hidden_diag + torch.unsqueeze(tempAttention_diag[:, :, 1], 2) * hidden_cat

        # proc-visit-attention
        proc_cat1 = torch.cat((x_emb_proc, hidden_proc), 2)
        proc_cat2 = torch.cat((x_emb_proc, hidden_cat), 2)
        tmp_proc = torch.matmul(proc_cat1, self.W_attention)
        tmp_proc = tmp_proc + self.b_attention
        mlpOutput_proc = torch.tanh(tmp_proc)
        preAttention_proc = torch.matmul(mlpOutput_proc, self.v_attention)
        preAttention_proc = torch.unsqueeze(preAttention_proc, 2)

        tmp_proc_cat = torch.matmul(proc_cat2, self.W_attention)
        tmp_proc_cat = tmp_proc_cat + self.b_attention
        mlpOutput_proc_cat = torch.tanh(tmp_proc_cat)
        preAttention_proc_cat = torch.matmul(mlpOutput_proc_cat, self.v_attention)
        preAttention_proc_cat = torch.unsqueeze(preAttention_proc_cat, 2)

        tempAttention_proc = torch.cat((preAttention_proc, preAttention_proc_cat), 2)
        tempAttention_proc = F.softmax(tempAttention_proc, dim=2)
        final_hidden_proc = torch.unsqueeze(tempAttention_proc[:, :, 0], 2) * hidden_proc + torch.unsqueeze(tempAttention_proc[:, :, 1], 2) * hidden_cat

        # diagnosis prediction
        output_diag = torch.sigmoid(torch.matmul(final_hidden_diag, self.W_output_diag) + self.b_output_diag)
        y_diag = output_diag * mask_diag[:, :, None]

        # procedure prediction
        y_tmp = torch.tanh(torch.matmul(y_diag.detach(), self.W_y) + self.b_y)
        final_hidden_proc = torch.cat((final_hidden_proc, y_tmp), 2)
        output_proc = torch.sigmoid(torch.matmul(final_hidden_proc, self.W_output_proc) + self.b_output_proc)
        y_proc = output_proc * mask_proc[:, :, None]

        # reconstructed interaction
        re_matrix = torch.mm(embList_diag[:self.inputDimSize_diag], embList_proc[:self.inputDimSize_proc].T)
        re_matrix = re_matrix/torch.sum(re_matrix, 1).unsqueeze(-1)

        return y_diag, y_proc, re_matrix

