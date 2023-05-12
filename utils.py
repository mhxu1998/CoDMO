import pickle
import numpy as np
import itertools


# A priori statistical co-occurrence of medical codes from patient’s history visits
def co_matrix_building(seqs_file, icd_nums, threshold):
    seqs = pickle.load(open(seqs_file, 'rb'))
    co_matrix = np.empty((icd_nums, icd_nums))

    # Statistical co-occurrence information
    for patient in seqs:
        for visit in patient:
            co = list(itertools.permutations(visit, 2))
            for item in co:
                co_matrix[item[0]][item[1]] += 1

    # find nearest neighbors with threshold
    for i in range(len(co_matrix[0])):
        if co_matrix[i].sum() == 0:
            continue
        co_matrix[i] = co_matrix[i]/co_matrix[i].sum()
    tmp = np.where(co_matrix >= threshold, co_matrix, 0)
    binary_matrix = np.where(tmp < threshold, tmp, 1)

    weight_matrix = co_matrix.copy()
    # nodes, neighbors, masks, weights = build_node_neighbor(binary_matrix, weight_matrix)

    return binary_matrix, weight_matrix


# A priori statistical co-occurrence of diagnosis and procedure codes from patient’s history visits
def co_matrix_with_diag_proc(seqs_file_diag, seqs_file_proc, icd_nums_diag, icd_nums_proc):
    diag_seqs = pickle.load(open(seqs_file_diag, 'rb'))
    proc_seqs = pickle.load(open(seqs_file_proc, 'rb'))
    matrix = np.empty((icd_nums_diag, icd_nums_proc))

    for (patient_d, patient_p) in zip(diag_seqs, proc_seqs):
        for (visit_d, visit_p) in zip(patient_d, patient_p):
            for i in visit_d:
                for j in visit_p:
                    matrix[i][j] += 1

    co_matrix = matrix/(matrix.sum(axis=1)[:, None]+0.000001)
    return co_matrix


# Construct nearest neighbor node
def build_node_neighbor(binary_matrix, weight_matrix):
    neighbors = []
    neighbor_weight = []
    for i in range(len(binary_matrix)):
        tmp_neighbors = []
        tmp_weight = []
        for j in range(len(binary_matrix[i])):
            if binary_matrix[i][j] == 1:
                tmp_neighbors.append(j)
                tmp_weight.append(weight_matrix[i][j])
        neighbors.append(tmp_neighbors)
        neighbor_weight.append(tmp_weight)
    nums = [len(neighbors[i]) for i in range(len(neighbors))]
    node_index = [i for i in range(len(binary_matrix))]

    node_info = zip(nums, node_index, neighbors, neighbor_weight)
    sortedinfo = sorted(node_info, key=lambda x: x[0])
    result = zip(*sortedinfo)
    neighbor_nums, all_nodes, all_neighbors, all_neighbors_weight = [list(x) for x in result]

    nums = np.array(neighbor_nums)
    maxsize = np.array(neighbor_nums).max()
    # Partition arrays according to the maximum number of neighbors to reduce storage space
    if maxsize < 50:
        divided = [0, 20, maxsize]
    elif maxsize < 100:
        divided = [0, 20, 50, maxsize]
    elif maxsize < 200:
        divided = [0, 20, 50, 100, maxsize]
    else:
        divided = [0, 20, 50, 100, 200, maxsize]
    divided_index = [0]
    for i in range(len(divided) - 1):
        tmp = len(np.where(nums < divided[i + 1])[0]) + 1
        divided_index.append(tmp)

    new_nodes = []              # nodes list
    new_neighbors = []          # the corresponding neighbors list
    masks = []                  # mask used to unify the length of the neighbors
    new_weights = []            # the weight of corresponding neighbors
    for i in range(len(divided_index) - 1):
        nodes = all_nodes[divided_index[i]:divided_index[i + 1]]
        neighbors = all_neighbors[divided_index[i]:divided_index[i + 1]]
        weights = all_neighbors_weight[divided_index[i]:divided_index[i + 1]]
        max_neighbor = max(len(x) for x in neighbors) + 1
        mask = []
        weight = []
        neighbor = []
        for n, ne, w in zip(nodes, neighbors, weights):
            if len(ne) == 0:
                continue
            # Adds the current node itself and sets its weight
            k = [n] + ne
            h = [sum(w)] + w
            cur_mask = [0.0] * len(k)
            if len(k) < max_neighbor:
                cur_mask += [-10 ** 17] * (max_neighbor - len(k))
                k += [0] * (max_neighbor - len(k))
                h += [-10 ** 17] * (max_neighbor - len(h))
            mask.append(cur_mask)
            neighbor.append(k)
            weight.append(h)

        node = []
        for n, ne in zip(nodes, neighbors):
            if len(ne) == 0:
                continue
            node.append([n] * max_neighbor)
        node = np.array(node)
        neighbor = np.array(neighbor)
        mask = np.array(mask)
        weight = np.array(weight)
        new_nodes.append(node)
        new_neighbors.append(neighbor)
        masks.append(mask)
        new_weights.append(weight)
    return new_nodes, new_neighbors, masks, new_weights


# Build a bidirectional hierarchy
def build_hierarchical_tree(p2cFile, level):
    p2c, c2p = pickle.load(open(p2cFile, 'rb'))

    # Down to Top (in a set form)
    p2c_parent = []
    p2c_children = []
    p2c_mask = []
    for i, p2c_i in enumerate(p2c[::-1][:level]):
        children = p2c_i.values()
        parent = p2c_i.keys()
        parent = np.expand_dims(list(parent), axis=1)
        max_n_children = max(len(x) for x in children) + 1
        mask = []
        child = []
        for p, c in zip(parent, children):
            k = list(p) + list(c)
            cur_mask = [0.0] * len(k)
            if len(k) < max_n_children:
                cur_mask += [-10**17] * (max_n_children - len(k))
                k += [0] * (max_n_children - len(k))
            mask.append(cur_mask)
            child.append(k)
        children = child
        parents = []
        for p in p2c_i.keys():
            parents.append([p] * max_n_children)
        parents = np.array(parents)
        children = np.array(children)
        mask = np.array(mask)
        p2c_parent.append(parents)
        p2c_children.append(children)
        p2c_mask.append(mask)

    # Top to Down (in a path form)
    c2p_parents = []
    c2p_child = []
    c2p_mask = []

    for i, c2p_i in enumerate(c2p[:]):
        parents = c2p_i.values()
        child = c2p_i.keys()
        child = np.expand_dims(list(child), axis=1)
        max_n_parents = max(len(list(list(x)[0])) for x in parents) + 1
        mask = []
        parent = []
        for c, p in zip(child, parents):
            k = list(list(p)[0]) + list(c)
            cur_mask = [0.0] * len(k)
            if len(k) < max_n_parents:
                cur_mask += [-10**17] * (max_n_parents - len(k))
                k += [0] * (max_n_parents - len(k))
            mask.append(cur_mask)
            parent.append(k)
        parents = parent
        children = []
        for c in c2p_i.keys():
            children.append([c] * max_n_parents)
        parents = np.array(parents)
        children = np.array(children)
        mask = np.array(mask)
        c2p_parents.append(parents)
        c2p_child.append(children)
        c2p_mask.append(mask)
    return p2c_parent, p2c_children, p2c_mask, c2p_parents, c2p_child, c2p_mask


def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in set(visit):
                codeSet.add(code)
    return max(codeSet) + 1


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    #print(list(tree.values()))
    rootCode = list(tree.values())[0][1]
    return rootCode


def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()



