import argparse
import pickle
import random
import time
import os
import torch
from model.codmo import CoDMO
from utils import *
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from process_mimic import LabelsForData


# Description of the CCS category
class Description(object):
    def __init__(self, ccs_single_file):
        self.ccs_single_dx_df = pd.read_csv(ccs_single_file, header=0, dtype=object)
        self.cat2description = {}
        self.build_maps()

    def build_maps(self):
        for i, row in self.ccs_single_dx_df.iterrows():
            single_cat = row[1][1:-1].strip()
            description = row[2][1:-1]
            self.cat2description[single_cat] = description


# process the input of test sample
def process_input(adm_file, diag_file, proc_file):
    # ICD->CCS single category mapping
    diag_ccs_single_file = 'data/ccs_single_dx_tool_2015.csv'
    proc_ccs_single_file = 'data/ccs_single_pr_tool_2015.csv'
    label4data = LabelsForData(diag_ccs_single_file)
    label4data_proc = LabelsForData(proc_ccs_single_file)

    # print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}          # patient id->admission id
    admDateMap = {}         # admission id->admission date
    infd = open(adm_file, 'r')      # open admission table
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y/%m/%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    # print('Building admission-diagList mapping')
    admDiagMap = {}  # admission id->diag ICD
    admDiagMap_ccs_single = {}  # admission id->diag ICD (ccs single)
    infd = open(diag_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        if len(tokens[4]) == 0:
            continue
        if tokens[4][0]!='"':
            diag = tokens[4]
        else:
            diag = tokens[4][1:-1]
        if len(diag) == 0:
            continue

        try:
            diagStr_ccs_single = 'D_' + label4data.code2single_dx[diag]
        except KeyError:
            try:
                diag = str('0' + diag)
                diagStr_ccs_single = 'D_' + label4data.code2single_dx[diag]
            except KeyError:
                diag = str('0' + diag)
                diagStr_ccs_single = 'D_' + label4data.code2single_dx[diag]
        # diagStr_ccs_single = 'D_' + label4data.code2single_dx[diag]  # e.g. D_99
        diagStr = 'D_' + diag  # e.g. D_40301

        if admId in admDiagMap:
            admDiagMap[admId].append(diagStr)
        else:
            admDiagMap[admId] = [diagStr]

        if admId in admDiagMap_ccs_single:
            admDiagMap_ccs_single[admId].append(diagStr_ccs_single)
        else:
            admDiagMap_ccs_single[admId] = [diagStr_ccs_single]
    infd.close()

    # print('Building admission-procList mapping')
    admProcMap = {}  # admission id->proc ICD
    admProcMap_ccs_single = {}  # admission id->proc ICD (ccs single)
    infd = open(proc_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        if tokens[4][0]!='"':
            proc = tokens[4]
        else:
            proc = tokens[4][1:-1]
        if len(proc) == 0:
            continue

        try:
            procStr_ccs_single = 'P_' + label4data_proc.code2single_dx[proc]
        except KeyError:
            try:
                proc = str('0' + proc)
                procStr_ccs_single = 'P_' + label4data_proc.code2single_dx[proc]
            except KeyError:
                proc = str('0' + proc)
                procStr_ccs_single = 'P_' + label4data_proc.code2single_dx[proc]
        # procStr_ccs_single = 'P_' + label4data_proc.code2single_dx[proc]
        procStr = 'P_' + proc

        if admId in admProcMap:
            admProcMap[admId].append(procStr)
        else:
            admProcMap[admId] = [procStr]
        if admId in admProcMap_ccs_single:
            admProcMap_ccs_single[admId].append(procStr_ccs_single)
        else:
            admProcMap_ccs_single[admId] = [procStr_ccs_single]
    infd.close()

    # print('Building pid-sortedVisits mapping')
    pidDiagMap = {}  # pid->diag
    pidDiagMap_ccs_single = {}
    pidProcMap = {}  # pid->proc
    pidProcMap_ccs_single = {}
    for pid, admIdList in pidAdmMap.items():  # pid->admission id
        new_admIdList = []
        for admId in admIdList:
            if admId in (admDiagMap and admProcMap):
                new_admIdList.append(admId)
        if len(new_admIdList) < 2:  # Records of patients admitted less than 2 times are ignored
            continue

        # Sorted by the admission date
        sortedDiagList = sorted([(admDateMap[admId], admDiagMap[admId]) for admId in new_admIdList])
        pidDiagMap[pid] = sortedDiagList

        sortedDiagList_ccs_single = sorted(
            [(admDateMap[admId], admDiagMap_ccs_single[admId]) for admId in new_admIdList])
        pidDiagMap_ccs_single[pid] = sortedDiagList_ccs_single

        sortedProcList = sorted([(admDateMap[admId], admProcMap[admId]) for admId in new_admIdList])
        pidProcMap[pid] = sortedProcList

        sortedProcList_ccs_single = sorted(
            [(admDateMap[admId], admProcMap_ccs_single[admId]) for admId in new_admIdList])
        pidProcMap_ccs_single[pid] = sortedProcList_ccs_single

    patient_list = list(pidAdmMap.keys())

    # print('Building diagSeqs, diagSeqs_ccs_single, procSeqs, timeSeqs')
    diag_seqs = []
    diag_seqs_ccs_single = []
    proc_seqs = []
    proc_seqs_ccs_single = []
    time_seqs = []
    for pid, visits in pidDiagMap.items():  # diag ICD List of patient
        seq = []
        time = []
        first_time = visits[0][0]  # the first time of admission
        for i, visit in enumerate(visits):
            current_time = visit[0]
            interval = (current_time - first_time).days  # time interval
            seq.append(visit[1])
            time.append(interval)
        diag_seqs.append(seq)
        time_seqs.append(time)

    for pid, visits in pidDiagMap_ccs_single.items():  # diag ICD List (ccs single) of patient
        seq = []
        for i, visit in enumerate(visits):
            seq.append(visit[1])
        diag_seqs_ccs_single.append(seq)

    for pid, visits in pidProcMap.items():  # proc ICD List of patient
        seq = []
        for i, visit in enumerate(visits):
            seq.append(visit[1])
        proc_seqs.append(seq)

    for pid, visits in pidProcMap_ccs_single.items():  # proc ICD List (ccs single) of patient
        seq = []
        for i, visit in enumerate(visits):
            seq.append(visit[1])
        proc_seqs_ccs_single.append(seq)

    # print('Converting strSeqs to intSeqs, and making types for diag and proc code')
    dict_diag = pickle.load(open('processed/diag.dict', 'rb'))             # code->index   e.g. dict_css['D_101']=1
    new_diag_seqs = []          # index list
    # dict_diag['UNK'] = 0
    for patient in diag_seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in dict_diag:
                    # if dict_diag[code]==4622:
                    #     print(code)
                    newVisit.append(dict_diag[code])
                else:
                    dict_diag[code] = len(dict_diag)
                    newVisit.append(dict_diag[code])
            newPatient.append(newVisit)
        new_diag_seqs.append(newPatient)

    dict_diag_ccs_single = pickle.load(open('processed/diag_ccs_single.dict', 'rb'))
    new_diag_seqs_ccs_single = []
    # dict_diag_ccs_single['UNK'] = 0
    for patient in diag_seqs_ccs_single:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in dict_diag_ccs_single:
                    newVisit.append(dict_diag_ccs_single[code])
                else:
                    dict_diag_ccs_single[code] = len(dict_diag_ccs_single)
                    newVisit.append(dict_diag_ccs_single[code])
            newPatient.append(newVisit)
        new_diag_seqs_ccs_single.append(newPatient)

    dict_proc = pickle.load(open('processed/proc.dict', 'rb'))
    new_proc_seqs = []
    # dict_proc['UNK'] = 0
    for patient in proc_seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in dict_proc:
                    newVisit.append(dict_proc[code])
                else:
                    dict_proc[code] = len(dict_proc)
                    newVisit.append(dict_proc[code])
            newPatient.append(newVisit)
        new_proc_seqs.append(newPatient)

    dict_proc_ccs_single = pickle.load(open('processed/proc_ccs_single.dict', 'rb'))
    new_proc_seqs_ccs_single = []
    # dict_diag_ccs_single['UNK'] = 0
    for patient in proc_seqs_ccs_single:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in dict_proc_ccs_single:
                    newVisit.append(dict_proc_ccs_single[code])
                else:
                    dict_proc_ccs_single[code] = len(dict_proc_ccs_single)
                    newVisit.append(dict_proc_ccs_single[code])
            newPatient.append(newVisit)
        new_proc_seqs_ccs_single.append(newPatient)

    # print('Converting seqs to model inputs')
    inputs_diag_all = []
    inputs_proc_all = []
    max_visit_len = 0
    max_diag_len = 0
    max_proc_len = 0
    truncated_len = 21  # limitation on the maximum number of visits

    for i in range(len(new_diag_seqs)):
        length = len(new_diag_seqs[i])

        if length >= truncated_len:
            all_diag_seqs = new_diag_seqs[i][length - truncated_len:]
            all_proc_seqs = new_proc_seqs[i][length - truncated_len:]

        else:
            all_diag_seqs = new_diag_seqs[i]
            all_proc_seqs = new_proc_seqs[i]

        input_diag = all_diag_seqs[:]
        input_proc = all_proc_seqs[:]

        inputs_diag_all.append(input_diag)
        inputs_proc_all.append(input_proc)

        if len(input_diag) > max_visit_len:
            max_visit_len = len(input_diag)     # the max number of visits

        for visit in input_diag:
            if len(visit) > max_diag_len:
                max_diag_len = len(visit)       # the max number of diag ICD codes in a visit

        for visit in input_proc:
            if len(visit) > max_proc_len:
                max_proc_len = len(visit)       # the max number of proc ICD codes in a visit

    return inputs_diag_all, inputs_proc_all, patient_list


# transform the input diagnosis data
def build_trees_diag(infile, seqs, typeFile):
    infd = open(infile, 'r')
    _ = infd.readline()

    types = pickle.load(open(typeFile, 'rb'))

    startSet = set(types.keys())
    hitList = []
    missList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0
    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + cat4

        icd9 = 'D_' + icd9

        if icd9 not in types:
            missList.append(icd9)
        else:
            hitList.append(icd9)

        # ccs category->type index
        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)
        if len(cat4) > 0:
            if desc4 not in types:
                cat4count += 1
                types[desc4] = len(types)
    infd.close()

    rootCode = len(types)
    types['A_ROOT'] = rootCode

    newTypes = pickle.load(open('processed/tree.diag.types', 'rb'))
    rtypes = dict([(v, k) for k, v in types.items()])  # type_num->icd_code

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                # print(code, rtypes[code])
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    return newSeqs


# transform the input procedure data
def build_trees_proc(infile, seqs, typeFile):
    infd = open(infile, 'r')
    _ = infd.readline()

    types = pickle.load(open(typeFile, 'rb'))

    startSet = set(types.keys())
    hitList = []
    missList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'B_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'B_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'B_' + cat3

        icd9 = 'P_' + icd9

        if icd9 not in types:
            missList.append(icd9)
        else:
            hitList.append(icd9)

        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)

    infd.close()
    rootCode = len(types)
    types['B_ROOT'] = rootCode

    newTypes = pickle.load(open('processed/tree.proc.types', 'rb'))
    rtypes = dict([(v, k) for k, v in types.items()])  # type_num->icd_code

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                # print(code, rtypes[code])
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    return newSeqs


# Converts the sample data into a format for model input
def padMatrix_test(seqs, task):
    lengths = np.array([len(seq) for seq in seqs])
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    max_visit_len = 0
    for patient in seqs:
        for visit in patient:
            if len(visit) > max_visit_len:
                max_visit_len = len(visit)

    if task == 'diag':
        x = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
        code_mask = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
    else:
        x = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
        code_mask = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
    mask = np.zeros((maxlen, n_samples))

    for idx, seq in enumerate(seqs):
        # for yvec, subseq in zip(y[:, idx, :], lseq[1:]):
        #     yvec[subseq] = 1.

        for xvec, subseq, codevec in zip(x[:, idx, :], seq[:], code_mask[:, idx, :]):
            xvec[:len(subseq)] = subseq
            codevec[:len(subseq)] = 1
        mask[:lengths[idx], idx] = 1.
    lengths = np.array(lengths)

    return x, mask, lengths, code_mask


# get output
def test_model(seq_diag, seq_proc, patient_list, embFile_diag, embFile_proc, processed_path, output_path, device,
                inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc,
                embDimSize, hiddenDimSize, attentionDimSize, dropoutRate):
    options = locals().copy()

    p2cFile_diag = processed_path + 'tree.diag.p2c'
    p2cFile_proc = processed_path + 'tree.proc.p2c'
    level_diag = 5
    level_proc = 4
    p2c_parent_diag, p2c_children_diag, p2c_mask_diag, c2p_parents_diag, c2p_child_diag, c2p_mask_diag = build_hierarchical_tree(p2cFile_diag, level_diag)
    p2c_parent_proc, p2c_children_proc, p2c_mask_proc, c2p_parents_proc, c2p_child_proc, c2p_mask_proc = build_hierarchical_tree(p2cFile_proc, level_proc)

    threshold = 0.01

    # Load nearest neighbor nodes and weights
    binary_matrix_diag, weight_matrix_diag = pickle.load(open('processed/diag.comatrix', 'rb'))
    nodes_diag, neighbors_diag, masks_diag, weights_diag = build_node_neighbor(binary_matrix_diag, weight_matrix_diag)
    binary_matrix_proc, weight_matrix_proc = pickle.load(open('processed/proc.comatrix', 'rb'))
    nodes_proc, neighbors_proc, masks_proc, weights_proc = build_node_neighbor(binary_matrix_proc, weight_matrix_proc)

    model = CoDMO(inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc,
                  numClass_proc,
                  embDimSize, hiddenDimSize, attentionDimSize, dropoutRate, embFile_diag, embFile_proc, 20, device,
                  p2c_parent_diag, p2c_children_diag, p2c_mask_diag, c2p_parents_diag, c2p_child_diag, c2p_mask_diag,
                  nodes_diag, neighbors_diag, masks_diag, weights_diag,
                  p2c_parent_proc, p2c_children_proc, p2c_mask_proc, c2p_parents_proc, c2p_child_proc, c2p_mask_proc,
                  nodes_proc, neighbors_proc, masks_proc, weights_proc)

    model.load_state_dict(torch.load('model/codmo.model', map_location='cpu'))
    model.to(device)

    x_diag, mask_diag, lengths_diag, code_mask_diag = padMatrix_test(seq_diag, 'diag')
    x_proc, mask_proc, lengths_proc, code_mask_proc = padMatrix_test(seq_proc, 'proc')

    x_diag = torch.from_numpy(x_diag).to(device).float()
    mask_diag = torch.from_numpy(mask_diag).to(device).float()
    code_mask_diag = torch.from_numpy(code_mask_diag).to(device).float()
    x_proc = torch.from_numpy(x_proc).to(device).float()
    mask_proc = torch.from_numpy(mask_proc).to(device).float()
    code_mask_proc = torch.from_numpy(code_mask_proc).to(device).float()

    y_diag_hat, y_proc_hat, re_matrix = model(x_diag, mask_diag, code_mask_diag, x_proc, mask_proc, code_mask_proc)

    y_diag_sorted, indices_diag = torch.sort(y_diag_hat, dim=2, descending=True)
    indices_diag = indices_diag.transpose(0, 1)
    y_diag_sorted = y_diag_sorted.transpose(0, 1)
    y_proc_sorted, indices_proc = torch.sort(y_proc_hat, dim=2, descending=True)
    indices_proc = indices_proc.transpose(0, 1)
    y_proc_sorted = y_proc_sorted.transpose(0, 1)


    dict_diag_ccs_single = pickle.load(open('processed/diag_ccs_single.dict', 'rb'))
    dict_proc_ccs_single = pickle.load(open('processed/proc_ccs_single.dict', 'rb'))

    rtypes_diag_ccs_single = dict([(v, k) for k, v in dict_diag_ccs_single.items()])
    rtypes_proc_ccs_single = dict([(v, k) for k, v in dict_proc_ccs_single.items()])

    diag_ccs_single_file = 'data/ccs_single_dx_tool_2015.csv'
    proc_ccs_single_file = 'data/ccs_single_pr_tool_2015.csv'
    description_diag = Description(diag_ccs_single_file)
    description_proc = Description(proc_ccs_single_file)

    # output top k prediction and Converts them to the corresponding category description
    k = 5
    pred_diag_topk = []
    pred_proc_topk = []
    for i in range(len(lengths_diag)):
        final_visit = lengths_diag[i]-1
        tmp_diag = indices_diag[i][final_visit][:k].cpu().numpy().tolist()
        tmp_proc = indices_proc[i][final_visit][:k].cpu().numpy().tolist()
        for j in range(len(tmp_diag)):
            tmp_diag[j] = rtypes_diag_ccs_single[tmp_diag[j]]
            tmp_diag[j] = description_diag.cat2description[tmp_diag[j][2:]]
        for j in range(len(tmp_proc)):
            tmp_proc[j] = rtypes_proc_ccs_single[tmp_proc[j]]
            tmp_proc[j] = description_proc.cat2description[tmp_proc[j][2:]]

        pred_diag_topk.append(tmp_diag)
        pred_proc_topk.append(tmp_proc)

    head = []
    head.append('Pid')
    for i in range(k):
        head.append('Diag-top'+str(i+1))
    for i in range(k):
        head.append('Proc-top'+str(i+1))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path+'result.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(head)
        for i in range(len(patient_list)):
            writer.writerow([patient_list[i]] + pred_diag_topk[i] + pred_proc_topk[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, default='processed/', help='The path to the processed file')
    parser.add_argument('--output_path', type=str, default='output/', help='The path to the processed file')
    parser.add_argument('--split_seed', type=int, default=5, help='The random seed to split the dataset')
    parser.add_argument('--embed_file_diag', type=str, default='preTrain/diag/diag.npz', help='The pretrained file containing the representation of diagnosis codes.')
    parser.add_argument('--embed_file_proc', type=str, default='preTrain/proc/proc.npz', help='The pretrained file containing the representation of procedure codes.')
    parser.add_argument('--embed_dim', type=int, default=200, help='The dimension size of the medical code embedding')
    parser.add_argument('--hidden_dim', type=int, default=200, help='The dimension size of the hidden layer of the GRU')
    parser.add_argument('--attention_dim', type=int, default=100, help='The dimension size of hidden layer of the MLP that generates the attention weights')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate used for the hidden layer of GRU')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of training epochs')
    parser.add_argument('--L2', type=float, default=0.0001, help='L2 regularization coefficient')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of a single mini-batch')
    parser.add_argument('--coef_diag', type=float, default=2.5, help='The coefficient of the loss function of diagnosis prediction')
    parser.add_argument('--coef_proc', type=float, default=1, help='The coefficient of the loss function of procedure prediction')
    parser.add_argument('--coef_priori', type=float, default=3, help='The coefficient of the loss function of a priori interaction')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='The ratio of valid dataset')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='The ratio of test dataset')
    parser.add_argument('--device', type=str, default="cpu", help='cuda:number or cpu')
    args = parser.parse_args()

    if args.device != "cpu":
        device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    processed_path = args.processed_path
    output_path = args.output_path
    seqFile_diag = processed_path + 'tree.diag.seqs'
    seqFile_proc = processed_path + 'tree.proc.seqs'
    embFile_diag = args.embed_file_diag
    embFile_proc = args.embed_file_proc
    split_seed = args.split_seed
    embDimSize = args.embed_dim
    hiddenDimSize = args.hidden_dim
    attentionDimSize = args.attention_dim
    dropoutRate = args.dropout_rate
    epochs = args.n_epochs
    L2 = args.L2
    lr = args.lr
    batchSize = args.batch_size
    coef_diag = args.coef_diag
    coef_proc = args.coef_proc
    coef_priori = args.coef_priori
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio


    admissionFile = 'data/ADMISSIONS_sample.csv'
    diagnosisFile = 'data/DIAGNOSES_ICD_sample.csv'
    proceduresFile = 'data/PROCEDURES_ICD_sample.csv'
    inputs_diag_all, inputs_proc_all, patient_list = process_input(admissionFile, diagnosisFile, proceduresFile)

    diag_ccs_multi_file = 'data/ccs_multi_dx_tool_2015.csv'
    proc_ccs_multi_file = 'data/ccs_multi_pr_tool_2015.csv'

    seq_diag = build_trees_diag(diag_ccs_multi_file, inputs_diag_all, 'processed/diag.dict')
    seq_proc = build_trees_proc(proc_ccs_multi_file, inputs_proc_all, 'processed/proc.dict')

    # inputDimSize_diag = calculate_dimSize(seqFile_diag)
    # numAncestors_diag = get_rootCode(processed_path + 'tree.diag.level2.pk') - inputDimSize_diag + 1
    # numClass_diag = calculate_dimSize(labelFile_diag)
    # inputDimSize_proc = calculate_dimSize(seqFile_proc)
    # numAncestors_proc = get_rootCode(processed_path + 'tree.proc.level3.pk') - inputDimSize_proc + 1
    # numClass_proc = calculate_dimSize(labelFile_proc)
    inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc = 4662, 728, 270, 1504, 406, 209

    test_model(seq_diag, seq_proc, patient_list, embFile_diag, embFile_proc, processed_path, output_path, device,
                inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc,
                embDimSize, hiddenDimSize, attentionDimSize, dropoutRate)
