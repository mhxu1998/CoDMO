import argparse
import pickle
import random
import time
import os
import torch
import torch.nn as nn
import numpy as np
from model.codmo import CoDMO
from dataLoader import load_data, padMatrix
from loss import CrossEntropy
from tqdm import tqdm

from utils import *

seed = 4396
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_num_threads(3)


def train_model(seqFile_diag, seqFile_proc, labelFile_diag, labelFile_proc, embFile_diag, embFile_proc, processed_path, output_path, device,
                inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc, split_seed,
                embDimSize, hiddenDimSize, attentionDimSize, dropoutRate, max_epochs, L2, lr, batchSize, coef_diag, coef_proc, coef_priori, val_ratio, test_ratio):
    """

    :param seqFile_diag: The path to diagnosis input file 'tree.diag.seqs' extracted by 'build_trees.py'
    :param seqFile_proc: The path to procedure input file 'tree.proc.seqs' extracted by 'build_trees.py'
    :param labelFile_diag: The path to diagnosis label file 'next_diag_ccs_single.labels' extracted by 'process_mimic.py'
    :param labelFile_proc: The path to procedure label file 'next_proc_ccs_single.labels' extracted by 'process_mimic.py'
    :param embFile_diag: The path to pretrained diagnosis code embedding file
    :param embFile_proc: The path to pretrained procedure code embedding file
    :param processed_path: The path of processed data
    :param output_path: The output path
    :param device: Use cpu or Gpu
    :param inputDimSize_diag: Numbers of diagnosis root codes
    :param numAncestors_diag: Numbers of diagnosis ancestor codes
    :param numClass_diag: Numbers of diagnosis labels
    :param inputDimSize_proc: Numbers of procedure root codes
    :param numAncestors_proc: Numbers of procedure ancestor codes
    :param numClass_proc: Numbers of procedure labels
    :param split_seed: Random seed that splits the data set
    :param embDimSize: The dimension of diagnosis/procedure code embedding
    :param hiddenDimSize: The dimension of hidden layer
    :param attentionDimSize: The dimension of attention layer
    :param dropoutRate: The rate of dropout
    :param max_epochs: Max epochs for model training
    :param L2: L2 regularization
    :param lr: The learning rate
    :param batchSize: The mini-batch size
    :param coef_diag: The coefficient of the diagnosis prediction loss
    :param coef_proc: The coefficient of the procedure prediction loss
    :param coef_priori: The coefficient of the priori interaction loss
    :param val_ratio: The valid set ratio
    :param test_ratio: The test set ratio
    """

    options = locals().copy()

    # Load ontology structure
    p2cFile_diag = processed_path + 'tree.diag.p2c'
    p2cFile_proc = processed_path + 'tree.proc.p2c'
    level_diag = 5
    level_proc = 4
    p2c_parent_diag, p2c_children_diag, p2c_mask_diag, c2p_parents_diag, c2p_child_diag, c2p_mask_diag = build_hierarchical_tree(p2cFile_diag, level_diag)
    p2c_parent_proc, p2c_children_proc, p2c_mask_proc, c2p_parents_proc, c2p_child_proc, c2p_mask_proc = build_hierarchical_tree(p2cFile_proc, level_proc)

    # Threshold for binarization of a cooccurrence matrix
    threshold = 0.01

    # Construct nearest neighbor nodes and weights
    binary_matrix_diag, weight_matrix_diag = co_matrix_building(seqFile_diag, inputDimSize_diag, threshold)
    nodes_diag, neighbors_diag, masks_diag, weights_diag = build_node_neighbor(binary_matrix_diag, weight_matrix_diag)
    binary_matrix_proc, weight_matrix_proc = co_matrix_building(seqFile_proc, inputDimSize_proc, threshold)
    nodes_proc, neighbors_proc, masks_proc, weights_proc = build_node_neighbor(binary_matrix_proc, weight_matrix_proc)

    # A priori interaction matrix
    co_matrix = co_matrix_with_diag_proc(seqFile_diag, seqFile_proc, inputDimSize_diag, inputDimSize_proc)
    co_matrix = torch.from_numpy(co_matrix).float().to(device)

    print('Building the model ... ')
    model = CoDMO(inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc,
                  embDimSize, hiddenDimSize, attentionDimSize, dropoutRate, embFile_diag, embFile_proc, 20, device,
                  p2c_parent_diag, p2c_children_diag, p2c_mask_diag, c2p_parents_diag, c2p_child_diag, c2p_mask_diag, nodes_diag, neighbors_diag, masks_diag, weights_diag,
                  p2c_parent_proc, p2c_children_proc, p2c_mask_proc, c2p_parents_proc, c2p_child_proc, c2p_mask_proc, nodes_proc, neighbors_proc, masks_proc, weights_proc)

    model.to(device)

    loss_fn = CrossEntropy().to(device)
    loss_mse = nn.MSELoss(reduction='mean').to(device)

    print('Constructing the optimizer ... ')
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=L2)
    loss_factor = [coef_diag, coef_proc, coef_priori]

    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(seqFile_diag, labelFile_diag, seqFile_proc, labelFile_proc, val_ratio, test_ratio, split_seed)

    print('Data length:', len(trainSet[0]))
    train_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    val_batches = int(np.ceil(float(len(validSet[0])) / float(batchSize)))
    test_batches = int(np.ceil(float(len(testSet[0])) / float(batchSize)))

    print('Optimization start !!')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logFile = output_path + 'out_seed' + str(split_seed) + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '.log'

    buf = '[Parameter] Emb_dim_size:%d, Hidden_dim_size:%d, Attn_dim_size:%d, Dropout:%.2f, emb_file_diag:%s, emb_file_proc:%s, lr:%.2f, n_epochs:%d, coef_diag:%.2f, coef_proc:%.2f, coef_priori:%.2f' % (
        embDimSize, hiddenDimSize, attentionDimSize, dropoutRate, embFile_diag, embFile_proc, lr, max_epochs, coef_diag, coef_proc, coef_priori)
    print(buf)
    print2file(buf, logFile)

    bestAcc = 0

    bestTrainCost = 0.0
    bestValidCost = 100000.0
    bestTestCost = 0.0

    bestTrainAcc_diag = [0.0]*4
    bestValidAcc_diag = [0.0]*4
    bestTestAcc_diag = [0.0]*4

    bestTrainAcc_proc = [0.0]*4
    bestValidAcc_proc = [0.0]*4
    bestTestAcc_proc = [0.0]*4

    epochDuration = 0.0
    bestEpoch = 0

    random.seed(seed)
    for epoch in range(max_epochs):
        print('Epoch:'+str(epoch)+'  '+'*'*20)
        iteration = 0
        cost_vec = []
        model.train()

        r = [0]*4
        r_p = [0]*4
        to = [0]*4
        to_p = [0]*4

        acc = [0]*4
        acc_proc = [0]*4
        valid_acc = [0]*4
        valid_acc_proc = [0]*4
        test_acc = [0]*4
        test_acc_proc = [0]*4

        print('Train:')
        for index in tqdm(random.sample(range(train_batches), train_batches)):
            optimizer.zero_grad()
            batchX_diag = trainSet[0][index * batchSize:(index + 1) * batchSize]
            batchY_diag = trainSet[1][index * batchSize:(index + 1) * batchSize]
            batchX_proc = trainSet[2][index * batchSize:(index + 1) * batchSize]
            batchY_proc = trainSet[3][index * batchSize:(index + 1) * batchSize]
            x_diag, y_diag, mask_diag, lengths_diag, code_mask_diag = padMatrix(batchX_diag, batchY_diag, options, 'diag')
            x_proc, y_proc, mask_proc, lengths_proc, code_mask_proc = padMatrix(batchX_proc, batchY_proc, options, 'proc')
            x_diag = torch.from_numpy(x_diag).to(device).float()
            mask_diag = torch.from_numpy(mask_diag).to(device).float()
            code_mask_diag = torch.from_numpy(code_mask_diag).to(device).float()
            x_proc = torch.from_numpy(x_proc).to(device).float()
            mask_proc = torch.from_numpy(mask_proc).to(device).float()
            code_mask_proc = torch.from_numpy(code_mask_proc).to(device).float()

            y_diag_hat, y_proc_hat, re_matrix = model(x_diag, mask_diag, code_mask_diag, x_proc, mask_proc, code_mask_proc)

            y_diag = torch.from_numpy(y_diag).float().to(device)
            y_proc = torch.from_numpy(y_proc).float().to(device)
            lengths_diag = torch.from_numpy(lengths_diag).float().to(device)
            lengths_proc = torch.from_numpy(lengths_proc).float().to(device)
            loss_diag, right, total = loss_fn(y_diag_hat, y_diag, lengths_diag)
            loss_proc, right_proc, total_proc = loss_fn(y_proc_hat, y_proc, lengths_proc)

            loss_all = loss_factor[0]*loss_diag + loss_factor[1]*loss_proc + loss_factor[2]*loss_mse(co_matrix.view(-1), re_matrix.view(-1))
            loss_all.backward()
            optimizer.step()

            cost_vec.append(loss_all.item())

            for i in range(4):
                r[i] += right[i]
                to[i] += total[i]
                r_p[i] += right_proc[i]
                to_p[i] += total_proc[i]

            iteration += 1

        cost = np.mean(cost_vec)

        for i in range(4):
            acc[i] = r[i]/to[i]
            acc_proc[i] = r_p[i]/to_p[i]

        # for i in range(4):
        #     print('train code@',(i+1)*5,'  diag:',r[i],'/',to[i],'   proc:',r_p[i],'/',to_p[i])

        model.eval()
        with torch.no_grad():
            # valid
            cost_vec = []

            valid_r = [0] * 4
            valid_r_p = [0] * 4
            valid_to = [0] * 4
            valid_to_p = [0] * 4
            print('Valid:')
            for index in tqdm(range(val_batches)):
                validX_diag = validSet[0][index * batchSize:(index + 1) * batchSize]
                validY_diag = validSet[1][index * batchSize:(index + 1) * batchSize]
                validX_proc = validSet[2][index * batchSize:(index + 1) * batchSize]
                validY_proc = validSet[3][index * batchSize:(index + 1) * batchSize]
                val_x_diag, val_y_diag, mask_diag, lengths_diag, code_mask_diag = padMatrix(validX_diag, validY_diag, options, 'diag')
                val_x_proc, val_y_proc, mask_proc, lengths_proc, code_mask_proc = padMatrix(validX_proc, validY_proc, options, 'proc')
                val_x_diag = torch.from_numpy(val_x_diag).float().to(device)
                mask_diag = torch.from_numpy(mask_diag).float().to(device)
                code_mask_diag = torch.from_numpy(code_mask_diag).to(device).float()
                val_x_proc = torch.from_numpy(val_x_proc).float().to(device)
                mask_proc = torch.from_numpy(mask_proc).float().to(device)
                code_mask_proc = torch.from_numpy(code_mask_proc).to(device).float()

                val_y_diag_hat, val_y_proc_hat, val_re_matrix = model(val_x_diag, mask_diag, code_mask_diag, val_x_proc, mask_proc, code_mask_proc)

                val_y_diag = torch.from_numpy(val_y_diag).float().to(device)
                val_y_proc = torch.from_numpy(val_y_proc).float().to(device)
                lengths_diag = torch.from_numpy(lengths_diag).float().to(device)
                lengths_proc = torch.from_numpy(lengths_proc).float().to(device)
                valid_cost_diag, right, total = loss_fn(val_y_diag_hat, val_y_diag, lengths_diag)
                valid_cost_proc, right_proc, total_proc = loss_fn(val_y_proc_hat, val_y_proc, lengths_proc)

                valid_cost = loss_factor[0]*valid_cost_diag + loss_factor[1]*valid_cost_proc + loss_factor[2]*loss_mse(co_matrix.view(-1), val_re_matrix.view(-1))
                cost_vec.append(valid_cost.item())

                for i in range(4):
                    valid_r[i] += right[i]
                    valid_to[i] += total[i]
                    valid_r_p[i] += right_proc[i]
                    valid_to_p[i] += total_proc[i]

            valid_cost = np.mean(cost_vec)

            for i in range(4):
                valid_acc[i] = valid_r[i] / valid_to[i]
                valid_acc_proc[i] = valid_r_p[i] / valid_to_p[i]

            # for i in range(4):
            #     print('valid code@', (i + 1) * 5, '  diag:', valid_r[i], '/', valid_to[i], '   proc:', valid_r_p[i], '/', valid_to_p[i])

            # test
            cost_vec = []

            test_r = [0] * 4
            test_r_p = [0] * 4
            test_to = [0] * 4
            test_to_p = [0] * 4
            print('Test:')
            for index in tqdm(range(test_batches)):
                testX_diag = testSet[0][index * batchSize:(index + 1) * batchSize]
                testY_diag = testSet[1][index * batchSize:(index + 1) * batchSize]
                testX_proc = testSet[2][index * batchSize:(index + 1) * batchSize]
                testY_proc = testSet[3][index * batchSize:(index + 1) * batchSize]
                test_x_diag, test_y_diag, mask_diag, lengths_diag, code_mask_diag = padMatrix(testX_diag, testY_diag, options, 'diag')
                test_x_proc, test_y_proc, mask_proc, lengths_proc, code_mask_proc = padMatrix(testX_proc, testY_proc, options, 'proc')
                test_x_diag = torch.from_numpy(test_x_diag).float().to(device)
                mask_diag = torch.from_numpy(mask_diag).float().to(device)
                code_mask_diag = torch.from_numpy(code_mask_diag).to(device).float()
                test_x_proc = torch.from_numpy(test_x_proc).float().to(device)
                mask_proc = torch.from_numpy(mask_proc).float().to(device)
                code_mask_proc = torch.from_numpy(code_mask_proc).to(device).float()

                test_y_diag_hat, test_y_proc_hat, test_re_matrix = model(test_x_diag, mask_diag, code_mask_diag, test_x_proc, mask_proc, code_mask_proc)

                test_y_diag = torch.from_numpy(test_y_diag).float().to(device)
                test_y_proc = torch.from_numpy(test_y_proc).float().to(device)
                lengths_diag = torch.from_numpy(lengths_diag).float().to(device)
                lengths_proc = torch.from_numpy(lengths_proc).float().to(device)
                test_cost_diag, right, total = loss_fn(test_y_diag_hat, test_y_diag, lengths_diag)
                test_cost_proc, right_proc, total_proc = loss_fn(test_y_proc_hat, test_y_proc, lengths_proc)

                test_cost = loss_factor[0]*test_cost_diag + loss_factor[1]*test_cost_proc + loss_factor[2]*loss_mse(co_matrix.view(-1), test_re_matrix.view(-1))
                cost_vec.append(test_cost.item())

                for i in range(4):
                    test_r[i] += right[i]
                    test_to[i] += total[i]
                    test_r_p[i] += right_proc[i]
                    test_to_p[i] += total_proc[i]

            test_cost = np.mean(cost_vec)

            for i in range(4):
                test_acc[i] = test_r[i] / test_to[i]
                test_acc_proc[i] = test_r_p[i] / test_to_p[i]

            # for i in range(4):
            #     print('test code@', (i + 1) * 5, '  diag:', test_r[i], '/', test_to[i], '   proc:', test_r_p[i], '/', test_to_p[i])

        # print the loss
        buf = '[Epoch:%d], Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (epoch, cost, valid_cost, test_cost)
        print(buf)
        print2file(buf, logFile)
        buf = '----Diagnosis'
        print(buf)
        print2file(buf, logFile)
        for i in range(4):
            buf = '------Acc@%d, Train:%f, Valid:%f, Test:%f' % ((i+1)*5, acc[i], valid_acc[i], test_acc[i])
            print(buf)
            print2file(buf, logFile)

        buf = '----Procedure'
        print(buf)
        print2file(buf, logFile)
        for i in range(4):
            buf = '------Acc@%d, Train:%f, Valid:%f, Test:%f' % ((i+1)*5, acc_proc[i], valid_acc_proc[i], test_acc_proc[i])
            print(buf)
            print2file(buf, logFile)

        buf = '***************************'
        print(buf)
        print2file(buf, logFile)

        # save the best model
        if valid_cost < bestValidCost:
            bestValidCost = valid_cost
            bestTestCost = test_cost
            bestTrainCost = cost
            bestEpoch = epoch

            for i in range(4):
                bestTrainAcc_diag[i] = acc[i]
                bestValidAcc_diag[i] = valid_acc[i]
                bestTestAcc_diag[i] = test_acc[i]
                bestTrainAcc_proc[i] = acc_proc[i]
                bestValidAcc_proc[i] = valid_acc_proc[i]
                bestTestAcc_proc[i] = test_acc_proc[i]

        # torch.save(model.state_dict(), outFile + f'.{epoch}')

    buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        bestEpoch, bestTrainCost, bestValidCost, bestTestCost)
    print(buf)
    print2file(buf, logFile)
    buf = '----Diagnosis'
    print(buf)
    print2file(buf, logFile)
    for i in range(4):
        buf = '------Acc@%d, Train:%f, Valid:%f, Test:%f' % ((i + 1) * 5, bestTrainAcc_diag[i], bestValidAcc_diag[i], bestTestAcc_diag[i])
        print(buf)
        print2file(buf, logFile)
    buf = '----Procedure'
    print(buf)
    print2file(buf, logFile)
    for i in range(4):
        buf = '------Acc@%d, Train:%f, Valid:%f, Test:%f' % ((i + 1) * 5, bestTrainAcc_proc[i], bestValidAcc_proc[i], bestTestAcc_proc[i])
        print(buf)
        print2file(buf, logFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, default='processed/', help='The path to the processed file')
    parser.add_argument('--output_path', type=str, default='output/', help='The path to the processed file')
    parser.add_argument('--split_seed', type=int, default=5, help='The random seed to split the dataset')
    parser.add_argument('--embed_file_diag', type=str, default='', help='The pretrained file containing the representation of diagnosis codes.')
    parser.add_argument('--embed_file_proc', type=str, default='', help='The pretrained file containing the representation of procedure codes.')
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
    labelFile_diag = processed_path + 'next_diag_ccs_single.labels'
    labelFile_proc = processed_path + 'next_proc_ccs_single.labels'
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

    inputDimSize_diag = calculate_dimSize(seqFile_diag)
    print('Diagnosis inputDimSize:%d' % inputDimSize_diag)

    numAncestors_diag = get_rootCode(processed_path + 'tree.diag.level2.pk') - inputDimSize_diag + 1
    print('Diagnosis numAncestors:%d' % numAncestors_diag)

    numClass_diag = calculate_dimSize(labelFile_diag)
    print('Diagnosis numClass:%d' % numClass_diag)

    inputDimSize_proc = calculate_dimSize(seqFile_proc)
    print('Procedure inputDimSize:%d' % inputDimSize_proc)

    numAncestors_proc = get_rootCode(processed_path + 'tree.proc.level3.pk') - inputDimSize_proc + 1
    print('Procedure numAncestors:%d' % numAncestors_proc)

    numClass_proc = calculate_dimSize(labelFile_proc)
    print('Procedure numClass:%d' % numClass_proc)

    train_model(seqFile_diag, seqFile_proc, labelFile_diag, labelFile_proc, embFile_diag, embFile_proc, processed_path, output_path, device,
                inputDimSize_diag, numAncestors_diag, numClass_diag, inputDimSize_proc, numAncestors_proc, numClass_proc, split_seed,
                embDimSize, hiddenDimSize, attentionDimSize, dropoutRate, epochs, L2, lr, batchSize, coef_diag,
                coef_proc, coef_priori, val_ratio, test_ratio)

