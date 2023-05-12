import sys, time, random
import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.ifelse import ifelse
from tqdm import tqdm
import argparse
import pickle
from collections import OrderedDict
import os


'''
Based on GRAM (https://github.com/mp2893/gram)
    --create_glove_comap.py
    --glove.py
'''


def augmentVisit(visit, code, treeList):
    for tree in treeList:
        if code in tree:
            visit.extend(tree[code][1:])
            break
    return


def countCooccurrenceProduct(visit, coMap):
    codeSet = set(visit)
    for code1 in codeSet:
        for code2 in codeSet:
            if code1 == code2: continue

            product = visit.count(code1) * visit.count(code2)
            key1 = (code1, code2)
            key2 = (code2, code1)

            if key1 in coMap:
                coMap[key1] += product
            else:
                coMap[key1] = product

            if key2 in coMap:
                coMap[key2] += product
            else:
                coMap[key2] = product


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def unzip(zipped):
    new_params = OrderedDict()
    for k, v in zipped.items():
        new_params[k] = v.get_value()
    return new_params


def init_params(options):
    params = OrderedDict()

    inputSize = options['inputSize']
    dimensionSize = options['dimensionSize']

    rng = np.random.RandomState(1234)
    params['w'] = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(inputSize, dimensionSize)),
                             dtype=theano.config.floatX)
    rng = np.random.RandomState(12345)
    params['w_tilde'] = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(inputSize, dimensionSize)),
                                   dtype=theano.config.floatX)

    params['b'] = np.zeros(inputSize).astype(theano.config.floatX)
    params['b_tilde'] = np.zeros(inputSize).astype(theano.config.floatX)

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for k, v in params.items():
        tparams[k] = theano.shared(v, name=k)
    return tparams


def build_model(tparams, options):
    weightVector = T.vector('weightVector', dtype=theano.config.floatX)
    iVector = T.vector('iVector', dtype='int32')
    jVector = T.vector('jVector', dtype='int32')
    cost = weightVector * (((tparams['w'][iVector] * tparams['w_tilde'][jVector]).sum(axis=1) + tparams['b'][iVector] +
                            tparams['b_tilde'][jVector] - T.log(weightVector)) ** 2)

    return weightVector, iVector, jVector, cost.sum()


def adadelta(tparams, grads, weightVector, iVector, jVector, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in
                      tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([weightVector, iVector, jVector], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in
             zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update


def weightFunction(x):
    if x < 100.0:
        return (x / 100.0) ** 0.75
    else:
        return 1


def load_data(infile):
    cooccurMap = pickle.load(open(infile, 'rb'))
    I = []
    J = []
    Weight = []
    for key, value in cooccurMap.items():
        I.append(key[0])
        J.append(key[1])
        Weight.append(weightFunction(value))
    shared_I = theano.shared(np.asarray(I, dtype='int32'), borrow=True)
    shared_J = theano.shared(np.asarray(J, dtype='int32'), borrow=True)
    shared_Weight = theano.shared(np.asarray(Weight, dtype=theano.config.floatX), borrow=True)
    return shared_I, shared_J, shared_Weight


def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


def train_glove(infile, inputSize=20000, batchSize=100, dimensionSize=100, maxEpochs=1000, outfile='result', x_max=100,
                alpha=0.75):
    options = locals().copy()
    print('initializing parameters')
    params = init_params(options)
    tparams = init_tparams(params)

    print('loading data')
    I, J, Weight = load_data(infile)
    n_batches = int(np.ceil(float(I.get_value(borrow=True).shape[0]) / float(batchSize)))

    print('building models')
    weightVector, iVector, jVector, cost = build_model(tparams, options)
    grads = T.grad(cost, wrt=list(tparams.values()))
    f_grad_shared, f_update = adadelta(tparams, grads, weightVector, iVector, jVector, cost)

    logFile = outfile + '.log'
    print('training start')
    for epoch in tqdm(range(maxEpochs)):
        costVector = []
        iteration = 0
        for batchIndex in tqdm(random.sample(range(n_batches), n_batches)):
            cost = f_grad_shared(Weight.get_value(borrow=True, return_internal_type=True)[
                                 batchIndex * batchSize:(batchIndex + 1) * batchSize],
                                 I.get_value(borrow=True, return_internal_type=True)[
                                 batchIndex * batchSize: (batchIndex + 1) * batchSize],
                                 J.get_value(borrow=True, return_internal_type=True)[
                                 batchIndex * batchSize: (batchIndex + 1) * batchSize])
            f_update()
            costVector.append(cost)

            if (iteration % 1000 == 0):
                buf = 'epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, cost)
                # print(buf)
                print2file(buf, logFile)
            iteration += 1
        trainCost = np.mean(costVector)
        buf = 'epoch:%d, cost:%f' % (epoch, trainCost)
        print(buf)
        print2file(buf, logFile)
        tempParams = unzip(tparams)
        np.savez_compressed(outfile + '.' + str(epoch), **tempParams)


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    return list(tree.values())[0][1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, default='../processed/', help='The path to the processed file')
    parser.add_argument('--embed_dim', type=int, default=200, help='The dimension size of the medical code embedding')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs')
    parser.add_argument('--out_path', type=str, default='')
    args = parser.parse_args()

    embDimSize = args.embed_dim
    batchSize = args.batch_size
    maxEpochs = args.n_epochs

    seqFile_diag = args.processed_path + 'tree.diag.seqs'
    seqFile_proc = args.processed_path + 'tree.proc.seqs'

    outFile_diag = 'cooccurrence_map_diag.pk'
    outFile_proc = 'cooccurrence_map_proc.pk'

    maxLevel_diag = 5
    maxLevel_proc = 4
    seqs_diag = pickle.load(open(seqFile_diag, 'rb'))
    treeList_diag = [pickle.load(open(args.processed_path + 'tree.diag.level' + str(i) + '.pk', 'rb')) for i in range(1, maxLevel_diag + 1)]

    if os.path.exists(outFile_diag):
        print('Diagnosis Cooccurrence file already exist...')
    else:
        print('Count Diagnosis Cooccurrence...')
        coMap_diag = {}
        count = 0
        for patient in seqs_diag:
            if count % 1000 == 0:
                print(count)
            count += 1
            for visit in patient:
                for code in visit:
                    augmentVisit(visit, code, treeList_diag)
                countCooccurrenceProduct(visit, coMap_diag)

        pickle.dump(coMap_diag, open(outFile_diag, 'wb'), -1)

    seqs_proc = pickle.load(open(seqFile_proc, 'rb'))
    treeList_proc = [pickle.load(open(args.processed_path + 'tree.proc.level' + str(i) + '.pk', 'rb')) for i in range(1, maxLevel_proc + 1)]

    if os.path.exists(outFile_proc):
        print('Procedure Cooccurrence file already exist...')
    else:
        print('Count Procedure Cooccurrence...')
        coMap_proc = {}
        count = 0
        for patient in seqs_proc:
            if count % 1000 == 0:
                print(count)
            count += 1
            for visit in patient:
                for code in visit:
                    augmentVisit(visit, code, treeList_proc)
                countCooccurrenceProduct(visit, coMap_proc)

        pickle.dump(coMap_proc, open(outFile_proc, 'wb'), -1)

    # Train Glove
    print('Train diagnosis code embedding...')
    inputDimSize_diag = get_rootCode(args.processed_path + 'tree.diag.level2.pk') + 1

    if not os.path.exists(args.out_path + 'diag'):
        os.makedirs(args.out_path + 'diag')

    emb_file_diag = args.out_path + 'diag/diag'
    train_glove(outFile_diag, inputSize=inputDimSize_diag, batchSize=batchSize, dimensionSize=embDimSize, maxEpochs=maxEpochs, outfile=emb_file_diag)

    print('Train procedure code embedding...')
    inputDimSize_proc = get_rootCode(args.processed_path + 'tree.proc.level3.pk') + 1

    if not os.path.exists(args.out_path + 'proc'):
        os.makedirs(args.out_path + 'proc')

    emb_file_proc = args.out_path + 'proc/proc'
    train_glove(outFile_proc, inputSize=inputDimSize_proc, batchSize=batchSize, dimensionSize=embDimSize, maxEpochs=maxEpochs, outfile=emb_file_proc)

    print('')
