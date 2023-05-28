import numpy as np
import pickle


def len_argsort(seq):
    """
    Sort the input data from least to most visits
    :param seq: input data
    :return: sorted data
    """
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))


def load_data(diagSeqFile, diagLabelFile, procSeqFile, procLabelFile, test_ratio=0.1, valid_ratio=0.1, seed=5):
    """
    Split data into train/valid/test set
    :param diagSeqFile: The path to diagnosis input file 'tree.diag.seqs' extracted by 'build_trees.py'
    :param diagLabelFile: The path to diagnosis label file 'next_diag_ccs_single.labels' extracted by 'process_mimic.py'
    :param procSeqFile: The path to procedure input file 'tree.proc.seqs' extracted by 'build_trees.py'
    :param procLabelFile: The path to procedure label file 'next_proc_ccs_single.labels' extracted by 'process_mimic.py'
    :param test_ratio: The test set ratio
    :param valid_ratio: The valid set ratio
    :param seed: Random seed that splits the data set
    :return: train_set, valid_set, test_set
    """
    diag_sequences = np.array(pickle.load(open(diagSeqFile, 'rb')), dtype=object)
    diag_labels = np.array(pickle.load(open(diagLabelFile, 'rb')), dtype=object)
    proc_sequences = np.array(pickle.load(open(procSeqFile, 'rb')), dtype=object)
    proc_labels = np.array(pickle.load(open(procLabelFile, 'rb')), dtype=object)

    np.random.seed(seed)
    dataSize = len(diag_labels)
    ind = np.random.permutation(dataSize)
    nTest = int(test_ratio * dataSize)
    nValid = int(valid_ratio * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x_diag = diag_sequences[train_indices]
    train_set_y_diag = diag_labels[train_indices]
    train_set_x_proc = proc_sequences[train_indices]
    train_set_y_proc = proc_labels[train_indices]
    test_set_x_diag = diag_sequences[test_indices]
    test_set_y_diag = diag_labels[test_indices]
    test_set_x_proc = proc_sequences[test_indices]
    test_set_y_proc = proc_labels[test_indices]
    valid_set_x_diag = diag_sequences[valid_indices]
    valid_set_y_diag = diag_labels[valid_indices]
    valid_set_x_proc = proc_sequences[valid_indices]
    valid_set_y_proc = proc_labels[valid_indices]

    train_sorted_index = len_argsort(train_set_x_diag)
    train_set_x_diag = [train_set_x_diag[i] for i in train_sorted_index]
    train_set_y_diag = [train_set_y_diag[i] for i in train_sorted_index]
    train_set_x_proc = [train_set_x_proc[i] for i in train_sorted_index]
    train_set_y_proc = [train_set_y_proc[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x_diag)
    valid_set_x_diag = [valid_set_x_diag[i] for i in valid_sorted_index]
    valid_set_y_diag = [valid_set_y_diag[i] for i in valid_sorted_index]
    valid_set_x_proc = [valid_set_x_proc[i] for i in valid_sorted_index]
    valid_set_y_proc = [valid_set_y_proc[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x_diag)
    test_set_x_diag = [test_set_x_diag[i] for i in test_sorted_index]
    test_set_y_diag = [test_set_y_diag[i] for i in test_sorted_index]
    test_set_x_proc = [test_set_x_proc[i] for i in test_sorted_index]
    test_set_y_proc = [test_set_y_proc[i] for i in test_sorted_index]

    train_set = (train_set_x_diag, train_set_y_diag, train_set_x_proc, train_set_y_proc)
    valid_set = (valid_set_x_diag, valid_set_y_diag, valid_set_x_proc, valid_set_y_proc)
    test_set = (test_set_x_diag, test_set_y_diag, test_set_x_proc, test_set_y_proc)

    return train_set, valid_set, test_set


def padMatrix(seqs, labels, options, task):
    """
    Converts the data into a format for model input
    :param seqs: input data
    :param labels: output label
    :param options: local variable
    :param task: 'diag' or 'proc'
    :return: x, y, mask, lengths, code_mask
    """
    lengths = np.array([len(seq) for seq in seqs])-1
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    max_visit_len = 0
    for patient in seqs:
        for visit in patient:
            if len(visit) > max_visit_len:
                max_visit_len = len(visit)

    if task == 'diag':
        x = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
        y = np.zeros((maxlen, n_samples, options['numClass_diag']), dtype=float)
        code_mask = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
    else:
        x = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
        y = np.zeros((maxlen, n_samples, options['numClass_proc']), dtype=float)
        code_mask = np.zeros((maxlen, n_samples, max_visit_len), dtype=float)
    mask = np.zeros((maxlen, n_samples))

    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        for yvec, subseq in zip(y[:, idx, :], lseq[1:]):
            yvec[subseq] = 1.

        for xvec, subseq, codevec in zip(x[:, idx, :], seq[:-1], code_mask[:, idx, :]):
            xvec[:len(subseq)] = subseq
            codevec[:len(subseq)] = 1
        mask[:lengths[idx], idx] = 1.
    lengths = np.array(lengths)

    return x, y, mask, lengths, code_mask
