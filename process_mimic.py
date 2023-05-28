import sys
import pickle
import os
import pandas as pd
import argparse
from datetime import datetime


# obtain the code of ccs single file
class LabelsForData(object):
    def __init__(self, ccs_single_file):
        self.ccs_single_dx_df = pd.read_csv(ccs_single_file, header=0, dtype=object)
        self.code2single_dx = {}        # ICD code to CCS single category
        self.build_maps()

    def build_maps(self):
        for i, row in self.ccs_single_dx_df.iterrows():
            code = row[0][1:-1].strip()
            single_cat = row[1][1:-1].strip()
            self.code2single_dx[code] = single_cat      # e.g. code[01000] = 1


def processing(adm_file, diag_file, proc_file, diag_ccs_single_file, proc_ccs_single_file, out_dir):
    """
    Process raw MIMIC data
    :param adm_file: The path to 'ADMISSIONS.csv'
    :param diag_file: The path to 'DIAGNOSES_ICD.csv'
    :param proc_file: The path to 'PROCEDURES_ICD.csv'
    :param diag_ccs_single_file: The path to 'ccs_single_dx_tool_2015.csv'
    :param proc_ccs_single_file: The path to 'ccs_single_pr_tool_2015.csv'
    :param out_dir: The output path
    """

    # ICD->CCS single category mapping
    label4data = LabelsForData(diag_ccs_single_file)
    label4data_proc = LabelsForData(proc_ccs_single_file)

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}          # patient id->admission id
    admDateMap = {}         # admission id->admission date
    infd = open(adm_file, 'r')      # open admission table
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-diagList mapping')
    admDiagMap = {}                 # admission id->diag ICD
    admDiagMap_ccs_single = {}      # admission id->diag ICD (ccs single)
    infd = open(diag_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        diag = tokens[4][1:-1]
        if len(diag) == 0:
            continue

        diagStr = 'D_' + diag       # e.g. D_40301
        diagStr_ccs_single = 'D_' + label4data.code2single_dx[diag]     # e.g. D_99

        if admId in admDiagMap:
            admDiagMap[admId].append(diagStr)
        else:
            admDiagMap[admId] = [diagStr]

        if admId in admDiagMap_ccs_single:
            admDiagMap_ccs_single[admId].append(diagStr_ccs_single)
        else:
            admDiagMap_ccs_single[admId] = [diagStr_ccs_single]
    infd.close()

    print('Building admission-procList mapping')
    admProcMap = {}                 # admission id->proc ICD
    admProcMap_ccs_single = {}      # admission id->proc ICD (ccs single)
    infd = open(proc_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        proc = tokens[4][1:-1]
        if len(proc) == 0:
            continue

        procStr = 'P_' + proc
        procStr_ccs_single = 'P_' + label4data_proc.code2single_dx[proc]

        if admId in admProcMap:
            admProcMap[admId].append(procStr)
        else:
            admProcMap[admId] = [procStr]
        if admId in admProcMap_ccs_single:
            admProcMap_ccs_single[admId].append(procStr_ccs_single)
        else:
            admProcMap_ccs_single[admId] = [procStr_ccs_single]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidDiagMap = {}                 # pid->diag
    pidDiagMap_ccs_single = {}
    pidProcMap = {}                 # pid->proc
    pidProcMap_ccs_single = {}
    for pid, admIdList in pidAdmMap.items():        # pid->admission id
        new_admIdList = []
        for admId in admIdList:
            if admId in (admDiagMap and admProcMap):
                new_admIdList.append(admId)
        if len(new_admIdList) < 2:      # Records of patients admitted less than 2 times are ignored
            continue

        # Sorted by the admission date
        sortedDiagList = sorted([(admDateMap[admId], admDiagMap[admId]) for admId in new_admIdList])
        pidDiagMap[pid] = sortedDiagList

        sortedDiagList_ccs_single = sorted([(admDateMap[admId], admDiagMap_ccs_single[admId]) for admId in new_admIdList])
        pidDiagMap_ccs_single[pid] = sortedDiagList_ccs_single

        sortedProcList = sorted([(admDateMap[admId], admProcMap[admId]) for admId in new_admIdList])
        pidProcMap[pid] = sortedProcList

        sortedProcList_ccs_single = sorted([(admDateMap[admId], admProcMap_ccs_single[admId]) for admId in new_admIdList])
        pidProcMap_ccs_single[pid] = sortedProcList_ccs_single

    print('Building diagSeqs, diagSeqs_ccs_single, procSeqs, timeSeqs')
    diag_seqs = []
    diag_seqs_ccs_single = []
    proc_seqs = []
    proc_seqs_ccs_single = []
    time_seqs = []
    for pid, visits in pidDiagMap.items():                  # diag ICD List of patient
        seq = []
        time = []
        first_time = visits[0][0]       # the first time of admission
        for i, visit in enumerate(visits):
            current_time = visit[0]
            interval = (current_time - first_time).days         # time interval
            seq.append(visit[1])
            time.append(interval)
        diag_seqs.append(seq)
        time_seqs.append(time)

    for pid, visits in pidDiagMap_ccs_single.items():       # diag ICD List (ccs single) of patient
        seq = []
        for i, visit in enumerate(visits):
            seq.append(visit[1])
        diag_seqs_ccs_single.append(seq)

    for pid, visits in pidProcMap.items():                  # proc ICD List of patient
        seq = []
        for i, visit in enumerate(visits):
            seq.append(visit[1])
        proc_seqs.append(seq)

    for pid, visits in pidProcMap_ccs_single.items():       # proc ICD List (ccs single) of patient
        seq = []
        for i, visit in enumerate(visits):
            seq.append(visit[1])
        proc_seqs_ccs_single.append(seq)

    print('Converting strSeqs to intSeqs, and making types for diag and proc code')
    dict_diag = {}              # code->index   e.g. dict_css['D_101']=1
    new_diag_seqs = []          # index list
    # dict_diag['UNK'] = 0
    for patient in diag_seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in dict_diag:
                    newVisit.append(dict_diag[code])
                else:
                    dict_diag[code] = len(dict_diag)
                    newVisit.append(dict_diag[code])
            newPatient.append(newVisit)
        new_diag_seqs.append(newPatient)

    dict_diag_ccs_single = {}
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

    dict_proc = {}
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

    dict_proc_ccs_single = {}
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

    print('Converting seqs to model inputs')
    inputs_diag_all = []
    inputs_proc_all = []
    labels_diag = []
    labels_diag_ccs_single = []
    labels_proc = []
    labels_proc_ccs_single = []
    labels_next_diag = []
    labels_next_diag_ccs_single = []
    labels_next_proc = []
    labels_next_proc_ccs_single = []
    max_visit_len = 0
    max_diag_len = 0
    max_proc_len = 0
    truncated_len = 21      # limitation on the maximum number of visits

    for i in range(len(new_diag_seqs)):
        length = len(new_diag_seqs[i])

        if length >= truncated_len:
            all_diag_seqs = new_diag_seqs[i][length - truncated_len:]
            all_diag_seqs_ccs_single = new_diag_seqs_ccs_single[i][length - truncated_len:]
            all_proc_seqs = new_proc_seqs[i][length - truncated_len:]
            all_proc_seqs_ccs_single = new_proc_seqs_ccs_single[i][length - truncated_len:]
            all_time = time_seqs[i][length - truncated_len:]
        else:
            all_diag_seqs = new_diag_seqs[i]
            all_diag_seqs_ccs_single = new_diag_seqs_ccs_single[i]
            all_proc_seqs = new_proc_seqs[i]
            all_proc_seqs_ccs_single = new_proc_seqs_ccs_single[i]
            all_time = time_seqs[i]

        input_diag = all_diag_seqs[:]         # diag ICD list
        input_proc = all_proc_seqs[:]         # proc ICD list
        label_diag = all_diag_seqs[-1]          # diag ICD of the last visit
        label_diag_ccs_single = all_diag_seqs_ccs_single[-1]    # diag ICD (ccs single) of the last visit
        label_proc = all_proc_seqs[-1]          # proc ICD of the last visit
        label_proc_ccs_single = all_proc_seqs_ccs_single[-1]    # proc ICD (ccs single) of the last visit
        label_next_diag = all_diag_seqs[:]     # diag ICD list removing the first visit
        label_next_diag_ccs_single = all_diag_seqs_ccs_single[:]       # diag ICD (ccs single) list removing the first visit
        label_next_proc = all_proc_seqs[:]     # proc ICD list removing the first visit
        label_next_proc_ccs_single = all_proc_seqs_ccs_single[:]       # proc ICD (ccs single) list removing the first visit

        inputs_diag_all.append(input_diag)
        inputs_proc_all.append(input_proc)
        labels_diag.append(label_diag)
        labels_diag_ccs_single.append(label_diag_ccs_single)
        labels_proc.append(label_proc)
        labels_proc_ccs_single.append(label_proc_ccs_single)
        labels_next_diag.append(label_next_diag)
        labels_next_diag_ccs_single.append(label_next_diag_ccs_single)
        labels_next_proc.append(label_next_proc)
        labels_next_proc_ccs_single.append(label_next_proc_ccs_single)

        if len(input_diag) > max_visit_len:
            max_visit_len = len(input_diag)     # the max number of visits

        for visit in input_diag:
            if len(visit) > max_diag_len:
                max_diag_len = len(visit)       # the max number of diag ICD codes in a visit

        for visit in input_proc:
            if len(visit) > max_proc_len:
                max_proc_len = len(visit)       # the max number of proc ICD codes in a visit

    # print(max_visit_len, max_diag_len, max_proc_len)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pickle.dump(inputs_diag_all, open(os.path.join(out_dir, 'diag.inputs'), 'wb'), -1)
    pickle.dump(inputs_proc_all, open(os.path.join(out_dir, 'proc.inputs'), 'wb'), -1)
    pickle.dump(labels_diag, open(os.path.join(out_dir, 'diag.labels'), 'wb'), -1)
    pickle.dump(labels_diag_ccs_single, open(os.path.join(out_dir, 'diag_ccs_single.labels'), 'wb'), -1)
    pickle.dump(labels_proc, open(os.path.join(out_dir, 'proc.labels'), 'wb'), -1)
    pickle.dump(labels_proc_ccs_single, open(os.path.join(out_dir, 'proc_ccs_single.labels'), 'wb'), -1)
    pickle.dump(labels_next_diag, open(os.path.join(out_dir, 'next_diag.labels'), 'wb'), -1)
    pickle.dump(labels_next_diag_ccs_single, open(os.path.join(out_dir, 'next_diag_ccs_single.labels'), 'wb'), -1)
    pickle.dump(labels_next_proc, open(os.path.join(out_dir, 'next_proc.labels'), 'wb'), -1)
    pickle.dump(labels_next_proc_ccs_single, open(os.path.join(out_dir, 'next_proc_ccs_single.labels'), 'wb'), -1)

    pickle.dump(dict_diag, open(os.path.join(out_dir, 'diag.dict'), 'wb'), -1)
    pickle.dump(dict_diag_ccs_single, open(os.path.join(out_dir, 'diag_ccs_single.dict'), 'wb'), -1)
    pickle.dump(dict_proc, open(os.path.join(out_dir, 'proc.dict'), 'wb'), -1)
    pickle.dump(dict_proc_ccs_single, open(os.path.join(out_dir, 'proc_ccs_single.dict'), 'wb'), -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adm_file', type=str, default='data/ADMISSIONS.csv')
    parser.add_argument('--diag_file', type=str, default='data/DIAGNOSES_ICD.csv')
    parser.add_argument('--proc_file', type=str, default='data/PROCEDURES_ICD.csv')
    parser.add_argument('--diag_ccs_single_file', type=str, default='data/ccs_single_dx_tool_2015.csv')
    parser.add_argument('--proc_ccs_single_file', type=str, default='data/ccs_single_pr_tool_2015.csv')
    parser.add_argument('--out_path', type=str, default='processed/')
    args = parser.parse_args()

    admissionFile = args.adm_file
    diagnosisFile = args.diag_file
    proceduresFile = args.proc_file
    diagSingleFile = args.diag_ccs_single_file
    procSingleFile = args.proc_ccs_single_file
    outDir = args.out_path

    processing(admissionFile, diagnosisFile, proceduresFile, diagSingleFile, procSingleFile, outDir)
