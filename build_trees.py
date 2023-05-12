# coding=utf-8
import sys, copy
import pickle
import numpy as np
import argparse
from collections import defaultdict, OrderedDict


def build_trees_diag(infile, seqFile, typeFile, outPath):
    infd = open(infile, 'r')
    _ = infd.readline()

    seqs = pickle.load(open(seqFile, 'rb'))
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
    print('Number of diag ccs multi type:'+str(rootCode))

    # pickle.dump(types, open(outPath + 'tree.diag.ancestors.types', 'wb'), -1)

    # CCS each level count
    print('cat1count: %d' % cat1count)
    print('cat2count: %d' % cat2count)
    print('cat3count: %d' % cat3count)
    print('cat4count: %d' % cat4count)
    print('Number of total ancestors: %d' % (cat1count + cat2count + cat3count + cat4count + 1))
    print('hit count: %d' % len(set(hitList)))
    print('miss count: %d' % len(startSet - set(hitList)))
    # missSet-> ICD without ccs category information
    # missList-> ICD not occurred in MIMIC-III dataset
    missSet = startSet - set(hitList)


    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    infd = open(infile, 'r')
    infd.readline()

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
            continue

        icdCode = types[icd9]

        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]


    newFiveMap = {}
    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    rtypes = dict([(v, k) for k, v in types.items()])       # type_num->icd_code


    codeCount = 0
    for icdCode, ancestors in fiveMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        # newFiveMap-> [newTypeNum, ancestors(rootCode, code1, code2, code3, code4)]
        newFiveMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in fourMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newFourMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in threeMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newThreeMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in twoMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newTwoMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in oneMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newOneMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    # build mappings from parents to children and the reversed
    p2c = [defaultdict(set) for _ in range(5)]
    c2p = [defaultdict(set) for _ in range(5)]
    for icdCode, ancestors in newFiveMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(5):
            parent, child = tmp[i], tmp[i+1]
            pp = tmp[:i+1]
            p2c[i][parent].add(child)
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newFourMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(4):
            parent, child = tmp[i], tmp[i+1]
            pp = tmp[:i + 1]
            p2c[i][parent].add(child)
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newThreeMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(3):
            parent, child = tmp[i], tmp[i+1]
            pp = tmp[:i + 1]
            p2c[i][parent].add(child)
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newTwoMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(2):
            parent, child = tmp[i], tmp[i+1]
            pp = tmp[:i + 1]
            p2c[i][parent].add(child)
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newOneMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(1):
            parent, child = tmp[i], tmp[i+1]
            pp = tmp[:i + 1]
            p2c[i][parent].add(child)
            c2p[i][child].add(tuple(pp))

    pickle.dump(newFiveMap, open(outPath + 'tree.diag.level5.pk', 'wb'), -1)
    pickle.dump(newFourMap, open(outPath + 'tree.diag.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outPath + 'tree.diag.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outPath + 'tree.diag.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outPath + 'tree.diag.level1.pk', 'wb'), -1)
    pickle.dump(newTypes, open(outPath + 'tree.diag.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outPath + 'tree.diag.seqs', 'wb'), -1)
    pickle.dump((p2c, c2p), open(outPath + 'tree.diag.p2c', 'wb'), -1)


def build_trees_proc(infile, seqFile, typeFile, outPath):
    infd = open(infile, 'r')
    _ = infd.readline()

    seqs = pickle.load(open(seqFile, 'rb'))
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
    print('Number of proc ccs multi type:'+str(rootCode))

    # pickle.dump(types, open(outPath + 'tree.proc.ancestors.types', 'wb'), -1)


    print('cat1count: %d' % cat1count)
    print('cat2count: %d' % cat2count)
    print('cat3count: %d' % cat3count)
    print('Number of total ancestors: %d' % (cat1count + cat2count + cat3count + 1))
    print('hit count: %d' % len(set(hitList)))
    print('miss count: %d' % len(startSet - set(hitList)))
    missSet = startSet - set(hitList)




    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    infd = open(infile, 'r')
    infd.readline()

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
            continue

        icdCode = types[icd9]

        if len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            print(icd9)
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]


    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    rtypes = dict([(v, k) for k, v in types.items()])

    codeCount = 0
    for icdCode, ancestors in fourMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newFourMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in threeMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newThreeMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in twoMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newTwoMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in oneMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newOneMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)


    p2c = [defaultdict(set) for _ in range(4)]
    c2p = [defaultdict(set) for _ in range(4)]
    for icdCode, ancestors in newFourMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(4):
            parent, child = tmp[i], tmp[i+1]
            p2c[i][parent].add(child)
            pp = tmp[:i+1]
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newThreeMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(3):
            parent, child = tmp[i], tmp[i+1]
            p2c[i][parent].add(child)
            pp = tmp[:i+1]
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newTwoMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(2):
            parent, child = tmp[i], tmp[i+1]
            p2c[i][parent].add(child)
            pp = tmp[:i+1]
            c2p[i][child].add(tuple(pp))
    for icdCode, ancestors in newOneMap.items():
        tmp = ancestors[1:] + [ancestors[0]]
        for i in range(1):
            parent, child = tmp[i], tmp[i+1]
            p2c[i][parent].add(child)
            pp = tmp[:i+1]
            c2p[i][child].add(tuple(pp))

    pickle.dump(newFourMap, open(outPath + 'tree.proc.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outPath + 'tree.proc.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outPath + 'tree.proc.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outPath + 'tree.proc.level1.pk', 'wb'), -1)
    pickle.dump(newTypes, open(outPath + 'tree.proc.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outPath + 'tree.proc.seqs', 'wb'), -1)
    pickle.dump((p2c, c2p), open(outPath + 'tree.proc.p2c', 'wb'), -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_seq', type=str, default='processed/diag.inputs')
    parser.add_argument('--proc_seq', type=str, default='processed/proc.inputs')
    parser.add_argument('--diag_type', type=str, default='processed/diag.dict')
    parser.add_argument('--proc_type', type=str, default='processed/proc.dict')
    parser.add_argument('--diag_ccs_multi_file', type=str, default='data/ccs_multi_dx_tool_2015.csv')
    parser.add_argument('--proc_ccs_multi_file', type=str, default='data/ccs_multi_pr_tool_2015.csv')
    parser.add_argument('--out_path', type=str, default='processed/')
    args = parser.parse_args()

    build_trees_diag(args.diag_ccs_multi_file, args.diag_seq, args.diag_type, args.out_path)
    build_trees_proc(args.proc_ccs_multi_file, args.proc_seq, args.proc_type, args.out_path)
