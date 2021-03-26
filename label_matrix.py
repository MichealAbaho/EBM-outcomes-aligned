import numpy as np
import pickle
import argparse
import pandas as pd
from tabulate import tabulate
from scipy.spatial.distance import euclidean, cosine

def label_matrix(args):
    if args.file1.__contains__('pickle') and args.file2.__contains__('pickle'):
        source = pickle.load(open(args.file1, 'rb'))
        target = pickle.load(open(args.file2, 'rb'))
    else:
        source = args.file1
        target = args.file2

    labe_matrix = {}
    #obtain the distance between the labe representations in the source and target dataset using any one of the methods
    for s in source:
        _s_ = {}
        for t in target:
            sE, tE = source[s].cpu().numpy(), target[t].cpu().numpy()
            if args.distance.lower() == 'euclidean':
                d = euclidean(sE, tE)
            elif args.distance.lower() == 'cosine':
                d = cosine(sE, tE)
            _s_[t] = d
        labe_matrix[s] = _s_

    labe_matrix_frame = pd.DataFrame(labe_matrix)
    print(tabulate(labe_matrix_frame, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("--file1", help="source dataset")
    par.add_argument("--file2", help="target dataset")
    par.add_argument("--distance", default="cosine", help="method for calculating the distance between the vectors")
    args = par.parse_args()
    label_matrix(args)