#   -*- coding: utf-8 -*-
#
#   main.py
#   
#   Developed by Tianyi Liu on 2020-11-25 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import argparse
from factorization import iNMF
from utils import load_data, load_cache


def parse_args():
    """
    Argparser
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--read_cache",
                       dest="cache",
                       help="Read data from cache.")
    group.add_argument("--read_raw",
                       dest="raw",
                       help="Read data from raw data.")
    parser.add_argument("--t",
                        dest = "t",
                        action="store_true",
                        help = "Transpose to #Gene x #Cell, Default = FALSE")
    parser.add_argument("--groups",
                        dest="groups",
                        default="",
                        help="Group file")
    parser.add_argument("--groups_col",
                        dest="groups_col",
                        default="",
                        help="Group column")
    parser.add_argument("--batches",
                        dest="batches",
                        default="",
                        help="Batch file")
    parser.add_argument("--batches_col",
                        dest="batches_col",
                        default="",
                        help="Batches column")
    parser.add_argument("--k",
                        dest="k",
                        type=int,
                        default=64,
                        help="inner dimension")
    parser.add_argument("--lam",
                        dest="lam",
                        type=float,
                        default=1,
                        help="Lambda")
    parser.add_argument("--penalty",
                        dest="penalty",
                        action="store_false",
                        help="Additional penalty, Default = TRUE.")
    parser.add_argument("--gam",
                        dest="gam",
                        type=float,
                        default=0.5,
                        help="Gamma")
    parser.add_argument("--metric",
                        dest="metric",
                        default="Frobenius",
                        choices=["Frobenius", "kld"],
                        help="Distance metric in objective function.")
    args = parser.parse_args()

    print('\n', " Call with Arguments ".center(50, "="), sep='')
    for item in args.__dict__:
        print("{:18}".format(item), "->\t", args.__dict__[item])
    return args


if __name__ == "__main__":
    args = parse_args()
    # Data # Genes x # Cells
    if args.raw is not None:
        data_dict = load_data(args.raw, args.t, args.groups, args.groups_col, args.batches, args.batches_col)
    else:
        data_dict = load_cache(args.cache)
    inmf = iNMF(data_dict, args.k, args.lam, args.gam, args.penalty, args.metric)
    print(inmf)

    for i in range(100000):
        obj_val = inmf.cal_objective()
        inmf.cvg.update_ma(obj_val)
        if i == 0 or (i + 1) % 100 == 0:
            print("Iteration: {}\tObjective Value: {}".format(i + 1, obj_val))
        if inmf.cvg.is_converge():
            print("Convergence Criterion Reached at Iteration: {}".format(i + 1))
            break
        inmf.update_par()

    inmf.plot_obj()
    inmf.run_dr("PCA", original=True)
    inmf.cal_alignment()
    inmf.plot_embedding("PCA")

