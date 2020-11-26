#   -*- coding: utf-8 -*-
#
#   main.py
#   
#   Developed by Tianyi Liu on 2020-11-25 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

import numpy as np
from factorization import iNMF


if __name__ == "__main__":
    # Data # Genes x # Cells
    x1 = np.abs(np.random.randn(1000, 800)) + 1
    x2 = np.abs(np.random.randn(1000, 800)) + 3

    inmf = iNMF(x1, x2, 64, 1, 0.2, False)

    for i in range(100000):
        obj_val = inmf.cal_objective()
        print("Objective Value: {}".format(obj_val))
        inmf.cvg.update_ma(obj_val)
        if inmf.cvg.is_converge():
            print("Convergence Criterion Reached.")
            break

        inmf.update_par()

    inmf.plot_obj()
    inmf.run_dr("PCA", original=True)
    inmf.plot_embedding(np.concatenate((np.ones(800), np.ones(800) * 2)), None, "PCA")

