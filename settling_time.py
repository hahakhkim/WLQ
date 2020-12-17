import numpy as np
import argparse
import os

import math_lib
import plot_lib
import Finite_horizon_controller
import Infinite_horizon_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_number", default=300, type=int) # Number of stages
    parser.add_argument("--test_number", default=1000, type=int) # Number of test cases
    parser.add_argument("--theta", default=0.5) # Radius of the Wasserstein ambiguity set
    parser.add_argument("--sample_number", default=10, type=int)  # Number of samples
    parser.add_argument("--sample_mean", default=0.02)  # Mean of generated samples
    parser.add_argument("--sample_sigma", default=0.01)  # Sigma of generated samples
    parser.add_argument("--use_saved_sample", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("./results/infinite_horizon"):
        os.makedirs("./results/infinite_horizon")
    if not os.path.exists("./results/finite_horizon"):
        os.makedirs("./results/finite_horizon")

    # Data Input
    A = np.load("./inputs/A.npy") # (n x n) matrix
    B = np.load("./inputs/B.npy") # (n x m) matrix
    Xi = np.load("./inputs/Xi.npy") # (n x k) matrix
    Q = np.load("./inputs/Q.npy") # (n x n) matrix
    Q_f = np.load("./inputs/Q_f.npy") # (n x n) matrix
    R = np.load("./inputs/R.npy") # (m x m) matrix
    x_0 = np.load("./inputs/x_0.npy") # (n x 1) vector



    ########### Finite Horizon Control ###########
    if args.use_saved_sample:
        # Load sample data for each stage
        multi_stage_sample = np.load("./inputs/multi_stage_sample.npy") # (Stage number x Sample number x k x 1) matrix
        multi_stage_sample_mean = np.load("./inputs/multi_stage_sample_mean.npy") # (Stage number x k x 1) matrix
    else:
        # Generate sample data from normal distribution
        multi_stage_sample, multi_stage_sample_mean = math_lib.generate_multi_sample(sample_number=args.sample_number, stage_number=args.stage_number,
                                                         dim=len(Xi[0]), mean=args.sample_mean, sigma=args.sample_sigma)
        np.save('./inputs/multi_stage_sample', multi_stage_sample)
        np.save('./inputs/multi_stage_sample_mean', multi_stage_sample_mean)

    for gen in range(10):
        x_0 = np.zeros((20, 1))
        x_0[2*gen+1][0] = 1.0
        kwargs = {
                "A": A, "B": B, "Xi": Xi, "Q": Q, "Q_f": Q_f, "R": R, "x_0": x_0,
                "sample": multi_stage_sample,
                "sample_mean": multi_stage_sample_mean,
                "stage_number": args.stage_number,
                "test_number": args.test_number,
                "theta": args.theta
            }
        finite_controller = Finite_horizon_controller.Finite_horizon_controller(**kwargs)
        finite_controller.optimize_penalty()
        finite_controller.simulate()
        np.save("./results/settling_time/"+str(gen+1)+"/Standard_LQG_X", finite_controller.X_standard)
        np.save("./results/settling_time/"+str(gen+1)+"/Minimax_LQR_X", finite_controller.X_minimax)

    for gen in range(10):
        X1 = np.load("./results/settling_time/"+str(gen+1)+"/Standard_LQG_X.npy")
        X2 = np.load("./results/settling_time/"+str(gen+1)+"/Minimax_LQR_X.npy")
        plot_lib.plot_median(X1, X2, state_index=2*gen+1, figure_number=gen+1)