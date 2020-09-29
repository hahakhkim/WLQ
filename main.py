import numpy as np
import argparse
import os
import math_lib
import Minimax_LQR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_number", default=150, type=int) # Number of stages
    parser.add_argument("--test_number", default=100, type=int) # Number of test cases
    parser.add_argument("--theta", default=2.0) # Radius of the Wasserstein ambiguity set
    parser.add_argument("--penalty", default=2.0) # Coefficient of the Wasserstein penalty
    parser.add_argument("--sample_number", default=10, type=int)  # Number of samples
    parser.add_argument("--sample_mean", default=0.02)  # Mean of generated samples
    parser.add_argument("--sample_sigma", default=0.01)  # Sigma of generated samples
    parser.add_argument("--use_saved_sample", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Data Input
    A = np.load("./inputs/A.npy") # (n x n) matrix
    B = np.load("./inputs/B.npy") # (n x m) matrix
    Xi = np.load("./inputs/Xi.npy") # (n x k) matrix
    Q = np.load("./inputs/Q.npy") # (n x n) matrix
    Q_f = np.load("./inputs/Q_f.npy") # (n x n) matrix
    R = np.load("./inputs/R.npy") # (m x m) matrix
    x_0 = np.load("./inputs/x_0.npy") # (n x 1) vector

    if args.use_saved_sample:
        # Load sample
        sample = np.load("./inputs/Sample.npy") # (Sample number x Stage number x k x 1) matrix
        sample_mean = np.load("./inputs/Sample_mean.npy") # (Stage number x k x 1) matrix
    else:
        # Generate sample
        sample, sample_mean = math_lib.generate_multi_sample(sample_number=args.sample_number, stage_number=args.stage_number,
                                                         dim=len(Xi[0]), mean=args.sample_mean, sigma=args.sample_sigma)
        np.save('./inputs/Sample', sample)
        np.save('./inputs/Sample_mean', sample_mean)
    kwargs = {
            "A": A, "B": B, "Xi": Xi, "Q": Q, "Q_f": Q_f, "R": R, "x_0": x_0,
            "sample": sample,
            "sample_mean": sample_mean,
            "stage_number": args.stage_number,
            "test_number": args.test_number,
            "theta": args.theta,
            "penalty": args.penalty
        }
    controller = Minimax_LQR.Finite_horizon_controller(**kwargs)
    controller.simulate()
    controller.plot_results(state_index=19, figure_number=1, controller="Standard LQG")
    controller.plot_results(state_index=19, figure_number=2, controller="Minimax LQR")
