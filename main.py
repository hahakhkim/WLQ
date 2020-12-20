import numpy as np
import argparse
import os

import math_lib
import plot_lib
import Finite_horizon_controller
import Infinite_horizon_controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_number", default=150, type=int) # Number of stages
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
    finite_controller.plot_both_controllers(state_index=19, figure_number=1) # Compare the results of two controllers
    finite_controller.save()
    
    ########### Compare the result based on two different values of theta (0.5, 1.0)   ###########
    kwargs["theta"] = 0.5
    finite_controller = Finite_horizon_controller.Finite_horizon_controller(**kwargs)
    finite_controller.optimize_penalty()
    finite_controller.simulate()
    data1 = finite_controller.X_standard
    data2 = finite_controller.X_minimax

    kwargs["theta"] = 1.0
    finite_controller = Finite_horizon_controller.Finite_horizon_controller(**kwargs)
    finite_controller.optimize_penalty()
    finite_controller.simulate()
    data3 = finite_controller.X_standard
    data4 = finite_controller.X_minimax
    plot_lib.plot_compare_data(data1, data2, data3, data4, figure_number=2)

    '''
    ########### Calculate optimal penalty, control energy, and reliability with respect to theta ###########
    theta_list1 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    optimal_penalty_result = []
    control_energy_result = []
    for theta in theta_list1:
        print(theta)
        kwargs["theta"] = theta
        finite_controller = Finite_horizon_controller.Finite_horizon_controller(**kwargs)
        finite_controller.optimize_penalty()
        finite_controller.simulate()
        optimal_penalty_result.append(finite_controller.optimal_penalty)
        control_energy_result.append(finite_controller.control_energy)

    theta_list2 = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    reliability_result = [0 for i in range(10)]
    for i, theta in enumerate(theta_list2):
        print(theta)
        kwargs["theta"] = theta
        finite_controller = Finite_horizon_controller.Finite_horizon_controller(**kwargs)
        finite_controller.optimize_penalty()
        finite_controller.simulate_ground_truth_disturbance()
        reliability_result[i] = reliability_result[i] + finite_controller.reliability
    np.save('./results/optimal_penalty_result', optimal_penalty_result)
    np.save('./results/control_energy_result', control_energy_result)
    np.save('./results/reliability_result', reliability_result)
    plot_lib.plot_all_data(theta_list1, optimal_penalty_result, control_energy_result, theta_list2, reliability_result,
                           figure_number=3)

    '''
    ########### Infinite Horizon Control ###########
    if args.use_saved_sample:
        # Load sample data (stationary at each stage)
        single_stage_sample = np.load("./inputs/single_stage_sample.npy") # (Stage number x Sample number x k x 1) matrix
        single_stage_sample_mean = np.load("./inputs/single_stage_sample_mean.npy") # (Stage number x k x 1) matrix
    else:
        # Generate sample data from normal distribution
        single_stage_sample, single_stage_sample_mean, _ = math_lib.generate_single_sample(sample_number=args.sample_number,
                                                         dim=len(Xi[0]), mean=args.sample_mean, sigma=args.sample_sigma)
        np.save('./inputs/single_stage_sample', single_stage_sample)
        np.save('./inputs/single_stage_sample_mean', single_stage_sample_mean)

    kwargs = {
            "A": A, "B": B, "Xi": Xi, "Q": Q, "Q_f": Q_f, "R": R, "x_0": x_0,
            "sample": single_stage_sample,
            "sample_mean": single_stage_sample_mean,
            "stage_number": args.stage_number,
            "test_number": args.test_number,
            "theta": args.theta,
            "penalty": 1.5
        }
    infinite_controller = Infinite_horizon_controller.Infinite_horizon_controller(**kwargs)
    infinite_controller.simulate()
    infinite_controller.plot_both_controllers(state_index=19, figure_number=2)
    infinite_controller.save()

