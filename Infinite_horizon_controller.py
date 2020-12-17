import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import math_lib
import plot_lib

class Infinite_horizon_controller(object):
    def __init__(self, A, B, Xi, Q, Q_f, R, x_0, sample, sample_mean, theta, penalty, stage_number, test_number):
        self.A = A
        self.B = B
        self.Xi = Xi
        self.Q = Q
        self.Q_f = Q_f
        self.R = R
        self.x_0 = x_0
        self.sample = sample
        self.sample_mean = sample_mean
        self.theta = theta
        self.penalty = penalty
        self.stage_number = stage_number
        self.test_number = test_number
        self.dim_x = len(A)
        self.dim_u = len(B[0])
        self.dim_w = len(Xi[0])

        # Matrix Container for Standard LQG Controller
        self.P_standard = np.zeros((self.dim_x, self.dim_x))
        self.K_standard = np.zeros((self.dim_u, self.dim_x))
        self.X_standard = np.zeros((self.test_number, self.stage_number + 1, self.dim_x, 1))
        self.J_standard = np.zeros((self.test_number, self.stage_number + 1))

        # Matrix Container for Minimax LQR Controller
        self.P_minimax = np.zeros((self.dim_x, self.dim_x))
        self.r_minimax = np.zeros((self.dim_x, 1))
        self.z_minimax = np.zeros((self.stage_number + 1, 1, 1))
        self.K_minimax = np.zeros((self.dim_u, self.dim_x))
        self.L_minimax = np.zeros((self.dim_u, 1))
        self.X_minimax = np.zeros((self.test_number, self.stage_number + 1, self.dim_x, 1))
        self.J_minimax = np.zeros((self.test_number, self.stage_number + 1))

    def objective(self, x):
        return math_lib.objective_function_infinite(A=self.A, B=self.B, Xi=self.Xi, Q=self.Q, Q_f=self.Q_f, R=self.R,
                penalty=x, theta=self.theta, sample=self.sample, sample_mean=self.sample_mean, error_bound=1e-6, max_iteration=10000)


    def simulate(self):
        # Calculate P_t from recursion
        self.P_standard = math_lib.solve_standard_Riccati(self.A, self.B, self.Q, self.Q_f, self.R, error_bound=1e-6, max_iteration=10000)

        self.P_minimax, self.r_minimax, _ = math_lib.solve_minimax_Riccati(self.A, self.B, self.Xi, self.Q, self.Q_f, self.R,
                                                         self.sample, self.sample_mean, self.penalty, error_bound=1e-6,
                                                         max_iteration=10000)

       # Calculate control gain K_t from P_t
        self.K_standard = math_lib.standard_LQG_control_gain(self.A, self.B, self.P_standard, self.R)
        self.K_minimax, self.L_minimax = math_lib.minimax_LQR_control_gain(self.A, self.B, self.Xi, self.Q, self.R,
                                                    self.P_minimax, self.r_minimax, self.sample_mean, self.penalty)
        for itr in range(self.test_number):
            x_standard = np.zeros((self.stage_number + 1, self.dim_x, 1)) # Container for system state x_t
            x_minimax = np.zeros((self.stage_number + 1, self.dim_x, 1))
            x_standard[0] = np.reshape(self.x_0, (self.dim_x, 1)) # Initial state x_0
            x_minimax[0] = np.reshape(self.x_0, (self.dim_x, 1))
            v_standard = np.zeros((self.stage_number + 1)) # Container for the cost
            v_minimax = np.zeros((self.stage_number + 1))
            u_standard = np.zeros((self.stage_number + 1, self.dim_u, 1)) # Container for the control input u_t
            u_minimax = np.zeros((self.stage_number + 1, self.dim_u, 1))

            for t in range(self.stage_number):
                # Determining control input
                u_standard[t] = np.matmul(self.K_standard, x_standard[t])
                u_minimax[t] = np.matmul(self.K_minimax, x_minimax[t]) + self.L_minimax

                # Adversarial (worst-case) distribution generated from Wasserstein penalty problem
                disturbance_standard = math_lib.worst_case_distribution(self.sample, x_standard[t], u_standard[t],
                                                                        self.A, self.B, self.Xi, self.P_minimax, self.r_minimax, self.penalty)
                disturbance_minimax = math_lib.worst_case_distribution(self.sample, x_minimax[t], u_minimax[t],
                                                                       self.A, self.B, self.Xi, self.P_minimax, self.r_minimax, self.penalty)

                # System equation
                x_standard[t + 1] = np.matmul(self.A, x_standard[t]) + np.matmul(self.B, u_standard[t])\
                                    + np.matmul(self.Xi, disturbance_minimax)
                x_minimax[t + 1] = np.matmul(self.A, x_minimax[t]) + np.matmul(self.B, u_minimax[t])\
                                    + np.matmul(self.Xi, disturbance_minimax)

                # Computing Cost
                v_standard[t + 1] = v_standard[t] + math_lib.matmul3(np.transpose(x_standard[t]), self.Q, x_standard[t]) \
                                    + math_lib.matmul3(np.transpose(u_standard[t]), self.R, u_standard[t])
                v_minimax[t + 1] = v_minimax[t] + math_lib.matmul3(np.transpose(x_minimax[t]), self.Q, x_minimax[t]) \
                                   + math_lib.matmul3(np.transpose(u_minimax[t]), self.R, u_minimax[t])

            # Cost for the final stage.
            v_standard[self.stage_number] = v_standard[self.stage_number] + math_lib.matmul3(
                np.transpose(x_standard[self.stage_number]), self.Q_f, x_standard[self.stage_number])
            v_minimax[self.stage_number] = v_minimax[self.stage_number] + math_lib.matmul3(
                np.transpose(x_minimax[self.stage_number]), self.Q_f, x_minimax[self.stage_number])

            # Store the result (state and cost)
            self.X_standard[itr] = x_standard
            self.X_minimax[itr] = x_minimax
            self.J_standard[itr] = v_standard
            self.J_minimax[itr] = v_minimax
        return

    def plot_single_controllers(self, state_index, figure_number, controller="Minimax LQR"):
        # Select data
        if controller == "Minimax LQR":
            X_data = self.X_minimax
        elif controller == "Standard LQG":
            X_data = self.X_standard
        else:
            print("Plot ERROR: Unidentified Controller")
            return
        plot_lib.plot_single_data(X_data, state_index, figure_number, title=controller)
        return

    def plot_both_controllers(self, state_index, figure_number):
        plot_lib.plot_both_data(self.X_standard, self.X_minimax, state_index, figure_number)
        return

    def save(self):
        np.save("./results/infinite_horizon/Standard_LQG_P", self.P_standard)
        np.save("./results/infinite_horizon/Standard_LQG_K", self.K_standard)
        np.save("./results/infinite_horizon/Standard_LQG_X", self.X_standard)
        np.save("./results/infinite_horizon/Standard_LQG_J", self.J_standard)

        np.save("./results/infinite_horizon/Minimax_LQR_P", self.P_minimax)
        np.save("./results/infinite_horizon/Minimax_LQR_r", self.r_minimax)
        np.save("./results/infinite_horizon/Minimax_LQR_K", self.K_minimax)
        np.save("./results/infinite_horizon/Minimax_LQR_L", self.L_minimax)
        np.save("./results/infinite_horizon/Minimax_LQR_X", self.X_minimax)
        np.save("./results/infinite_horizon/Minimax_LQR_J", self.J_minimax)
        return

