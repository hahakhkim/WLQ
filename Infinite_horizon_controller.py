import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import math_lib

class Finite_horizon_controller(object):
    def __init__(self, A, B, Xi, Q, Q_f, R, x_0, sample, sample_mean, theta, stage_number, test_number):
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
        self.penalty = 0
        self.stage_number = stage_number
        self.test_number = test_number
        self.dim_x = len(A)
        self.dim_u = len(B[0])
        self.dim_w = len(Xi[0])

        # Matrix Container for Standard LQG Controller
        self.P_standard = np.zeros((self.stage_number + 1, self.dim_x, self.dim_x)) #
        self.K_standard = np.zeros((self.stage_number, self.dim_u, self.dim_x))
        self.X_standard = np.zeros((self.test_number, self.stage_number + 1, self.dim_x, 1))
        self.J_standard = np.zeros((self.test_number, self.stage_number + 1))

        # Matrix Container for Minimax LQR Controller
        self.P_minimax = np.zeros((self.stage_number + 1, self.dim_x, self.dim_x))
        self.r_minimax = np.zeros((self.stage_number + 1, self.dim_x, 1))
        self.z_minimax = np.zeros((self.stage_number + 1, 1, 1))
        self.K_minimax = np.zeros((self.stage_number, self.dim_u, self.dim_x))
        self.L_minimax = np.zeros((self.stage_number, self.dim_u, 1))
        self.X_minimax = np.zeros((self.test_number, self.stage_number + 1, self.dim_x, 1))
        self.J_minimax = np.zeros((self.test_number, self.stage_number + 1))

    def objective(self, x):
        return math_lib.objective_function(A=self.A, B=self.B, Xi=self.Xi, Q=self.Q, Q_f=self.Q_f, R=self.R, lam=x,
                                    theta=self.theta, stage_number=self.stage_number, initial_state=self.x_0, sample=self.sample, sample_mean=self.sample_mean)
    def optimize_penalty(self):
        self.penalty = minimize(self.objective, x0=np.array([5.0]), method='nelder-mead', options={'xatol': 1e-6, 'disp': True}).x[0]
        print("Optimal penalty:", self.penalty)
        return

    def simulate(self):
        # Calculate P_t from recursion
        self.P_standard[self.stage_number] = self.Q_f
        self.P_minimax[self.stage_number] = self.Q_f
        for t in range(self.stage_number, 0, -1):
            self.P_standard[t - 1] = math_lib.standard_Riccati_iteration(self.A, self.B, self.P_standard[t], self.Q, self.R)
            self.P_minimax[t - 1], self.r_minimax[t-1], self.z_minimax[t-1] = math_lib.minimax_Riccati_iteration(self.A, self.B, self.Xi, self.Q, self.R,
                                self.P_minimax[t], self.r_minimax[t], self.z_minimax[t], self.sample[t-1], self.sample_mean[t-1], self.penalty)

        # Calculate control gain K_t from P_t
        for t in range(self.stage_number):
            self.K_standard[t] = math_lib.standard_LQG_control_gain(self.A, self.B, self.P_standard[t + 1], self.R)
            self.K_minimax[t], self.L_minimax[t] = math_lib.minimax_LQR_control_gain(self.A, self.B, self.Xi, self.Q, self.R,
                                                                                     self.P_minimax[t+1], self.r_minimax[t+1], self.sample_mean[t], self.penalty)
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
                # Control input
                u_standard[t] = np.matmul(self.K_standard[t], x_standard[t])
                u_minimax[t] = np.matmul(self.K_minimax[t], x_minimax[t]) + self.L_minimax[t]

                # Worst-case distribution generated from Wasserstein penalty problem
                disturbance_standard = math_lib.worst_case_distribution(self.sample[t], x_standard[t], u_standard[t],
                                                                        self.A, self.B, self.Xi, self.P_minimax[t + 1], self.r_minimax[t + 1], self.penalty)
                disturbance_minimax = math_lib.worst_case_distribution(self.sample[t], x_minimax[t], u_minimax[t],
                                                                       self.A, self.B, self.Xi, self.P_minimax[t + 1], self.r_minimax[t + 1], self.penalty)

                # System equation
                x_standard[t + 1] = np.matmul(self.A, x_standard[t]) + np.matmul(self.B, u_standard[t])\
                                    + np.matmul(self.Xi, disturbance_minimax)
                x_minimax[t + 1] = np.matmul(self.A, x_minimax[t]) + np.matmul(self.B, u_minimax[t])\
                                    + np.matmul(self.Xi, disturbance_minimax)

                # Cost function
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

    def plot_single_result(self, state_index, figure_number, controller="Minimax LQR"):
        # Select data
        if controller == "Minimax LQR":
            X_data = self.X_minimax
        elif controller == "Standard LQG":
            X_data = self.X_standard
        else:
            print("Plot ERROR: Unidentified Controller")
            return

        # Rearrange data
        data = [[0 for itr in range(self.test_number)] for t in range(self.stage_number)]
        for t in range(self.stage_number):
            for itr in range(self.test_number):
                data[t][itr] = X_data[itr][t][state_index][0]

        # Box plot
        fig = plt.figure(figure_number)
        ax = fig.add_subplot(1, 1, 1)
        plt.title(controller)
        b = ax.boxplot(data, showcaps=False, showfliers=False, widths=0.001, patch_artist=True)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(b[element], color='cyan')
        plt.setp(b['boxes'], color='blue')
        plt.setp(b['medians'], color='black')
        ax.set_xticks([20 * i for i in range(int(self.stage_number / 20) + 1)])
        ax.set_xticklabels([20 * i for i in range(int(self.stage_number / 20) + 1)])
        ax.set_xlabel('stage')
        ax.set_ylabel('value')
        fig = plt.gcf()
        fig.savefig('./results/figure' + str(figure_number) + '.png', dpi=300)
        plt.show()
        return

    def plot_both_results(self, state_index, figure_number):
        # Rearrange data
        data = np.zeros((2, self.test_number, self.stage_number))
        for t in range(self.stage_number):
            for itr in range(self.test_number):
                data[0, itr, t] = self.X_standard[itr, t, state_index, 0]
                data[1, itr, t] = self.X_minimax[itr, t, state_index, 0]

        # Box plot
        fig = plt.figure(figure_number)
        ax = []
        plt.subplots_adjust(hspace=0.5)
        plt.rc('font', size=10)
        for p in range(2):
            ax.append(fig.add_subplot(1, 2, p + 1))
            b = ax[p].boxplot(data[p], showcaps=False, showfliers=False, widths=0.001, patch_artist=True)
            for element in ['whiskers', 'fliers', 'means', 'caps']:
                plt.setp(b[element], color='cyan')
            plt.setp(b['boxes'], color='blue')
            plt.setp(b['medians'], color='black')
            plt.grid(linestyle=':', linewidth=1.0, alpha=0.8)
            ax[p].set_xticks([20 * i for i in range(int(self.stage_number / 20) + 1)])
            ax[p].set_xticklabels([20 * i for i in range(int(self.stage_number / 20) + 1)])
            ax[p].set_xlabel('stage')
            if p == 0:
                plt.title("Standard LQG")
                ax[p].set_ylabel('value')
            else:
                plt.title("Minimax LQR")
        fig.savefig('./results/figure' + str(figure_number) + '.png', dpi=300)
        plt.show()
        return

    def save(self):
        np.save("./results/Standard_LQG_infinite_P", self.P_standard)
        np.save("./results/Standard_LQG_infinite_K", self.K_standard)
        np.save("./results/Standard_LQG_infinite_X", self.X_standard)
        np.save("./results/Standard_LQG_infinite_J", self.J_standard)

        np.save("./results/Minimax_LQR_infinite_P", self.P_minimax)
        np.save("./results/Minimax_LQR_infinite_r", self.r_minimax)
        np.save("./results/Minimax_LQR_infinite_K", self.K_minimax)
        np.save("./results/Minimax_LQR_infinite_L", self.L_minimax)
        np.save("./results/Minimax_LQR_infinite_X", self.X_minimax)
        np.save("./results/Minimax_LQR_infinite_J", self.J_minimax)
        return

