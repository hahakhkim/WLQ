import copy
import numpy as np
import matplotlib.pyplot as plt
import math_lib

class Finite_horizon_controller(object):
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
        self.P_standard = np.zeros((self.stage_number + 1, self.dim_x, self.dim_x)) #
        self.K_standard = np.zeros((self.stage_number, self.dim_u, self.dim_x))
        self.X_standard = np.zeros((self.test_number, self.stage_number + 1, self.dim_x, 1))
        self.J_standard = np.zeros((self.test_number, self.stage_number + 1))

        # Matrix Container for Minimax LQR Controller
        self.P_minimax = np.zeros((self.stage_number + 1, self.dim_x, self.dim_x))
        self.r_minimax = np.zeros((self.stage_number + 1, self.dim_x, 1))
        self.K_minimax = np.zeros((self.stage_number, self.dim_u, self.dim_x))
        self.L_minimax = np.zeros((self.stage_number, self.dim_u, 1))
        self.X_minimax = np.zeros((self.test_number, self.stage_number + 1, self.dim_x, 1))
        self.J_minimax = np.zeros((self.test_number, self.stage_number + 1))

    def simulate(self):
        # Calculate P_t from recursion
        self.P_standard[self.stage_number] = self.Q_f
        self.P_minimax[self.stage_number] = self.Q_f
        for t in range(self.stage_number, 0, -1):
            self.P_standard[t - 1] = math_lib.standard_Riccati_iteration(self.A, self.B, self.P_standard[t], self.Q, self.R)
            self.P_minimax[t - 1], self.r_minimax[t-1] = math_lib.minimax_Riccati_iteration(self.A, self.B, self.Xi, self.Q, self.R,
                                                                                            self.P_minimax[t], self.r_minimax[t], self.sample_mean[t-1], self.penalty)

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
                                    + np.matmul(self.Xi, disturbance_standard)
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

    def plot_results(self, state_index, figure_number, controller="Minimax LQR"):
        if controller == "Minimax LQR":
            X_data = self.X_minimax
        elif controller == "Standard LQG":
            X_data = self.X_standard
        else:
            print("Plot ERROR: Unidentified Controller")
            return

        data = [[0 for itr in range(self.test_number)] for t in range(self.stage_number)]
        for t in range(self.stage_number):
            for itr in range(self.test_number):
                data[t][itr] = X_data[itr][t][state_index][0]
        fig = plt.figure(figure_number)
        ax = fig.add_subplot(1, 1, 1)
        plt.title(controller)
        b = ax.boxplot(data, showcaps=False, showfliers=False, widths=0.001, patch_artist=True)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(b[element], color='cyan')
        plt.setp(b['boxes'], color='blue')
        plt.setp(b['medians'], color='black')
        ax.set_xlabel('stage')
        ax.set_ylabel('value')
        fig = plt.gcf()
        fig.savefig('./results/figure' + str(figure_number) + '.png', dpi=300)
        plt.show()
        return

    def save(self):
        np.save("Standard_LQG_finite_P", self.P_standard)
        np.save("Standard_LQG_finite_K", self.K_standard)
        np.save("Standard_LQG_finite_X", self.X_standard)
        np.save("Standard_LQG_finite_J", self.J_standard)

        np.save("Minimax_LQR_finite_P", self.P_minimax)
        np.save("Minimax_LQR_finite_r", self.r_minimax)
        np.save("Minimax_LQR_finite_K", self.K_minimax)
        np.save("Minimax_LQR_finite_L", self.L_minimax)
        np.save("Minimax_LQR_finite_X", self.X_minimax)
        np.save("Minimax_LQR_finite_J", self.J_minimax)
        return

