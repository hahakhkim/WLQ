import numpy as np
from scipy.optimize import minimize
import math
import math_lib
import plot_lib

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
        self.optimal_penalty = 0
        self.infimum_penalty = 0
        self.control_energy = 0
        self.stage_number = stage_number
        self.test_number = test_number
        self.dim_x = len(A)
        self.dim_u = len(B[0])
        self.dim_w = len(Xi[0])
        self.reliability = 0
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
        if x < self.infimum_penalty:
            return math.inf
        return math_lib.objective_function_finite(A=self.A, B=self.B, Xi=self.Xi, Q=self.Q, Q_f=self.Q_f, R=self.R, penalty=x,
                                    theta=self.theta, stage_number=self.stage_number, initial_state=self.x_0, sample=self.sample, sample_mean=self.sample_mean)

    def optimize_penalty(self):
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption 1)
        print("Computing penalty boundary...")
        self.infimum_penalty = math_lib.binarysearch_infimum_penalty_finite(self.stage_number, self.A, self.B, self.Xi, self.Q, self.Q_f, self.R)
        print("Boundary penalty (lambda_hat):", self.infimum_penalty)
        # Optimize penalty using nelder-mead method
        self.optimal_penalty = minimize(self.objective, x0=np.array([2*self.infimum_penalty]), method='nelder-mead', options={'xatol': 1e-6, 'disp': True}).x[0]
        print("Optimal penalty (lambda_star):", self.optimal_penalty)
        return

    def simulate(self):
        # Calculate P_t from recursion
        self.P_standard[self.stage_number] = self.Q_f
        self.P_minimax[self.stage_number] = self.Q_f
        for t in range(self.stage_number, 0, -1):
            self.P_standard[t - 1] = math_lib.standard_Riccati_iteration(self.A, self.B, self.P_standard[t], self.Q, self.R)
            self.P_minimax[t - 1], self.r_minimax[t-1], self.z_minimax[t-1] = math_lib.minimax_Riccati_iteration(self.A, self.B, self.Xi, self.Q, self.R,
                                self.P_minimax[t], self.r_minimax[t], self.z_minimax[t], self.sample[t-1], self.sample_mean[t-1], self.optimal_penalty)

        # Calculate control gain K_t from P_t
        for t in range(self.stage_number):
            self.K_standard[t] = math_lib.standard_LQG_control_gain(self.A, self.B, self.P_standard[t + 1], self.R)
            self.K_minimax[t], self.L_minimax[t] = math_lib.minimax_LQR_control_gain(self.A, self.B, self.Xi, self.Q, self.R,
                                                                                     self.P_minimax[t+1], self.r_minimax[t+1], self.sample_mean[t], self.optimal_penalty)
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
                u_standard[t] = np.matmul(self.K_standard[t], x_standard[t])
                u_minimax[t] = np.matmul(self.K_minimax[t], x_minimax[t]) + self.L_minimax[t]

                # Worst-case distribution generated from Wasserstein penalty problem
                disturbance_standard = math_lib.worst_case_distribution(self.sample[t], x_standard[t], u_standard[t],
                                                                        self.A, self.B, self.Xi, self.P_minimax[t + 1], self.r_minimax[t + 1], self.optimal_penalty)
                disturbance_minimax = math_lib.worst_case_distribution(self.sample[t], x_minimax[t], u_minimax[t],
                                                                       self.A, self.B, self.Xi, self.P_minimax[t + 1], self.r_minimax[t + 1], self.optimal_penalty)

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
            control_energy_itr = 0
            for t in range(50):
                control_energy_itr = control_energy_itr + np.linalg.norm(u_minimax[t]) / 50.0

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
            self.control_energy = self.control_energy + control_energy_itr/self.test_number
        return

    def plot_single_controller(self, state_index, figure_number, controller="Minimax LQR"):
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
        np.save("./results/finite_horizon/Standard_LQG_P", self.P_standard)
        np.save("./results/finite_horizon/Standard_LQG_K", self.K_standard)
        np.save("./results/finite_horizon/Standard_LQG_X", self.X_standard)
        np.save("./results/finite_horizon/Standard_LQG_J", self.J_standard)

        np.save("./results/finite_horizon/Minimax_LQR_P", self.P_minimax)
        np.save("./results/finite_horizon/Minimax_LQR_r", self.r_minimax)
        np.save("./results/finite_horizon/Minimax_LQR_K", self.K_minimax)
        np.save("./results/finite_horizon/Minimax_LQR_L", self.L_minimax)
        np.save("./results/finite_horizon/Minimax_LQR_X", self.X_minimax)
        np.save("./results/finite_horizon/Minimax_LQR_J", self.J_minimax)
        return

    def simulate_ground_truth_disturbance(self):
        # Calculate P_t from recursion
        self.P_minimax[self.stage_number] = self.Q_f
        for t in range(self.stage_number, 0, -1):
            self.P_minimax[t - 1], self.r_minimax[t-1], self.z_minimax[t-1] = math_lib.minimax_Riccati_iteration(self.A, self.B, self.Xi, self.Q, self.R,
                                self.P_minimax[t], self.r_minimax[t], self.z_minimax[t], self.sample[t-1], self.sample_mean[t-1], self.optimal_penalty)

        # Calculate control gain K_t from P_t
        for t in range(self.stage_number):
           self.K_minimax[t], self.L_minimax[t] = math_lib.minimax_LQR_control_gain(self.A, self.B, self.Xi, self.Q, self.R,
                                                                                     self.P_minimax[t+1], self.r_minimax[t+1], self.sample_mean[t], self.optimal_penalty)
        for itr in range(self.test_number):
            x_minimax = np.zeros((self.stage_number + 1, self.dim_x, 1))
            x_minimax[0] = np.reshape(self.x_0, (self.dim_x, 1))
            v_minimax = np.zeros((self.stage_number + 1))
            u_minimax = np.zeros((self.stage_number + 1, self.dim_u, 1))

            for t in range(self.stage_number):
                # Determining control input
                u_minimax[t] = np.matmul(self.K_minimax[t], x_minimax[t]) + self.L_minimax[t]

                # ground truth distribution generated from Normal distribution
                disturbance_minimax = np.reshape(np.random.normal(0.02, 0.01, len(self.Xi[0])), (len(self.Xi[0]), 1))

                # System equation
                x_minimax[t + 1] = np.matmul(self.A, x_minimax[t]) + np.matmul(self.B, u_minimax[t])\
                                    + np.matmul(self.Xi, disturbance_minimax)

                # Computing Cost
                v_minimax[t + 1] = v_minimax[t] + math_lib.matmul3(np.transpose(x_minimax[t]), self.Q, x_minimax[t]) \
                                   + math_lib.matmul3(np.transpose(u_minimax[t]), self.R, u_minimax[t])

            # Cost for the final stage.
            v_minimax[self.stage_number] = v_minimax[self.stage_number] + math_lib.matmul3(
                np.transpose(x_minimax[self.stage_number]), self.Q_f, x_minimax[self.stage_number])

            # Store the result (state and cost)
            self.X_minimax[itr] = x_minimax
            self.J_minimax[itr] = v_minimax
            #print("self.J_minimax[itr][self.stage_number]:", self.J_minimax[itr][self.stage_number])
            #print("opt:", math_lib.matmul3(np.transpose(self.x_0), self.P_minimax[0], self.x_0) + 2*np.matmul(np.transpose(self.r_minimax[0]), self.x_0) + self.z_minimax[0])
            if self.J_minimax[itr][self.stage_number] < math_lib.matmul3(np.transpose(self.x_0), self.P_minimax[0], self.x_0) + 2*np.matmul(np.transpose(self.r_minimax[0]), self.x_0) + self.z_minimax[0]:
                self.reliability = self.reliability + 1/self.test_number
        return