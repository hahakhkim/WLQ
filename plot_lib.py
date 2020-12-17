import matplotlib.pyplot as plt
import numpy as np

def plot_single_data(X_data, state_index, figure_number, title):
    test_number = len(X_data)
    stage_number = len(X_data[0])
    # Rearrange data
    data = [[0 for itr in range(test_number)] for t in range(stage_number)]
    for t in range(stage_number):
        for itr in range(test_number):
            data[t][itr] = X_data[itr][t][state_index][0]

    # Box plot
    fig = plt.figure(figure_number)
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    b = ax.boxplot(data, showcaps=False, showfliers=False, widths=0.001, patch_artist=True)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(b[element], color='cyan')
    plt.setp(b['boxes'], color='blue')
    plt.setp(b['medians'], color='black')
    ax.set_xticks([20 * i for i in range(int(stage_number / 20) + 1)])
    ax.set_xticklabels([20 * i for i in range(int(stage_number / 20) + 1)])
    ax.set_xlabel('stage')
    ax.set_ylabel('value')
    fig = plt.gcf()
    fig.savefig('./results/figure' + str(figure_number) + '.png', dpi=300)
    plt.show()
    return

def plot_both_data(X_data1, X_data2, state_index, figure_number):
    # Rearrange data
    test_number = len(X_data1)
    stage_number = len(X_data1[0])
    data = np.zeros((2, test_number, stage_number))
    for t in range(stage_number):
        for itr in range(test_number):
            data[0, itr, t] = X_data1[itr, t, state_index, 0]
            data[1, itr, t] = X_data2[itr, t, state_index, 0]

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
        ax[p].set_xticks([20 * i for i in range(int(stage_number / 20) + 1)])
        ax[p].set_xticklabels([20 * i for i in range(int(stage_number / 20) + 1)])
        ax[p].set_xlabel('stage')
        if p == 0:
            plt.title("Standard LQG")
            ax[p].set_ylabel('value')
        else:
            plt.title("Minimax LQR")
    fig.savefig('./results/figure' + str(figure_number) + '.png', dpi=300)
    plt.show()
    return

def plot_median(X_data1, X_data2, state_index, figure_number):
    # Rearrange data
    stage_number = len(X_data1[0])
    data = np.zeros((2, stage_number))
    for t in range(stage_number):
            data[0, t] = np.median(X_data1[:, t, state_index, 0])
            data[1, t] = np.median(X_data2[:, t, state_index, 0])
    error_bound = 0.03
    # Box plot
    fig = plt.figure(figure_number)
    ax = []
    plt.subplots_adjust(hspace=0.5)
    plt.rc('font', size=10)
    for p in range(2):
        ax.append(fig.add_subplot(1, 2, p + 1))
        b = ax[p].plot(data[p])
        avg = np.mean(data[p, stage_number-150:stage_number-50], axis=0)
        ax[p].plot([avg+error_bound for i in range(stage_number)], color='blue')
        ax[p].plot([avg-error_bound for i in range(stage_number)], color='blue')
        ax[p].set_xlabel('time(s)')
        if p == 0:
            plt.title("Standard LQG")
            ax[p].set_ylabel('value')
        else:
            plt.title("Minimax LQR")
    plt.show()
    return


def plot_all_data(theta_list1, optimal_penalty_result, control_energy_result, theta_list2, reliability_result, figure_number):
    ax = []
    plt.subplots_adjust(hspace=0.5)
    fig = plt.figure(figure_number)

    ax.append(fig.add_subplot(1, 3, 1))
    plt.plot(theta_list1, optimal_penalty_result, color='blue', marker='o', markerfacecolor='cyan', linestyle='solid', linewidth=1,
             markersize=5)

    plt.xlabel('θ', labelpad=-0.5)
    plt.ylabel('λ', labelpad=-0.5)
    plt.xscale('log')
    plt.grid(linestyle=':', linewidth=1.0, alpha=0.8)

    ax.append(fig.add_subplot(1, 3, 2))
    plt.plot(theta_list1, control_energy_result, color='blue', marker='o', markerfacecolor='cyan', linestyle='solid', linewidth=1,
             markersize=5)
    plt.xlabel('θ', labelpad=-0.5)
    plt.ylabel('Energy')
    plt.xscale('log')
    plt.grid(linestyle=':', linewidth=1.0, alpha=0.8)

    ax.append(fig.add_subplot(1, 3, 3))
    plt.plot(theta_list2, reliability_result, color='blue', marker='o', markerfacecolor='cyan', linestyle='solid', linewidth=1,
             markersize=5)
    plt.xlabel('θ', labelpad=-0.5)
    plt.ylabel('Reliability')
    plt.xscale('log')
    plt.grid(linestyle=':', linewidth=1.0, alpha=0.8)
    plt.show()

def plot_compare_data(data1, data2, data3, data4, figure_number):
    data_num = len(data1)
    stage_num = len(data1[0])
    stage_num = 100
    dim = len(data1[0][0])
    data = np.zeros((4, dim, data_num, stage_num))
    for r in range(dim):
        for t in range(stage_num):
            for n in range(data_num):
                data[0, r, n, t] = data1[n, t, r, 0]
                data[1, r, n, t] = data2[n, t, r, 0]
                data[2, r, n, t] = data3[n, t, r, 0]
                data[3, r, n, t] = data4[n, t, r, 0]

    fig = plt.figure(figure_number)
    ax = []
    plt.subplots_adjust(hspace=0.5)
    plt.rc('font', size=10)
    for p in range(4):
        ax.append(fig.add_subplot(2, 2, p + 1))
        b = ax[p].boxplot(data[p][19], showcaps=False, showfliers=False, widths=0.001, patch_artist=True)
        for element in ['whiskers', 'fliers', 'means', 'caps']:
            plt.setp(b[element], color='cyan')
        plt.setp(b['boxes'], color='blue')
        plt.setp(b['medians'], color='black')
        plt.ylim(-0.4, 1.2)
        plt.grid(linestyle=':', linewidth=1.0, alpha=0.8)

        ax[p].set_xticks([10 * i for i in range(int(stage_num / 10) + 1)])
        ax[p].set_xticklabels([i for i in range(int(stage_num / 10) + 1)])
        ax[p].set_yticks([-0.4, 0.0, 0.4, 0.8, 1.2])

        ax[p].set_xlabel('time(s)')
        if p == 0 or p == 2:
            ax[p].set_ylabel('frequency(rad/s)')
    fig.savefig('./results/finite_horizon/figure' + str(figure_number) + '.png', dpi=300)
    plt.show()