"""
    Creates plots for the questions with
    data generated during training
"""
from sys import argv

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plot():
    """
        Script Input
    """
    qst = handle_input()

    os.chdir('cs285/.figures')
    qst = '_'.join([x for x in qst])
    globals()[qst]()


def q_1():
    """
        Basic DQN
    """
    paths = glob.glob('*q1*')

    fig, axs = plt.subplots(2)
    fig.suptitle('Q1 PongNoFrameskip-v4', x=0.2, y=0.97, size=11)

    mean_path = paths[paths.index('q1_mean_return.csv')]
    best_path = paths[paths.index('q1_best_return.csv')]

    draw_plot(axs[0], mean_path.split('.')[0], path=mean_path)
    draw_plot(axs[1], best_path.split('.')[0], path=best_path)
    fig.tight_layout()

    plt.show()


def q_2():
    """
        Seeded Double DQN
    """
    dqn_paths = glob.glob('*q2_dqn*')
    ddqn_paths = glob.glob('*q2_ddqn*')

    dqn_figs = average_seed(dqn_paths)
    ddqn_figs = average_seed(ddqn_paths)

    fig, axs = plt.subplots(2)
    fig.suptitle('Q2', x=0.2, y=0.97, size=11)

    draw_plot(axs[0], title='DQN', figures=dqn_figs, change_ticks=1)
    draw_plot(axs[1], title='DDQN', figures=ddqn_figs, change_ticks=1)
    fig.tight_layout()

    plt.show()


def q_3():
    """
        Hyperparameter search
    """
    paths = glob.glob('*q3*')

    legends = [
        draw_plot_plt(path, path.split('.')[0]) for path in paths]

    plt.title('Q3 plot for different Learning rates',
              fontdict={'size': 11})
    plt.legend(handles=legends)
    plt.show()


def draw_plot_plt(path, title):
    """
        Draw non-axis plots
    """
    title = title.replace('_', ' ')
    figures = np.loadtxt(path,
                         delimiter=',',
                         skiprows=1,
                         )

    legend = plt.plot(figures[:, 1], figures[:, 2], label=title)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    return legend[0]


def average_seed(paths):
    """
        Gets the mean of seeded runs
    """
    fig = np.asanyarray([np.loadtxt(path,
                                    delimiter=',',
                                    skiprows=1,
                                    max_rows=49) for path in paths])
    return fig.mean(axis=0)


def draw_plot(axis, title, path=None, figures=None, change_ticks=False):
    """
        Draws plots using loaded csv data
    """
    title = title.replace('_', ' ')
    figures = np.loadtxt(path, delimiter=',',
                         skiprows=1) if not np.any(figures) else figures
    axis.plot(figures[:, 1], figures[:, 2], label=title)
    axis.set_title(title, fontdict={'size': 10})
    axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    if change_ticks:
        axis.yaxis.set_major_locator(MultipleLocator(100))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axis.yaxis.set_minor_locator(MultipleLocator(20))

    axis.grid()


def handle_input():
    """
        Handles input error
    """

    choices = ['q1', 'q2', 'q3', 'q4', 'q5']

    if len(argv) < 2:
        err_msg = 'Missing the problem number'
        exit(err_msg)
    problem = argv[1].lower()

    if problem not in choices:
        msg = 'Invalid choice: ' + 'choices = ' + ', '.join(choices)
        exit(msg)

    return problem


if __name__ == '__main__':
    plot()
