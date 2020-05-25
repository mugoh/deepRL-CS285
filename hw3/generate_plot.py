"""
    Creates plots for the questions with
    data generated during training
"""
from sys import argv

import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def plot():
    """
        Script Input
    """
    qst = handle_input()

    os.chdir('cs285/.figures')
    globals()[qst]()


def q1():
    """
        Basic DQN
    """
    paths = glob.glob('*q1*')

    fig, axs = plt.subplots(2)
    fig.suptitle('Q1 PongNoFrameskip-v4', x=0.2, y=0.97, size=11)

    mean_path = paths[paths.index('q1_mean_return.csv')]
    best_path = paths[paths.index('q1_best_return.csv')]

    draw_plot(axs[0], mean_path, mean_path.split('.')[0])
    draw_plot(axs[1], best_path, best_path.split('.')[0])
    fig.tight_layout()

    plt.show()


def draw_plot(axis, path, title):
    """
        Draws plots using loaded csv data
    """
    title = title.replace('_', ' ')
    figures = np.loadtxt(path, delimiter=',', skiprows=1)
    axis.plot(figures[:, 1], figures[:, 2], label=title)
    axis.set_title(title, fontdict={'size': 10})
    axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))


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
