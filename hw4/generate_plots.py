
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


def p_3():
    """
       On policy data collection
    """
    paths = glob.glob('*p3*')
    titles = ['obstacles', 'reacher', 'cheetah']

    legends = [
        draw_plot_plt(path, get_title(titles,
                                      path.split('.')[0])) for path in paths]

    plt.title('P3 Iterative training on different envs',
              fontdict={'size': 11})
    plt.legend(handles=legends)
    plt.xlabel('step  (s)')
    plt.ylabel('evaluation return')

    plt.show()


def p_2():
    """
        Action selection
    """

    paths = glob.glob('*p2*')
    eval_path = ''.join([path for path in paths if 'Eval' in path])
    train_path = ''.join([path for path in paths if 'Train' in path])

    fig, axes = plt.subplots(2)
    fig.suptitle('P2 MPC Action selection', x=0.2, y=0.97, size=11)

    legend = [draw_plot(axes[0], 'Evaluation Return', eval_path)]
    label_axis(axes[0],
               ax_info={'title': 'Evaluation', 'legend': legend})

    legend = [draw_plot(axes[1], 'Train Return', train_path)]
    label_axis(axes[1],
               ax_info={'title': 'Train', 'legend': legend})

    fig.tight_layout()

    plt.show()


def p_4():
    """
        Hyperparameters
    """
    horizon_paths = glob.glob('*horizon*')
    horizon_titles = ['horizon5', 'horizon15', 'horizon30']

    fig, axes = plt.subplots(3)
    fig.suptitle('P5 Hyperparameters', x=0.2, y=0.97, size=11)

    legends = [draw_plot(axes[0], get_title(horizon_titles, path), path)
               for path in horizon_paths]
    label_axis(axes[0],
               ax_info={'title': 'Horizon length', 'legend': legends})

    seq_paths = glob.glob('*numseq*')
    seq_titles = ['seq100', 'seq1000']
    legends = [draw_plot(axes[1], get_title(seq_titles, path), path)
               for path in seq_paths]

    label_axis(axes[1],
               ax_info={'title': 'Action sequence length',
                        'legend': legends})

    ens_paths = glob.glob('*ensemble*')
    ens_titles = ['ensemble1', 'ensemble3', 'ensemble5']
    legends = [draw_plot(axes[2], get_title(ens_titles, path), path)
               for path in ens_paths]

    label_axis(axes[2],
               ax_info={'title': 'Ensemble size', 'legend': legends})

    fig.tight_layout()

    plt.show()


def label_axis(a_x, ax_info):
    """
        Labels subplot
    """
    a_x.set_title(ax_info.get('title', 'title'), fontdict={'size': 10})
    a_x.set_ylabel(ax_info.get('ylabel', 'reward'))
    a_x.set_xlabel(ax_info.get('xlabel', 'step (s)'))
    a_x.legend(handles=ax_info.get('legend'))


def get_title(titles, path):
    """
        Extracts titles from path
    """
    for title in titles:
        if title in path:
            return title
    return "title"


def draw_plot_plt(path, title, sci_ticks=True):
    """
        Draw non-axis plots
    """
    figures = np.loadtxt(path,
                         delimiter=',',
                         skiprows=1,
                         )

    legend = plt.plot(figures[:, 1], figures[:, 2], label=title)
    if sci_ticks:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    return legend[0]


def draw_plot(axis, title, path=None, figures=None, change_ticks=False,
              sci_ticks=True):
    """
        Draws plots using loaded csv data
    """
    figures = np.loadtxt(path, delimiter=',',
                         skiprows=1) if not np.any(figures) else figures
    try:
        # 1 dim figures
        legend = axis.plot(figures[:, 1], figures[:, 2], label=title)
    except IndexError:
        legend = axis.plot(figures[1], figures[2], 'r+', label=title)

    if sci_ticks:
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    if change_ticks:
        axis.yaxis.set_major_locator(MultipleLocator(100))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axis.yaxis.set_minor_locator(MultipleLocator(20))

    axis.grid()
    return legend[0]


def handle_input():
    """
        Handles input error
    """

    choices = ['p2', 'p3', 'p4']

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
