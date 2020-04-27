"""
    Plots the outputs from figures of the runs
"""
import matplotlib.pyplot as plt
import numpy as np

from sys import argv
from os import path


def plot():
    if len(argv) < 2:
        print(argv)
        exit("Missing the numberof problem to generate the plot for")
    problem = argv[1].lower()
    smooth = argv[1] if len(argv) > 2 else None

    if problem == '3':
        paths = [
            '.figures_csv/run_pg_sb_rtg_na_CartPole-v0_25-04-2020_11-45-12-tag-Eval_AverageReturn.csv',
            '.figures_csv/run_pg_sb_no_rtg_dsa_CartPole-v0_25-04-2020_11-40-20-tag-Eval_AverageReturn.csv',
            '.figures_csv/run_pg_sb_rtg_dsa_CartPole-v0_25-04-2020_11-42-37-tag-Eval_AverageReturn.csv',
            'sb',
            '.figures_csv/run_pg_lb_rtg_na_CartPole-v0_25-04-2020_13-22-41-tag-Eval_AverageReturn.csv',
            '.figures_csv/run_pg_lb_no_rtg_dsa_CartPole-v0_25-04-2020_11-47-10-tag-Eval_AverageReturn.csv',
            '.figures_csv/run_pg_lb_rtg_dsa_CartPole-v0_25-04-2020_11-53-57-tag-Eval_AverageReturn.csv',
            'lb'
        ]
        sb_path = 'plot_3_{size_}.png'
        for path in paths:
            if len(path) < 3:
                plt.legend()
                plt.savefig(sb_path.format(size_=path))
                plt.title(sb_path.format(size_=path))
                plt.close()
                print(f'Saved plot in current dir as plot_3_{path}.png')
            else:
                figures = np.loadtxt(path, delimiter=',', skiprows=1)
                x = figures[:, 1]
                y = figures[:, 2]
                x_ = x
                y_ = y
                if smooth:

                    from scipy.interpolate import make_interp_spline
                    x_ = np.linspace(x.min(), x.max(), 300)
                    Bs_pline = make_interp_spline(x, y)
                    y_ = Bs_pline(x_)
                plt.plot(x_, y_, label=path[20:30])

    elif problem == '4':
        path = '.figures_csv/run_pg_ip_b5000_r2e-2_InvertedPendulum-v2_25-04-2020_17-18-32-tag-Eval_AverageReturn.csv'
        title_ = 'problem 4'
        create_plt(path, title_)
    else:
        print(f'Unable to find the problem number: {problem}')


def create_plt(path, title_):
    figures = np.loadtxt(path, delimiter=',', skiprows=1)
    plt.plot(figures[:, 1], figures[:, 2], label=path[20:30])
    plt.title(title_)
    plt.legend()
    clean_p = title_.replace(' ', '_')
    plt.savefig(f'plot_{clean_p}.png')
    plt.close()
    print(f'Saved plit in current dir as plot_{clean_p}.png')


if __name__ == '__main__':
    plot()
