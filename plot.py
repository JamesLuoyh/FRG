from seldonian.utils.io_utils import load_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    # Load loan spec file
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--log_id', help='log_id')
    args = parser.parse_args()

    log_id = args.log_id
    cs_file = f'logs/candidate_selection_log{log_id}.p'
    solution_dict = load_pickle(cs_file)
    
    fig = plot_gradient_descent(solution_dict,
        primary_objective_name='vae loss',
        save=False)
    plt.savefig(f'./SeldonianExperimentResults/{log_id}.png')