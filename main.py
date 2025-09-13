from re import VERBOSE
from itertools import cycle
import miceforest as mf
import pandas as pd
import numpy as np
import os
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ridge_regression
import cvxpy as cp
print(cp.installed_solvers())
import numpy as np
import copy


import traceback


import argparse

from make import make_dictionary_data, make_dictionary_method, make_info_axis, make_probabilities
from plot import plot_res
from run import run_multiple_experiments


parser = argparse.ArgumentParser()

parser.add_argument('--nbr', type=int, default=1, help='identifying number of the experiment')
parser.add_argument('--nbr_exp', type=int, default=1, help='number of times the experiments is run')
parser.add_argument('--dim', type=int, default=3, help='dimension space')
parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
parser.add_argument('--adv_rad_times_delta_dts_max', type=float, default=1, help='')
parser.add_argument('--adv_rad_times_delta_mis_max', type=float, default=1, help='')
parser.add_argument('--eps_adv_rad_times_delta_dts', type=float, default=1e-5, help='')
parser.add_argument('--eps_adv_rad_times_delta_mis', type=float, default=1e-5, help='')
parser.add_argument('--eps_alpha_ridge_reg', type=float, default=1e-5, help='')
parser.add_argument('--n_a_dts', type=int, default=15, help='')
parser.add_argument('--n_a_mis', type=int, default=3, help='')
parser.add_argument('--n_a_rid', type=int, default=15, help='')
parser.add_argument('--alpha_ridge_reg_max', type=float, default=100, help='')
parser.add_argument('--img_folder_path', type=str, default='results/')

args = parser.parse_args()

if __name__ == "__main__":
    number_experiment = args.nbr_exp
    #info_axis = 'train'
    #n_train = [400, 800, 1200, 1600, 2000]
    #p_seen = make_probabilities([0.8, 0.8, 0.8, 0.8, 0.8])
    #main_vec = n_train if info_axis == 'train' else p_seen
    #info_x_axis = make_info_axis(main_vec, info_axis)

    # def get_path(X, y, estimator, amax, dts_max, mis_max, S_dict, eps_amax=1e-4, eps_dts_max=1e-3, eps_mis_max=1e-3, n_alphas=100, n_deltas_dts=2, n_deltas_mis=3):
    gen = 'fixed'
    dim = args.dim
    info_axis = 'train'  # train or p_seen
    n_train = [200, 350, 500]  # check how dataset are generated, there should be some problems with 'fixed'
    lenght_vec = len(n_train)
    p_seen_both = [0.7, 1, 1]
    length_vec = len(p_seen_both)
    error_vec =  [0] * length_vec
    p_seen = [p_seen_both] * length_vec
    if info_axis == 'train':
        main_vec = n_train
        fix_vec = 'prob_seen:' + str(p_seen_both[0])
    elif info_axis == 'p_seen':
        main_vec = np.cumprod(p_seen_both)  # p_seen_both
        fix_vec = 'n_train:' + str(n_train[0])
    elif info_axis == 'error':
        main_vec = error_vec
    info_x_axis = make_info_axis(main_vec, info_axis)
    number_test = 20000

    beta_gt = np.random.randn(dim)
    print(beta_gt)
    dim = len(beta_gt)
    nbr_feature = int(np.sqrt(dim))
    mean = np.array([0] * dim)
    matr = np.random.randn(dim, dim) * 1
    cov = matr.T @ matr + np.eye(dim) * 0.5
    # np.array([[1, cov_var], [cov_var, 1]])
    data_type = 'Uniform'
    #cov = np.array([1, 1, 1])

    dictio = make_dictionary_data(
        nbr_experiments= len(main_vec), n_train = n_train, n_test=number_test,
        data = {'data': data_type, 'mean': mean, 'cov': cov},
        beta_gt = beta_gt,
        p_miss = p_seen,
        err_vector = ['Gaussian_on_X', error_vec],
        plots = [] #['points', 'l1_vs_coef', '1/alpha_vs_coef']
    )
    #dicc = dicc | {'generation':gen}
    '''
    dicc['what_to_plot'] = ['l2_dist_best_coeff_gt', 'best_score', 'best_alpha_dts']
    dicc = dicc | {'generation': gen, 'title_infer_error':' inference_error', 'title_test_error':'  test_error'}
    dicc = dicc | {'title_dts_radius': 'dts_radius', 'title_mis_radius': 'mis_radius'}
    dicc = dicc | {'info_algo': {'adv_rad_times_delta_dts_max': args.adv_rad_times_delta_dts_max, 
                                 'adv_rad_times_delta_mis_max': args.adv_rad_times_delta_mis_max, 
                                 'alpha_ridge_reg_max': args.alpha_ridge_reg_max,
                                 'eps_adv_rad_times_delta_dts': args.eps_adv_rad_times_delta_dts, 
                                 'eps_adv_rad_times_delta_mis': args.eps_adv_rad_times_delta_mis, 
                                 'eps_alpha_ridge_reg': args.eps_alpha_ridge_reg,
                                 'n_a_dts': args.n_a_dts, 
                                 'n_a_mis': args.n_a_mis, 
                                 'n_a_rid': args.n_a_dts}}
    dicc['what_to_plot'] = ['l2_dist_best_coeff_gt', 'best_score', 'best_alpha_dts', 'best_alpha_mis']
    '''
    dictionary_problem = dictio | {
        'folder_img': args.img_folder_path + str(args.nbr) + '/',
        'nbr': args.nbr,
        'what_to_plot': ['l2_dist_best_coeff_gt', 'best_score', 'best_alpha_dts'],
        'generation': 'fixed',  # generate new data when we change the parameter (mask, or cardinality training set), or keep the same data 
        'title_infer_error':' inference_error', 
        'title_test_error':'  test_error',
        'title_dts_radius': 'dts_radius', 'title_mis_radius': 'mis_radius',
        'info_algo': {'adv_rad_times_delta_dts_max': args.adv_rad_times_delta_dts_max, 
                                 'adv_rad_times_delta_mis_max': args.adv_rad_times_delta_mis_max, 
                                 'alpha_ridge_reg_max': args.alpha_ridge_reg_max,
                                 'eps_adv_rad_times_delta_dts': args.eps_adv_rad_times_delta_dts, 
                                 'eps_adv_rad_times_delta_mis': args.eps_adv_rad_times_delta_mis, 
                                 'eps_alpha_ridge_reg': args.eps_alpha_ridge_reg,
                                 'n_a_dts': args.n_a_dts, 
                                 'n_a_mis': args.n_a_mis, 
                                 'n_a_rid': args.n_a_dts},
        'what_to_plot': ['l2_dist_best_coeff_gt', 'best_score', 'best_alpha_dts', 'best_alpha_mis'],
        'name_image': 'To_Be_Defined',
    }

    for key, value in dictionary_problem.items():
        print(key, ": " , value)

    # (imp method, cov strategy, mi_nbr)
    #list_imp_cov_methods = [('BR_si', 'sd'), ('l_d', 'sd'), ('mi', 'sd', 1)]

    #list_methods_strategy = make_dictionary_method(list_imp_cov_methods)
    mi_nbr = 5
    nbr_cand = 10
    it_mc = 2  # iteartion markov chain, for multiple imputation method
    # def get_path(X, y, estimator, amax, dts_max, mis_max, S_dict, eps_amax=1e-4, eps_dts_max=1e-3, eps_mis_max=1e-3, n_alphas=100, n_deltas_dts=2, n_deltas_mis=3):

    list_methods_strategy = [#{'imp_method': 'BR_si', 'cov_strategy': 'std_nan', 'algo_superv_learn': 'adv', 'color': 'b'},  #, 'multip_dataset': 3, 'multip_missing':0},

                            #{'imp_method': 'BR_si', 'cov_strategy': 'std_nan', 'algo_superv_learn': 'ridge', 'color': 'k'},  #, 'multip_dataset': 3, 'multip_missing':0},
                            #{'imp_method': 'l_d', 'cov_strategy': 'std_nan', 'multip_dataset': 3, 'multip_missing':3},
                            #{'imp_method': 'oracle', 'cov_strategy': 'zero', 'algo_superv_learn': 'adv'},
                            #{'imp_method': 'oracle', 'cov_strategy': 'zero', 'algo_superv_learn': 'ridge'},  #, 'multip_dataset': 3, 'multip_missing': 0}

                            # {'imp_method': 'oracle', 'cov_strategy': 'sd', 'algo_superv_learn': 'adv', 'color': 'orange'},
                            {'imp_method': 'gc', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'algo_superv_learn':'adv', 'color':'b'},
                            #{'imp_method': 'oracle', 'cov_strategy': 'sd', 'algo_superv_learn': 'ridge', 'color': 'purple'},
                            #{'imp_method': 'BR_si', 'cov_strategy': 'std_nan', 'algo_superv_learn': 'ridge'},#, 'multip_dataset': 3, 'multip_missing':0},
                            #{'imp_method': 'l_d', 'cov_strategy': 'std_nan', 'multip_dataset': 3, 'multip_missing':3},
                            #{'imp_method': 'oracle', 'cov_strategy': 'sd', 'algo_superv_learn':'ridge'},#, 'multip_dataset': 3, 'multip_missing': 0},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'nbr_feature': dim, 'algo_superv_learn':'adv', 'color':'g'},  #, 'multip_dataset': 3, 'multip_missing': 3},
                            {'imp_method': 'mf_imp1', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'it_mc': it_mc, 'nbr_candidates_mm': nbr_cand, 'algo_superv_learn':'adv', 'color':'r'},  #, 'multip_dataset': 3, 'multip_missing': 3},
                            {'imp_method': 'mf_imp', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'it_mc': it_mc, 'nbr_candidates_mm': 0, 'algo_superv_learn':'adv', 'color':'purple'},  #, 'multip_dataset': 3, 'multip_missing': 3},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'zero', 'mi_nbr': mi_nbr, 'algo_superv_learn':'adv', 'color': 'g'}, #, 'multip_dataset': 3, 'multip_missing': 3},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'zero', 'cov_strategy': 'zero', 'mi_nbr': mi_nbr, 'algo_superv_learn':'adv', 'color': 'r'}
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'algo_superv_learn':'adv'}
                            #{'imp_method': 'oracle', 'cov_strategy': 'sd', 'multip_dataset': 0, 'multip_missing': 0},
                            #{'imp_method': 'mi', 'cov_strategy': 'RR', 'mi_nbr': 1},
                            #{'imp_method': 'mi', 'cov_strategy': 'RR', 'mi_nbr': 3},
                            #{'imp_method': 'mi', 'cov_strategy': 'std_mi', 'mi_nbr': mi_nbr},
                            #{'imp_method': 'mi_pure', 'cov_strategy': 'cond_var', 'cov_strategy_within': 'sd', 'mi_nbr': 5},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'zero', 'mi_nbr': mi_nbr, 'multip_betw': 1, 'multip_with': 1},
                            #{'imp_method': 'mi_mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'eye', 'mi_nbr': 5},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'zero', 'mi_nbr': mi_nbr, 'multip_betw': 0, 'multip_with': 0},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'RR', 'mi_nbr': mi_nbr, 'multip_betw': 1, 'multip_with': 0.2},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'RR', 'mi_nbr': mi_nbr, 'multip_betw': 1, 'multip_with': 0.4},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'RR', 'mi_nbr': mi_nbr, 'multip_betw': 1, 'multip_with': 0.6},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'multip_dataset': 0, 'multip_missing': 0},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'multip_dataset': 0, 'multip_missing': 1},
                            #{'imp_method': 'mi', 'post_imp':'mean', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr, 'multip_dataset': 3, 'multip_missing': 0},
                            #{'imp_method': 'mi', 'post_imp':'conc', 'cov_strategy_between': 'cond_var', 'cov_strategy': 'std_nan', 'mi_nbr': mi_nbr}#, 'multip_dataset': 3, 'multip_missing': 3}
                            #{'imp_method': 'mi', 'cov_strategy': 'RR', 'mi_nbr': 5},
                            ]
    print(list_methods_strategy)
    for el in list_methods_strategy:
        for key, value in el.items():
            print(key,": " , value)

    print("----> Starting experiments")

    '''
    nbr_exp = 2
    #res[key_tuple]['l2_dist_best_coeff_gt'].append(l2_dist)
    #res[key_tuple]['best_coeff'].append(coeff_round)
    #res[key_tuple]['best_score'].append(score_round)
    #res[key_tuple]['best_alpha'].append(alpha_round)
    res_l2 = []

    rdm_seed = 4654321
    np.random.seed(rdm_seed)
    res = run_experiments(dicc, list_methods_strategy)
    plot_res(info_x_axis, res, dicc)
    if nbr_exp > 1:
    for k in res:
        for h in res[k]:
        res[k][h] = [res[k][h]]
    for i in range(nbr_exp-1):
        print("--------------------------------------------------------------------------------------nbr_experiment external ---------------> ", i+2, "-", i+2, " ", i+2, "-", i+2, " ", i+2)
        #np.random.seed(rdm_seed * (i+2))
        res_partial = run_experiments(dicc, list_methods_strategy)
        plot_res(info_x_axis, res_partial, dicc)
        print(res)
        for k in res:
        res[k]['l2_dist_best_coeff_gt'].append(res_partial[k]['l2_dist_best_coeff_gt'])
        res[k]['best_score'].append(res_partial[k]['best_score'])
        res[k]['best_alpha'].append(res_partial[k]['best_alpha'])
        #res[k]['best_coeff'].append(res_partial[k]['best_coeff
        #res.append(res['l2_dist_best_coeff_gt'])

    print("final ")
    print(res)
    for k in res:
    print(k)
    print(np.array(res[k]['l2_dist_best_coeff_gt']))
    print(np.mean(np.array(res[k]['l2_dist_best_coeff_gt']), axis=0))
    print(np.mean(res[k]['l2_dist_best_coeff_gt'], axis=0))
    #mean_res = {k: np.mean(v, axis=0) for k, v in res.items()}
    mean_res = {k: {v: np.mean(w, axis=0) for v, w in res[k].items()} for k in res}
    for k, v in mean_res.items():
    print("k:   ", k)
    for s, t in v.items():
        print(s, ": ", t)
    #print(np.mean(res, axis=0))
    '''

    nbr_exp = args.nbr_exp
    seed = args.seed
    #dictionary_problem['name_image'] = 
    folder_path = args.img_folder_path + str(args.nbr)
    os.makedirs(folder_path, exist_ok=True)
    mean_res = run_multiple_experiments(nbr_exp, seed, dictionary_problem, list_methods_strategy, info_x_axis)
    print("PLOT OF THE MEANS")
    dictionary_problem['title_infer_error'] = 'seed: ' + str(seed) + ', nbr_exp: ' + str(nbr_exp) + ', data: '+ data_type + ', dim: ' + str(dim) # ', cov: ' + str(cov_var)
    dictionary_problem['title_test_error'] = 'sigma_err: ' + str(error_vec[0]) + ', ' + fix_vec + ', n_test: ' + str(number_test)
    #dicc['title_dts_radius'] = 'dts_radius'
    #dicc['title_mis_radius'] = 'mis_radius'
    #dicc = dicc | {'title_dts_radius': 'title_dts_radius', 'title_mis_radius': 'title_mis_radius'}
    #dicc = dicc | {'generation':gen, 'title_infer_error':'mean_infer_error, rep: ' + str(nbr_exp), 'title_mean_error':'mean_test_error'}
    #dicc['what_to_plot'] = ['l2_dist_best_coeff_gt', 'best_score', 'best_alpha_dts', 'best_alpha_mis']
    dictionary_problem['name_image'] = dictionary_problem['folder_img'] + str(dictionary_problem['nbr'])+ '_' + str(args.nbr_exp + 1)
    plot_res(info_x_axis, mean_res, dictionary_problem)

    ## you can see if you manage to take the index i that maximize alpha







