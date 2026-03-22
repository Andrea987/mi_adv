import numpy as np
import time
from generate import generate_mask_with_bounded_flip
from tsp_imputation import impute_matrix_under_parametrized, impute_matrix_overparametrized
from tsp import gibb_sampl_no_modification
from utils import flip_matrix_manual, rk_1_update_inverse, swm_formula, matrix_switches, split_upd, s, update_covariance
from utils import make_centered_kernel_matrix, update_inverse_rk2_sym
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge 
from scipy.sparse import csr_matrix
from hyppo.ksample import Energy

















