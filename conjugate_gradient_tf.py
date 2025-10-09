import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
import tensorflow_probability as tfp
from scipy.linalg import cho_solve
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from utils import initialize
import time



class LinearOperator_Main_System(tf.linalg.LinearOperator):
    """
    Define the main system by computing X.T @ N @ X@v,
    where X is the full matrix, N is a column of the 
    binary matrix that contains the masks
    """
    def __init__(self, X, col_mask, is_non_singular=True, name="LinearOperatorScale"):
        self.shapee = list(X.shape)
        self.shapee[-2] = X.shape[-1]
        self.shapee = tuple(self.shapee)
        self.X_part = tf.convert_to_tensor(X, name="X_part")
        self.col_mask = tf.convert_to_tensor(col_mask, name="mask_part", dtype=self.X_part.dtype)
        # Determine the shape (N, N) and dtype for the base class
        # Scale diag has shape [B1, ..., Bk, N]
        #self.shape = tf.TensorShape(self.chol.shape)
        #print("shape tensor", shape)
        # Static batch shape and event dimension (N)
        self._batch_shape = X.shape[:-2]
        self._num_rows = X.shape[-1]
        self._num_cols = X.shape[-1]
        #print("num rows \n ", self._num_rows)
        #print("num colm \n ", self._num_cols)
        if self._num_rows is not None and self._num_rows != self._num_cols:
             raise ValueError("Should be a square matrix.")

        super().__init__(
            dtype=self.X_part.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=True,
            is_positive_definite=True,
            is_square=True,
            name=name
        )

# ----------------------------------------------------
    # Required Abstract Methods
# ----------------------------------------------------

    def _shape(self):
        """Returns the static shape of the conceptual matrix (B, M, N)."""
        # LinearOperator represents a conceptual matrix of shape [..., M, N].
        #sh = self.X_part.shape.as_list
        #print(sh)
        #sh[-2] = sh[-1] 
        return tf.TensorShape(self.shapee)  #self._batch_shape.concatenate([self._num_row, self._num_cols])
    
    # NOTE: The base class requires _matmul for the general case of A @ X.
    # We explicitly implement _matmul to satisfy the abstract base class and 
    # handle both matvec (X is vector) and matmul (X is matrix) efficiently.

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        """Implements the matrix-matrix product: A @ X or A* @ X."""
        # A diagonal matrix D @ X is element-wise multiplication on the first matrix dimension.
        # D has shape [..., N, N] (conceptually)
        # scale_diag has shape [..., N]
        # X has shape [..., N, K] where K=1 for matvec.
        
        # We need to broadcast scale_diag from [..., N] to [..., N, 1] for matmul
        #scale_diag_reshaped = self.scale_diag[..., tf.newaxis]
        
        # For a diagonal matrix, A @ X == A* @ X, so 'adjoint' flag is ignored.
        #if tf.size(x) < tf.size(self.chol):
        #    print("x is a single vector, increase the size adding a 1 at the end")
        #    x = tf.expand_dims(x, axis=-1)
        #res = tf.linalg.cholesky_solve(self.chol, x, name=None)
        #print(1-self.col_mask)
        if tf.rank(self.X_part) == 2:
            perm = [1, 0]
        else:
            perm = [0, 2, 1]
        
        X_T = tf.transpose(self.X_part, perm)
        #print("dad", self.col_mask)
        #print("XT ", X_T)
        #print("self X ", self.X_part)
        #print("x ", x)
        #print("col mask ", self.col_mask)
        X_x = tf.transpose(self.X_part @ x, perm) 
        #print("X_x ", X_x)
        pt1 = (1-self.col_mask) * X_x
        #print("pt1 ", pt1)

        res = X_T @ tf.transpose(pt1, perm) + x * 1e-5  # small regulirizer
        return res


def test_linear_opeartor_main_system():
    d = 3
    n = 4
    C = np.random.randint(1, 9, size=(n, d)) + 0.0
    m = np.random.binomial(1, 0.3, d)  + 0.0
    m = np.array([0, 0, 1, 1]) 
    np_matrix = np.swapaxes(C, -1, -2) @ C + np.eye(d)[None, :, :]
    # Convert to TensorFlow tensor
    #x = np.random.randint(2, d, (2, 3, 4))
    # print(x.shape[:-1])
    chol = np.linalg.cholesky(np_matrix)
    L = LinearOperator_Main_System(C, m)
    xx1 = np.random.rand(d, 1)
    #xx2 = tf.convert_to_tensor(xx1)
    print("shape L ", L.shape)
    v = L.matmul(xx1)
    print(v)
    res1 = C.T @ (((1-m) * C.T).T) @ xx1
    print(res1)
    print("end test linear operator main system")
    #print(L.shape)



class LinearOperator_Inv_Chol(tf.linalg.LinearOperator):
    """
    A LinearOperator that implements multiplication by the inverse of a matrix A,
    decomposed by it's cholesky decomposition
    """
    def __init__(self, chol, is_non_singular=True, name="LinearOperatorScale"):
        self.chol = tf.convert_to_tensor(chol, name="chol")
        
        # Determine the shape (N, N) and dtype for the base class
        # Scale diag has shape [B1, ..., Bk, N]
        #self.shape = tf.TensorShape(self.chol.shape)
        #print("shape tensor", shape)
        # Static batch shape and event dimension (N)
        self._batch_shape = chol.shape[:-1]
        self._num_rows = chol.shape[-2]
        self._num_cols = chol.shape[-1]
        #print("num rows \n ", self._num_rows)
        #print("num colm \n ", self._num_cols)
        if self._num_rows is not None and self._num_rows != self._num_cols:
             raise ValueError("Should be a square matrix.")

        super().__init__(
            dtype=self.chol.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=True,
            is_positive_definite=True,
            is_square=True,
            name=name
        )

# ----------------------------------------------------
    # Required Abstract Methods
# ----------------------------------------------------

    def _shape(self):
        """Returns the static shape of the conceptual matrix (B, M, N)."""
        # LinearOperator represents a conceptual matrix of shape [..., M, N].
        return tf.TensorShape(self.chol.shape)
    
    # NOTE: The base class requires _matmul for the general case of A @ X.
    # We explicitly implement _matmul to satisfy the abstract base class and 
    # handle both matvec (X is vector) and matmul (X is matrix) efficiently.

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        """Implements the matrix-matrix product: A @ X or A* @ X."""
        # A diagonal matrix D @ X is element-wise multiplication on the first matrix dimension.
        # D has shape [..., N, N] (conceptually)
        # scale_diag has shape [..., N]
        # X has shape [..., N, K] where K=1 for matvec.
        
        # We need to broadcast scale_diag from [..., N] to [..., N, 1] for matmul
        #scale_diag_reshaped = self.scale_diag[..., tf.newaxis]
        
        # For a diagonal matrix, A @ X == A* @ X, so 'adjoint' flag is ignored.
        #if tf.size(x) < tf.size(self.chol):
        #    print("x is a single vector, increase the size adding a 1 at the end")
        #    x = tf.expand_dims(x, axis=-1)
        res = tf.linalg.cholesky_solve(self.chol, x, name=None)
        return res



def test_Lin_Op():
    d = 3
    C = np.random.randint(1, 9, size=(2, d, d))
    np_matrix = np.swapaxes(C, -1, -2) @ C + np.eye(d)[None, :, :]
    # Convert to TensorFlow tensor
    #x = np.random.randint(2, d, (2, 3, 4))
    # print(x.shape[:-1])
    chol = np.linalg.cholesky(np_matrix)
    L = LinearOperator_Inv_Chol(chol)
    xx1 = np.random.rand(2, d, 2)
    xx2 = tf.convert_to_tensor(xx1)
    v = L.matmul(xx2)
    print(L.shape)
    #print(xx1[:, :, 0][:, :, None])
    print( np.linalg.inv(np_matrix) @ xx1)
    print( np.linalg.inv(np_matrix) @ xx1[:, :, 0][:, :, None]) 
    print(v)


def test_cg():
    d = 3
    m = 2
    C = np.random.randint(1, 9, size=(d, d))
    np_matrix = np.swapaxes(C, -1, -2) @ C + np.eye(d)[None, :, :]
    # Convert to TensorFlow tensor
    #x = np.random.randint(2, d, (2, 3, 4))
    # print(x.shape[:-1])
    chol = np.linalg.cholesky(np_matrix)
    L = LinearOperator_Inv_Chol(chol)
    xx1 = np.random.rand(2, 2, d)
    xx1 = np.random.randint(1, 4, (2, d)) + 0.0
    xx2 = tf.convert_to_tensor(xx1)


    B = np_matrix + np.eye(d) * 1e-1
    print("B\n", B)
    print("xx2\n ", xx2)
    B_Linear_Op = tf.linalg.LinearOperatorFullMatrix(B, is_self_adjoint=True, is_positive_definite=True)
    res = tf.linalg.experimental.conjugate_gradient(B_Linear_Op, xx2, preconditioner=L)
    
    
    print(res.x)
    res_np = res.x.numpy()
    print(res_np)
    print(res_np @ B)
    

def spectral_dec_matrix(U):
    # given U = [v|w], and R = u@v.T + v@u.T, find 
    # a and b vectors such that R = a@a.T - b.T. To do that observe that 
    # we need the spectral decomposition of (0, 1 | 1 0)
    S = np.array([[1, 1], [-1, 1]]) * (1 / np.sqrt(2))
    return U @ S  # = [b|a]


#test_linear_opeartor_main_system()
#test_Lin_Op()
#test_cg()



def solve_system(X_ini, n_j, chol_t, crr_j):
    j = crr_j + 0
    vj = X_ini[:, j]
    X_del_j = np.delete(X_ini, j, axis=1)
    #print("X_del_j \n", X_del_j)
    #km = X_del_j.T @ X_del_j
    #print("kernel matrix \n", km)
    #K_maskj = Lin_op_mask(X_del_j, nj) # first system to solve
    bj = convert_to_tensor(X_del_j.T @ (n_j * vj))
    #solj = cg(K_maskj, bj, x0=warm_start_j, rtol=1e-05, atol=1e-05, M=K_j_inv)[0]
    main_syst = LinearOperator_Main_System(X_del_j, n_j)
    prec = LinearOperator_Inv_Chol(chol_t)
    #print("b j ", bj)
    #print("chol t ", chol_t)
    #print("shape main system ", main_syst.shape)
    solj = tf.linalg.experimental.conjugate_gradient(operator=main_syst, rhs=bj, preconditioner=prec)
    warm_start_j = solj
    return solj
    

def update_column(x_i, m_i, X_i_del, theta):
    # x_i, (, n), i-th column of the original mask
    # m_i, (, n), mask associated to x_i
    first = x_i * (1 - m_i)
    # third = m_i * X_i_del.T
    #print(m_i)
    #print(X_i_del)
    #print(theta)
    #print(m_i * X_i_del.T)
    second = (m_i * X_i_del.T).T @ theta.numpy()  # np.random.randn(len(theta)) * 0.1)  # sampling part
    second = second  # + m_i * np.random.randn(len(second)) * np.std(second)
    return first + second
    

def conjugate_gradient_tf(info):
    X = info['X']
    n, d = X.shape
    m = info['nbr_multiple_imputation']
    #S = np.zeros((n, m))  # sampling
    S = np.zeros(n)
    M = info['masks']
    R = info['it_MC']
    upd_j = np.zeros((d-1, 2))
    print(M)
    list_N = [M[:, i] for i in range(d)]
    print(list_N)
    X_nan = info['X_nan']
    #X_init = np.array([initialize(info)])  # should be (1, n, d)
    #X_ini = np.tile(X_init, (m, 1, 1))  # should be (m , n, d)
    X_ini = initialize(info)
    #print(X_ini)
    info['X_ini'] = X_ini
    #print("X_ini, the final result\n ", X_ini)
    info['list_N'] = list_N
    X_ini_del = np.delete(X_ini, 0, axis=1)
    #K_j = X_ini_del.T @ X_ini_del  # + np.eye(d-1) * 1e-8 # (d, d)
    K = X_ini_del.T @ X_ini_del
    #print("print K \n", K)
    tf_tensor = convert_to_tensor(K)
    chol = np.linalg.cholesky(K)
    chol_t = convert_to_tensor(chol)
    print("starting program \n")
    for i in range(R):
        for crr_j in range(d):
            #print("\ncolum nbr: ", crr_j, "\n")
            j = crr_j
            n_j = list_N[j]
            sol_j = solve_system(X_ini, n_j, chol_t, j).x
            vj = X_ini[:, j]
            X_del_j = np.delete(X_ini, j, axis=1)
            #print("X_del_j \n", X_del_j)
            #Q = X_del_j.T @ X_del_j
            #print("kernel matrix actual, hand made\n", Q)
            #print("v_j ", vj)
            #print("update columns ", update_column(vj, n_j, X_del_j, sol_j))
            S = update_column(vj, n_j, X_del_j, sol_j)
            S = S + n_j * np.random.randn(n) * np.std(S) * 1e-6 # random perturbation, must be changed in the future, that's just for checking that everything is ok
            X_ini[:, j] = S
            vj_upd = S
            if crr_j == d-1:
                j=0
            upd_j[j, 0] = 1
            pt_j = (vj_upd - X_del_j[:, j]) @ X_del_j  # partial_j
            pt_j[j] = np.sum((vj_upd - X_del_j[:, j]) * (vj_upd + X_del_j[:, j])) / 2
            upd_j[:, 1] = pt_j
            C = np.array([[0, 1], [1, 0]])
            Idm = np.array([[-1, 0], [0, 1]])
            #print("upd j \n", upd_j)

            #km_check2 = K + upd_j @ C @ upd_j.T
            #print("kernel matrix km check2 \n", km_check2)
            #CC = np.linalg.cholesky(km_check2)
            #print("CC cholesky \n", CC)
            upd_j_rotate = spectral_dec_matrix(upd_j)
            #km_check3 = K + upd_j_rotate @ Idm @ upd_j_rotate.T
            #print("kernel matrix km check 3 \n", km_check3)
            #K = km_check3
            #print("upd j \n", upd_j)
            #print("upd j rotate \n", upd_j_rotate)
            u_t = convert_to_tensor(upd_j_rotate)
            #print("chol km check 3 \n", np.linalg.cholesky(km_check3))
            #print("u_t \n", u_t)
            #print("chol t before updt ", chol_t)
            chol_t = tfp.math.cholesky_update(chol_t, u_t[:, 1], multiplier = 1 )
            #print("first upd \n", chol_t)
            chol_t = tfp.math.cholesky_update(chol_t, u_t[:, 0], multiplier = -1)
            #print("second chol updt \n", chol_t)
            upd_j[j, 0] = 0
            #print("X_ini, the final result (after update)\n ", X_ini)




np.random.seed(42)
n, d = 600, 20
m = np.random.binomial(1, 0.2, size=(n, d))
X = np.random.rand(n, d)
#X = np.random.randint(0, 5, (n, d)) + 0.0
X_nan = X.copy()
X_nan[m==1] = np.nan
print('wewew ', X_nan)

infoo = {'masks': m, 
         'X': X,
         'initialize': 'mean',
         'X_nan': X_nan,
         'nbr_multiple_imputation': 1,
         'it_MC': 1
        }
print(infoo)
start1 = time.time()   # tic
res = conjugate_gradient_tf(infoo)
end1 = time.time()     # toc
print(f"Elapsed time no   prec: {end1 - start1:.4f} seconds")

ice = IterativeImputer(estimator=BayesianRidge(), max_iter=d, initial_strategy='mean')
start2 = time.time()   # tic
res1 = ice.fit_transform(X_nan)
end2 = time.time()     # toc
print(f"Elapsed time no   prec: {end2 - start2:.4f} seconds")
#print("res \n", res1)





























    
















