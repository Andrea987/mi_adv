import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
import tensorflow_probability as tfp
from scipy.linalg import cho_solve

from utils import initialize

np.random.seed(42)


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
        print("num rows \n ", self._num_rows)
        print("num colm \n ", self._num_cols)
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
        print(1-self.col_mask)
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

        res = X_T @ tf.transpose(pt1, perm)
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
        print("num rows \n ", self._num_rows)
        print("num colm \n ", self._num_cols)
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
    


test_linear_opeartor_main_system()
#test_Lin_Op()
#test_cg()



def define_system():
    x = 1

def solve_systems(X, m, chol_t):
    x = 1
    main_syst = LinearOperator_Main_System()
    prec = LinearOperator_Inv_Chol(chol_t)

    


def conjugate_gradient_tf(info):
    X = info['X']
    n, d = X.shape
    m = info['nbr_multiple_imputation']
    M = info['masks']
    R = info['it_MC']
    print(M)
    list_N = [M[:, i] for i in range(d)]
    print(list_N)
    X_nan = info['X_nan']
    X_ini = initialize(info)
    info['X_ini'] = X_ini
    info['list_N'] = list_N
    K = X_ini.T @ X_ini

    tf_tensor = convert_to_tensor(K)
    chol = np.linalg.cholesky(K)
    chol_t = convert_to_tensor(chol)






    
















