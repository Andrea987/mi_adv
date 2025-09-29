# code that solve multiple imputation with gibb sampling. The underlying models are linear, 
# and the systems are solved by coniugate gradient

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
import tensorflow_probability as tfp
from scipy.linalg import cho_solve



np.random.seed(42)
# Example NumPy matrix

def test_cholesky_tf():
    d = 3
    C = np.random.randint(1, 9, size=(2, d, d))
    np_matrix = np.swapaxes(C, -1, -2) @ C + np.eye(d)[None, :, :]
    # Convert to TensorFlow tensor
    tf_tensor = convert_to_tensor(np_matrix)
    print("NumPy array:\n", np_matrix)
    print("TensorFlow tensor:\n", tf_tensor)

    chol = np.linalg.cholesky(np_matrix)
    print(chol)

    chol_t = convert_to_tensor(chol)

    u = np.random.randint(1, 9, size=(2, 3)) + 0.0
    u_t = convert_to_tensor(u)
    mult = -0.1
    mult_t = convert_to_tensor([mult], dtype=np.float64)
    print(mult_t)

    chol_upd = tfp.math.cholesky_update(chol_t, u_t, multiplier=mult_t)
    print(chol_upd)
    print("dim chol ", chol_upd.shape)
    #print("test numpy")
    #print(np.outer(u, u))
    #C_fin = np_matrix + np.outer(u, u) * mult
    print(u[:, None, :])
    print(u[:, :, None])
    print(u[:, :, None] @ u[:, None, :])
    C_fin = np_matrix + u[:, :, None] @ u[:, None, :] * mult
    chol_fin = np.linalg.cholesky(C_fin)
    print(chol_fin)
    np.testing.assert_allclose(chol_upd, chol_fin)
    print("test cholesky passed")

def test_cholesky_solver_tf():
    print("start test_cholesky_solver_tf")
    d = 10
    b = 10 # dimension batch
    C = np.random.randint(1, 9, size=(b, d, d))
    np_matrix = np.swapaxes(C, -1, -2) @ C + np.eye(d)[None, :, :]
    # Convert to TensorFlow tensor
    tf_tensor = convert_to_tensor(np_matrix)

    #chol_t = convert_to_tensor(chol)
    #print("tf tensor \n ", tf_tensor)
    chol_t = tf.linalg.cholesky(tf_tensor)  # return L such that L @ L.T = original matrix
    #print("chol_t\n ", chol_t)
    check0 = tf.linalg.matmul(chol_t, chol_t, transpose_b=True)  # correct
    #print("check 0 ", check0)
    u = np.random.randint(1, 9, size=(b, d, 1)) + 0.0
    #print("u: \n", u)
    RHS = convert_to_tensor(u)
    #u_t = tf.keras.random.randint((b, d, 1), 1, 9, dtype='int32')
    #print(chol_t)
    res = tf.linalg.cholesky_solve(chol_t, RHS)
    #print("final res", res)
    check1 = tf.linalg.matvec(chol_t, tf.squeeze(res), transpose_a=True)
    check2 = tf.linalg.matvec(chol_t, check1, transpose_a=False)
    #print("final check \n ", check2)
    np.testing.assert_allclose(np.squeeze(u), check2)
    print("\ntest on cho solver passed")


def test_cg_tf():
    print("start test_cg_tf")
    d = 3
    b = 2 # dimension batch
    C = np.random.randint(1, 9, size=(b, d, d))
    E = np.random.randint(1, 9, size=(b, d, d))
    np_matrix = np.swapaxes(C, -1, -2) @ C + np.eye(d)[None, :, :]
    err = np.swapaxes(E, -1, -2) @ E * 1e-2
    # Convert to TensorFlow tensor
    tf_tensor = convert_to_tensor(np_matrix)
    err_tensor = convert_to_tensor(err)
    a = tf_tensor + err_tensor
    print("op a ", a)

    chol_t = tf.linalg.cholesky(tf_tensor)  # return L such that L @ L.T = original matrix
    u = np.random.randint(1, 9, size=(b, d, 1)) + 0.0
    #print("u: \n", u)
    RHS = convert_to_tensor(u)
    #u_t = tf.keras.random.randint((b, d, 1), 1, 9, dtype='int32')
    #print(chol_t)
    #res = tf.linalg.cholesky_solve(chol_t, RHS)
    def prec(x):
        return tf.linalg.cholesky_solve(chol_t, x)
    
    q = tuple(a.shape)
    print("type ", type(q))
    print(q)
    print(q[0])
    print("icajsdfjia")
    class MyLinearOperator(tf.linalg.LinearOperator):
        def __init__(self, operator, dtype=np.float64, is_square=True, name=None):
            parameters = dict(
                dtype=dtype,
                operator=operator,
                is_square=is_square,
                name=name
            )
            super().__init__(parameters=parameters, dtype=np.float64)

        #def __init__(self, mat):
        #    self.mat = mat
        #    super().__init__(dtype=mat.dtype, is_self_adjoint=True, is_positive_definite=True)
        
        @property
        def _shape(self):
            return self.mat.shape
        
        def _matvec(self, x):
            return tf.linalg.matvec(self.mat, x)
        
        def _matmul(self, x):
            return tf.linalg.matmul(self.mat, x, adjoint=False, adjoint_arg=False)

    
    op_a = MyLinearOperator(a)
    print(op_a.o)
    print(op_a.shape)
    
    
    
    res = tf.linalg.experimental.conjugate_gradient(
        op_a,
        u,
        preconditioner=None,
        x=None,
        tol=1e-05,
        max_iter=20,
        name='conjugate_gradient'
    )
    print("final res\n ", res.x)
    #print("final res", res)
    print("\ntest cg passed")
#test_cholesky_tf()
#test_cholesky_solver_tf()
test_cg_tf()

print("\n\nOther tests\n")
matrices = np.array([ [[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.0], [0.0, 5.0]] ]) 
# shape (2, 2, 2) # Wrap them as a batch of LinearOperators 
ops = tf.linalg.LinearOperatorFullMatrix(matrices) 
print(ops)
# Check shape: batch_shape is (2,), operator acts on dimension (2, 2) 
print("Batch shape:", ops.batch_shape_tensor().numpy()) 
print("Operator shape:", ops.shape_tensor().numpy()) 
# Apply to a batch of vectors (2 vectors in parallel) 
vecs = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64) # shape (2, 2) 
result = ops.matvec(vecs) 
print("Result:", result.numpy())































