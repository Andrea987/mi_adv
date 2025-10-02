import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
import tensorflow_probability as tfp
from scipy.linalg import cho_solve
import tensorflow as tf





import tensorflow as tf

class LinearOperatorScale(tf.linalg.LinearOperator):
    """
    A LinearOperator that implements multiplication by a diagonal matrix
    defined by a scale vector 'd'.
    A @ X = D @ X, where D is a diagonal matrix with diagonal d.
    """
    def __init__(self, scale_diag, is_non_singular=True, name="LinearOperatorScale"):
        self.scale_diag = tf.convert_to_tensor(scale_diag, name="scale_diag")
        
        # Determine the shape (N, N) and dtype for the base class
        # Scale diag has shape [B1, ..., Bk, N]
        diag_shape = tf.TensorShape(self.scale_diag.shape)
        
        # Static batch shape and event dimension (N)
        self._batch_shape = diag_shape[:-1]
        self._num_rows = diag_shape[-1]
        self._num_cols = diag_shape[-1]
        
        if self._num_rows is not None and self._num_rows != self._num_cols:
             raise ValueError("Diagonal operator must be square.")

        super().__init__(
            dtype=self.scale_diag.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=True,
            is_positive_definite=False,
            is_square=True,
            name=name
        )

# ----------------------------------------------------
    # Required Abstract Methods
# ----------------------------------------------------

    def _shape(self):
        """Returns the static shape of the conceptual matrix (B, M, N)."""
        # LinearOperator represents a conceptual matrix of shape [..., M, N].
        return self._batch_shape.concatenate([self._num_rows, self._num_rows])
    
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
        scale_diag_reshaped = self.scale_diag[..., tf.newaxis]
        
        # For a diagonal matrix, A @ X == A* @ X, so 'adjoint' flag is ignored.
        return scale_diag_reshaped * x

# ----------------------------------------------------
    # Optional/Utility Methods (Included from before)
# ----------------------------------------------------

    def _diag(self):
        """Returns the conceptual diagonal of the matrix."""
        return self.scale_diag

    def _determinant(self):
        """Returns the determinant of the matrix."""
        return tf.reduce_prod(self.scale_diag, axis=-1)

# ----------------------------------------------------
    # Usage Example
# ----------------------------------------------------

# 1. Define the scaling vector
scale = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

# 2. Instantiate the custom operator
D = LinearOperatorScale(scale)

# 3. Define an input vector (now works via D.matvec)
x_vec = tf.constant([10.0, 20.0, 30.0], dtype=tf.float32)

# 4. Define an input matrix (uses D.matmul)
X_mat = tf.constant([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=tf.float32)

# 5. Use the LinearOperator
result_matvec = D.matvec(x_vec) # Uses _matmul internally for correctness
result_matmul = D.matmul(X_mat)

print(f"Result (D @ x_vec): {result_matvec.numpy()}")
print(f"Result (D @ X_mat):\n{result_matmul.numpy()}")






























class LinearOperatorScale(tf.linalg.LinearOperator):
    def __init__(self, scale_diag, is_non_singular=True, name="LinearOperatorScale"):
        self.scale_diag = tf.convert_to_tensor(scale_diag, name="scale_diag")
        
        self
        # Determine the shape (N, N) and dtype for the base class
        # Scale diag has shape [B1, ..., Bk, N]
        diag_shape = tf.TensorShape(self.scale_diag.shape)
        
        # Static batch shape and event dimension (N)
        self._batch_shape = diag_shape[:-1]
        self._num_rows = diag_shape[-1]
        self._num_cols = diag_shape[-1]
        
        if self._num_rows is not None and self._num_rows != self._num_cols:
             raise ValueError("Diagonal operator must be square.")

        super().__init__(
            dtype=self.scale_diag.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=True,
            is_positive_definite=False,
            is_square=True,
            name=name
        )

# ----------------------------------------------------
    # Required Abstract Methods
# ----------------------------------------------------

    def _shape(self):
        """Returns the static shape of the conceptual matrix (B, M, N)."""
        # LinearOperator represents a conceptual matrix of shape [..., M, N].
        return self._batch_shape.concatenate([self._num_rows, self._num_rows])
    
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
        scale_diag_reshaped = self.scale_diag[..., tf.newaxis]
        
        # For a diagonal matrix, A @ X == A* @ X, so 'adjoint' flag is ignored.
        return scale_diag_reshaped * x

# ----------------------------------------------------
    # Optional/Utility Methods (Included from before)
# ----------------------------------------------------

    def _diag(self):
        """Returns the conceptual diagonal of the matrix."""
        return self.scale_diag

    def _determinant(self):
        """Returns the determinant of the matrix."""
        return tf.reduce_prod(self.scale_diag, axis=-1)














