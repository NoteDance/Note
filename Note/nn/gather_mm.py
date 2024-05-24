import tensorflow as tf

def gather_mm(a, b, idx_b):
    """
    Gather data according to the given indices and perform matrix multiplication.

    Parameters
    ----------
    a : tf.Tensor
        A 2-D tensor of shape (N, D1)
    b : tf.Tensor
        A 3-D tensor of shape (R, D1, D2)
    idx_b : tf.Tensor, optional
        A 1-D integer tensor of shape (N,)

    Returns
    -------
    tf.Tensor
        The output dense matrix of shape (N, D2)
    """
    # Gather the appropriate slices from b according to idx_b
    gathered_b = tf.gather(b, idx_b)
    
    # Perform the batch matrix multiplication
    result = tf.einsum('ij,ijk->ik', a, gathered_b)
    
    return result