import tensorflow as tf

def gather_mm(a, b, idx_b):
    """
    Gather data according to the given indices and perform matrix multiplication.

    Parameters
    ----------
    a : tf.Tensor
        A 3-D tensor of shape (N, M, D1) or a 2-D tensor of shape (N, D1)
    b : tf.Tensor
        A 3-D tensor of shape (R, D1, D2)
    idx_b : tf.Tensor, optional
        A 1-D integer tensor of shape (N,)

    Returns
    -------
    tf.Tensor
        The output dense matrix of shape (N, M, D2) if a is 3-D, or (N, D2) if a is 2-D
    """
    # Gather the appropriate slices from b according to idx_b
    gathered_b = tf.gather(b, idx_b)
    
    # If a is 2-D, expand its dimensions to 3-D for consistent batch matrix multiplication
    if len(a.shape) == 2:
        a = tf.expand_dims(a, axis=1)  # Shape becomes (N, 1, D1)
        expanded = True
    else:
        expanded = False

    # Perform the batch matrix multiplication
    result = tf.einsum('nij,njk->nik', a, gathered_b)
    
    # If a was originally 2-D, squeeze the extra dimension
    if expanded:
        result = tf.squeeze(result, axis=1)  # Shape becomes (N, D2)
    
    return result
