import numpy as np
rng = np.random.default_rng()
from scipy.sparse import csc_array, csr_array

def k_cap(input, cap_size):
    if np.all(input == 0):
        return []
    else:
        return input.argsort(axis=-1)[...,-cap_size:]

def idx_to_vec(idx, shape):
    vec = np.zeros(idx.shape[:-1] + (shape,))
    np.put_along_axis(vec, idx, 1, axis=-1)
    return vec

def random_sparse_array(n, m, density):
    nnz = rng.binomial(n, density, size=m)
    row_idx = np.zeros(nnz.sum(), dtype=int)
    ind_ptr = np.zeros(m+1, dtype=int)
    ind_ptr[1:] = nnz.cumsum()
    
    for i in range(m):
        row_idx[ind_ptr[i]:ind_ptr[i+1]] = rng.choice(n, size=nnz[i], replace=False)
    
    return csc_array((np.ones(nnz.sum()), row_idx, ind_ptr), shape=(n, m))

def random_block_array(block_graph, b_rows, b_cols, density):
    n_rows = block_graph.shape[0] * b_rows
    n_cols = block_graph.shape[1] * b_cols
    col_nnz = rng.binomial(block_graph.sum(axis=0)[:, np.newaxis]*b_rows, density, size=(block_graph.shape[1], b_cols))
    
    row_idx = np.zeros(col_nnz.sum(), dtype=int)
    ind_ptr = np.zeros(n_cols + 1, dtype=int)
    ind_ptr[1:] = col_nnz.reshape(-1).cumsum()

    for j in range(block_graph.shape[1]):
        for k in range(b_cols):
            b_nnz = rng.multivariate_hypergeometric(block_graph[:, j] * b_rows, col_nnz[j, k])
            b_ptr = np.zeros(block_graph.shape[0]+1, dtype=int)
            b_ptr[1:] = b_nnz.cumsum()
            for i in range(block_graph.shape[0]):
                if block_graph[i, j]:
                    row_idx[ind_ptr[j*b_cols+k]+b_ptr[i]:ind_ptr[j*b_cols+k]+b_ptr[i+1]] = rng.choice(b_rows, size=b_nnz[i], replace=False)+i*b_rows

    return csc_array((np.ones(col_nnz.sum()), row_idx, ind_ptr), shape=(n_rows, n_cols))

