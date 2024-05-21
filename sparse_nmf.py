import numpy as np

def sparse_nmf(X, rank, spar, tol=1e-4, seed=None, verbose=0, 
               maxiter=1000, W = None, H = None):
    """
    Learns a sparse NMF model given data X and the rank.

    Modified from: 
    Block Coordinate Descent for Sparse NMF
    http://arxiv.org/abs/1301.3527
    
    """
    from collections import deque
    if W is None and H is None:
        assert seed is not None
        W, H = init_nmf(X, rank, spar, seed)
    prev_errors = deque(maxlen=10)
    norm = np.linalg.norm(X, 'fro')
    
    for i in range(maxiter):
        W = update_W(X, W, H, spar)
        H = update_H(X, W, H)
        error = np.linalg.norm(X - np.dot(W, H), 'fro') / norm
        if verbose > 1:
            print(f'{i=}, {error=:0.10f}')
        if i >= 10 and 0 < prev_errors[-10] - error < tol:
            if verbose > 0:
                print(f'Convergence reached at iteration {i}')
            break
        prev_errors.append(error)
    return W, H


def init_nmf(X, rank, spar, seed):
    """ Initialize the matrix factors for NMF.

    Use Gaussian random numbers in [-1,1] to initialize

    Parameters
    ----------
    X: {array}, shape = [n_features, n_samples]
    rank: rank of factorization

    Returns
    -------
    W : {array}
        Feature matrix of the factorization
    H : {array}
        Weight matrix of the factorization

    where X ~ WH
    """
    np.random.seed(seed)
    m, n = np.shape(X)
    W = np.zeros((m, rank))
    k = np.sqrt(m) - spar * (np.sqrt(m) - 1)
    for i in range(rank):
        W[:, i] = sparse_opt(np.sort(np.random.rand(m))[::-1], k)

    W = np.random.rand(m, rank)
    H = np.random.rand(rank, n)
    return W, H


def update_W(X, W, H, spar):
    """Update the feature matrix based on user-defined sparsity"""
    m = np.shape(X)[0]
    m, rank = np.shape(W)
    cach = np.zeros((m, rank))
    HHt = np.dot(H, H.T)
    cach = -np.dot(X, H.T) + np.dot(W, np.dot(H, H.T))
    for i in range(rank):
        W, cach = W_sparse_ith(W, HHt, cach, spar, i)
    return W


def update_H(X, W, H):
    """Update the weight matrix using the regular multiplicative updates"""
    WtX = np.dot(W.T, X)
    WtW = np.dot(W.T, W)
    for _ in range(10):
        H = H * WtX / (np.dot(WtW, H) + np.spacing(1))
    return H


def W_sparse_ith(W, HHt, cach, spar, i):
    """ Update the columns sequentially"""
    m = np.shape(W)[0]
    C = cach[:, i] - W[:, i] * HHt[i, i]
    V = np.zeros(m)
    k = np.sqrt(m) - spar * (np.sqrt(m) - 1)
    a = sparse_opt(np.sort(-C)[::-1], k)
    ind = np.argsort(-C)[::-1]
    V[ind] = a
    cach = cach + np.outer(V - W[:, i], HHt[i, :])
    W[:, i] = V
    return W, cach


def sparse_opt(b, k):
    """ Project a vector onto a sparsity constraint

    Solves the projection problem by taking into account the
    symmetry of l1 and l2 constraints.

    Parameters
    ----------
    b : sorted vector in decreasing value
    k : Ratio of l1/l2 norms of a vector

    Returns
    -------
    z : closest vector satisfying the required sparsity constraint.

    """
    m = len(b)
    sumb = np.cumsum(b)
    normb = np.cumsum(b * b)
    pnormb = np.arange(1, m + 1) * normb
    y = (pnormb - sumb * sumb) / (np.arange(1, m + 1) - k * k)
    bot = int(np.ceil(k * k))
    z = np.zeros(m)
    #print(f'{bot=}, {m=}, {k=}')
    if bot >= m:
        raise ValueError(
            'Looks like the sparsity measure is not between 0 and 1')
    with np.errstate(invalid='ignore'):
        obj = (-np.sqrt(y) * (np.arange(1, m + 1) + k) + sumb) / np.arange(1, m + 1)
    indx = np.argmax(obj[bot:m])
    p = indx + bot - 1
    p = min(p, m - 1)
    p = max(p, bot)
    lam = np.sqrt(y[p])
    mue = -sumb[p] / (p + 1) + k / (p + 1) * lam
    z[:p + 1] = (b[:p + 1] + mue) / lam
    return z