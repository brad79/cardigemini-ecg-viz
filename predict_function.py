# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:53:03 2025
import sklearn
sklearn.__version__
import sys
print(sys.version)
@author: BOX
"""

import numpy as np
try:
    import cvxpy as cp
except ImportError:
    cp = None
from sklearn.linear_model import OrthogonalMatchingPursuit
import warnings
warnings.filterwarnings("ignore")
# conda install -c conda-forge ecos
# python -m pip install ecos


def sparse_represent_kernelized(K_y, K_Tr, K_te, sp_level):
    n = max(np.shape(K_Tr))
    N = sp_level*n
    x = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(x,K_Tr,assume_PSD=True)+K_y-cp.transpose(K_te)@cp.transpose(x)-x@K_te)
    # obj = cp.Minimize(cp.quad_form(x,K_Tr)+K_y-cp.transpose(K_te)@cp.transpose(x)-x@K_te)
    constrain = [cp.norm(x,1)<=N]
    prob = cp.Problem(obj, constrain)
    try:
        prob.solve(solver = 'ECOS')
    except Exception as e:
        print(e)
    return x.value

def sparse_represent(Test, Train, sp_level):
    n, p = np.shape(Train)
    k, pt = np.shape(Test)
    if p != pt:
        print('training data and test data must have the same dimensionality')
    elif k != 1:
        print('put the testing data one by one (size: 1x60)')
    else:
        X = Train
        K_Tr = np.matmul(X,np.transpose(X))
        y = Test
        K_te = np.matmul(X,np.transpose(y))
        K_y = np.matmul(y,np.transpose(y))       
    return  sparse_represent_kernelized(K_y, K_Tr, K_te, sp_level)

def src(Traindata, Trainlabels, Testdata, sp_level):
    A = sparse_represent(Testdata, Traindata,sp_level)
    uniqlabels = np.unique(Trainlabels)
    src_scores = np.zeros((len(uniqlabels),1))
    c = max(np.shape(uniqlabels))
    for i in range(c):
        idx = np.where(Trainlabels==uniqlabels[i])[0]
        R = Testdata-np.matmul(A[idx],Traindata[idx])
        src_scores[i] = np.sqrt(np.sum(np.multiply(R,R),axis=1))
    prediction = uniqlabels[np.where(src_scores==min(src_scores))[0]]
    return prediction


def src_predict(Testdata, sp_level=0.3, **kwargs):
    d = np.load("src_trainset.npz")
    Traindata = d["Traindata"]
    Trainlabels = d["Trainlabels"]
    Testdata = np.asarray(Testdata, dtype=float).reshape(1, -1)  # (1, 60)
    pred = src(Traindata, Trainlabels, Testdata, sp_level, **kwargs)
    return pred





def src_predict_fast(
    TestFeature,
    train_npz_path="src_trainset.npz",
    k=1000,
    n_nonzero=20,
    use_abs_sim=True
):
    d = np.load(train_npz_path)
    X = d["Traindata"].astype(float)          # (n, p)
    ylab = d["Trainlabels"].reshape(-1)       # (n,)

    y = np.asarray(TestFeature, dtype=float).reshape(-1)  # (p,)
    n, p = X.shape
    if y.size != p:
        raise ValueError(f"TestFeature 維度不對：{y.size}，應為 {p}")

    # 1) top-k by similarity
    sim = X @ y
    if use_abs_sim:
        idx = np.argpartition(np.abs(sim), -k)[-k:]
    else:
        idx = np.argpartition(sim, -k)[-k:]
    idx = idx[np.argsort(np.abs(sim[idx]))[::-1]]

    Xk = X[idx, :]           # (k, p)
    lk = ylab[idx]           # (k,)

    # 2) OMP sparse coding: y ≈ D a, D = Xk.T (p, k)
    D = Xk.T
    omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=min(int(n_nonzero), D.shape[1]),
        fit_intercept=False
    )
    omp.fit(D, y)
    a = omp.coef_            # (k,)

    # 3) class-wise residual
    uniq = np.unique(lk)
    best_lab = None
    best_res = np.inf

    for lab in uniq:
        m = (lk == lab)
        if not np.any(m):
            continue
        recon = D[:, m] @ a[m]
        res = np.linalg.norm(y - recon)
        if res < best_res:
            best_res = res
            best_lab = lab

    return best_lab


def src_predict_fast_with_confidence(
    TestFeature,
    train_npz_path="src_trainset.npz",
    k=1000,
    n_nonzero=20,
    use_abs_sim=True
):
    """與 src_predict_fast 相同邏輯，額外回傳 0~1 信心度"""
    d = np.load(train_npz_path)
    X = d["Traindata"].astype(float)
    ylab = d["Trainlabels"].reshape(-1)

    y = np.asarray(TestFeature, dtype=float).reshape(-1)
    n, p = X.shape
    if y.size != p:
        raise ValueError(f"TestFeature 維度錯誤：{y.size}，應為 {p}")

    sim = X @ y
    if use_abs_sim:
        idx = np.argpartition(np.abs(sim), -k)[-k:]
    else:
        idx = np.argpartition(sim, -k)[-k:]
    idx = idx[np.argsort(np.abs(sim[idx]))[::-1]]

    Xk = X[idx, :]
    lk  = ylab[idx]
    D   = Xk.T

    omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=min(int(n_nonzero), D.shape[1]),
        fit_intercept=False
    )
    omp.fit(D, y)
    a = omp.coef_

    uniq = np.unique(lk)
    residuals = {}
    for lab in uniq:
        m = lk == lab
        if not np.any(m):
            continue
        recon = D[:, m] @ a[m]
        residuals[int(lab)] = float(np.linalg.norm(y - recon))

    best_lab = min(residuals, key=residuals.get)
    total    = sum(residuals.values())
    confidence = 1.0 - residuals[best_lab] / total if total > 0 else 0.5

    return int(best_lab), round(confidence, 4)
