import tensorly as tl
import tensorly.tenalg as tg
from tensorly import random
from functools import reduce
import numpy as np
from numpy import linalg as lg
import argparse
import csv
import os
import multiprocessing as mp
from scipy.io import savemat
import pickle
import pandas as pd
# Basic Functions and Measurements
# outer product of multiple vectors.
def cp_outer(vs):
    return reduce(np.multiply.outer, vs)

# extract an array containing (r+1)th columns for all matrices.
def cp_column(matrix_list, r):
    return [matrix[:,r] for matrix in matrix_list]

# convert factor matrices to estimated tensor.
def cp_est(matrix_list, boolvec=-1):
    est = np.zeros([matrix.shape[0] for matrix in matrix_list])
    if isinstance(boolvec, int):
        for r in range(matrix_list[1].shape[1]):
            est = est + cp_outer(cp_column(matrix_list,r))
    else:
        for r in range(matrix_list[1].shape[1]):
            if boolvec[r]==True:
                est = est + cp_outer(cp_column(matrix_list,r))
    return est

# extract result for a list of cp_linked_s objective.
def extract(result, measure):
    r=[]
    if isinstance(result[0][measure], list):
        for k in range(len(result)):
            r.append(result[k][measure][-1])
    else:
        for k in range(len(result)):
            r.append(result[k][measure])
    return np.array(r)

def singular_value(A):
    sig_val = np.prod([lg.norm(A[i], axis=0) for i in range(len(A))], axis=0)
    return sig_val.tolist()

# objective function for single tensor with l2 penalty
def obj_single(tensor, A, sigma):
    if isinstance(sigma, np.ndarray): # assuming normality of factor matrix entries
        res = lg.norm(tensor - cp_est(A))**2
        penalty_vec = np.array([sum([lg.norm(Ak[:,r])**2 for Ak in A])*sig for r,sig in zip(range(A[0].shape[1]), sigma)])
        penalty_vec[np.isnan(penalty_vec)]=0
        return res+sum(penalty_vec)
    else:
        res = lg.norm(tensor - cp_est(A))**2
        penalty = sigma*sum([lg.norm(A[k])**2 for k in range(len(A))])
        return res+penalty
    
# log-likelihood for single tensor 
def ll_single(tensor, A, gamma, sigma):  
    if isinstance(sigma, np.ndarray):
        obj = -obj_single(tensor, A, sigma)/(2*gamma**2)
        sigma_factor = np.sqrt(gamma**2/sigma)
        sigma_factor[sigma_factor==0] = 1 #log(1)=0, avoid inf value
        sigma_relate = -np.prod(tensor.shape)*np.log(gamma)-sum(tensor.shape)*sum(np.log(sigma_factor)) 
        return obj + sigma_relate
    else:
        obj = -obj_single(tensor, A, sigma)/(2*gamma**2)
        sigma_relate = -np.prod(tensor.shape)*np.log(gamma)-sum(tensor.shape)*A[0].shape[1]*np.log(np.sqrt(gamma**2/sigma))
        return obj + sigma_relate

def diff_str_single(A_new, A_cur):
    ss=0
    for k in range(len(A_new)):
        ss+=lg.norm(A_new[k]- A_cur[k])/lg.norm(A_new[k])
    return ss

def line_search_single(tensor, A_cur, A_pre, RLS):
    loss_cur = lg.norm(tensor - cp_est(A_cur)) 
    #loss_cur = diff_str_single(A_cur, A_pre)
    A_new=[]
    for k in range(len(A_pre)):
        A_new.append(A_pre[k] + RLS*(A_cur[k] - A_pre[k])) 

    loss_new = lg.norm(tensor - cp_est(A_new)) 
    if loss_new < loss_cur:
        return A_new, 1
    else:
        return A_cur, 0 

def ind_split(tensor1, nfolds, seed, missType):
    if missType == 'random':
        flat_tensor1 = tensor1.ravel()
        
        # Find indices of the non-NaN elements for each tensor
        non_nan_indices1 = np.arange(flat_tensor1.size)[~np.isnan(flat_tensor1)]
        
        # Shuffle the indices based on the tensor with fewer non-NaN elements
        np.random.seed(seed)
        shuffled_indices1 = np.random.permutation(non_nan_indices1)
        cvIndList1 = np.array_split(shuffled_indices1, nfolds)

        missIndList = []
        for fold_indices1 in cvIndList1:
            mask1 = np.zeros_like(flat_tensor1, dtype=bool)    
            mask1[fold_indices1] = True
            
            missIndList.append(mask1.reshape(tensor1.shape))
    

    elif missType == 'tensor-wise':
        tensor = np.zeros(size)
        flat_tensor = tensor.ravel()
        indices = np.arange(flat_tensor.size)
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = indices.reshape(nfolds, -1)

        missIndList = []
        indList = np.arange(size[0])
        np.random.shuffle(indList)
        indFolds = np.split(indList, nfolds)
        
        for ind in indFolds:
            tens = np.zeros(size)
            tens[ind,:] = 1
            missInd = tens == 1
            missIndList.append(missInd)

    return missIndList

# single tensor simulation 
def simu_single(array, R, noise=True, SNR = 1, seed=123):
    np.random.seed(seed)
    A = [np.random.normal(0,1,size=(dim,R)) for dim in array]
    signal = cp_est(A)
    if(noise == True):
        noise = np.random.normal(size=array)
        c = np.sqrt(SNR*np.var(noise)/np.var(signal))
        noise = noise/c
        tensor = signal + noise
        return {'tensor': tensor, 'signal': signal, 'noise': noise, 'noiseRatio': np.var(signal)/np.var(noise)}
    else:
        signal = signal
        tensor = signal
        return {'tensor': tensor, 'signal': signal}

# single tensor cp decomposition with l2 penalty
def cp_penalized(simu, R, w, initA=None, cutoff=0.001, maxiter=25, LS=False, seed=123):
    tensor = simu['tensor']
    signal = simu['signal']
    if (initA==None):
        np.random.seed(seed)
        A = [np.random.normal(size=(dim,R)) for dim in tensor.shape]
    else:
        A = initA.copy()
    niter = 1
    resFrob = [lg.norm(tensor-cp_est(A))]
    RSE = [(lg.norm(signal - cp_est(A)))/lg.norm(signal)]
    obsRSE = [(lg.norm(tensor - cp_est(A)))/lg.norm(tensor)]
    objFun = [obj_single(tensor, A, w)]
    diff = []
    nacc, failtime = 0, 0
    while(niter<=maxiter):
        A_cur = A.copy()
        for i in range(len(A)):
            V = tg.khatri_rao(A[:i]+A[i+1:]) 
            A[i] = tl.unfold(tensor,i) @ V @ lg.inv(V.T@V+w*np.identity(R))
        if LS == True and niter>6 and failtime < 5:
            A, acc = line_search_single(tensor, A, A_cur, RLS = niter**(1/3))
            nacc += acc
            failtime += np.abs(acc-1)
        elif LS == True and niter>6 and failtime >= 5:
            A, acc = line_search_single(tensor, A, A_cur, RLS = niter**(1/4))
            nacc += acc
            failtime += np.abs(acc-1)
        resFrob.append(lg.norm(tensor-cp_est(A)))
        RSE.append((lg.norm(signal - cp_est(A)))/lg.norm(signal))
        obsRSE.append((lg.norm(tensor - cp_est(A)))/lg.norm(tensor))
        objFun.append(obj_single(tensor, A, w))
        diff.append(diff_str_single(A, A_cur))

        if(np.abs(objFun[niter]-objFun[niter-1]) < cutoff): break
        niter += 1

    return {'A': A, 'resFrob': resFrob, 'RSE': RSE, 'obsRSE': obsRSE, 'objFun': objFun, 'diff': diff, 'nacc':nacc, 'failtime': failtime, 'niter': niter}

# single tensor with missing values: cp decomposition with l2 penalty
def cp_cv(simu, R, w, cvInd, cutoff=0.001, maxiter=25, initA=None, seed = 123):
    tensor = simu['tensor'].copy()
    impuTensor = tensor.copy()
    nanInd = np.isnan(tensor)
    missInd = nanInd | cvInd

    impuTensor[missInd]=np.random.normal(size=impuTensor.shape)[missInd] #initialize missing elements with random imputed values.
    niter = 1
    if initA is not None:
        A = initA.copy()
    else:
        np.random.seed(seed)
        A = [np.random.normal(size=(dim,R)) for dim in impuTensor.shape]

    if np.sum(cvInd) != 0:
        obsRSE_impute = [lg.norm((tensor - cp_est(A))[cvInd])/lg.norm(tensor[cvInd])]
    else:
        obsRSE_impute = []
    objFun = [obj_single(tensor, A, w)]

    while(niter<=maxiter):
        for i in range(len(A)):
            V = tg.khatri_rao(A[:i]+A[i+1:]) 
            A[i] = tl.unfold(impuTensor,i) @ V @ lg.inv(V.T@V+w*np.identity(R))
            
        impuTensor[missInd]=cp_est(A)[missInd]

        if np.sum(cvInd) != 0:
            obsRSE_impute.append(lg.norm((tensor - cp_est(A))[cvInd])/lg.norm(tensor[cvInd]))
        objFun.append(obj_single(tensor, A, w))
        
        if(np.abs(objFun[niter]-objFun[niter-1])/objFun[niter] < cutoff): break
        niter += 1

    return {'A': A, 'objFun': objFun, 'obsRSE_impute': obsRSE_impute,  'niter': niter - 1}

def write_to_csv(csv_file_path, data, column_names):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline='') as file:
        csvwriter = csv.writer(file)
        # Write column names if the file is new
        if not file_exists:
            csvwriter.writerow(column_names)
        
        csvwriter.writerow(data)

