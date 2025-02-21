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

# extract an array containing (r+1)th columns for all factor matrices.
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

def obj_s(tensor1, tensor2, A1, A2, w, sigma):
    res1 = lg.norm(tensor1 - cp_est(A1))**2
    res2 = lg.norm(tensor2 - cp_est(A2))**2
    penalty = w*sigma*(lg.norm(A1[0])**2 + sum([lg.norm(A1[k])**2 for k in range(1, len(A1))]) + sum([lg.norm(A2[k])**2 for k in range(1, len(A2))]))
    return res1+res2+penalty

# objective function for two tensors with normality assumption for entries
def obj_eb(tensor1, tensor2, A1, A2, sigma_0, sigma_1, sigma_2):
    R = A1[0].shape[1]
    res1 = lg.norm(tensor1 - cp_est(A1))**2
    res2 = lg.norm(tensor2 - cp_est(A2))**2
    penalty_vec0 = np.array([lg.norm(A1[0][:,r])**2*sig for r,sig in zip(range(R), sigma_0)])
    penalty_vec0[np.isnan(penalty_vec0)]=0
    
    penalty_vec1 = np.array([sum([lg.norm(Ak[:,r])**2 for Ak in A1[1:]])*sig for r,sig in zip(range(R), sigma_1)]) 
    penalty_vec1[np.isnan(penalty_vec1)]=0

    penalty_vec2 = np.array([sum([lg.norm(Ak[:,r])**2 for Ak in A2[1:]])*sig for r,sig in zip(range(R), sigma_2)]) 
    penalty_vec2[np.isnan(penalty_vec2)]=0

    return res1 + res2 + sum(penalty_vec0) + sum(penalty_vec1) + sum(penalty_vec2)

def diff_str(A1_new, A2_new, A1_cur, A2_cur):
    ss=0
    for k in range(len(A1_new)):
        ss+=lg.norm(A1_new[k]- A1_cur[k])/lg.norm(A1_new[k])
    for k in range(len(A2_new)):
        ss+=lg.norm(A2_new[k]- A2_cur[k])/lg.norm(A2_new[k])
    return ss

def line_search(tensor1, tensor2, A1_cur, A2_cur, A1_pre, A2_pre, RLS):
    #loss_cur = lg.norm(tensor1 - cp_est(A1_cur)) + lg.norm(tensor2 - cp_est(A2_cur))
    loss_cur = diff_str(A1_cur, A2_cur, A1_pre, A2_pre)
    A1_new, A2_new= [],[]
    for k in range(len(A1_pre)):
        A1_new.append(A1_pre[k] + RLS*(A1_cur[k] - A1_pre[k]))
    for k in range(len(A2_pre)):
        A2_new.append(A2_pre[k] + RLS*(A2_cur[k] - A2_pre[k]))
    #loss_new = lg.norm(tensor1 - cp_est(A1_new)) + lg.norm(tensor2 - cp_est(A2_new))
    loss_new = diff_str(A1_new, A2_new, A1_pre, A2_pre)
    if loss_new < loss_cur:
        return A1_new, A2_new, 1
    else:
        return A1_cur, A2_cur, 0

def singular_value(A):
    sig_val = np.prod([lg.norm(A[i], axis=0) for i in range(len(A))], axis=0)
    return sig_val.tolist()

def rse(tensor, estTensor, selectInd = None):
    if selectInd is None:
        selectInd = np.ones(tensor.shape, dtype=bool)
    if lg.norm(tensor[selectInd]) == 0:
        return np.nan
    else:
        return lg.norm((tensor-estTensor)[selectInd])/lg.norm(tensor[selectInd])

# identify shared and individual components
def identifier(A1, A2):
    # identify column indices with non-zero singular value
    ind1 = np.array([a > 0.00001 for a in singular_value(A1)])
    ind2 = np.array([a > 0.00001 for a in singular_value(A2)])
    sharedInd = ind1 & ind2
    indivInd1 = ind1 & ~sharedInd
    indivInd2 = ind2 & ~sharedInd
    
    # identify components
    estShared1 = cp_est(A1, sharedInd)
    estShared2 = cp_est(A2, sharedInd)
    estIndiv1 = cp_est(A1, indivInd1)
    estIndiv2 = cp_est(A2, indivInd2)
    estTensor1 = estShared1 + estIndiv1
    estTensor2 = estShared2 + estIndiv2
    return estShared1, estShared2, estIndiv1, estIndiv2, estTensor1, estTensor2

# extract result for a list of cp_linked objective.
def extract(result, measure): 
    r=[]
    if isinstance(result[0][measure], list): # measurement is a list
        for k in range(len(result)):
            r.append(result[k][measure][-1])
    else: # measurement is a scalar or a nparray
        for k in range(len(result)):
            r.append(result[k][measure])
    return np.array(r)

# identify shared and individual ranks
def rank_identifier(A1, A2):
    ind1 = np.array([a > 0.00001 for a in singular_value(A1)])
    ind2 = np.array([a > 0.00001 for a in singular_value(A2)])
    sharedInd = ind1 & ind2
    indivInd1 = ind1 & ~sharedInd
    indivInd2 = ind2 & ~sharedInd

    ranks = [sum(sharedInd), sum(indivInd1), sum(indivInd2)]
    return ranks


def constraint(A_list, ranks, A_type):
    if A_type==1:
        Ind = ranks[0] + ranks[1] + np.arange(ranks[2]) # set ranks[2] columns to be 0 
        for matrix in A_list[1:]:
            matrix[:, Ind] = 0
    elif A_type==2:
        Ind = ranks[0] + np.arange(ranks[1])# set ranks[1] columns to be 0 
        for matrix in A_list[1:]:
            matrix[:, Ind] = 0
    return A_list

def constraint_factor_identifier(A1, A2):
    ind1 = np.array([a > 0.00001 for a in singular_value(A1)])
    ind2 = np.array([a > 0.00001 for a in singular_value(A2)])
    sharedInd = ind1 & ind2
    indivInd1 = ind1 & ~sharedInd
    indivInd2 = ind2 & ~sharedInd
    ranks = [sum(sharedInd), sum(indivInd1), sum(indivInd2)]
    S1 = [matrix[:,sharedInd] for matrix in A1]
    S2 = [matrix[:,sharedInd] for matrix in A2]
    I1 = [matrix[:,indivInd1] for matrix in A1]
    I2 = [matrix[:,indivInd2] for matrix in A2]

    # dimensions of each tensor
    dim1 = [factor.shape[0] for factor in A1]
    dim2 = [factor.shape[0] for factor in A2]

    # factor matrices of zeros used to create sparsity
    zero_factor1 = [np.zeros((dim, ranks[2])) for dim in dim1]
    zero_factor2 = [np.zeros((dim, ranks[1])) for dim in dim2]

    A1 = [np.hstack((a, b, c)) for a, b, c in zip(S1, I1, zero_factor1)]
    A2 = [np.hstack((a, b, c)) for a, b, c in zip(S2, zero_factor2, I2)]
    
    # shared matrix A0 doestn't have sparsity
    ind = list(range(sum(ranks[:2]), sum(ranks)))
    A1[0][:,ind] = A2[0][:,ind]
    A2[0] = A1[0]

    return A1, A2

def identify_missing_id(tensor):
    is_missing = np.all(np.isnan(tensor), axis=tuple(range(1, tensor.ndim)))
    return is_missing

# creating index for validation set.
def ind_split(tensor1, tensor2, nfolds, seed, missType):
    if missType=='random':
        flat_tensor1 = tensor1.ravel()
        flat_tensor2 = tensor2.ravel()
        
        # Find indices of the non-NaN elements for each tensor
        non_nan_indices1 = np.arange(flat_tensor1.size)[~np.isnan(flat_tensor1)]
        non_nan_indices2 = np.arange(flat_tensor2.size)[~np.isnan(flat_tensor2)]
        
        # Shuffle the indices based on the tensor with fewer non-NaN elements
        np.random.seed(seed)
        shuffled_indices1 = np.random.permutation(non_nan_indices1)
        shuffled_indices2 = np.random.permutation(non_nan_indices2)
        cvIndList1 = np.array_split(shuffled_indices1, nfolds)
        cvIndList2 = np.array_split(shuffled_indices2, nfolds)

        cvIndList = []
        for fold_indices1, fold_indices2 in zip(cvIndList1, cvIndList2):
            mask1 = np.zeros_like(flat_tensor1, dtype=bool)
            mask2 = np.zeros_like(flat_tensor2, dtype=bool)
            
            mask1[fold_indices1] = True
            mask2[fold_indices2] = True
            
            cvIndList.append({'cvInd1': mask1.reshape(tensor1.shape), 'cvInd2': mask2.reshape(tensor2.shape)})
    elif missType=='tensor':
        print('hello')

    elif missType=='both':
        np.random.seed(seed)
        missing_id_tensor1 = identify_missing_id(tensor1)
        missing_id_tensor2 = identify_missing_id(tensor2)
        missing_entry_tensor1 = np.isnan(tensor1)
        missing_entry_tensor2 = np.isnan(tensor2)
        id_common = np.arange(tensor1.shape[0])[~missing_id_tensor1 & ~missing_id_tensor2]
        np.random.shuffle(id_common)
        indFolds = np.array_split(id_common, nfolds)
        cvIndList = []
        for i, j in zip(np.arange(nfolds), np.concatenate((np.arange(1,nfolds), 0), axis=None)):
            tens1 = np.zeros(tensor1.shape)
            tens2 = np.zeros(tensor2.shape)
            ind1, ind2 = indFolds[i], indFolds[j]
            
            # mark tensor-wise missing part as 1
            tens1[ind1,:]=1
            tens2[ind2,:]=1
            cvInd1_tensor = (tens1==1) & ~missing_entry_tensor1 # tensor-wise validation set
            cvInd2_tensor = (tens2==1) & ~missing_entry_tensor2 

            # mark entry-wise missing part as 1
            non_missing_tens1 = np.argwhere((tens1 == 0) & ~missing_entry_tensor1)
            non_missing_tens2 = np.argwhere((tens2 == 0) & ~missing_entry_tensor2)

            num_to_select_tens1 = np.sum(cvInd1_tensor) 
            num_to_select_tens2 = np.sum(cvInd2_tensor) 

            selected_entries_tens1 = non_missing_tens1[np.random.choice(non_missing_tens1.shape[0], size=num_to_select_tens1, replace=False)]
            tens1_entry = np.zeros(tensor1.shape)
            for entry in selected_entries_tens1:
                tens1[tuple(entry)] = 1
                tens1_entry[tuple(entry)] = 1 

            selected_entries_tens2 = non_missing_tens2[np.random.choice(non_missing_tens2.shape[0], size=num_to_select_tens2, replace=False)]
            tens2_entry = np.zeros(tensor2.shape)
            for entry in selected_entries_tens2:
                tens2[tuple(entry)] = 1
                tens2_entry[tuple(entry)] = 1

            cvInd1_entry = tens1_entry==1 # entry-wise validation set
            cvInd2_entry = tens2_entry==1
            
            cvInd1 = (tens1==1) & ~missing_entry_tensor1
            cvInd2 = (tens2==1) & ~missing_entry_tensor2
            
            cvIndList.append({'cvInd1': cvInd1, 'cvInd2': cvInd2, 'cvInd1_tensor': cvInd1_tensor, 
                              'cvInd2_tensor': cvInd2_tensor, 'cvInd1_entry': cvInd1_entry, 'cvInd2_entry': cvInd2_entry})
    return cvIndList

# filter index of nonzero columns
def nonzero_col_index(matrix):
    nonzero_columns = np.any(matrix != 0, axis=0)
    indices = np.where(nonzero_columns)[0]  
    return indices.tolist()


def simu_linked(dim1, dim2, R, SNR=1, seed=1):
    np.random.seed(seed)
    S1 = [np.random.normal(0,1,size=(dim,R[0])) for dim in dim1]
    I1 = [np.random.normal(0,1,size=(dim,R[1])) for dim in dim1]
    S2 = [np.random.normal(0,1,size=(dim,R[0])) for dim in dim2]
    I2 = [np.random.normal(0,1,size=(dim,R[2])) for dim in dim2]
    S2[0] = S1[0]

    signal1 = cp_est(S1)+cp_est(I1)
    signal2 = cp_est(S2)+cp_est(I2)
    shared1 = cp_est(S1)
    shared2 = cp_est(S2)
    indiv1 = cp_est(I1)
    indiv2 = cp_est(I2)
    
    if SNR==None:       
        tensor1 = signal1 
        tensor2 = signal2 
        return {'S1': S1, 'S2': S2, 'I1':I1, 'I2':I2, 'tensor1': tensor1, 'tensor2': tensor2, 'signal1': signal1, 'signal2': signal2, 
                'shared1': shared1, 'shared2': shared2, 'indiv1': indiv1, 'indiv2': indiv2}
    else:
        noise1 = np.random.normal(size=dim1)
        noise2 = np.random.normal(size=dim2)
        c1 = np.sqrt(SNR*np.var(noise1)/np.var(signal1))
        c2 = np.sqrt(SNR*np.var(noise2)/np.var(signal2))
        noise1 = noise1/c1
        noise2 = noise2/c2
        tensor1 = signal1 + noise1
        tensor2 = signal2 + noise2

        return {'S1': S1, 'S2': S2, 'I1':I1, 'I2':I2, 'tensor1': tensor1, 'tensor2': tensor2, 'signal1': signal1, 'signal2': signal2, 'noise1': noise1,'noise2': noise2 , 
                'shared1': shared1, 'shared2': shared2, 'indiv1': indiv1, 'indiv2': indiv2,
               'SNR1': np.var(signal1)/np.var(noise1), 'SNR2': np.var(signal2)/np.var(noise2)}
   
def idenfity_slice_missing(missInd):
    sampleInd = np.all(missInd, tuple(range(1, missInd.ndim)))
    reshaped_sampleInd = sampleInd.reshape([sampleInd.shape[0]] + [1] * (missInd.ndim - 1))
    sliceInd = np.broadcast_to(reshaped_sampleInd, missInd.shape)
    return sliceInd

def ALS(tensor1, tensor2, A1, A2, missInd1, missInd2, R=20, maxiter=500, cutoff=0.001, sigma=0, w_list=[1]):
    impuTensor1 = tensor1.copy()
    impuTensor2 = tensor2.copy()
    
    # Initial imputation with random values
    impuTensor1[missInd1] = np.random.normal(size=impuTensor1.shape)[missInd1]
    impuTensor2[missInd2] = np.random.normal(size=impuTensor2.shape)[missInd2]
    
    # identify missing indices
    sliceInd1 = idenfity_slice_missing(missInd1)
    sliceInd2 = idenfity_slice_missing(missInd2)
    entryInd1 = ~sliceInd1 & missInd1
    entryInd2 = ~sliceInd2 & missInd2
    nonzero_col1 = np.any(A1[1] != 0, axis=0)
    nonzero_col2 = np.any(A2[1] != 0, axis=0)

    obj = [obj_s(impuTensor1, impuTensor2, A1, A2, w_list[0], sigma)]
    niter = 0

    for w in w_list:
        count = 0
        while count <= maxiter:
            X1_1 = tl.unfold(impuTensor1, 0)
            X2_1 = tl.unfold(impuTensor2, 0)
            V1 = tg.khatri_rao(A1[1:])
            V2 = tg.khatri_rao(A2[1:])
            A1[0] = (X1_1 @ V1 + X2_1 @ V2)  @ lg.inv(V1.T @ V1 + V2.T @ V2 + w * sigma * np.identity(R))
            A2[0] = A1[0]

            for k in range(1, len(A1)):
                X1_k = tl.unfold(impuTensor1, k)
                V1 = tg.khatri_rao(A1[:k] + A1[k + 1:])
                V1 = V1[:, nonzero_col1]
                A1[k][:, nonzero_col1] = (X1_k @ V1) @ lg.inv(V1.T @ V1 + w * sigma * np.identity(np.sum(nonzero_col1)))
            for k in range(1, len(A2)):
                X2_k = tl.unfold(impuTensor2, k)
                V2 = tg.khatri_rao(A2[:k] + A2[k + 1:])
                V2 = V2[:, nonzero_col2]
                A2[k][:, nonzero_col2] = (X2_k @ V2) @ lg.inv(V2.T @ V2 + w * sigma * np.identity(np.sum(nonzero_col2)))

            estShared1, estShared2, estIndiv1, estIndiv2, estTensor1, estTensor2 = identifier(A1, A2)
            
            # impute missing values using estimated structures: entry with full structure and slice with shared structure.
            impuTensor1[entryInd1] = estTensor1[entryInd1]
            impuTensor2[entryInd2] = estTensor2[entryInd2]
            impuTensor1[sliceInd1] = estShared1[sliceInd1]
            impuTensor2[sliceInd2] = estShared2[sliceInd2]

            obj.append(obj_s(impuTensor1, impuTensor2, A1, A2, w, sigma))
            count += 1
            niter += 1
            if np.abs(obj[niter] - obj[niter - 1]) < cutoff:
                break

    return A1, A2, estShared1, estShared2, estIndiv1, estIndiv2, estTensor1, estTensor2, obj, niter


def cp_linked(obj1, obj2, cvInd={}, initA1=None, initA2=None, R=20, sigma=0, cutoff=0.001, maxiter=500, w_list=[1], seed=1):
    tensor1 = obj1.copy()
    tensor2 = obj2.copy()

    # intialized factor matrices
    if initA1 is not None:    # pre-specified factor matrices
        A1 = initA1.copy()
        A2 = initA2.copy()
        if isinstance(R, list): # constraint factor matrices
            rank = sum(R)
            R = rank
    else:                    # randomly initialized factor matrices
        if isinstance(R, list): # constraint factor matrices
            rank = sum(R)
            np.random.seed(seed)
            A1 = [np.random.normal(size=[dim, rank]) for dim in tensor1.shape]
            A1 = constraint(A1, R, 1)
            A2 = [np.random.normal(size=[dim, rank]) for dim in tensor2.shape]
            A2 = constraint(A2, R, 2)
            R = rank
        else:
            np.random.seed(seed)
            A1 = [np.random.normal(size=[dim, R]) for dim in tensor1.shape]
            A2 = [np.random.normal(size=[dim, R]) for dim in tensor2.shape]

    # identify missing indices
    cvInd1 = cvInd.get('cvInd1', np.zeros(tensor1.shape, dtype=bool))
    cvInd2 = cvInd.get('cvInd2', np.zeros(tensor2.shape, dtype=bool))
    nanInd1 = np.isnan(tensor1)
    nanInd2 = np.isnan(tensor2)
    missInd1 = nanInd1 | cvInd1
    missInd2 = nanInd2 | cvInd2

    # ALS algorithm
    A1, A2, estShared1, estShared2, estIndiv1, estIndiv2, estTensor1, estTensor2, obj, niter = ALS(
        tensor1, tensor2, A1, A2, missInd1, missInd2, R, maxiter, cutoff, sigma, w_list)

    # cv imputation performance
    obsRSE_obs = np.array([rse(tensor1, estTensor1, ~missInd1), rse(tensor2, estTensor2, ~missInd2)])
    obsRSE_cv = np.array([rse(tensor1, estTensor1, cvInd1), rse(tensor2, estTensor2, cvInd2)])
    
    cvInd1_tensor = cvInd.get('cvInd1_tensor', np.zeros(tensor1.shape, dtype=bool))
    cvInd2_tensor = cvInd.get('cvInd2_tensor', np.zeros(tensor2.shape, dtype=bool))
    cvInd1_entry = cvInd.get('cvInd1_entry', np.zeros(tensor1.shape, dtype=bool))
    cvInd2_entry = cvInd.get('cvInd2_entry', np.zeros(tensor2.shape, dtype=bool))
    obsRSE_cv_tensor = np.array([rse(tensor1, estTensor1, cvInd1_tensor), rse(tensor2, estTensor2, cvInd2_tensor)])
    obsRSE_cv_entry = np.array([rse(tensor1, estTensor1, cvInd1_entry), rse(tensor2, estTensor2, cvInd2_entry)])

    return {
        'A1': A1, 'A2': A2, 'estShared1': estShared1, 'estShared2': estShared2, 'estIndiv1': estIndiv1, 
        'estIndiv2': estIndiv2, 'estTensor1': estTensor1, 'estTensor2': estTensor2, 
        'obsRSE_obs': obsRSE_obs, 'obsRSE_cv': obsRSE_cv, 'obsRSE_cv_tensor': obsRSE_cv_tensor, 
        'obsRSE_cv_entry': obsRSE_cv_entry, 'niter': niter, 'obj': obj
    }

