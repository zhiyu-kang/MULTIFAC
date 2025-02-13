from SingleTensor import *

def main(args):
    if args.action=='simulation': #simulate couple tensors and calculate initial factor matrices  
        simu = simu_single(args.shape, args.rank, True, args.snr, args.task)
        simu['tensor_true'] = simu['tensor'].copy()

        np.random.seed(args.task)
        num_element = np.prod(simu['tensor'].shape)
        nan_Ind = np.zeros(num_element, dtype=bool)
        true_indices = np.random.choice(num_element, size=int(num_element*0.1), replace=False)
        nan_Ind[true_indices] = True
        nan_tensor = nan_Ind.reshape(simu['tensor'].shape)
        simu['tensor'][nan_tensor] = np.nan

        if args.format=='python':
            with open(f"data/dataset_{args.task}.pkl", "wb") as file:
                pickle.dump(simu, file)
        elif args.format=='matlab':
            savemat("data/dataset_"+str(args.task)+".mat", {'Tensor': simu['tensor'], 'Signal': simu['signal']})
        elif args.format=='both':
            with open(f"data/dataset_{args.task}.pkl", "wb") as file:
                pickle.dump(simu, file)
            savemat("data/dataset_"+str(args.task)+".mat", {'Tensor': simu['tensor'], 'Signal': simu['signal']})
            
        
    elif args.action=='tuning': #get imputation score for given sigma value by cross validation
        with open(f"data/dataset_{args.task}.pkl", "rb") as file:
            simu = pickle.load(file)
        
        missIndList = ind_split(simu['tensor'], args.nfolds, args.task, 'random') # tensor-wise missing won't work 
        ########## unconstrained decomposition: determine rank
        # initialize factor matrices by unpenalized imputation
        AList = []
        for missInd in missIndList:
            init_cv_temp = [cp_cv(simu, args.cprank, 0, missInd, args.cutoff, args.maxiter, None, args.nfolds*args.task + seed) for seed in range(args.nfolds)]
            init_cv = init_cv_temp[np.argmin(extract(init_cv_temp, 'objFun'))]
            AList.append(init_cv['A'])

        # list of sigma values
        sigma_list = np.logspace(np.log10(args.sigma_lower), np.log10(args.sigma_upper), num=args.sigma_num)
        mean_se_score = []
        for sigma in sigma_list:
            cp_result = [cp_cv(simu, args.cprank, sigma, missInd, 0.001, 300, AList[k], args.task) for k, missInd in zip(range(args.nfolds), missIndList)]
            AList = [cp_result[k]['A'].copy() for k in range(args.nfolds)]
            score = extract(cp_result, 'obsRSE_impute')
            mean_score = np.mean(score)
            se_score = np.std(score)/np.sqrt(args.nfolds)
            mean_se_score.append([sigma, mean_score, se_score])
        mean_se_score_df = pd.DataFrame(mean_se_score, columns=['sigma', 'mean_RSE', 'se_RSE'])
        mean_se_score_df.round(6).to_csv(f"scores/score_{args.task}.txt", index=False)

        ####### constrained decomposition: select optimal sigma value
        sigma_best = mean_se_score_df['sigma'][mean_se_score_df['mean_RSE'].idxmin()]
        min_score_mix = mean_se_score_df['mean_RSE'].min()
        target_score_mix = min_score_mix + mean_se_score_df.loc[mean_se_score_df['mean_RSE'].idxmin(), 'se_RSE']
        sigma_1se = mean_se_score_df[mean_se_score_df['mean_RSE'] <= target_score_mix]['sigma'].max()

        temp_init_cp = [cp_cv(simu, args.cprank, 0, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, None, args.nfolds*args.task + seed) for seed in range(args.nfolds)]
        init_cp = temp_init_cp[np.argmin(extract(temp_init_cp, 'objFun'))]
        initA = init_cp['A']

        cp_sigma = cp_cv(simu, args.cprank, sigma_1se, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, initA, args.task)

        ind = np.array([a > 0.00001 for a in singular_value(cp_sigma['A'])])
        est_rank = np.sum(ind)

        AList = []
        for missInd in missIndList:
            init_cv_temp = [cp_cv(simu, est_rank, 0, missInd, args.cutoff, args.maxiter, None, args.nfolds*args.task + seed) for seed in range(args.nfolds)]
            init_cv = init_cv_temp[np.argmin(extract(init_cv_temp, 'objFun'))]
            AList.append(init_cv['A'])

        mean_se_score = []
        for sigma in sigma_list:
            cp_result = [cp_cv(simu, est_rank, sigma, missInd, 0.001, 300, AList[k], args.task) for k, missInd in zip(range(args.nfolds), missIndList)]
            AList = [cp_result[k]['A'].copy() for k in range(args.nfolds)]
            score = extract(cp_result, 'obsRSE_impute')
            mean_score = np.mean(score)
            se_score = np.std(score)/np.sqrt(args.nfolds)
            mean_se_score.append([sigma, mean_score, se_score])
        mean_se_score_df = pd.DataFrame(mean_se_score, columns=['sigma', 'mean_RSE', 'se_RSE'])
        mean_se_score_df.round(6).to_csv(f"scores/score_constraint_{args.task}.txt", index=False)

    elif args.action=='solving':
        with open(f"data/dataset_{args.task}.pkl", "rb") as file:
            simu = pickle.load(file)
            
        df = pd.read_csv(f"scores/score_{args.task}.txt", sep=',')
        sigma_best = df['sigma'][df['mean_RSE'].idxmin()]
        min_score_mix = df['mean_RSE'].min()
        target_score_mix = min_score_mix + df.loc[df['mean_RSE'].idxmin(), 'se_RSE']
        sigma_1se = df[df['mean_RSE'] <= target_score_mix]['sigma'].max()
        df_constraint = pd.read_csv(f"scores/score_constraint_{args.task}.txt", sep=',')
        sigma_constraint = df_constraint['sigma'][df_constraint['mean_RSE'].idxmin()]

        ### unconstrained cp decomposition
        # initialization with unpenalized cp decomposition
        temp_init_cp = [cp_cv(simu, args.cprank, 0, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, None, args.nfolds*args.task + seed) for seed in range(args.nfolds)]
        init_cp = temp_init_cp[np.argmin(extract(temp_init_cp, 'objFun'))]
        initA = init_cp['A']
        # cp decomp with large selected sigma to find ranks and initial factor matrices for constrained cp decomp
        cp_sigma = cp_cv(simu, args.cprank, sigma_1se, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, initA, args.task)

        ### constrained cp decomposition
        ind = np.array([a > 0.00001 for a in singular_value(cp_sigma['A'])])
        est_rank = np.sum(ind)
        temp_init_cp = [cp_cv(simu, est_rank, 0, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, None, args.nfolds*args.task + seed) for seed in range(args.nfolds)]
        init_cp = temp_init_cp[np.argmin(extract(temp_init_cp, 'objFun'))]
        initA = init_cp['A']

        # unpenalized
        cp_con_unpenalized = cp_cv(simu, est_rank, 0, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, initA, args.task)
        # penalized
        cp_con_penalized = cp_cv(simu, est_rank, sigma_constraint, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, initA, args.task)

        ### cp decomposition with true rank
        temp_init_cp = [cp_cv(simu, args.rank, 0, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, None, args.nfolds*args.task + seed) for seed in range(args.nfolds)]
        init_cp = temp_init_cp[np.argmin(extract(temp_init_cp, 'objFun'))]
        initA = init_cp['A']
        
        cp_true = cp_cv(simu, args.rank, 0, np.zeros(simu['tensor'].shape, dtype=bool), args.cutoff, args.maxiter, initA, args.task)

        nanInd = np.isnan(simu['tensor'])
        signal = simu['signal']
        RSE_sigma_impute = lg.norm((signal - cp_est(cp_sigma['A']))[nanInd])/lg.norm(signal[nanInd])
        RSE_con_unpenalized_impute = lg.norm((signal - cp_est(cp_con_unpenalized['A']))[nanInd])/lg.norm(signal[nanInd])
        RSE_con_penalized_impute = lg.norm((signal - cp_est(cp_con_penalized['A']))[nanInd])/lg.norm(signal[nanInd])
        RSE_true_impute = lg.norm((signal - cp_est(cp_true['A']))[nanInd])/lg.norm(signal[nanInd])

        column_names = ['Task', 'RSE_sigma_impute', 'RSE_con_unpenalized_impute', 'RSE_con_penalized_impute', 'RSE_true_impute', 'sigma_step1', 'sigma_step2', 'est_rank']
        result = np.array([args.task, RSE_sigma_impute, RSE_con_unpenalized_impute, RSE_con_penalized_impute, RSE_true_impute, sigma_1se, sigma_constraint, est_rank])
        
        write_to_csv("Summary_single_tensor.csv", np.round(result, 6), column_names)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help = 'Task ID')
    parser.add_argument("--action", type=str, required=True, choices = ['simulation', 'tuning', 'solving'], help = 'The operation to be performed.')
    parser.add_argument("--format", type=str, default='python', choices = ['python', 'matlab', 'both'], help = 'Ouput format for generated data.')

    # parameter for data generation
    parser.add_argument("--shape", type=int, nargs = '+', help = 'shape of tensor')
    parser.add_argument("--rank", type=int,  help = 'rank for the underlying signal')
    parser.add_argument("--snr", type=float, help = 'signal-to-noise ratio')
    
    
    #parameters for factor tuning
    parser.add_argument("--sigma_lower", type=float, help = 'lower bound of grid search')
    parser.add_argument("--sigma_upper", type=float, help = 'upper bound of grid search')
    parser.add_argument("--sigma_num", type=int, help = 'number of sigma values in grid search')
    parser.add_argument("--nfolds", type=int, default=5, help = 'number of folds in cross validation')
    
    parser.add_argument("--cprank", type=int, default=20, help = 'rank used to solve cp decomposition')
    parser.add_argument("--sigma", type=float, default=0, help = 'penalty factor')
    parser.add_argument("--cutoff", type=float, default=0.00001, help = 'stopping criteria')
    parser.add_argument("--maxiter", type=int, default=300, help = 'maximum number of iteration')

    args = parser.parse_args()
    if args.action == 'simulation':
        if not all(arg is not None for arg in [args.shape, args.rank, args.snr]):
            parser.error("When action is 'simulation', --shape, --rank, and --snr are required.")
    elif args.action == 'tuning':
        if not all(arg is not None for arg in [args.sigma_lower, args.sigma_upper, args.sigma_num]):
            parser.error("When action is 'tuning', --sigma_lower, --sigma_upper, and --sigma_num are required.")

    main(args)