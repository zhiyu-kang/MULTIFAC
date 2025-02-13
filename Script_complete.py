from CoupleTensor_functions import *

def main(args):
    if args.action=='simulation': #simulate couple tensors and calculate initial factor matrices
        simu = simu_linked(args.shape1, args.shape2, args.rank, args.snr, args.task)

        if args.stand=='True':
            N = sum(args.shape1) + sum(args.shape2)
            c1 = N/lg.norm(simu['tensor1'])
            simu['tensor1'] = simu['tensor1'] * c1
            simu['signal1'] = simu['signal1'] * c1
            simu['shared1'] = simu['shared1'] * c1
            simu['indiv1'] = simu['indiv1'] * c1

            c2 = N/lg.norm(simu['tensor2'])
            simu['tensor2'] = simu['tensor2'] * c2
            simu['signal2'] = simu['signal2'] * c2
            simu['shared2'] = simu['shared2'] * c2
            simu['indiv2'] = simu['indiv2'] * c2
        if args.format=='python':
            with open(f"simulation_data/dataset_{args.task}.pkl", "wb") as file:
                pickle.dump(simu, file)
        elif args.format=='matlab':
            savemat("simulation_data/dataset_"+str(args.task)+".mat", {'T1': simu['tensor1'], 'T2': simu['tensor2'], 'S1': simu['signal1'], 'S2': simu['signal2'], 
                                      'shared1': simu['shared1'], 'shared2': simu['shared2'], 'indiv1': simu['indiv1'], 'indiv2': simu['indiv2']})
        elif args.format=='both':
            with open(f"simulation_data/dataset_{args.task}.pkl", "wb") as file:
                pickle.dump(simu, file)
            savemat("simulation_data/dataset_"+str(args.task)+".mat", {'T1': simu['tensor1'], 'T2': simu['tensor2'], 'S1': simu['signal1'], 'S2': simu['signal2'], 
                                      'shared1': simu['shared1'], 'shared2': simu['shared2'], 'indiv1': simu['indiv1'], 'indiv2': simu['indiv2']})
        
    elif args.action=='tuning': #get imputation score for given sigma value by tensor-wise cross validation
        if (args.tensor1 == 'None') & (args.tensor2 == 'None'):
            with open(f"simulation_data/dataset_{args.task}.pkl", "rb") as file:
                simu = pickle.load(file)
                tensor1 = simu['tensor1']
                tensor2 = simu['tensor2']
            score_file_name = f"imputation_score/score_{args.task}.csv"
            score_file_name_constraint = f"imputation_score/score_constraint_{args.task}.csv"
        else:
            with open(f"{args.tensor1}.pkl", "rb") as file:
                tensor1 = pickle.load(file)
            with open(f"{args.tensor2}.pkl", "rb") as file:
                tensor2 = pickle.load(file)
            score_file_name = f"score_{args.tensor1}_{args.tensor2}.csv"
            score_file_name = f"score_constraint_{args.tensor1}_{args.tensor2}.csv"

        cvIndList = ind_split(tensor1, tensor2, args.nfolds, args.task, 'both')

        # initialize factor matrices by unpenalized imputation
        A1List = []
        A2List = []
        for cvInd in cvIndList:
            init_cv_temp = [cp_linked(tensor1, tensor2, cvInd, None, None, args.cprank, 0, args.cutoff, args.maxiter, [1], 5*args.task + seed) for seed in range(5)]
            init_cv = init_cv_temp[np.argmin(extract(init_cv_temp, 'obj'))]
            A1List.append(init_cv['A1'])
            A2List.append(init_cv['A2'])

        # loop for logorithimically spaced sigma values, updating initial factor matrices after each looping.
        sigma_log_values = np.linspace(np.log(args.sigma_lower), np.log(args.sigma_upper), num=args.sigma_num)
        sigma_list = np.exp(sigma_log_values)

        mean_se_score = []
        for sigma in sigma_list:
            cp_cv = [cp_linked(tensor1, tensor2, cvInd, A1List[k], A2List[k], args.cprank, sigma, args.cutoff, args.maxiter, [1], args.task) for k, cvInd in zip(range(args.nfolds), cvIndList)]
            A1List = [cp_cv[k]['A1'].copy() for k in range(args.nfolds)]
            A2List = [cp_cv[k]['A2'].copy() for k in range(args.nfolds)]
            score = np.mean(extract(cp_cv, 'obsRSE_cv'), axis=1) # average across tensor dimension.
            mean_score = np.mean(score)
            se_score = np.std(score)/np.sqrt(args.nfolds)
            mean_score_tensor=np.mean(extract(cp_cv, 'obsRSE_cv_tensor'), axis=0).tolist() # average across validation sets dimension.
            mean_score_entry=np.mean(extract(cp_cv, 'obsRSE_cv_entry'), axis=0).tolist() # average across validation sets dimension.
            mean_se_score.append([sigma, mean_score, se_score] + mean_score_tensor + mean_score_entry)
        mean_se_score_df = pd.DataFrame(mean_se_score, columns=['sigma', 'mean_RSE', 'se_RSE', 'mean_RSE_tensor1', 'mean_RSE_tensor2', 'mean_RSE_entry1', 'mean_RSE_entry2'])
        mean_se_score_df.round(6).to_csv(score_file_name, index=False)

        # cross validation for constraint cp decomposition
        sigma_best = mean_se_score_df['sigma'][mean_se_score_df['mean_RSE'].idxmin()]
        min_score_mix = mean_se_score_df['mean_RSE'].min()
        target_score_mix = min_score_mix + mean_se_score_df.loc[mean_se_score_df['mean_RSE'].idxmin(), 'se_RSE']
        sigma_1se = mean_se_score_df[mean_se_score_df['mean_RSE'] <= target_score_mix]['sigma'].max()

        temp_init_cp = [cp_linked(tensor1, tensor2, {}, None, None, args.cprank, 0, args.cutoff, args.maxiter, [1], 10*args.task + seed) for seed in range(10)]
        init_cp = temp_init_cp[np.argmin(extract(temp_init_cp, 'obj'))]
        initA1 = init_cp['A1']
        initA2 = init_cp['A2']

        # cp decomp with large selected sigma to find ranks and initial factor matrices for constrained cp decomp
        cp_sigma = cp_linked(tensor1, tensor2, {}, initA1, initA2, args.cprank, sigma_1se, args.cutoff, args.maxiter, [1/sigma_1se,1], args.task)

        # calculate ranks
        ranks = rank_identifier(cp_sigma['A1'], cp_sigma['A2'])

        # initialization for constraint cp decomposition
        A1List = []
        A2List = []
        for cvInd in cvIndList:
            init_cv_temp = [cp_linked(tensor1, tensor2, cvInd, None, None, ranks, 0, args.cutoff, args.maxiter, [1], 5*args.task + seed) for seed in range(5)]
            init_cv = init_cv_temp[np.argmin(extract(init_cv_temp, 'obj'))]
            A1List.append(init_cv['A1'])
            A2List.append(init_cv['A2'])

        mean_se_score = []
        for sigma in sigma_list:
            cp_cv = [cp_linked(tensor1, tensor2, cvInd, A1List[k], A2List[k], ranks, sigma, args.cutoff, args.maxiter, [1], args.task) for k, cvInd in zip(range(args.nfolds), cvIndList)]
            A1List = [cp_cv[k]['A1'].copy() for k in range(args.nfolds)]
            A2List = [cp_cv[k]['A2'].copy() for k in range(args.nfolds)]
            score = np.mean(extract(cp_cv, 'obsRSE_cv'), axis=1) # average across tensor dimension.
            mean_score = np.mean(score)
            se_score = np.std(score)/np.sqrt(args.nfolds)
            mean_score_tensor=np.mean(extract(cp_cv, 'obsRSE_cv_tensor'), axis=0).tolist() # average across validation sets dimension.
            mean_score_entry=np.mean(extract(cp_cv, 'obsRSE_cv_entry'), axis=0).tolist() # average across validation sets dimension.
            mean_se_score.append([sigma, mean_score, se_score] + mean_score_tensor + mean_score_entry)
        mean_se_score_df = pd.DataFrame(mean_se_score, columns=['sigma', 'mean_RSE', 'se_RSE', 'mean_RSE_tensor1', 'mean_RSE_tensor2', 'mean_RSE_entry1', 'mean_RSE_entry2'])
        mean_se_score_df.round(6).to_csv(score_file_name_constraint, index=False)


    elif args.action=='solving':
        if (args.tensor1 == 'None') & (args.tensor2 == 'None'):
            with open(f"simulation_data/dataset_{args.task}.pkl", "rb") as file:
                simu = pickle.load(file)
                tensor1 = simu['tensor1']
                tensor2 = simu['tensor2']
                signal1 = simu['signal1']
                signal2 = simu['signal2']
                shared1 = simu['shared1']
                shared2 = simu['shared2']
                indiv1 = simu['indiv1']
                indiv2 = simu['indiv2']

            score_file_name = f"imputation_score/score_{args.task}.csv"
            score_file_name_constraint = f"imputation_score/score_constraint_{args.task}.csv"
        else:
            with open(f"{args.tensor1}.pkl", "rb") as file:
                tensor1 = pickle.load(file)
            with open(f"{args.tensor2}.pkl", "rb") as file:
                tensor2 = pickle.load(file)
            score_file_name = f"score_{args.tensor1}_{args.tensor2}_sig.csv"
            score_file_name = f"score_constraint_{args.tensor1}_{args.tensor2}.csv"
             
        # Load data and Define column names 
        df = pd.read_csv(score_file_name, sep=',')
        sigma_best = df['sigma'][df['mean_RSE'].idxmin()]
        min_score_mix = df['mean_RSE'].min()
        target_score_mix = min_score_mix + df.loc[df['mean_RSE'].idxmin(), 'se_RSE']
        sigma_1se = df[df['mean_RSE'] <= target_score_mix]['sigma'].max()

        # initialization with unpenalized cp decomposition
        temp_init_cp = [cp_linked(tensor1, tensor2, {}, None, None, args.cprank, 0, args.cutoff, args.maxiter, [1], 10*args.task + seed) for seed in range(10)]
        init_cp = temp_init_cp[np.argmin(extract(temp_init_cp, 'obj'))]
        initA1 = init_cp['A1']
        initA2 = init_cp['A2']

        # cp decomp with large selected sigma to find ranks and initial factor matrices for constrained cp decomp
        cp_sigma = cp_linked(tensor1, tensor2, {}, initA1, initA2, args.cprank, sigma_1se, args.cutoff, args.maxiter, [1/sigma_1se,1], args.task)

        ranks = rank_identifier(cp_sigma['A1'], cp_sigma['A2'])
        A1, A2 = constraint_factor_identifier(cp_sigma['A1'], cp_sigma['A2'])

        # cp constraint: add small penalty to avoid singularity
        small_sigma = 0
        cp_con = cp_linked(tensor1, tensor2, {}, A1, A2, ranks, small_sigma, args.cutoff, args.maxiter, [1], args.task)
        ranks_con = rank_identifier(cp_con['A1'], cp_con['A2'])

        # cp constraint: use same sigma
        cp_con_same = cp_linked(tensor1, tensor2, {}, A1, A2, ranks, sigma_1se, args.cutoff, args.maxiter, [1], args.task)
        ranks_con_same = rank_identifier(cp_con_same['A1'], cp_con_same['A2'])

        df_constraint = pd.read_csv(score_file_name_constraint, sep=',')
        sigma_constraint = df_constraint['sigma'][df_constraint['mean_RSE'].idxmin()]
        cp_con_best = cp_linked(tensor1, tensor2, {}, A1, A2, ranks, sigma_constraint, args.cutoff, args.maxiter, [1], args.task)
        ranks_con_best = rank_identifier(cp_con_best['A1'], cp_con_best['A2'])

        # shared and individual simulation result
        if (args.tensor1 == 'None') & (args.tensor2 == 'None'):
            RSE_sigma = [rse(signal1, cp_sigma['estTensor1']), rse(signal2, cp_sigma['estTensor2'])]
            RSE_con = [rse(signal1, cp_con['estTensor1']), rse(signal2, cp_con['estTensor2'])]
            RSE_con_same = [rse(signal1, cp_con_same['estTensor1']), rse(signal2, cp_con_same['estTensor2'])]
            RSE_con_best = [rse(signal1, cp_con_best['estTensor1']), rse(signal2, cp_con_best['estTensor2'])]
            
            # shared structure
            shared_RSE_sigma = [rse(shared1, cp_sigma['estShared1']), rse(shared2, cp_sigma['estShared2'])]
            shared_RSE_con = [rse(shared1, cp_con['estShared1']), rse(shared2, cp_con['estShared2'])]
            shared_RSE_con_same = [rse(shared1, cp_con_same['estShared1']), rse(shared2, cp_con_same['estShared2'])]
            shared_RSE_con_best = [rse(shared1, cp_con_best['estShared1']), rse(shared2, cp_con_best['estShared2'])]

            # individual structure
            indiv_RSE_sigma = [rse(indiv1, cp_sigma['estIndiv1']), rse(indiv2, cp_sigma['estIndiv2'])]
            indiv_RSE_con = [rse(indiv1, cp_con['estIndiv1']), rse(indiv2, cp_con['estIndiv2'])]
            indiv_RSE_con_same = [rse(indiv1, cp_con_same['estIndiv1']), rse(indiv2, cp_con_same['estIndiv2'])]
            indiv_RSE_con_best = [rse(indiv1, cp_con_best['estIndiv1']), rse(indiv2, cp_con_best['estIndiv2'])]

            column_names = ['Task', 'RSE1', 'RSE1', 'shared_RSE1', 'shared_RSE2', 'indiv_RSE1', 'indiv_RSE2', 'sigma', 'rank']
            result_sigma = np.array([args.task] + RSE_sigma + shared_RSE_sigma + indiv_RSE_sigma + [sigma_1se] + ranks)
            result_con = np.array([args.task] + RSE_con + shared_RSE_con + indiv_RSE_con + [small_sigma] + ranks_con)
            result_con_same = np.array([args.task] + RSE_con_same + shared_RSE_con_same + indiv_RSE_con_same + [sigma_1se] + ranks_con_same)
            result_con_best = np.array([args.task] + RSE_con_best + shared_RSE_con_best + indiv_RSE_con_best + [sigma_constraint] + ranks_con_best)
            
            write_to_csv("Summary_complete_sigma.csv", np.round(result_sigma, 6), column_names)
            write_to_csv("Summary_complete_con.csv", np.round(result_con, 6), column_names)
            write_to_csv("Summary_complete_con_best.csv", np.round(result_con_best, 6), column_names)
            write_to_csv("Summary_complete_con_same.csv", np.round(result_con_same, 6), column_names)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help = 'Task ID')
    parser.add_argument("--action", type=str, required=True, choices = ['simulation', 'tuning', 'solving'], help = 'The operation to be performed.')
    parser.add_argument("--format", type=str, default='python', choices = ['python', 'matlab', 'both'], help = 'Ouput format for generated data.')

    # parameter for data generation
    parser.add_argument("--shape1", type=int, nargs = '+', help = 'shape of tensor 1')
    parser.add_argument("--shape2", type=int, nargs = '+', help = 'shape of tensor 2')
    parser.add_argument("--rank", type=int, nargs = '+', help = 'rank for the underlying signal')
    parser.add_argument("--snr", type=float, help = 'signal-to-noise ratio')
    parser.add_argument("--stand", type=str, default='False', help = 'standardization of tensor and data structures')
    
    
    #parameters for factor tuning
    parser.add_argument("--tensor1", type=str, default='None', help = 'name of tensor1 data')
    parser.add_argument("--tensor2", type=str, default='None', help = 'name of tensor2 data')
    parser.add_argument("--sigma_lower", type=float, help = 'lower bound of grid search')
    parser.add_argument("--sigma_upper", type=float, help = 'upper bound of grid search')
    parser.add_argument("--sigma_num", type=int, help = 'number of sigma values in grid search')
    parser.add_argument("--nfolds", type=int, default=10, help = 'number of folds to create in cross validation')
    parser.add_argument("--cprank", type=int, default=20, help = 'rank used to solve cp decomposition')
    parser.add_argument("--sigma", type=float, default=0, help = 'penalty factor')
    parser.add_argument("--cutoff", type=float, default=0.00001, help = 'stopping criteria')
    parser.add_argument("--maxiter", type=int, default=300, help = 'maximum number of iteration')

    args = parser.parse_args()
    if args.action == 'simulation':
        if not all([args.shape1, args.shape2, args.rank, args.snr]):
            parser.error("When action is 'simulation', --shape1, --shape2, --rank, and --snr are required.")
    elif args.action == 'tuning':
        if any(param is None for param in [args.sigma_lower, args.sigma_upper, args.sigma_num]):
            parser.error("When action is 'tuning', --sigma_lower, --sigma_upper, and --sigma_num are required.")
    
    main(args)
