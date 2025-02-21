from MULTIFAC_functions import *

# Function to write to the CSV file
def write_to_csv(csv_file_path, data, column_names):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline='') as file:
        csvwriter = csv.writer(file)
        # Write column names if the file is new
        if not file_exists:
            csvwriter.writerow(column_names)
        
        csvwriter.writerow(data)

def main(args):
    if args.action=='simulation': #simulate couple tensors and calculate initial factor matrices
        simu = simu_linked(args.shape1, args.shape2, args.rank, args.snr, args.task)
        simu['tensor1_true'] = simu['tensor1'].copy()
        simu['tensor2_true'] = simu['tensor2'].copy()
        np.random.seed(args.task)
        # randomly select approx 5% of the tensor slice to be missing
        sample_Ind1 = np.random.choice(np.arange(simu['tensor1'].shape[0]), size = np.ceil(0.05 * simu['tensor1'].shape[0]).astype(int), replace = False)
        sample_Ind2 = np.random.choice(np.arange(simu['tensor2'].shape[0]), size = np.ceil(0.05 * simu['tensor2'].shape[0]).astype(int), replace = False)
        slice_Ind1 = np.zeros(simu['tensor1'].shape, dtype=bool)
        slice_Ind2 = np.zeros(simu['tensor2'].shape, dtype=bool)
        slice_Ind1[sample_Ind1, :] = True
        slice_Ind2[sample_Ind2, :] = True

        # Randomly select 5% of the elements from the non-missing part
        remaining_Ind1 = np.argwhere(~slice_Ind1)
        remaining_Ind2 = np.argwhere(~slice_Ind2)

        entry_Ind1 = np.zeros(simu['tensor1'].shape, dtype=bool)
        entry_Ind2 = np.zeros(simu['tensor2'].shape, dtype=bool)

        # Select 5% of the remaining elements to be missing
        selected_entries_tens1 = remaining_Ind1[np.random.choice(remaining_Ind1.shape[0], size=np.sum(slice_Ind1), replace=False)]
        for entry in selected_entries_tens1:
            entry_Ind1[tuple(entry)] = True

        selected_entries_tens2 = remaining_Ind2[np.random.choice(remaining_Ind2.shape[0], size=np.sum(slice_Ind2), replace=False)]
        for entry in selected_entries_tens2:
            entry_Ind2[tuple(entry)] = True

        nanInd1 = entry_Ind1 | slice_Ind1
        nanInd2 = entry_Ind2 | slice_Ind2

        simu['tensor1'][nanInd1] = np.nan
        simu['tensor2'][nanInd2] = np.nan
        simu['nanInd1'] = nanInd1
        simu['nanInd2'] = nanInd2
        simu['entry_Ind1'] = entry_Ind1
        simu['entry_Ind2'] = entry_Ind2
        simu['slice_Ind1'] = slice_Ind1
        simu['slice_Ind2'] = slice_Ind2
        
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
                tensor1_true = simu['tensor1_true']
                tensor2_true = simu['tensor2_true']
                nanInd1 = simu['nanInd1']
                nanInd2 = simu['nanInd2']
                entry_Ind1 = simu['entry_Ind1']
                entry_Ind2 = simu['entry_Ind2']
                slice_Ind1 = simu['slice_Ind1']
                slice_Ind2 = simu['slice_Ind2']

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

        # cp constraint: use same sigma
        cp_con_same = cp_linked(tensor1, tensor2, {}, A1, A2, ranks, sigma_1se, args.cutoff, args.maxiter, [1], args.task)

        df_constraint = pd.read_csv(score_file_name_constraint, sep=',')
        sigma_constraint = df_constraint['sigma'][df_constraint['mean_RSE'].idxmin()]
        cp_con_best = cp_linked(tensor1, tensor2, {}, A1, A2, ranks, sigma_constraint, args.cutoff, args.maxiter, [1], args.task)
        ranks_con_best = rank_identifier(cp_con_best['A1'], cp_con_best['A2'])

        final_result = {'A1': cp_con_best['A1'], 'A2': cp_con_best['A2'], 'estTensor1': cp_con_best['estTensor1'], 'estTensor2': cp_con_best['estTensor2'], 'estShared1': cp_con_best['estShared1'], 'estShared2': cp_con_best['estShared2'], 'estIndiv1': cp_con_best['estIndiv1'], 'estIndiv2': cp_con_best['estIndiv2'], 'ranks': ranks_con_best}
        with open("MULTIFAC_result.pkl", "wb") as f:
            pickle.dump(final_result, f)
        
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
