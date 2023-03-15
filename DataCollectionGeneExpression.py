# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
from HelperFunctionsForMain import *

def main():

    #prepare HostTx Data
    filename = 'hosttxcounts_normalized_lmm.csv'  #Uncomment for HostTxCounts
    data, sample_names, gene_names, gene_df = prepareGeneExpression(filename)
    dic, label_dict = prepareTranscriptomicsSummary('hmp2_metadata.csv')  # Uncomment for HostTxCounts
    subject_data0, subject_label0, names0 = CombineDataGeneExpression(data, label_dict, sample_names,dic)  # Uncomment for HostTxCounts

    # # filename = 'metabolomics_hilicneg.csv'
    # data, sample_names, gene_names, gene_df = prepareGeneExpression(filename)
    # dic,label_dict = prepareTranscriptomicsSummary('hmp2_metadata.csv')  #Uncomment for HostTxCounts
    # # dic, label_dict, num = prepareMetabolomicsSummary('Book1.csv')
    # # print(dic)
    #
    # subject_data0, subject_label0, names0 = CombineDataGeneExpression(data, label_dict, sample_names, dic) #Uncomment for HostTxCounts
    # # subject_data, subject_label, names = CombineDataMetabolomics(data, label_dict, sample_names, dic,num)
    # with open('parrot_txc.pkl', 'rb') as f:
    #     names0, subject_data0, subject_label0 = pickle.load(f)


    # Prepare Metabolomics data: First run DataCollectionMetabolomics.py
    with open('parrot_metac18neg.pkl', 'rb') as f:
        names1, subject_data1, subject_label1 = pickle.load(f)
    with open('parrot_metac18pos.pkl', 'rb') as f:
        names2, subject_data2, subject_label2 = pickle.load(f)
    with open('parrot_metahilicneg.pkl', 'rb') as f:
        names3, subject_data3, subject_label3 = pickle.load(f)
    with open('parrot_metahilicpos.pkl', 'rb') as f:
        names4, subject_data4, subject_label4 = pickle.load(f)
    week_num = returns_week_nums_metabolomics('hmp2_metadata.csv')
    week_list3 = CalculateWeeks(subject_data3, week_num)

    # Prepare Metagenomics data: First run DataCollectionMetaGenomics.py
    with open('parrot_metagenome_lmm.pkl', 'rb') as f:
        names5, subject_data5, subject_label5, week_list5 = pickle.load(f)


    # List of subjects which are common to all the datasets
    common_names = sorted(list(set(names0) & set(names5) & set(names3)))

    plotTheWeeks(names5, week_list, common_names)
    #
    # week_num = returns_week_nums_metabolomics('hmp2_metadata.csv')
    # week_list = CalculateWeeks(subject_data3, week_num)
    # plotTheWeeks(names3, week_list, common_names)

    new_subject_data5 = ReduceToCommonTimeFrame(subject_data5, week_list, period=2)

    # plotFPCA(new_subject_data5, common_names, names5, subject_label5)

    index = -10
    nc= 1

    X0, y0, fpcplots0 = transformDataToTensor(names0, subject_data0, subject_label0, common_names, 'mean', nc)


    # X0, y0, fpcplots0,index0 = FindImportantFeatures(names0, subject_data0, subject_label0, common_names, index)
    # X1, y1, fpcplots1 = transformDataToTensor(names1, subject_data1, subject_label1, common_names, 'mean', 1)
    # X2, y2, fpcplots2 = transformDataToTensor(names2, subject_data2, subject_label2, common_names,'mean',1)
    X3, y3, fpcplots3 = transformDataToTensor(names3, subject_data3, subject_label3, common_names,'nothing', 1)
    X4, y4, fpcplots4 = transformDataToTensor(names4, subject_data4, subject_label4, common_names,'nothing', 2)
    X5, y5, fpcplots5 = transformDataToTensor(names5, new_subject_data5, subject_label5, common_names,'nothing', 1)

    print(X0.shape)
    print(X3.shape)
    print(X5.T.shape)





    # X5,y5,fpcplots5, index5 = FindImportantFeaturesUsingFrechet(names5, new_subject_data5, subject_label5, common_names, nc, index)
    print(common_names)
    meta_data = pd.read_csv('hmp2_metadata.csv')
    nf = 3      #Number of non-longitudinal data
    Xf = torch.zeros(nf, len(common_names))
    for j in range(len(common_names)):
        l = len(meta_data['Participant ID'].to_list())
        for i in range(l):
            if common_names[j] == meta_data['Participant ID'].to_list()[i]:
                Xf[0,j] = meta_data['consent_age'].to_list()[i]
                if meta_data['Antibiotics'].to_list()[i] == 'No':
                    Xf[1,j] = 0
                else:
                    Xf[1,j] = 1
                if meta_data['site_name'].to_list()[i] == 'Cedars-Sinai':
                    Xf[2,j] = 0
                elif meta_data['site_name'].to_list()[i] == 'Emory':
                    Xf[2,j] = 1
                elif meta_data['site_name'].to_list()[i] == 'MGH Pediatrics':
                    Xf[2,j] = 2
                elif meta_data['site_name'].to_list()[i] == 'Cincinnati':
                    Xf[2,j] = 3
                else:
                    Xf[2,j] = 4
                break
    Xf[0,20]=0
    # print(Xf)
    # return



    avg = torch.zeros(2)
    max_iter = 10
    rs = 7
    # for rs in range(max_iter):
    #     indices_train, indices_test, y_train0, y_test0 = train_test_split(range(90), y0, stratify=y0, test_size=0.2, random_state = rs)
    #
    X_train0, X_test0, y_train0, y_test0 = train_test_split(X0.T, y0, stratify=y0, test_size=0.2, random_state = rs)
    # X_train1, X_test1, y_train1, y_test1 = train_test_split(X1.T, y0, stratify=y0, test_size=0.2, random_state = rs)
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X2.T, y0, stratify=y0, test_size=0.2, random_state = rs)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3.T, y0, stratify=y0, test_size=0.2, random_state = rs)
    X_train4, X_test4, y_train4, y_test4 = train_test_split(X4.T, y0, stratify=y0, test_size=0.2, random_state = rs)
    X_train5, X_test5, y_train5, y_test5 = train_test_split(X5.T, y0, stratify=y0, test_size=0.2, random_state = rs)
    #     xx1 = tuple(slice(x) for x in indices_train)
    #     xx2 = tuple(slice(x) for x in indices_test)
    #
    #     X_train5, y_train5, fpcplots5, index5, fpcs = FindImportantFeaturesUsingFrechet(
    #         [names5[i] for i in indices_train], [new_subject_data5[i] for i in indices_train],
    #         [subject_label5[i] for i in indices_train],
    #         common_names, nc, [], index)
    #
    #     X_test5, y_test5, fpcplots5, index5, fpcs = FindImportantFeaturesUsingFrechet([names5[i] for i in indices_test],
    #                                                                                   [new_subject_data5[i] for i in
    #                                                                                    indices_test],
    #                                                                                   [subject_label5[i] for i in
    #                                                                                    indices_test],
    #                                                                                   common_names, nc, fpcs, index5)
    #
    #     X_train0, y_train0, fpcplots0, index0 = FindImportantFeatures([names0[i] for i in indices_train] , [subject_data0[i] for i in indices_train], [subject_label0[i] for i in indices_train], common_names, index)
    #     X_test0, y_test0, fpcplots0, index0 = FindImportantFeatures([names0[i] for i in indices_test], [subject_data0[i] for i in indices_test], [subject_label0[i] for i in indices_test], common_names, index0)
    #
    #
    #
    #     print(X_test5)
    # results = DeepIDA_nonBootstrap([X_train0,  X_train3, X_train5],
    #                                    [X_test0[0:8, :], X_test3[0:8, :], X_test5[0:8, :]],
    #                                    [X_test0[8:17, :], X_test3[8:17, :], X_test5[8:17, :]],
    #                                    y_train0, y_test0[0:8], y_test0[8:17],
    #                                    [[256, 64, 20], [256, 64, 20]],
    #                                    [5, 5, 5, 5], [0, 1, 1], [2, 256, 20], 0.0001, 100)
    # print(results)

    # print(results[0:6])
    #     #[200,100,20],[200,100,20] 0.01
    #
    #     print(y_test0[8:17])
    #     print('The result is ', results[0:6])
    #     avg[0] = avg[0] + results[0]
    #     avg[1] = avg[1] + results[1]
    #
    #
    # print(avg/max_iter)
    aa = 0
    avg1 = torch.zeros(2)
    avg2 = torch.zeros(3)
    avg3 = torch.zeros(3)
    X0 = X0.T
    X3 = X3.T
    X5 = X5.T
    Xf = Xf.T
    print(y0)
    i=0
    while i <= 89:
        mask = np.ones(90, dtype=bool)
        mask[i] = 0
        # mask[i+1] = 0
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X0[mask,:], y0[mask], stratify=y0[mask], test_size=0.05, random_state=0)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3[mask,:], y0[mask], stratify=y0[mask], test_size=0.05, random_state=0)
        X_train5, X_test5, y_train5, y_test5 = train_test_split(X5[mask, :], y0[mask], stratify=y0[mask], test_size=0.05, random_state=0)
        X_trainf, X_testf, y_trainf, y_testf = train_test_split(Xf[mask, :], y0[mask], stratify=y0[mask], test_size=0.05, random_state=0)
        # results = DeepIDA_nonBootstrap([X0[mask], X5[mask], Xf[mask]],
        #                                [X_test0, X_test5, X_testf],
        #                                [X0[i:i+1, :], X5[i:i+1, :], Xf[i:i+1,:]],
        #                                y0[mask], y_test0, y0[i:i+1],
        #                                [[256, 64, 20], [256, 64, 20], [256,64,20]],
        #                                [5, 5,5], lr_rate=0.001, n_epoch=60)
        var_names0 = np.array(range(X0.shape[1]))
        var_names3 = np.array(range(X3.shape[1]))
        var_names5 = np.array(range(X5.shape[1]))
        var_namesf = np.array(range(Xf.shape[1]))
        variables_name_list = [pd.Index(['View1Var%d' % i for i in range(X0.shape[1])]), pd.Index(['View2Var%d' % i for i in range(X3.shape[1])]),
                               pd.Index(['View3Var%d' % i for i in range(X5.shape[1])]), pd.Index(['View4Var%d' % i for i in range(Xf.shape[1])])]

        print(X5[mask].shape)


        try:
            # results = DeepIDA_Bootstrap([X0[mask],X3[mask], X5[mask]],
            #                             [X_test0, X_test3, X_test5],
            #                             [X0[i:i+1, :],X3[i:i+1, :], X5[i:i+1, :]],
            #                             y0[mask], y_test0, y0[i:i + 1],
            #                             [[200, 100, 20], [200, 100, 20], [200, 100, 20]], 40,20, variables_name_list,
            #                             [5,5,5,5], 'test'+str(i)+'_view')
            # results = DeepIDA_nonBootstrap([X_train0, X_train3, X_train5],
            #                                [X_test0[0:8, :], X_test3[0:8, :], X_test5[0:8, :]],
            #                                [X_test0[8:17, :], X_test3[8:17, :], X_test5[8:17, :]],
            #                                y_train0, y_test0[0:8], y_test0[8:17],
            #                                [[256, 64, 20], [256, 64, 20]],
            #                                [5, 5, 5, 5], [0, 1, 1], [2, 256, 20], 0.0001, 100)
            results = DeepIDA_nonBootstrap([X0[mask], X3[mask], X5[mask]],
                                           [X_test0, X_test3, X_test5],
                                           [X0[i:i + 1, :], X3[i:i + 1, :], X5[i:i + 1, :]],
                                           y0[mask], y_test0, y0[i:i + 1],
                                           [[256, 64, 20], [256, 64, 20]],
                                           [5, 5, 5, 5], [0, 1, 1], [2, 256, 20], 0.0001, 100)
            pass;
        except:
            print("Error occured")
            continue
        else:


            # avg2 = avg2 + torch.tensor(results[4])
            # avg3 = avg3 + torch.tensor(results[5])
            with open('results_test_'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(results[0:2])
            aa = aa+results[1]
            print(aa, " out of ", i+1)
            i=i+1
            # print(avg1, " in ", i, " iterations ")
            # print(avg2, " in ", i, " iterations ")
            # print(avg3, " in ", i, " iterations ")
    # print(avg1)
    # print(avg2)
    # print(avg3)
if __name__ == "__main__":
    main()