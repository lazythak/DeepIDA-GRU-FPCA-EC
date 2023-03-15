# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
from HelperFunctionsForMain import *
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import csv

def main():
    # prepare HostTx Data: No week list because all the data was collected in the same week
    filename = 'hosttxcounts_normalized_lmm.csv'  # Uncomment for HostTxCounts
    data, sample_names, gene_names, gene_df = prepareGeneExpression(filename)
    dic, label_dict = prepareTranscriptomicsSummary('hmp2_metadata.csv')  # Uncomment for HostTxCounts
    subject_data0, subject_label0, names0 = CombineDataGeneExpression(data, label_dict, sample_names,
                                                                      dic)  # Uncomment for HostTxCounts


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

    with open('parrot_metagenome.pkl', 'rb') as f:
        names6, subject_data6, subject_label6, week_list6 = pickle.load(f)


    # List of subjects which are common to all the datasets
    common_names = sorted(list(set(names0) & set(names5) & set(names3)))

    # Prepare Clinical Data
    meta_data = pd.read_csv('hmp2_metadata.csv')
    nf = 3  # Consent Age, Antibiotic Use, Site Name
    Xf = torch.zeros(nf, len(common_names))
    for j in range(len(common_names)):
        l = len(meta_data['Participant ID'].to_list())
        for i in range(l):
            if common_names[j] == meta_data['Participant ID'].to_list()[i]:
                Xf[0, j] = meta_data['consent_age'].to_list()[i]
                if meta_data['Antibiotics'].to_list()[i] == 'No':
                    Xf[1, j] = 0
                else:
                    Xf[1, j] = 1
                if meta_data['site_name'].to_list()[i] == 'Cedars-Sinai':
                    Xf[2, j] = 0
                elif meta_data['site_name'].to_list()[i] == 'Emory':
                    Xf[2, j] = 1
                elif meta_data['site_name'].to_list()[i] == 'MGH Pediatrics':
                    Xf[2, j] = 2
                elif meta_data['site_name'].to_list()[i] == 'Cincinnati':
                    Xf[2, j] = 3
                else:
                    Xf[2, j] = 4
                break
    Xf[0, 20] = 0



    # For metabolomics and metagenomics, if you want to visualize which weeks the data was collected in, uncomment the following two lines.
    # plotTheWeeks(names3, week_list3, common_names)
    # plotTheWeeks(names5, week_list5, common_names)


    # For metabolomics and metagenomics, each subject has different set of week-points the data was collected in. The following two lines make time-points uniform by imputing the data.
    new_subject_data5 = ReduceToCommonTimeFrame(subject_data5, week_list5, period=5)
    new_subject_data3 = ReduceToCommonTimeFrame(subject_data3, week_list3, period=5)
    new_subject_data6 = ReduceToCommonTimeFrame(subject_data6, week_list6, period=2)


    # Conversion of HostTx, Metabolomics and Metagenomics data into tensor form by doing fPCA, EC, mean or nothing
    X0, y0, fpcplots0 = transformDataToTensor(names0, subject_data0, subject_label0, common_names, 'mean', 1)  # P x N
    X3, y3, fpcplots3 = transformDataToTensor(names3, new_subject_data3, subject_label3, common_names, 'nothing', 1)  # P x T x N
    X5, y5, fpcplots5 = transformDataToTensor(names5, new_subject_data5, subject_label5, common_names, 'nothing', 1)  # P x T x N
    X6, y6, fpcplots6 = transformDataToTensor(names6, new_subject_data6, subject_label6, common_names, 'EC',1)  # P x T x N

    # with open('JPTA_exampleDataset1.npy', 'wb') as f:
    #     np.save(f, X3.reshape(X3.shape[2],X3.shape[0],X3.shape[1]).numpy())
    # with open('JPTA_exampleDataset2.npy', 'wb') as f:
    #     np.save(f, X6.reshape(X6.shape[2],X6.shape[0],X6.shape[1]).numpy())
    #
    # metabol_JPTA_ind = genfromtxt("metabol_JPTAFeatures.csv", delimiter=',')
    # metagenom_JPTA_ind = genfromtxt("metagenom_JPTAFeatures.csv", delimiter=',')
    # X6_jpta = X6[metagenom_JPTA_ind].T
    # X3_jpta = X3[metabol_JPTA_ind].T

    # 1-fold testing
    avg1 = torch.zeros(2)
    avg2 = torch.zeros(3)
    avg3 = torch.zeros(3)
    X0 = X0.T
    X3 = X3.T
    X5 = X5.T
    Xf = Xf.T

    i = 0
    A = 0
    while i <= 89:
        mask = np.ones(90, dtype=bool)
        mask[i] = 0
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X0[mask, :], y0[mask], stratify=y0[mask],
                                                                test_size=0.015, random_state=0)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3[mask, :], y0[mask], stratify=y0[mask],
                                                                test_size=0.015, random_state=0)
        X_train5, X_test5, y_train5, y_test5 = train_test_split(X5[mask, :], y0[mask], stratify=y0[mask],
                                                                test_size=0.015, random_state=0)
        X_trainf, X_testf, y_trainf, y_testf = train_test_split(Xf[mask, :], y0[mask], stratify=y0[mask],
                                                                test_size=0.015, random_state=0)


        # Uncomment the following paragraph for bootstrap version of DeepIDA
        var_names0 = np.array(range(X0.shape[1]))
        var_names3 = np.array(range(X3.shape[1]))
        var_names5 = np.array(range(X5.shape[1]))
        var_namesf = np.array(range(Xf.shape[1]))
        variables_name_list = [pd.Index(['View1Var%d' % i for i in range(X0.shape[1])]),
                               pd.Index(['View2Var%d' % i for i in range(X3.shape[1])]),
                               pd.Index(['View3Var%d' % i for i in range(X5.shape[1])]),
                               pd.Index(['View4Var%d' % i for i in range(Xf.shape[1])])]

        results = DeepIDA_Bootstrap([X0[mask], X3[mask], X5[mask]],
                                    [X_test0, X_test3, X_test5],
                                    [X0[i : i + 1, :], X3[i : i + 1, :], X5[i:i + 1, :]],
                                    y0[mask], y_test0, y0[i:i + 1],
                                    [[200, 100, 20], [200, 100, 20], [200, 100, 20]], 20, 20, variables_name_list,
                                    [5, 5, 5, 5], [0, 1, 1], [3, 256, 20], 0.0001, 60)
        return

        try:

            # Uncomment the following paragraph for bootstrap version.
            results = DeepIDA_Bootstrap([X0[mask],X3[mask], X5[mask]],
                                        [X_test0, X_test3, X_test5],
                                        [X0[i:i+1, :],X3[i:i+1, :], X5[i:i+1, :]],
                                        y0[mask], y_test0, y0[i:i + 1],
                                        [[200, 100, 20], [200, 100, 20], [200, 100, 20]], 40,20, variables_name_list,
                                        [5,5,5,5],[0, 1, 1], [3,256,20], 0.0001, 60)

            # Cross-sectional Data: NxP and Longitudinal Data: NxTxP
            # results = DeepIDA_nonBootstrap([X0[mask], X3[mask], X5[mask]],
            #                                [X_test0, X_test3, X_test5],
            #                                [X0[i:i+1, :], X3[i:i+1, :], X5[i:i+1, :]],
            #                                y0[mask], y_test0, y0[i:i+1],
            #                                [[256, 64, 20], [256, 64, 20], [256, 64, 20]],
            #                                [5, 5, 5, 5], [0, 0, 0], [3, 256, 20], 0.0001, 80)
            pass
        except:
            print("Error occured")
            continue
        else:

            with open('results_test_' + str(i) + '.pkl', 'wb') as f:
                pickle.dump(results, f)

            A = A + results[1]
            print(A, " out of ", i + 1)
            i = i + 1



if __name__ == "__main__":
    main()