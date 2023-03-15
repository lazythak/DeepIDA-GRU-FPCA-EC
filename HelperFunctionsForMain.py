# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import skfda
from skfda import FDataGrid
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
from main_functions import DeepIDA_nonBootstrap, DeepIDA_Bootstrap
import matplotlib.pyplot as plt
from EC import ECurve, ECtransform
import csv
# from scipy.stats import permutation_test





def prepareGeneExpression(filename):
    print('Preparing GeneExpression Data')

    # meta_df_AN001514 = pd.read_csv('GSE111889_host_tx_counts.tsv', sep='\t')
    # print(meta_df_AN001514)

    gene_df = pd.read_csv(filename)
    # print(gene_df)
    x = float('NaN')
    gene_df.replace(x, -999)
    gene_df = gene_df.fillna(-999)
    # print(gene_df)
    sample_id = []
    count = 0

    for cols in gene_df.columns:
        if count == 0:
            gene_names = gene_df[cols].to_list()[0:]
            count += 1
            pass
        else:
            sample_id.append(cols)
    gene_data = []
    for sid in sample_id:
        gene_data.append(gene_df[sid].to_list())
    # print(meta_data[0][0])

    # label = torch.zeros(len(sample_id))
    data = torch.zeros(len(gene_data[0]), len(sample_id))
    # print(data.shape)
    # print(meta_data[0][1:len(meta_data[0])])
    # print(type(meta_data[0][-3]))

    # print(list(map(float, meta_data[1][1:100])))
    for i in range(len(sample_id)):
        # x = gene_data[i][0]
        data[:, i] = torch.FloatTensor(list(map(float, gene_data[i][0:len(gene_data[0])])))
    # print(label)
    # print(data.shape)
    # print(len(metabolite_names))
    return data, sample_id, gene_names, gene_df



def prepareTranscriptomicsSummary(filename):
    dff = pd.read_csv(filename)
    subject_name = []
    sample_name = []
    label = []
    for i in range(len(dff['data_type'].to_list())):
        if dff['data_type'].to_list()[i] == 'host_transcriptomics':
            subject_name.append(dff['Participant ID'].to_list()[i])
            sample_name.append(dff['External ID'].to_list()[i])
            if dff['diagnosis'].to_list()[i] == 'CD':
                label.append(0)
            elif dff['diagnosis'].to_list()[i] == 'UC':
                label.append(1)
            else:
                label.append(2)



    res = dict(zip(sample_name, subject_name))
    label_dict = dict(zip(sample_name, label))
    return res,label_dict


def prepareMetabolomicsSummary(filename):
    dff = pd.read_csv(filename)
    subject_name = []
    sample_name = []
    label = []
    for i in range(len(dff['Sample name'].to_list())):
        if True:
            subject_name.append(dff['Subject name'].to_list()[i])
            sample_name.append(dff['Sample name'].to_list()[i])
            if dff['Diagnosis'].to_list()[i] == 'CD':
                label.append(0)
            elif dff['Diagnosis'].to_list()[i] == 'UC':
                label.append(1)
            else:
                label.append(2)

    num = np.unique(subject_name)
    res = dict(zip(sample_name, subject_name))
    label_dict = dict(zip(sample_name, label))
    return res,label_dict, num

def CombineDataGeneExpression(data, label_dict, sample_names, dic):
    label = []
    for i in range(len(sample_names)):
        label.append(label_dict[sample_names[i]])
        sample_names[i] = dic[sample_names[i]]

    names, nums = np.unique(sample_names,return_counts=True)
    # print(names)
    # print(nums)
    data_bysubjectname = []
    # print(data[0].shape)
    subject_label = []
    for name in names:
        lst = []

        for i in range(len(sample_names)):
            sample_name = sample_names[i]

            if sample_name == name:
                lst.append(data[:,i])
                ll = label[i]
        x = lst[0].reshape(-1,1)
        for j in range(1,len(lst)):
            x = torch.cat([x,lst[j].reshape(-1,1)], axis=1)
        data_bysubjectname.append(x)
        subject_label.append(ll)
    return data_bysubjectname,torch.Tensor(subject_label),names


def CombineDataMetabolomics(data, label, sample_names, dic, num):
    ss = []
    for i in range(len(sample_names)):
        ss.append(sample_names[i])
    for i in range(len(sample_names)):
        sample_names[i] = dic[sample_names[i].replace('.','-')]
    names, nums = np.unique(sample_names,return_counts=True)
    # print(names)
    # print(nums)
    data_bysubjectname = []
    # print(data[0].shape)
    subject_label = []
    for name in names:
        lst = []

        for i in range(len(sample_names)):
            sample_name = sample_names[i]

            if sample_name == name:
                lst.append(data[:,i])
                ll = label[ss[i].replace('.','-')]
        x = lst[0].reshape(-1,1)
        for j in range(1,len(lst)):
            x = torch.cat([x,lst[j].reshape(-1,1)], axis=1)

        data_bysubjectname.append(x)
        subject_label.append(ll)
    return data_bysubjectname,torch.Tensor(subject_label),names


def transformDataToTensor(names, subject_data, subject_label, common_names, method, nc=2):
    fpcplots = []
    if method == 'nothing':
        max_time = 0
        for i in range(len(subject_data)):
            if subject_data[i].shape[1]>max_time:
                max_time = subject_data[i].shape[1]

        X = torch.zeros((subject_data[0].shape[0],max_time,len(common_names)))
        y = torch.zeros(len(common_names))
        count = 0

        for i in range(len(names)):
            if names[i] in common_names:
                t = subject_data[i].shape[1]
                X[:,0:t,count] = subject_data[i]
                if subject_label[i] == 2:
                    y[count] = 0
                else:
                    y[count] = 1
                count = count+1
    elif method == 'mean':
        for i in range(len(subject_data)):
            subject_data[i] = torch.mean(subject_data[i],1)
        X = torch.zeros((subject_data[0].shape[0], len(common_names)))
        y = torch.zeros(len(common_names))

        count = 0
        for i in range(len(names)):
            if names[i] in common_names:
                X[:, count] = subject_data[i]
                if subject_label[i] == 2:
                    y[count] = 0
                else:
                    y[count] = 1
                count = count+1
    elif method == 'fpca':
        # functional representation of data
        X = torch.zeros((nc*subject_data[0].shape[0], len(common_names)))
        # print(X.shape)
        y = torch.zeros(len(common_names))
        count = 0
        fpcplots = []
        for i in range(len(subject_data)):
            if names[i] in common_names:
                names[i]
                x = subject_data[i].T
                # print(x.shape)
                fd = FDataGrid(x, range(x.shape[1]),
                               dataset_name='Time Series',
                               argument_names=['t'],
                               coordinate_names=['x(t)'])

                # fpca computation

                fpca_discretized = FPCA(n_components=nc)
                fpca_discretized.fit(fd)
                h = fpca_discretized.components_
                fpcplots.append(h)
                X[:, count] = torch.tensor(h.data_matrix.reshape(nc * x.shape[1], )).T
                count = count + 1
        count = 0
        for i in range(len(names)):
            if names[i] in common_names:
                if subject_label[i] == 2:
                    y[count] = 2
                elif subject_label[i] == 1:
                    y[count] = 1
                else:
                    y[count] = 0
                count = count + 1

    elif method == 'fpca_trans':
        # min_time = 100
        # for i in range(len(subject_data)):
        #     if names[i] in common_names:
        #         tt = subject_data[i].shape[1]
        #         if tt<min_time:
        #             min_time = tt

        count = 0
        X = torch.zeros((100, len(common_names)))
        y = torch.zeros(len(common_names))
        for i in range(len(subject_data)):
            if names[i] in common_names:
                x = subject_data[i]
                # print(x.shape)
                fd = FDataGrid(x, range(x.shape[1]),
                               dataset_name='Time Series',
                               argument_names=['t'],
                               coordinate_names=['x(t)'])

                # fpca computation

                fpca_discretized = FPCA(n_components=nc)
                fpca_discretized.fit(fd)
                h = fpca_discretized.components_
                fpcplots.append(h)
                X[0:nc * len(h.data_matrix[0]), count] = torch.tensor(h.data_matrix.reshape(nc * len(h.data_matrix[0]), )).T
                if subject_label[i] == 2:
                    y[count] = 2
                elif subject_label[i] == 1:
                    y[count] = 1
                else:
                    y[count] = 0
                count = count+1


    elif method == 'fpca_new':
        # functional representation of data
        X = torch.zeros((nc * subject_data[0].shape[0], len(common_names)))
        # print(X.shape)
        y = torch.zeros(len(common_names))
        count = 0
        fpcplots = []
        # print(subject_data)
        for i in range(subject_data[0].shape[0]):
            x = torch.zeros(len(common_names),subject_data[0].shape[1])
            ff = 0
            for sid in range(len(subject_data)):
                if names[sid] in common_names:
                    x[ff,:] = subject_data[sid][i,:]
                    ff = ff+1
            fd = FDataGrid(x, range(x.shape[1]),
                            dataset_name='Time Series',
                            argument_names=['t'],
                            coordinate_names=['x(t)'])

            # fpca computation

            fpca_discretized = FPCA(n_components=nc)
            fpca_discretized.fit(fd)
            h = fpca_discretized.transform(fd)
            for j in range(nc):
                X[i*nc+j,:] = torch.tensor(h[:,j].T)


        count = 0
        for i in range(len(names)):
            if names[i] in common_names:
                if subject_label[i] == 2:
                    y[count] = 2
                elif subject_label[i] == 1:
                    y[count] = 1
                else:
                    y[count] = 0
                count = count + 1

    elif method == 'EC':
        # functional representation of data
        np = 50
        X = torch.zeros((np+1, len(common_names)))
        # print(X.shape)
        y = torch.zeros(len(common_names))
        count = 0
        for i in range(len(subject_data)):
            if names[i] in common_names:
                from nilearn.connectome import ConnectivityMeasure
                tangent_measure = ConnectivityMeasure(kind="correlation", vectorize=False)
                x = subject_data[i]
                yy = x.T.numpy().reshape(1,x.T.shape[0],x.T.shape[1])
                M = tangent_measure.fit_transform(yy)
                h = ECurve(M, np)
                # print(ECurve(cov(x.T),np))
                X[:, count] = torch.tensor(h)
                count = count + 1
        count = 0
        for i in range(len(names)):
            if names[i] in common_names:
                if subject_label[i] == 2:
                    y[count] = 0
                else:
                    y[count] = 1
                count = count + 1
    else:
        X = 0
        y = 0
    return X,y,fpcplots


def FindImportantFeatures(names, subject_data, subject_label, common_names, index):
    fpcplots = []
    X = torch.zeros((subject_data[0].shape[0], len(subject_data)))
    print(X.shape)
    print(subject_data[0].shape)
    y = torch.zeros(len(subject_data))
    count = 0

    for i in range(len(names)):
        if names[i] in common_names:
            X[:, count] = subject_data[i]
            if subject_label[i] == 2:
                y[count] = 0
            else:
                y[count] = 1
            count = count + 1

    if index != -10:
        return X[index, :], y, fpcplots, index

    healthy_index = []
    disease_index = []
    for i in range(len(y)):
        if y[i] == 0:
            healthy_index.append(i)
        else:
            disease_index.append(i)

    from mlxtend.evaluate import permutation_test
    pvalue = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if i%100 == 0:
            print(i)
        pvalue[i] = permutation_test(X[i,healthy_index].tolist(),X[i,disease_index].tolist(),method='approximate', num_rounds=100,seed=0)
    #     print(i)
    #     plt.scatter(i,pvalue[i])
    # plt.show()

    # with open('pvalue_transcriptomics.pkl', 'wb') as f:
    #     pickle.dump(pvalue, f)

    # with open('pvalue_transcriptomics.pkl', 'rb') as f:
    #     pvalue = pickle.load(f)
    filter = []
    for i in range(len(pvalue)):
        if pvalue[i]<=0.05:
            filter.append(i)


    return X[filter, :], y, fpcplots, filter


def returns_week_nums_metabolomics(filename):
    dff = pd.read_csv(filename)
    subject_name = []
    sample_name = []
    week_num = []
    label = []
    print(len(dff['data_type'].to_list()))
    for i in range(len(dff['data_type'].to_list())):
        if dff['data_type'].to_list()[i] == 'metabolomics':
            subject_name.append(dff['Participant ID'].to_list()[i])
            sample_name.append(dff['External ID'].to_list()[i])
            week_num.append(dff['week_num'].to_list()[i])
            if dff['diagnosis'].to_list()[i] == 'CD':
                label.append(0)
            elif dff['diagnosis'].to_list()[i] == 'UC':
                label.append(1)
            else:
                label.append(2)

    print(sample_name)

    res = dict(zip(sample_name, subject_name))
    label_dict = dict(zip(sample_name, label))
    return week_num


def FindImportantFeaturesUsingFPCA(names, subject_data, subject_label, common_names,nc):
    X = torch.zeros((nc * subject_data[0].shape[0], len(subject_data)))
    # print(X.shape)
    y = torch.zeros(len(subject_data))
    count = 0
    fpcplots = []
    # print(subject_data)
    for i in range(subject_data[0].shape[0]):
        x = torch.zeros(len(common_names), subject_data[0].shape[1])
        ff = 0
        for sid in range(len(subject_data)):
            if names[sid] in common_names:
                x[ff, :] = subject_data[sid][i, :]
                ff = ff + 1
        fd = FDataGrid(x, range(x.shape[1]),
                       dataset_name='Time Series',
                       argument_names=['t'],
                       coordinate_names=['x(t)'])

        # fpca computation

        fpca_discretized = FPCA(n_components=nc)
        fpca_discretized.fit(fd)
        h = fpca_discretized.transform(fd)
        X[i, :] = torch.tensor(h.T)

    count = 0
    for i in range(len(names)):
        if names[i] in common_names:
            if subject_label[i] == 2:
                y[count] = 0
            else:
                y[count] = 1
            count = count + 1

    healthy_index = []
    disease_index = []
    for i in range(len(y)):
        if y[i]==0:
            healthy_index.append(i)
        else:
            disease_index.append(i)


    # from mlxtend.evaluate import permutation_test
    # pvalue = np.zeros(X.shape[0])
    # for i in range(X.shape[0]):
    #     pvalue[i] = permutation_test(X[i,healthy_index].tolist(),X[i,disease_index].tolist(),method='approximate', num_rounds=10000,seed=0)
    #     print(i)
    #     plt.scatter(i,pvalue[i])
    # plt.show()
    #
    # with open('pvalue_metagenomicsUsingFPCA.pkl', 'wb') as f:
    #     pickle.dump(pvalue, f)

    with open('pvalue_metagenomicsUsingFPCA.pkl', 'rb') as f:
        pvalue = pickle.load(f)

    print(X[pvalue <= 0.05, :].shape)

    return X[pvalue<=0.1,:],y,fpcplots


def FindImportantFeaturesUsingFrechet(names, subject_data, subject_label, common_names,nc,fpcs, index):


    X = torch.zeros((nc * subject_data[0].shape[0], len(subject_data)))
    H1 = torch.zeros((nc * subject_data[0].shape[0], subject_data[0].shape[1]))
    H2 = torch.zeros((nc * subject_data[0].shape[0], subject_data[0].shape[1]))
    # print(X.shape)
    y = torch.zeros(len(subject_data))

    count = 0
    for i in range(len(names)):
        if names[i] in common_names:
            if subject_label[i] == 2:
                y[count] = 0
            else:
                y[count] = 1
            count = count + 1


    healthy_index = []
    disease_index = []
    for i in range(len(y)):
        if y[i] == 0:
            healthy_index.append(i)
        else:
            disease_index.append(i)
    count = 0
    fpcplots = []

    frdistances = []
    # print(subject_data)
    for i in range(subject_data[0].shape[0]):
        x = torch.zeros(len(subject_data), subject_data[0].shape[1])
        ff = 0
        for sid in range(len(subject_data)):
            if names[sid] in common_names:
                x[ff, :] = subject_data[sid][i, :]
                ff = ff + 1
        fd = FDataGrid(x, range(x.shape[1]),
                       dataset_name='Time Series',
                       argument_names=['t'],
                       coordinate_names=['x(t)'])

        # fpca computation
        yy = x[healthy_index, :]

        fd1 = FDataGrid(x[healthy_index,:], range(x.shape[1]),
                       dataset_name='Time Series',
                       argument_names=['t'],
                       coordinate_names=['x(t)'])
        fd2 = FDataGrid(x[disease_index,:], range(x.shape[1]),
                        dataset_name='Time Series',
                        argument_names=['t'],
                        coordinate_names=['x(t)'])

        fpca_discretized = FPCA(n_components=nc)
        fpca_discretized.fit(fd)
        if index!=-10:
            h = fpcs.transform(fd)
        else:
            h = fpca_discretized.transform(fd)
        X[i, :] = torch.tensor(h.T)

        fpca_discretized1 = FPCA(n_components=nc)
        fpca_discretized1.fit(fd1)
        h1 = fpca_discretized1.components_

        fpca_discretized1 = FPCA(n_components=nc)
        fpca_discretized1.fit(fd2)
        h2 = fpca_discretized1.components_

        H1[i,:] = torch.tensor(h1.data_matrix.reshape(subject_data[0].shape[1]))
        H2[i,:] = torch.tensor(h2.data_matrix.reshape(subject_data[0].shape[1]))

        from frechetdist import frdist

        P = []
        Q = []
        for j in range(len(H1[i,:])):
            P.append([j,H1[i,j]])
            Q.append([j,H2[i,j]])
        frdistances.append(frdist(P,Q))

    plt.plot(sorted(frdistances))
    plt.show()
    filter = []
    threshold = 0.1
    for i in range(len(frdistances)):
        if frdistances[i]>=threshold:
            filter.append(i)


    if index != -10:
        print(X.shape)
        return X[index,:],y,fpcplots, index,fpca_discretized
    else:
        print(X[filter,:].shape)
        return X[filter,:],y,fpcplots, filter,fpca_discretized


def plotTheWeeks(names, week_list, common_names):
    # plotTheWeeks(names5, week_list, common_names)
    plt.figure()
    counter = 0
    for i in range(len(names)):
        if names[i] in common_names:
            for j in range(len(week_list[i])):
                plt.scatter(counter,week_list[i][j])
            counter = counter+1
    plt.xlabel('Subjects', fontsize = 30)
    plt.ylabel('Week of Data Collection', fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()

def ReduceToCommonTimeFrame(subject_data, week_list, period):
    subject_data_mod = []
    for i in range(len(subject_data)):
        X = torch.zeros(subject_data[0].shape[0],50//period)
        for c in range(40//period):
            weeksOfInterest = list(range(c*period,c*period+period))
            if len(set(weeksOfInterest).intersection(set(week_list[i])))<=0 and c>=1:
                X[:,c] = X[:,c-1]
            else:
                count = 0
                for j in range(len(weeksOfInterest)):
                    if weeksOfInterest[j] in week_list[i]:
                        id = week_list[i].index(weeksOfInterest[j])
                        X[:,c] = X[:,c] + subject_data[i][:,id]
                        count = count+1
                if count>=1:
                    X[:,c] = X[:,c]/count
        subject_data_mod.append(X)
    return subject_data_mod


def CalculateWeeks(subject_data, week_num):
    counter = 0
    weeks_list = []
    for i in range(len(subject_data)):
        x = subject_data[i].shape[1]
        weeks_list.append(week_num[counter:counter+x])
        counter = x
    return weeks_list



def plotFPCA(new_subject_data5, common_names, names5, subject_label5):
    plt.figure()
    ii = 0
    XX = []
    flag1, flag2 = True, True
    for i in range(len(new_subject_data5)):
        if names5[i] in common_names:
            XX.append(i)
            if subject_label5[i] == 2:
                if flag1:
                    plt.plot(new_subject_data5[i][ii, :], 'g', alpha=0.3, label='Healthy')
                    flag1 = False
                else:
                    plt.plot(new_subject_data5[i][ii, :], 'g', alpha=0.3)
            else:
                if flag2:
                    plt.plot(new_subject_data5[i][ii, :], 'r', alpha=0.3, label='Disease')
                    flag2 = False
                else:
                    plt.plot(new_subject_data5[i][ii, :], 'r', alpha=0.3)
    print(XX)
    plt.xlabel('Week', fontsize=15)
    plt.ylabel('Feature', fontsize=15)
    plt.title('Time series of one variable for all subjects')
    plt.legend()
    x = torch.zeros(len(common_names), new_subject_data5[0].shape[1])
    ff = 0
    y5 = []
    for sid in range(len(new_subject_data5)):
        if names5[sid] in common_names:
            x[ff, :] = new_subject_data5[sid][ii, :]
            ff = ff + 1
            if subject_label5[sid] == 2:
                y5.append(0)
            else:
                y5.append(1)
    fd = FDataGrid(x, range(x.shape[1]),
                   dataset_name='Time Series',
                   argument_names=['t'],
                   coordinate_names=['x(t)'])

    # fpca computation

    fpca_discretized = FPCA(n_components=3)
    fpca_discretized.fit(fd)
    h = fpca_discretized.components_
    # print(np.mean(h.data_matrix[1]))
    plt.show()
    plt.figure()
    plt.plot(h.data_matrix[0], 'k', linewidth=3, label='FPC-1')
    plt.plot(h.data_matrix[1], 'b', linewidth=3, label='FPC-2')
    plt.plot(h.data_matrix[2], 'c', linewidth=3, label='FPC-3')
    plt.legend()
    plt.show()

    from mpl_toolkits import mplot3d
    Z = fpca_discretized.transform(fd)
    plt.figure()
    ax = plt.axes(projection="3d")
    flag1 = True
    flag2 = True
    for i in range(Z.shape[0]):
        if y5[i] == 0:
            if flag1:
                ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='green', label = 'Healthy')
                flag1 = False
            else:
                ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='green')
        else:
            if flag2:
                ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='red', label = 'Disease')
                flag2 = False
            else:
                ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='red')
    plt.legend()
    plt.show()

    return

