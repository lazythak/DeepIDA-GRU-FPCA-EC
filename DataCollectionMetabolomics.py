# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
import pandas as pd
import numpy as np
import torch
import math
from DataCollectionGeneExpression import prepareGeneExpression,prepareTranscriptomicsSummary,CombineDataGeneExpression

def prepareMetabolomics(filename):
    print('Preparing Metabolomics')

    # meta_df_AN001514 = pd.read_csv('GSE111889_host_tx_counts.tsv', sep='\t')
    # print(meta_df_AN001514)

    meta_df = pd.read_csv(filename, sep = '\t')
    x = float('NaN')
    meta_df.replace(x, -999)
    meta_df = meta_df.fillna(-999)
    # print(meta_df)
    sample_id = []
    count = 0

    for cols in meta_df.columns:
        if count == 0:
            metabolite_names = meta_df[cols].to_list()[1:]
            count += 1
            pass
        else:
            sample_id.append(cols)
    meta_data = []
    for sid in sample_id:
        meta_data.append(meta_df[sid].to_list())
    # print(meta_data[0][0])

    label = torch.zeros(len(sample_id))
    data = torch.zeros(len(meta_data[0]) - 1, len(sample_id))
    # print(data.shape)
    # print(meta_data[0][1:len(meta_data[0])])
    # print(type(meta_data[0][-3]))

    # print(list(map(float, meta_data[1][1:100])))
    print(sample_id)
    for i in range(len(sample_id)):
        x = meta_data[i][0]
        data[:, i] = torch.FloatTensor(list(map(float, meta_data[i][1:len(meta_data[0])])))
        if x.startswith('Diagnosis:CD'):
            label[i] = 0  # 0 for CD
        elif x.startswith('Diagnosis:UC'):
            label[i] = 1  # 1 for UC
        elif x.startswith('Diagnosis:nonIBD'):
            label[i] = 2  # 2 for nonIBD
        else:
            label[i] = -999  # This should never occur
    # print(label)
    # print(data.shape)
    # print(len(metabolite_names))
    return data, label, sample_id, metabolite_names, meta_df


def prepareMetabolomicsSummary(filename):
    dff = pd.read_csv(filename)
    subject_name = dff['Subject name'].to_list()
    num = np.unique(subject_name)
    sample_name = dff['Sample name'].to_list()
    res = dict(zip(sample_name, subject_name))
    return res,num

def CombineDataMetabolomics(data, label, sample_names, dic, num):
    for i in range(len(sample_names)):
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







def main():

    #prepare dictionary of full summary

    # filename = 'ST000923_AN001516_Results (1).txt'
    filename = 'metabolomics_hilicpos.txt'
    data, label, sample_names, metabolite_names, meta_df = prepareMetabolomics(filename)
    print(meta_df)

    ## Get dictionary of subject ids and sample names, list of unique subject names
    dic, num = prepareMetabolomicsSummary('Book1.csv')

    ## Combine data using sample_names and subject_ids
    subject_data, subject_label, names = CombineDataMetabolomics(data, label, sample_names, dic, num)


    arr = []
    for i in range(len(names)):
        print(names[i], subject_data[i].shape, subject_label[i])
        arr.append(subject_data[i].shape[1])
    count = sum(i >= 4 for i in arr)
    print(count)




if __name__ == "__main__":
    main()

