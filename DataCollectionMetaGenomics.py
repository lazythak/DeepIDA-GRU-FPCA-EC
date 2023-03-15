import pandas as pd
import numpy as np
import torch
import pickle

def prepareGeneExpression(filename):
    print('Preparing GeneExpression Data')

    # meta_df_AN001514 = pd.read_csv('GSE111889_host_tx_counts.tsv', sep='\t')
    # print(meta_df_AN001514)

    gene_df = pd.read_csv(filename, sep=',')
    # print(gene_df)
    x = float('NaN')
    gene_df.replace(x, -999)
    gene_df = gene_df.fillna(-999)
    # print(meta_df)
    sample_id = []
    count = 0

    for cols in gene_df.columns:
        if count == 0:
            gene_names = gene_df[cols].to_list()[0:]
            count += 1
            pass
        else:
            sample_id.append(cols.replace('_pathabundance_cpm',''))
    gene_data = []
    for sid in sample_id:
        gene_data.append(gene_df[sid+'_pathabundance_cpm'].to_list())
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
    week_num = []
    label = []
    print(len(dff['data_type'].to_list()))
    for i in range(len(dff['data_type'].to_list())):
        if dff['data_type'].to_list()[i] == 'metagenomics':
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
    return res,label_dict, week_num


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

def CalculateWeeks(subject_data, week_num):
    counter = 0
    weeks_list = []
    for i in range(len(subject_data)):
        x = subject_data[i].shape[1]
        weeks_list.append(week_num[counter:counter+x])
        counter = x
    return weeks_list





def main():

    #prepare dictionary of full summary
    filename = 'metagenomics_normalized_lmm.csv'
    data, sample_names, gene_names, gene_df = prepareGeneExpression(filename)
    # print(len(sample_names))
    print(gene_df)
    print('CSM5MCXD' in sample_names)

    dic,label_dict,week_num = prepareTranscriptomicsSummary('hmp2_metadata.csv')

    # print(dic)

    subject_data, subject_label, names = CombineDataGeneExpression(data, label_dict, sample_names, dic)
    week_list = CalculateWeeks(subject_data, week_num)

    print('Number of subjects are: ', len(names))
    for i in range(len(names)):
        print(names[i], subject_data[i].shape, subject_label[i])

    with open('parrot_metagenome_lmm.pkl', 'wb') as f:
        pickle.dump([names, subject_data, subject_label, week_list], f)




    # prepare dictionary of full summary
    filename = 'metagenomics_normalized.csv'
    data, sample_names, gene_names, gene_df = prepareGeneExpression(filename)
    # print(len(sample_names))
    print(gene_df)
    print('CSM5MCXD' in sample_names)

    dic, label_dict, week_num = prepareTranscriptomicsSummary('hmp2_metadata.csv')

    # print(dic)

    subject_data, subject_label, names = CombineDataGeneExpression(data, label_dict, sample_names, dic)
    week_list = CalculateWeeks(subject_data, week_num)

    print('Number of subjects are: ', len(names))
    for i in range(len(names)):
        print(names[i], subject_data[i].shape, subject_label[i])

    with open('parrot_metagenome.pkl', 'wb') as f:
        pickle.dump([names, subject_data, subject_label, week_list], f)


if __name__ == "__main__":
    main()