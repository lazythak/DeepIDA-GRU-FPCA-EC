# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
import pandas
from DataCollectionGeneExpression import *

def main():
    Ind = list(range(0,87))

    path = 'C:\\Users\\sarth\\Documents\\DeepIDA\\DeepIDA\\Bootstraping_LMM_FPCA\\'

    view1 = []
    view2 = []
    view3 = []
    for i in Ind:
        print(i)
        view1.append(pandas.read_csv(path+'test'+str(i)+'_view1.csv'))
        view2.append(pandas.read_csv(path + 'test' + str(i) + '_view2.csv'))
        view3.append(pandas.read_csv(path + 'test' + str(i) + '_view3.csv'))

    plt.figure()
    plt.bar(range(10), view3[0]['eff_prop'].to_list()[0:10], tick_label = view3[0]['var_name'].to_list()[0:10])
    plt.title('Top 10 variables of fold 1', fontsize  = 20)
    plt.ylabel('Relative Importance', fontsize = 20)
    plt.xlabel('Variable Name', fontsize = 20)
    plt.show()
    view = view3
    S = set()
    Top100variables = []
    for i in range(len(view)):
        Sd = set(view[i]['var_name'].to_list()[1:10])

        S = S.union(Sd)
        Top100variables.append(Sd)
    print(Sd)
    F = dict()
    for s in S:
        if s not in F:
            F[s] = 0
        for j in range(len(Top100variables)):
            if s in Top100variables[j]:
                F[s] = F[s]+1
    F = dict(sorted(F.items(), key=lambda item: item[1],reverse = True))
    names = list(F.keys())
    values = list(F.values())
    plt.bar(range(5), values[0:5], tick_label=names[0:5])
    plt.ylabel('Frequency of being in top 10', fontsize = 15)
    plt.xlabel('Variable name', fontsize = 15)
    plt.show()

    print(names)


if __name__ == "__main__":
    main()