# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
from EC import *
from DataCollectionGeneExpression import *
def main():

    #Euler Curves
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, y, X_v1, X_v2 = generate_data()
    print(X_v1.shape)
    print(X_v2.shape)
    for i in range(2):
        if y[i] == 0:
            plotin3D(X_v1[:, :, i], i, ax, 'g', 'Class 1')
        else:
            plotin3D(X_v1[:, :, i], i, ax, 'r', 'Class 2')
    plt.ylabel('Time', fontsize=15)
    plt.legend()
    plt.show()

    plt.figure()

    c1 = 1
    c2 = 1
    for i in range(X_v1.shape[2]):
        print(i)
        from nilearn.connectome import ConnectivityMeasure

        tangent_measure = ConnectivityMeasure(kind="correlation", vectorize=False)
        x = X_v1[:, :, i]
        yy = x.T.numpy().reshape(1, x.T.shape[0], x.T.shape[1])
        M = tangent_measure.fit_transform(yy)
        h = ECurve(M, 1000)

        if y[i] == 0:
            if c1 > 0:
                plt.plot(h, 'g', label='Class 1')
                c1 = c1 - 1
            else:
                plt.plot(h, 'g')
        else:
            if c2 > 0:
                plt.plot(h, 'r', label='Class 2')
                c2 = c2 - 1
            else:
                plt.plot(h, 'r')
    plt.ylabel('Euler Characteristic', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.title('Euler Curves for different classes', fontsize=15)
    plt.legend()
    plt.show()


    # Functional-PCA

    for ff in range(1): # For each variable
        x = torch.zeros(X_v1.shape[2], X_v1.shape[1]) # Number of subjects by number of time points
        for sid in range(X_v1.shape[2]):
            u = X_v1[ff, :, sid]
            x[sid, :] = u
            # ff = ff + 1
        fd = FDataGrid(x, range(x.shape[1]),
                    dataset_name='Time Series',
                    argument_names=['t'],
                    coordinate_names=['x(t)'])

        # fpca computation

        fpca_discretized = FPCA(n_components=3)
        fpca_discretized.fit(fd)
        h = fpca_discretized.components_
        plt.figure()
        plt.plot(h.data_matrix[0], 'k', linewidth=3, label='FPC-1')
        plt.plot(h.data_matrix[1], 'b', linewidth=3, label='FPC-2')
        plt.plot(h.data_matrix[2], 'c', linewidth=3, label='FPC-3')
        plt.legend()
        plt.show()

        Z = fpca_discretized.transform(fd)
        plt.figure()
        ax = plt.axes(projection="3d")
        flag1 = True
        flag2 = True
        for i in range(Z.shape[0]):
            if y[i] == 0:
                if flag1:
                    ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='green',
                                 label='Class 1')
                    flag1 = False
                else:
                    ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='green')
            else:
                if flag2:
                    ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='red', label='Class 2')
                    flag2 = False
                else:
                    ax.scatter3D(np.array(Z[i, 0]), np.array(Z[i, 1]), np.array(Z[i, 2]), color='red')
        plt.legend()
        plt.show()

    fig = plt.figure()


    c1 = 1
    c2 = 1
    for i in range(X_v1.shape[2]):
        if y[i] == 0:
            if c1>0:
                plt.plot(X_v1[1, :, i], 'g', label = 'Class 1')
                c1 = c1-1
            else:
                plt.plot(X_v1[1, :, i], 'g')
        else:
            if c2>0:
                plt.plot(X_v1[1, :, i], 'r', label = 'Class 2')
                c2 = c2-1
            else:
                plt.plot(X_v1[1, :, i], 'r')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Variable 1 values for all subjects', fontsize = 15)
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    main()