# Author: Sarthak Jain (Supervisor: Professor Sandra Safo)
import numpy as np
import numpy
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import skfda
from skfda import FDataGrid
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial
from mpl_toolkits import mplot3d
from main_functions import DeepIDA_nonBootstrap
from scipy.stats import random_correlation
import math
import pickle
# from statsmodels.tsa.arima_process import ArmaProcess

# def ECurve(M, npoints = -10):
#     # M : Matrix for edge-weighted graph
#     # Step : step-size for computing ECurve
#     # npoints: For computing npoints EC curve
#     # Output: EC curve in vector form
#     if npoints>-10:
#         step = M.max()/npoints
#     else:
#         step = M.max()/10
#     high = M.max()
#     low = M.min()
#
#     V = len(M)
#
#     count = 0
#     rat = np.zeros((npoints+1,1))
#     thr_list = np.linspace(low,high,npoints+1)
#     for thr in thr_list:
#         E = 0
#         for i in range(0,len(M)):
#             for j in range(i,len(M)):
#                 if M[i,j] <= thr:
#                     E = E+1
#
#         rat[count] = V-E
#         count = count+1
#         thr = thr+step
#     return rat


def ECurve(M, npoints = -10):
    # M : Matrix for edge-weighted graph
    # Step : step-size for computing ECurve
    # npoints: For computing npoints EC curve
    # Output: EC curve in vector form
    if npoints>-10:
        step = M.max()/npoints
    else:
        step = M.max()/10
    # high = M.max()

    high = 5
    # low = M.min()
    low = 0


    V = len(M)

    count = 0
    rat = np.zeros((npoints+1,1))
    # thr_list = np.linspace(low,high,npoints+1)
    thr_list = np.linspace(0,1,npoints+1)
    rat = euler_char(M,thr_list)
    arr = numpy.array(rat)
    return arr

def euler_char(A, thresh):
    ECs = []
    # A = A - np.identity(np.shape(A)[0])
    # print(A)

    for t in thresh:
        M = np.array((A <= t) * 1)

        # Number Edges
        Edges = np.sum(M) / 2

        # Number Vertices
        Vertices = np.shape(A)[0]

        # Betti 1
        EC = Vertices - Edges

        ECs.append(EC)

    return ECs





def ECtransform(X_v, np = 50):
    # input: multi-variate time-series data of single view for multiple subjects in tensor X
    from nilearn.connectome import ConnectivityMeasure
    tangent_measure = ConnectivityMeasure(kind="precision", vectorize = False)
    npoints = np  # number of points in the EC curve
    X_trans = torch.zeros(npoints+1,X_v.shape[2])
    M = X_v.numpy().reshape(X_v.shape[2], X_v.shape[1], X_v.shape[0])
    M = tangent_measure.fit_transform(M)
    print(M[2,:,:])

    for i in range(X_v.shape[2]):
        # M = tangent_measure.fit_transform(X_v[:,:,i].numpy())

        # C = np.linalg.inv(np.cov(X_v[:,:,i],rowvar=True))
        X_trans[:, i] = torch.tensor(ECurve(M[i,:,:], npoints).reshape(npoints + 1, ))



    return X_trans

def multiviewCovMat(n1, n2):
    #Generates block covariance matrix for multiple views
    # n1: Number of variables in first view
    # n2: Number of variables in second view
    C = np.random.uniform(-1,1,(n1+n2,n1+n2))
    C = (C+C.T)/2
    for i in range(C.shape[0]):
        C[i,i] = np.random.uniform(0,1)
    return np.dot(C,C.transpose())



def timeseries(C, T, ar, ma):
    # Simulate a time series of length T with covariance matrix C
    # ar and ma are vectors for arma parameters
    x = np.zeros((C.shape[0],T))
    w = np.zeros((C.shape[0], T))
    mu = np.zeros(C.shape[0])
    w[:,0] = np.random.multivariate_normal(mu,C)
    x[:,0] = w[:,0]

    for t in range(1,T):
        w[:,t] = np.random.multivariate_normal(mu,C)
        x[:,t] = ar[1]*x[:,t-1]+ma[0]*w[:,t]+ma[1]*w[:,t-1]
    return x

def plotin3D(X,zz,ax,col, lab=''):
    # X: multivariate time-series with different columns corresponding to times
    # z: Subject number
    cc = 1
    for i in range(X.shape[0]):
        z = zz * np.ones(X.shape[1])
        x = X[i,:]
        y = np.linspace(1, X.shape[1], X.shape[1])
        if cc>0:
            ax.plot3D(x, y, z,col, label = lab)
            cc = cc-1
        else:
            ax.plot3D(x, y, z, col)

def generate_data():
    time_length_v1 = 20   # This is the number of time samples in view v1 for both classes
    time_length_v2 = 20   # This is the number of time samples in view v2 for both classes
    nfeat_v1 = 10          # Number of features in view v1 for both classes
    nfeat_v2 = 10         # Number of features in view v2 for both classes
    total_subjects = 100   # Total number of subjects

    ar_c1 = [1, 0.5]      # ar parameters for all views in class c1
    ar_c2 = [1, -0.9]     # ar parameters for all views in class c2
    # ar_c2 = [1, 0.25]
    # ar_c2 = ar_c1
    ma_c1 = [1, 0.6]      # ma parameters for all views of class c1
    ma_c2 = [1, -0.6]      # ma parameters for all views of class c2
    # ma_c2 = ma_c1
    # ma_c2 = [1, 0]

    # covariance matrix for class c1 (for both views combined)
    C_c1 = np.random.random((nfeat_v1 + nfeat_v2, nfeat_v1 + nfeat_v2))
    C_c1 = np.dot(C_c1, C_c1.transpose())
    # C_c1 = np.linalg.inv(C_c1)
    # rng = np.random.default_rng()
    # eig = np.random.rand(nfeat_v1+nfeat_v2)
    # eig = eig*(nfeat_v1+nfeat_v2)/sum(eig)
    # C_c1 = random_correlation.rvs(eig, random_state=rng)


    # covariance matrix for class c2 (for both views combined)
    C_c2 = np.random.random((nfeat_v1 + nfeat_v2, nfeat_v1 + nfeat_v2))
    C_c2 = np.dot(C_c2, C_c2.transpose())
    C_c2 = np.random.power(10,(nfeat_v1 + nfeat_v2, nfeat_v1 + nfeat_v2))
    C_c2 = np.dot(C_c2, C_c2.transpose())
    # C_c2 = np.linalg.inv(C_c2)

    C_c2 = C_c1

    # print(C_c1)
    # print(C_c2)
    # C_c2 = np.random.lognormal(0.5,0.5, (nfeat_v1 + nfeat_v2, nfeat_v1 + nfeat_v2))
    # C_c2 = np.dot(C_c2, C_c2.transpose())
    # rng = np.random.default_rng()
    # eig = np.random.rand(nfeat_v1 + nfeat_v2)
    # eig = eig * (nfeat_v1 + nfeat_v2) / sum(eig)
    # C_c2 = random_correlation.rvs(eig, random_state=rng)

    # tensor for data of both classes and all views and all subjects
    X = torch.zeros(nfeat_v1+nfeat_v2, max(time_length_v1,time_length_v2), total_subjects)
    y = torch.zeros(total_subjects)
    for i in range(total_subjects):
        if np.random.rand()>=0.5:
            X[:, :, i] = torch.tensor(timeseries(C_c1, max(time_length_v1, time_length_v2), ar_c1, ma_c1))
            y[i] = 0
        else:
            X[:, :, i] = torch.tensor(timeseries(C_c2, max(time_length_v1, time_length_v2), ar_c2, ma_c2))
            y[i] = 1

    # Separate whole data X into data for each views
    # print(y)
    X_v1 = X[0:nfeat_v1,0:time_length_v1,:]
    X_v2 = X[nfeat_v1:nfeat_v1+nfeat_v2,0:time_length_v2,:]

    return X, y, X_v1, X_v2


def generatedataJPT():
    N = 200
    T = 20
    P = 30
    Q = 30

    X1 = torch.zeros(P, T, N)
    X2 = torch.zeros(Q, T, N)
    C_c1 = np.random.random((P+Q, P+Q))
    C_c1 = np.multiply(C_c1,C_c1)
    C_c1 = np.dot(C_c1, C_c1.transpose())

    C_c2 = np.random.random((P+Q, P+Q))
    C_c2 = np.dot(C_c2, C_c2.transpose())
    # C_c2 = np.linalg.inv(C_c2)
    y = torch.zeros(N)
    k = 5
    for i in range(N):
        print(i)
        if np.random.random(1) >= 0.5:

            for t in range(T):
                noise = np.random.multivariate_normal(100*np.random.random(P + Q), C_c1)
                for p in range(P):
                    X1[p, t, i] = (p < 10) * math.sin(2 * k* math.pi * t/T) + (p < 15 and p >= 10) * math.sin(1 *k* math.pi * t/T) + (p < 20 and p >= 15) * math.sin(0.5 *k* math.pi * t/T) + noise[p]
                    X2[p, t, i] = (p < 10) * math.sin(4 *k* math.pi * t/T) + (p < 15 and p >= 10) * math.sin(2 *k* math.pi * t/T) + (p < 20 and p >= 15) * math.sin(1 *k* math.pi * t/T) + noise[P+p]
                    y[i] = 0
        else:
            for t in range(T):
                noise = np.random.multivariate_normal(np.zeros(P + Q), C_c1)
                for p in range(P):
                    X1[p,t,i] = (p < 10) * math.sin(2 * k* math.pi * t/T) + (p < 15 and p >= 10) * math.sin(1 *k* math.pi * t/T) + (p < 20 and p >= 15) * math.sin(0.5 *k* math.pi * t/T) + noise[p]
                    X2[p, t, i] = (p < 10) * math.sin(4 *k* math.pi * t/T) + (p < 15 and p >= 10) * math.sin(2 *k* math.pi * t/T) + (p < 20 and p >= 15) * math.sin(1 *k* math.pi * t/T) + noise[P+p]
                    y[i] = 1

    for i in range(N):
        for p in range(P):
            X1[p, :, i] = torch.from_numpy(np.multiply(np.fft.fft(X1[p, :, i]),np.fft.fft(X1[p, :, i])))
            X2[p, :, i] = torch.from_numpy(np.multiply(np.fft.fft(X2[p, :, i]), np.fft.fft(X2[p, :, i])))

    print(X1.shape)
    return X1,X2,y
def twobytwoplotexample():
    n1 = 10
    n2 = 10

    ax = plt.axes(projection='3d')

    # C = multiviewCovMat(n1, n2)
    # C = np.array([[1, 0.6, 0.2, 0.2],[0.6, 1, 0.1, 0.1],[0.2, 0.1,1,0.8],[0.2,0.1,0.8,0.1]])
    C = np.random.random((n1+n2,n1+n2))
    C = np.dot(C,C.transpose())
    C_healthy = C
    C1 = C[0:n1,0:n1]
    C2 = C[n1:n1+n2,n1:n1+n2]

    x1 = timeseries(C1,10,[1, 0.8],[1,0.1])
    x2 = timeseries(C2, 10, [1, 0.8], [1, 0.1])
    X_healthy1 = []
    X_healthy2 = []
    for i in range(3):
        x1 = timeseries(C1, 10, [1, -0.6], [0.4, 0.1])
        x2 = timeseries(C2, 10, [1, -0.6], [0.4, 0.1])
        plotin3D(x1,i,ax,'b:')
        plotin3D(x2,i,ax,'k')
        X_healthy1.append(x1)
        X_healthy2.append(x2)


    # C = np.array([[1, 0.4, 0.1, 0.1], [0.4, 1, 0.1, 0.1], [0.1, 0.1, 1, 0.8], [0.1, 0.1, 0.8, 0.1]])
    C = np.random.uniform(-1,1,(n1 + n2, n1 + n2))
    C = np.dot(C, C.transpose())
    C_disease = C
    C1 = C[0:n1, 0:n1]
    C2 = C[n1:n1 + n2, n1:n1 + n2]

    X_disease1 = []
    X_disease2 = []
    for i in range(3,6):
        x1 = timeseries(C1, 10, [1, -0.6], [1, 0.1])
        x2 = timeseries(C2, 10, [1, -0.6], [1, 0.1])
        plotin3D(x1,i,ax,'r:')
        plotin3D(x2,i,ax,'g')
        X_disease1.append(x1)
        X_disease2.append(x2)
    plt.show()

    # plt.figure()
    # E1 = ECurve(C_healthy,50)
    # plt.plot(E1,'g')
    # E2 = ECurve(C_disease,50)
    # plt.plot(E2,'r')

    for i in range(len(X_healthy1)):
        x = X_healthy1[i]
        C = np.cov(x, rowvar=True)
        plt.plot(ECurve(C,1000), 'g:')

    for i in range(len(X_healthy2)):
        x = X_healthy2[i]
        C = np.cov(x, rowvar=True)
        plt.plot(ECurve(C,1000), 'g')

    for i in range(len(X_disease1)):
        x = X_disease1[i]
        C = np.cov(x, rowvar=True)
        plt.plot(ECurve(C,1000), 'r:')

    for i in range(len(X_disease2)):
        x = X_disease2[i]
        C = np.cov(x, rowvar=True)
        plt.plot(ECurve(C,1000), 'r')


    plt.show()
    plt.plot(x1)
    plt.show()

    fd = FDataGrid(x1, range(x1.shape[1]),
                   dataset_name='Time Series',
                   argument_names=['t'],
                   coordinate_names=['x(t)'])
    fd.plot()
    plt.show()
    fpca_discretized = FPCA(n_components=2)
    fpca_discretized.fit(fd)
    fpca_discretized.components_.plot()
    plt.show()


    # for i in range(0, C1.shape[0]):
    #     plt.plot(x1[i, :], 'g:')
    # for i in range(0, C2.shape[0]):
    #     plt.plot(x2[i, :], 'g')
    # plt.show()




def fPCAtransform(X):

    nc = 3
    X_trans = torch.zeros(nc*X.shape[1],X.shape[2])

    for i in range(0,X.shape[2]):
        x = X[:,:,i]

        #functional representation of data
        fd = FDataGrid(x, range(x.shape[1]),
                       dataset_name='Time Series',
                       argument_names=['t'],
                       coordinate_names=['x(t)'])

        # fpca computation

        fpca_discretized = FPCA(n_components=nc)
        fpca_discretized.fit(fd)
        h = fpca_discretized.components_
        X_trans[:,i] = torch.tensor(h.data_matrix.reshape(nc*X.shape[1],))

    return X_trans


def main():

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, y, X_v1, X_v2 = generate_data()
    print(X_v1.shape)
    print(X_v2.shape)
    for i in range(2):
        if y[i]==0:
            plotin3D(X_v1[:,:,i], i,ax, 'r', 'Class 1')
        else:
            plotin3D(X_v1[:,:,i], i,ax, 'b', 'Class 2')
    plt.ylabel('Time', fontsize = 15)
    plt.legend()
    plt.show()
    npoints = 1000
    data_v1 = torch.zeros((X_v1.shape[2], npoints + 1))
    data_v2 = torch.zeros((X_v2.shape[2], npoints + 1))
    plt.figure()
    for i in range(X_v1.shape[2]):
        print(i)
        from nilearn.connectome import ConnectivityMeasure
        tangent_measure = ConnectivityMeasure(kind="correlation", vectorize=False)
        x = X_v1[:, :, i]
        x2 = X_v2[:,:,i]
        yy = x.T.numpy().reshape(1, x.T.shape[0], x.T.shape[1])
        yy2 = x2.T.numpy().reshape(1, x2.T.shape[0], x2.T.shape[1])
        M = tangent_measure.fit_transform(yy)
        M2 = tangent_measure.fit_transform(yy2)
        h = ECurve(M, npoints)
        h2 = ECurve(M2, npoints)
        data_v1[i,:]=torch.from_numpy(h)
        data_v2[i,:]=torch.from_numpy(h2)
        c1 = 1
        c2 = 1
        if y[i] == 0:
            if c1>0:
                plt.plot(h, 'r', label = 'Class 1')
                c1 = c1-1
            else:
                plt.plot(h, 'r')
        else:
            if c2>0:
                plt.plot(h, 'b', label = 'Class 2')
                c2 = c2-1
            else:
                plt.plot(h, 'b')
    plt.ylabel('Euler Characteristic', fontsize = 15)
    plt.xlabel('Threshold', fontsize = 15)
    plt.title('Euler Curves for different classes',fontsize = 15)
    plt.show()

    X_train1, X_test1, y_train1, y_test1 = train_test_split(data_v1, y, stratify=y,
                                                            test_size=0.30, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data_v2, y, stratify=y,
                                                            test_size=0.30, random_state=0)

    results = DeepIDA_nonBootstrap([X_train1, X_train2],
                                   [X_test1[0:15,:], X_test2[0:15,:]],
                                   [X_test1[15:30, :], X_test2[15:30,:]],
                                   y_train1, y_test1[0:15], y_test2[15:30],
                                   [[256, 64, 20], [256, 64, 20]],
                                   [5, 5], [0, 0], [3, 256, 20], 0.001, 80)

    print(results[0:2])


    # FPCA
    nc=3
    data_v1 = torch.zeros((X_v1.shape[2], nc * X_v1.shape[0]))
    data_v2 = torch.zeros((X_v2.shape[2], nc * X_v2.shape[0]))
    for i in range(X_v1.shape[0]):
        x = X_v1[i, :,: ]
        x = x.T


        fd = FDataGrid(x, range(x.shape[1]),
                    dataset_name='Time Series',
                    argument_names=['t'],
                    coordinate_names=['x(t)'])
        fpca_discretized = FPCA(n_components=nc)
        fpca_discretized.fit(fd)
        h = fpca_discretized.components_

        print(np.mean(h.data_matrix[1]))
        if i==0:
            plt.show()
            plt.figure()
            plt.plot(h.data_matrix[0], 'k', linewidth=3, label='FPC-1')
            plt.plot(h.data_matrix[1], 'b', linewidth=3, label='FPC-2')
            plt.plot(h.data_matrix[2], 'c', linewidth=3, label='FPC-3')
            plt.legend()
            plt.show()

        h = fpca_discretized.transform(fd)

        for j in range(nc):
            data_v1[:,i * nc + j] = torch.tensor(h[:, j])

    for i in range(X_v2.shape[0]):
        print(i)
        x = X_v2[i, :,: ]
        x = x.T

        fd = FDataGrid(x, range(x.shape[1]),
                    dataset_name='Time Series',
                    argument_names=['t'],
                    coordinate_names=['x(t)'])
        fpca_discretized = FPCA(n_components=nc)
        fpca_discretized.fit(fd)
        # h = fpca_discretized.components_

        # print(np.mean(h.data_matrix[1]))
        # if i==0:
        #     plt.show()
        #     plt.figure()
        #     plt.plot(h.data_matrix[0], 'k', linewidth=3, label='FPC-1')
        #     plt.plot(h.data_matrix[1], 'b', linewidth=3, label='FPC-2')
        #     plt.plot(h.data_matrix[2], 'c', linewidth=3, label='FPC-3')
        #     plt.legend()
        #     plt.show()

        h = fpca_discretized.transform(fd)
        print(h.shape)
        for j in range(nc):
            data_v2[:,i * nc + j] = torch.tensor(h[:, j])

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    for i in range(data_v1.shape[0]):
        if y[i]==0:
            ax.scatter3D(data_v1[i,0], data_v1[i,1], data_v1[i,2], color="green")
        else:
            ax.scatter3D(data_v1[i, 0], data_v1[i, 1], data_v1[i, 2], color="red")

    # show plot
    plt.show()


    X_train1, X_test1, y_train1, y_test1 = train_test_split(data_v1, y, stratify=y,
                                                            test_size=0.30, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data_v2, y, stratify=y,
                                                            test_size=0.30, random_state=0)
    print(y_train1-y_train2)
    results = DeepIDA_nonBootstrap([X_train1, X_train2],
                                   [X_test1[0:15, :], X_test2[0:15, :]],
                                   [X_test1[15:30, :], X_test2[15:30, :]],
                                   y_train1, y_test1[0:15], y_test1[15:30],
                                   [[256, 64, 20], [256, 64, 20]],
                                   [5, 5], [0, 0], [3, 256, 20], 0.001, 80)

    print(results[0:2])


    # DeepIDA-GRU
    data_v1 = X_v1.T
    data_v2 = X_v2.T
    print(data_v1.shape)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(data_v1, y, stratify=y,
                                                            test_size=0.30, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data_v2, y, stratify=y,
                                                            test_size=0.30, random_state=0)
    results = DeepIDA_nonBootstrap([X_train1, X_train2],
                                   [X_test1[0:15, :], X_test2[0:15, :]],
                                   [X_test1[15:30, :], X_test2[15:30, :]],
                                   y_train1, y_test1[0:15], y_test1[15:30],
                                   [[256, 64, 20], [256, 64, 20]],
                                   [5, 5], [1, 1], [3, 256, 20], 0.001, 80)

    print(results[0:2])

    return
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



    # DeepIDA-GRU




    return




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


    return




    # M = np.array([[1.1, 0.6, 0.8, 0.7, 0.1],
    #      [0.6, 1.1, 0.5, 0.65, 0.2],
    #      [0.8, 0.5, 1.1, 0.55, 0.23],
    #      [0.7, 0.65, 0.55, 1.1, 0.3],
    #      [0.1, 0.2, 0.23, 0.2, 1.1]])
    npoints = 100
    thres = np.linspace(0,1,npoints+1)
    print(thres)
    X = euler_char(M, thres)
    print(X)
    plt.figure()
    plt.plot(thres, X)

    plt.ylabel('Euler characteristic', fontsize = 15)
    plt.xlabel('Threshold', fontsize = 15)
    plt.title('Euler Curve', fontsize = 15)
    plt.show()
    return

    # M=np.random.random((10,10))
    # M=[[1,0.2,0.8,0.5,0.1],
    #    [0.2,1,0.3,0.2,0.5],
    #    [0.8,0.3,1,0.1,0.8],
    #    [0.5,0.2,0.1,1,0.3],
    #    [0.1,0.5,0.8,0.3,1]]
    # M1=np.random.standard_normal([10,10])
    # M2=np.random.standard_normal([10, 10])


    # s1 = np.zeros((1000,10))
    # s2 = np.zeros((1000, 10))
    # for i in range(0,10):
    #     ar1 = np.array([1, 0.2])
    #     ma1 = np.array([1, -0.2])
    #     ar2 = np.array([1, 0.4, 0.5,0.2])
    #     ma2 = np.array([1, -0.4,0.5,0.2])
    #     s1[:, i] = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)
    #     s2[:, i] = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)
    #     # x(t) = 0.2*x(t-1)+w(t)-0.2*w(t-1)
    #     # Multivariate normal
    # M1 = (np.cov(s1, rowvar=False))
    # M2 = (np.cov(s2, rowvar=False))
    # print(M1)
    # plt.figure()
    # # im = plt.imshow(M1, cmap="copper_r")
    # plt.plot(s1[:,0])
    # plt.plot(s2[:,0])
    # # im = plt.imshow(M2, cmap="copper_r")
    # # plt.colorbar(im)
    # #(n,p,t): s
    #
    #
    # X1 = ECurve(M1,100)
    # X2 = ECurve(M2,100)
    # xlist = np.linspace(0,M1.max(),len(X1))
    # plt.figure()
    # plt.plot(xlist,X1)
    # plt.plot(xlist, X2)
    # # plt.plot(s1)
    # plt.show()
    # ax = plt.axes(projection = '3d')
    #
    # z = np.zeros((1,10)).reshape(10,)
    # x = np.random.random((1,10)).reshape(10,)
    # y = np.linspace(1,10,10).reshape(10,)
    # ax.plot3D(x,y,z)
    # plt.show()


    # Data Generated according to random covariance matrix
    # X, y, X_v1, X_v2 =generate_data()
    #
    # train_percent = 0.6
    # test_percent = 0.2
    # valid_percent = 1-train_percent-test_percent
    # train = range(int(X.shape[2]*train_percent//1))
    # test = range(int(X.shape[2]*train_percent//1),int(X.shape[2]*(test_percent+train_percent)//1))
    # valid = range(int(X.shape[2]*(test_percent+train_percent)//1),X.shape[2])
    #
    #
    #
    # X_v1_EC = ECtransform(X_v1)
    # X_v2_EC = ECtransform(X_v2)
    # results = DeepIDA_nonBootstrap([X_v1_EC[:,train].T, X_v2_EC[:,train].T], [X_v1_EC[:,valid].T, X_v2_EC[:,valid].T], [X_v1_EC[:,test].T, X_v2_EC[:,test].T], y[train], y[valid], y[test],
    #                      [[256,64,20],[256,64,20]], [5,5], lr_rate=0.01, n_epoch=10)
    #
    # print(results[5])
    #
    # X_v1_fPCA = fPCAtransform(X_v1)  #fPCA for view 1
    # X_v2_fPCA = fPCAtransform(X_v2)  #fPCA for view 2
    # X_v12_fPCA = torch.cat((X_v1_fPCA,X_v2_fPCA))  #concatenating the two fPCAs
    #
    # # Using SVM after fPCA for classification
    # X_train, X_test, y_train, y_test = train_test_split(X_v12_fPCA.T, y, test_size=0.25)
    # classifier = SVC(kernel='rbf')
    # classifier.fit(X_train,y_train)
    # y_pred = classifier.predict(X_test)
    # print(classification_report(y_test,y_pred))
    #
    # from numpy import save
    # # print(X_v1[].shape)
    #
    # data_v1_c1 = X_v1[:,:,y==0].detach().numpy()
    # data_v2_c1 = X_v2[:,:,y==0].detach().numpy()
    # data_v1_c2 = X_v1[:, :, y == 1].detach().numpy()
    # data_v2_c2 = X_v2[:, :, y == 1].detach().numpy()
    #
    # print(data_v1_c1.shape)
    # print(data_v1_c2.shape)
    #
    # save('tensor_v1_c1.npy', data_v1_c1)
    # save('tensor_v2_c1.npy', data_v2_c1)
    # save('tensor_v1_c2.npy', data_v1_c2)
    # save('tensor_v2_c2.npy', data_v2_c2)



    # Data Generated according to JPT paper

    X_v1, X_v2, y = generatedataJPT()
    train_percent = 0.6
    test_percent = 0.2
    valid_percent = 1-train_percent-test_percent
    train = range(int(X_v1.shape[2]*train_percent//1))
    test = range(int(X_v1.shape[2]*train_percent//1),int(X_v1.shape[2]*(test_percent+train_percent)//1))
    valid = range(int(X_v1.shape[2]*(test_percent+train_percent)//1),X_v1.shape[2])
    X_v1_EC = ECtransform(X_v1)
    X_v2_EC = ECtransform(X_v2)

    from numpy import save
    # print(X_v1[].shape)

    data_v1_c1 = X_v1[:,:,y==0].detach().numpy()
    data_v2_c1 = X_v2[:,:,y==0].detach().numpy()
    data_v1_c2 = X_v1[:, :, y == 1].detach().numpy()
    data_v2_c2 = X_v2[:, :, y == 1].detach().numpy()
    #
    # print(data_v1_c1.shape)
    # print(data_v1_c2.shape)
    #
    save('tensorjp_v1_c1.npy', data_v1_c1)
    save('tensorjp_v2_c1.npy', data_v2_c1)
    save('tensorjp_v1_c2.npy', data_v1_c2)
    save('tensorjp_v2_c2.npy', data_v2_c2)



    # npoints = 100
    # with open('brains.pickle', 'rb') as f:
    #     x = pickle.load(f)
    # plt.figure()
    # X_trans = torch.zeros(npoints + 1, x.shape[0])
    # print("I am here")
    # for i in range(x.shape[0]):
    #     # M = tangent_measure.fit_transform(X_v[:,:,i].numpy())
    #
    #     # C = np.linalg.inv(np.cov(X_v[:,:,i],rowvar=True))
    #     X_trans[:, i] = torch.tensor(ECurve(x[i,:,:], npoints).reshape(npoints + 1, ))
    #     if i<=5:
    #         plt.plot(X_trans[:,i],'r')
    #     else:
    #         plt.plot(X_trans[:,i],'b')
    # plt.show()



    # for i in range(X_v1_EC.shape[1]):
    #     if y[i]==0:
    #         plt.plot(X_v1_EC[:, i], 'r')
    #     else:
    #         plt.plot(X_v1_EC[:, i], 'b')
    # plt.show()

    plt.figure()
    for i in range(X_v2_EC.shape[1]):
        if y[i] == 0:
            plt.plot(X_v2[0,:, i], 'r')
        else:
            plt.plot(X_v2[0,:, i], 'b')
    plt.show()

    plt.figure()
    for i in range(X_v2_EC.shape[1]):
        if y[i] == 0:
            plt.plot(X_v2_EC[:, i], 'r')
        else:
            plt.plot(X_v2_EC[:, i], 'b')
    plt.show()

    results = DeepIDA_nonBootstrap([X_v1_EC[:, train].T, X_v2_EC[:, train].T],
                                   [X_v1_EC[:, valid].T, X_v2_EC[:, valid].T], [X_v1_EC[:, test].T, X_v2_EC[:, test].T],
                                   y[train], y[valid], y[test],
                                    [[256,64,20],[256,64,20]], [10,10], lr_rate=0.01, n_epoch=10)
    print('The result is ',results[0],results[1])
    out_train = results[17]
    out_valid = results[18]
    eigen = results[19]
    scores = torch.zeros(2,120,1)
    print('I am here')
    for i in range(2):
        scores[i]=torch.matmul(out_train[0],eigen[0])
    # print(scores[0].shape)
    #
    # covE = torch.zeros(X_v1.shape[0],X_v1.shape[1],X_v1.shape[0],X_v1.shape[1])
    # summ = torch.sum(X_v1,2)/X_v1.shape[0]
    # print(X_v1.shape)
    # for i in range(X_v1.shape[2]):
    #     X_v1[:,:,i] = X_v1[:,:,i] - summ
    # for i1 in range(X_v1.shape[0]):
    #     print(i1)
    #     for i2 in range(X_v1.shape[1]):
    #         for n in range(X_v1.shape[2]):
    #             covE[i1,i2,:,:]=covE[i1,i2,:,:]+X_v1[i1,i2,n]*X_v1[:,:,n]
    #         covE[i1, i2,:, :] = covE[i1,i2,:,:]/X_v1.shape[2]
    #
    # print(covE.shape)
    #
    #
    #
    # covE = torch.cov(X_1.T)
    # print(covE.shape)
    # print(torch.matmul(torch.matmul(Lv, covE), Lv.T))
    # penalty = 0
    # # beta = torch.matmul(S_1.T, X_1)
    # # beta = beta.reshape(1,X_v1.shape[0],X_v1.shape[1])
    # # for p in range(beta.shape[1]):
    # #     penalty = penalty + torch.norm(beta[:,p,:])
    # # print(penalty)


    X_1 = X_v1.reshape(X_v1.shape[2], X_v1.shape[0], X_v1.shape[1])
    X_1 = X_1[0:120, :, :]
    X_1 = X_1.reshape(X_1.shape[0], X_1.shape[1] * X_1.shape[2])
    S_1 = scores[0]
    beta = torch.zeros(1, X_v1.shape[0] * X_v1.shape[1])

    def objective(beta):
        L = torch.matmul(S_1.T, X_1) - beta
        Lv = L.reshape(1,X_v1.shape[0]*X_v1.shape[1])
        covE = torch.cov(X_1.T)
        obj = torch.matmul(torch.matmul(Lv, covE), Lv.T)
        beta1 = beta.reshape(1, X_v1.shape[0], X_v1.shape[1])
        penalty = 0
        for p in range(beta1.shape[1]):
            penalty = penalty + torch.norm(beta1[:, p, :])
        return obj+100*penalty
    print(objective(beta))
    from scipy.optimize import minimize
    bnds = torch.ones(1, X_v1.shape[0] * X_v1.shape[1])
    # sol = minimize(objective, beta, method='SLSQP')
    # print(y[0])
    # plt.imshow(X_v1[:,:,0])
    # plt.colorbar()
    # plt.show()
    #
    # print(y[1])
    # plt.imshow(X_v1[:, :, 1])
    # plt.colorbar()
    # plt.show()

    X_v1_fPCA = fPCAtransform(X_v1)  #fPCA for view 1
    X_v2_fPCA = fPCAtransform(X_v2)  #fPCA for view 2
    X_v12_fPCA = torch.cat((X_v1_fPCA,X_v2_fPCA))  #concatenating the two fPCAs

    # Using SVM after fPCA for classification
    X_train, X_test, y_train, y_test = train_test_split(X_v12_fPCA.T, y, test_size=0.25)
    results = DeepIDA_nonBootstrap([X_v1_fPCA[:, train].T, X_v2_fPCA[:, train].T],
                                   [X_v1_fPCA[:, valid].T, X_v2_fPCA[:, valid].T], [X_v1_fPCA[:, test].T, X_v2_fPCA[:, test].T],
                                   y[train], y[valid], y[test],
                                   [[256, 64, 20], [256, 64, 20]], [10, 10], lr_rate=0.01, n_epoch=10)
    print('The result is ', results[0], results[1])


    classifier = SVC(kernel='rbf')
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test,y_pred))

if __name__ == "__main__":
    main()






# Github: plot with their
# simulate data with different times for peaks and lows
# 3D plot for different people and their time series
# Simulate another type of data
# fmri imaging data and their ec curve

# DeepIDA with EC curve
# Functional PC
# Joint principle trained analysis
# joint principle trend analysis
# functional