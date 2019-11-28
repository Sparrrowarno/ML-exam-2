import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

def GMMgen(N, alpha, mu, Sigma):
    dist = [0]
    dist.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, len(mu[0,:])))
    for i in range(len(alpha)):
        indices = np.squeeze(np.where(np.logical_and(u >= dist[i], u < dist[i+1]) == True))
        L[indices] = i * np.ones(len(indices))
        for k in indices:
            x[k,:] = np.random.multivariate_normal(mu[i,:], Sigma[:,:,i])
    return x, L

def MAPestimate(X, mu, Sigma, Pr):
    n = len(Pr)
    rv = np.zeros(4)
    for j in range(n):
        rv[j] = scipy.stats.multivariate_normal(np.reshape(mu[j],(len(mu[0]))), Sigma[:,:,j])
    D = np.zeros(len(X))
    for i in range(len(X)):
        posterior_i = np.zeros(n)
        for j in range(n):
            posterior_i[j] = rv[j].pdf(X[i,:]) * Pr[j]
        D[i] = np.squeeze(np.where(posterior_i == max(posterior_i)))
    return D

# selected data distribution
prior = np.array([0.1, 0.2, 0.3, 0.4])
mu_true = np.array([[1, 1, 1], [-1, 1, -1], [-1, -1, -1], [1, -1, 1]])
Sigma_true = np.zeros([3, 3, 4])
k = 0.3  # adjuastment probe
Sigma_true[:, :, 0] = k * np.array([[2, 1, 0],
                                    [1, 2, 1],
                                    [0, 1, 2]])

Sigma_true[:, :, 1] = k * np.array([[2, 0, 1],
                                    [0, 2, 1],
                                    [1, 1, 2]])

Sigma_true[:, :, 2] = k * np.array([[2, 1, 0],
                                    [1, 2, 1],
                                    [0, 1, 2]])

Sigma_true[:, :, 3] = k * np.array([[2, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 2]])

# true distributions
N = 1000
x, L = GMMgen(N, prior,mu_true, Sigma_true)
# plot true samples
fig_1 = plt.figure(figsize=(10,10))
ax_1 = Axes3D(fig_1)
clist=['r','g','b','m']
labellist = ['L=1','L=2','L=3','L=4']
for i in range(len(prior)):
    ax_1.scatter(x[np.where(L == i),0],x[np.where(L == i),1],x[np.where(L == i),2],marker='o', c=clist[i], label= labellist[i])
ax_1.set_title('Q1.1  distribution of 1000 samples with true labels',fontsize=20)
ax_1.legend(fontsize=20)

#MAP estimation
N_test = 10000
x_test, L_test = GMMgen(N_test, prior,mu_true, Sigma_true)
D_MAP = MAPestimate(x_test,mu_true, Sigma_true,prior)
P_error_MAP = np.count_nonzero(D_MAP != L_test) / N_test
# plot MAP classified samples
fig_2 = plt.figure(figsize=(10,10))
ax_2 = Axes3D(fig_2)
clist=['r','g','b','m']
labellist = ['D=1','D=2','D=3','D=4']
for i in range(len(prior)):
    ax_2.scatter(x_test[np.where(D_MAP == i),0],x_test[np.where(D_MAP == i),1],x_test[np.where(D_MAP == i),2],marker='o', c=clist[i], label= labellist[i])
ax_2.set_title('Q1.2  10000 samples classified by optimal MAP classifier',fontsize=20)
ax_2.legend(fontsize=20)
print('Probability of error:%.4f'%P_error_MAP)

#optimize nodes
N_list = [100, 1000,10000]
K = 10
fig_3 = plt.figure(figsize=(10,10))
ax_3 = plt.subplot(111)
node_list = [5, 10, 15, 20, 25, 30]

for q in len(N_list):               #different data sets
    x_MLP, L_MLP = GMMgen(N_list[q], prior,mu_true, Sigma_true)
    feed = np.ceil(np.linspace(0,N_list[q],K+1))
    index_partition_limit = np.zeros([K,2])
    node_best = np.zeros(len(N_list))
    for k in range(K):             #different folds
        index_partition_limit[k,:] = [feed[k],feed[k+1]-1]
    P_incorrect_mean = np.zeros((len(node_list)))
    P_correct_mean = np.zeros((len(node_list)))
    for i in range(len(node_list)):             #different nodes
        node = node_list[i]
        P_incorrect = np.zeros((K))
        for k in range(K):
            ind_valid = np.arange(index_partition_limit[k,0], index_partition_limit[k,1]+1)
            ind_valid = ind_valid.astype(int)
            x_valid = x_MLP[ind_valid,:]
            L_valid = L_MLP[ind_valid]
            ind_train = np.hstack((np.arange(0,index_partition_limit[k,0]),np.arange(index_partition_limit[k,1]+1, N_list[q])))
            ind_train = ind_train.astype(int)
            x_train = x_MLP[ind_train,:]
            L_train = L_MLP[ind_train]
            model = tf.keras.models.Sequential([tf.keras.layers.Dense(node, activation='tanh'),tf.keras.layers.Dense(4, activation='softplus')])
            model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
            model.fit(x_train,L_train,epochs=15)
            pre_MLP = model.predict(x_valid)
            Dis_MLP = np.argmax(pre_MLP[:],axis=1)
            P_incorrect[k] = np.count_nonzero(Dis_MLP != L_valid) / N_list[q]
        P_incorrect_mean[i] = np.mean(P_incorrect)
        P_correct_mean[i] = 1 - np.mean(P_incorrect)
    node_best_ind = np.squeeze(np.where(P_incorrect_mean == (min(P_incorrect_mean))))
    node_best[q] = node_list[node_best_ind]
    print(node_best)                  #opt nodes table
    ax_3.plot(node_list, P_correct_mean, marker='x', label='N = %d'%N_list[q])
ax_3.set_title('Probability of correct decisions with different nodes',fontsize=20)
ax_3.set_xlabel('Number of nodes',fontsize=20)
ax_3.set_ylabel('Probability of correct decisions', fontsize=20)
ax_3.legend(fontsize=20)

p = 0 # p=0, 1, 2  data set probes
N = N_list[p]
x_MLP, L_MLP = GMMgen(N, prior,mu_true, Sigma_true)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(node_best[p], activation='tanh'),tf.keras.layers.Dense(4, activation='softplus') ])
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_MLP,L_MLP,epochs=15,verbose=0)
pre_MLP = model.predict(x_test)
Dis_MLP = np.argmax(pre_MLP[:],axis=1)
P_error_MLP = np.count_nonzero(Dis_MLP != L_test) / N_test

fig_4 = plt.figure(figsize=(10,10))
ax_4 = Axes3D(fig_4)
clist=['r','g','b','m']
labellist = ['D_MLP=1','D_MLP=2','D_MLP=3','D_MLP=4']
for i in range(len(prior)):
    ax_4.scatter(x_test[np.where(Dis_MLP == i),0],x_test[np.where(Dis_MLP == i),1],x_test[np.where(Dis_MLP == i),2],marker='o', c=clist[i], label= labellist[i])
ax_4.set_title('Q1.3  10000 samples classified by MLP with optimized %d nodes'%(node_best[p]),fontsize=20)
ax_4.legend(fontsize=20)
print('Probability of error:%.4f'%P_error_MLP)