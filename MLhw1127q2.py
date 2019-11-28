import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats
import math

#transfer datagen from matlab
def GMMgen(N, alpha, mu, covEvalues):
    dist = [0]
    covEvectors = np.zeros([2,2,3])
    covEvectors[:,:,0] = np.array([[1, -1],[1, 1]]) / math.sqrt(2)
    covEvectors[:,:,1] = np.array([[1, 0],[0, 1]])
    covEvectors[:,:,2] = np.array([[1, -1],[1, 1]]) / math.sqrt(2)
    dist.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, len(mu[0,:])))
    for i in range(len(alpha)):
        indices = np.squeeze(np.where(np.logical_and(u >= dist[i], u < dist[i+1]) == True))
        L[indices] = i * np.ones(len(indices))
        for k in indices:
            x[k,:] = np.dot(np.dot(covEvectors[:,:,i],covEvalues),np.random.randn(2,1)).T + mu[i]
    return x, L

K = 10
N_train = 1000
N_test = 10000
prior = [0.33,0.34,0.33]
mu_true = np.array([[-18,-8],[0,0],[18,8]])
#covEvalues = [[3.2**2 ,0],[0 ,0.6**2]]
covEvalues = [[3.2 ,0],[0 ,0.6]]
x_train, L_train = GMMgen(N_train, prior, mu_true,covEvalues)
x_test, L_test = GMMgen(N_test,prior,mu_true,covEvalues)

#show data
colors=['#66ccff','#cc66ff','#ffcc66']
labels = ['L=1','L=2','L=3']
plt.figure(figsize=(20,10))
plt.subplot(121)
for i in range(len(prior)):
    plt.scatter(x_train[np.where(L_train == i),0],x_train[np.where(L_train == i),1],marker='.', c=colors[i], label= labels[i])
plt.title('training samples',fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('x1',fontsize=20)
plt.ylabel('x2',fontsize=20)
plt.subplot(122)
for i in range(len(prior)):
    plt.scatter(x_test[np.where(L_test == i),0],x_test[np.where(L_test == i),1],marker='.', c=colors[i], label= labels[i])
plt.title('tresting samples',fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('x1',fontsize=20)
plt.ylabel('x2',fontsize=20)

#find the best model
plt.figure(figsize=(10, 10))
fold_mark = np.ceil(np.linspace(0, N_train, K + 1))
index_partition_limit = np.zeros([K, 2])
for k in range(K):
    index_partition_limit[k, 0] = fold_mark[k]
    index_partition_limit[k, 1] = fold_mark[k + 1] - 1

act_list = np.array(['softplus', 'sigmoid'])
node_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
mse_mean = np.zeros((2, len(node_list)))

for j in range(len(act_list)):
    act = act_list[j]
    for i in range(len(node_list)):
        node = node_list[i]
        mse = np.zeros((K))
        for k in range(K):
            ind_validate = np.arange(index_partition_limit[k, 0], index_partition_limit[k, 1] + 1)
            ind_validate = ind_validate.astype(int)
            x_validate = x_train[ind_validate, :]
            ind_train_k = np.hstack(
                (np.arange(0, index_partition_limit[k, 0]), np.arange(index_partition_limit[k, 1] + 1, N_train)))
            ind_train_k = ind_train_k.astype(int)
            x_train_k = x_train[ind_train_k, :]
            # activation function
            model = tf.keras.models.Sequential(
                [tf.keras.layers.Dense(node, activation=act), tf.keras.layers.Dense(1, activation='linear')])
            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            model.fit(x_train_k[:, 0], x_train_k[:, 1], epochs=10, verbose=0)  # verbos=1,0

            model.compile(optimizer='RMSprop'(learning_rate=1 * 0.00001), loss='mse', metrics=['mse'])
            model.fit(x_train_k[:, 0], x_train_k[:, 1], epochs=3, verbose=0)
            x_pre = model.predict(x_validate[:, 0])
            mse[k] = np.mean((x_pre.reshape(1, -1) - x_validate[:, 1]) ** 2)
        mse_mean[j, i] = np.mean(mse)
indice1 = np.squeeze(np.where(mse_mean == (mse_mean.min())))
node_best = np.array(node_list)[indice1]
act_best = np.array(act_list)[indice1]
#para found
for j in range(len(act_list)):
    plt.plot(node_list, mse_mean[j, :], marker='x', c=colors[j], label=act_list[j])
plt.title('how to find the best model', fontsize=20)
plt.xlabel('number of nodes', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
print(node_best)
print(act_best)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(node_best, activation=act_best),tf.keras.layers.Dense(1, activation='linear')])
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.fit(x_train[:,0],x_train[:,1],epochs=10,verbose=0)
model.compile(optimizer='RMSprop'(learning_rate=1*0.00001),loss='mse',metrics=['mse'])
model.fit(x_train_k[:,0],x_train_k[:,1],epochs=3,verbose=0)
x_pre = model.predict(x_test[:,0])
mse = np.mean((x_pre.reshape(1,-1) - x_test[:,1])**2)


plt.figure(figsize=(10,10))
plt.scatter(x_test[:,0],x_test[:,1],marker='.',c='#66ccff', label='Test Samples')
plt.scatter(x_test[:,0],x_pre,marker='x',c='#ffcc66', label='Estimated Performance')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('x1',fontsize=20)
plt.ylabel('x2',fontsize=20)
plt.title('Performance of model with %d nodes and activation function of %s'%(node_best,act_best), fontsize=20)
plt.legend(fontsize=20)
print('mse = %f'%(mse))