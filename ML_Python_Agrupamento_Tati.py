#Importações
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

#Agrupamento KMeans
iris=datasets.load_iris()
unicos,quantidade=np.unique(iris.target,return_counts=True)

cluster = KMeans(n_clusters=3)
cluster.fit(iris.data)

centroides= cluster.cluster_centers_
previsoes= cluster.labels_

unicos2,quantidade2=np.unique(previsoes,return_counts=True)

resultados= confusion_matrix(iris.target,previsoes)

plt.scatter(iris.data[previsoes==1,0],iris.data[previsoes==1,1],
            c= "green",label="Setosa")
plt.scatter(iris.data[previsoes==2,0],iris.data[previsoes==2,1],
            c= "red",label="Versicolor")
plt.scatter(iris.data[previsoes==0,0],iris.data[previsoes==0,1],
            c= "blue",label="Virginica")
plt.legend()


# Fuzzy C-Means
r= skfuzzy.cmeans(data=iris.data.T, c=3, m=2, error=0.005, maxiter=1000,init=None)
previsoes_porcentagem=r[1]
previsoes_porcentagem[0][0]
previsoes_porcentagem[1][0]
previsoes_porcentagem[2][0]

previsoes2=previsoes_porcentagem.argmax(axis=0)
resultados2=confusion_matrix(iris.target,previsoes2)


# K-Medoids
cluster2= kmedoids(iris.data[:,0:2],[3,12,20])
cluster2.get_medoids()
cluster2.process()
previsoes3= cluster2.get_clusters()
medoides= cluster2.get_medoids()

v=cluster_visualizer()
v.append_clusters(previsoes3,iris.data[:,0:2])
v.append_cluster(medoides,iris.data[:,0:2],marker="*",markersize=15)
v.show()

lista_previsoes=[]
lista_real=[]
for i in range(len(previsoes3)):
    print("----")
    print(i)
    print("----")
    for j in range(len(previsoes3[i])):
        #print(j)
        print(previsoes3[i][j])
        lista_previsoes.append(i)
        lista_real.append(iris.target[previsoes3[i][j]])

lista_previsoes=np.asarray(lista_previsoes)
lista_real=np.asarray(lista_real)
resultados3=confusion_matrix(lista_real,lista_previsoes)
    