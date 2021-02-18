#Importações
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

credito=pd.read_csv('Credit.csv')
previsores=credito.iloc[:,0:20].values
classe=credito.iloc[:,20].values

#SKLearn não entende colunas categóricas, transformado em números para classificar
labelencoder= LabelEncoder()
previsores[:,0]=labelencoder.fit_transform(previsores[:,0])
previsores[:,2]=labelencoder.fit_transform(previsores[:,2])
previsores[:,3]=labelencoder.fit_transform(previsores[:,3])
previsores[:,5]=labelencoder.fit_transform(previsores[:,5])
previsores[:,6]=labelencoder.fit_transform(previsores[:,6])
previsores[:,8]=labelencoder.fit_transform(previsores[:,8])
previsores[:,9]=labelencoder.fit_transform(previsores[:,9])
previsores[:,11]=labelencoder.fit_transform(previsores[:,11])
previsores[:,13]=labelencoder.fit_transform(previsores[:,13])
previsores[:,14]=labelencoder.fit_transform(previsores[:,14])
previsores[:,16]=labelencoder.fit_transform(previsores[:,16])
previsores[:,18]=labelencoder.fit_transform(previsores[:,18])
previsores[:,19]=labelencoder.fit_transform(previsores[:,19])

#Dividindo 70% para treino. 30% para teste (semelhante técnica para amostragem)
X_treinamento,X_teste,y_treinamento,y_teste=train_test_split(previsores,classe,
                                                             test_size=0.3,
                                                             random_state=0)
#Naive Bayes
naive_bayes=GaussianNB()
naive_bayes.fit(X_treinamento,y_treinamento)

previsoes=naive_bayes.predict(X_teste)

confusao=confusion_matrix(y_teste,previsoes)
taxa_acerto=accuracy_score(y_teste,previsoes)
taxa_erro=1-taxa_acerto

novo_credito=pd.read_csv("NovoCredit.csv")
novo_credito=novo_credito.iloc[:,0:20].values
novo_credito[:,0]=labelencoder.fit_transform(novo_credito[:,0])
novo_credito[:,2]=labelencoder.fit_transform(novo_credito[:,2])
novo_credito[:,3]=labelencoder.fit_transform(novo_credito[:,3])
novo_credito[:,5]=labelencoder.fit_transform(novo_credito[:,5])
novo_credito[:,6]=labelencoder.fit_transform(novo_credito[:,6])
novo_credito[:,8]=labelencoder.fit_transform(novo_credito[:,8])
novo_credito[:,9]=labelencoder.fit_transform(novo_credito[:,9])
novo_credito[:,11]=labelencoder.fit_transform(novo_credito[:,11])
novo_credito[:,13]=labelencoder.fit_transform(novo_credito[:,13])
novo_credito[:,14]=labelencoder.fit_transform(novo_credito[:,14])
novo_credito[:,16]=labelencoder.fit_transform(novo_credito[:,16])
novo_credito[:,18]=labelencoder.fit_transform(novo_credito[:,18])
novo_credito[:,19]=labelencoder.fit_transform(novo_credito[:,19])

naive_bayes.predict(novo_credito)

#Árvore de Decisão
arvore=DecisionTreeClassifier()
arvore.fit(X_treinamento,y_treinamento)

export_graphviz(arvore, out_file="tree.dot")
previsoes2=arvore.predict(X_teste)
confusao2=confusion_matrix(y_teste,previsoes2)
taxa_acerto2=accuracy_score(y_teste,previsoes2)
taxa_erro2=1-taxa_acerto2

#Algoritmo SVM
svm=SVC()
svm.fit(X_treinamento,y_treinamento)
previsoes3=svm.predict(X_teste)
confusao3=confusion_matrix(y_teste,previsoes3)
taxa_acerto3=accuracy_score(y_teste,previsoes3)
taxa_erro3=1-taxa_acerto3

#Seleção de Atributos + SVM (mesmo algoritmo anterior para comparação)
forest=ExtraTreesClassifier()
forest.fit(X_treinamento,y_treinamento)
importancias= forest.feature_importances_
X_treinamento2=X_treinamento[:,[0,1,2,3]]
X_teste2=X_teste[:,[0,1,2,3]]
svm2=SVC()
svm2.fit(X_treinamento2,y_treinamento)
previsoes4=svm2.predict(X_teste2)
taxa_acerto4=accuracy_score(y_teste,previsoes4)
taxa_erro4=1-taxa_acerto4

#Vizinho Próximo (Baseado em Instância)
iris=datasets.load_iris()
stats.describe(iris.data)
previsores2=iris.data
classe2=iris.target
X_treinamento3,X_teste3,y_treinamento2,y_teste2=train_test_split(previsores2,classe2,
                                                             test_size=0.3,
                                                             random_state=0)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_treinamento3,y_treinamento2)
previsoes5=knn.predict(X_teste3)
confusao4=confusion_matrix(y_teste2,previsoes5)
taxa_acerto5=accuracy_score(y_teste2,previsoes5)
taxa_erro5=1-taxa_acerto5

#Ensamble Learning (florestas randômicas)
floresta=RandomForestClassifier(n_estimators=100)
floresta.fit(X_treinamento,y_treinamento)
previsoes6=floresta.predict(X_teste)
confusao5=confusion_matrix(y_teste,previsoes6)
taxa_acerto6=accuracy_score(y_teste,previsoes6)
taxa_erro6=1-taxa_acerto2