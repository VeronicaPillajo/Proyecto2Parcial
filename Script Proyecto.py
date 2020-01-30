# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:10:34 2020

@authors: Anthony Cárdenas, Edwin Iza, Verónica Pillajo
"""

import csv
import collections
import numpy as np
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



temas = []
areas = []

#METODO PARA LEER EL CSV Y OBTENER LO QUE NECESITEMOS

with open('temas.csv', newline='',encoding="utf8") as File:
    reader = csv.reader(File)
    next(reader, None)
    for row in reader:
        temas.append(row[1])
        areas.append(row[2])



#CODIFICACION DE LOS TEMAS POR LAS AREAS RESPECTIVAS 
val_areas =  (collections.Counter(areas))
labels_areas = list(collections.Counter(areas))

lista_val_areas = list(val_areas.values())
lista_keys_areas = list(val_areas.keys())

vec_keys = []
for i in range(0,len(lista_keys_areas)):
    vec_keys.append(i)

valores_session = []
for i in range(0,len(areas)):
    for j in range(0,len(vec_keys)):
        if(areas[i]==lista_keys_areas[j]):
            valores_session.append(vec_keys[j])
            
            
## MATRIZ JACCARD PARA VER LA SIMILITUD DE LOS TEMAS POR AREAS (SOFTWARE,MACHINE LEARNING etc...)

#METODO NORMALIZE QUE NOS PERMITE ELIMINAR LOS PUNTOS DE LOS DOCUMENTOS
def normalize(s):   #Metodo para reemplazar los acentuaciones de las vocales
    replacements = (
        (".", ""),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s


#METODO QUE NOS PERMITE HACER EL NLP
    
def limpieza(lista):  # Metodo para realizar el limpieza
    voca = []  # Vector
    acum = ""

    for i in lista:  # Recorrer el vector
        separacion = i.split()  # Separamos la lista mediante el argumento "" (espacio)
        for j in range(0, len(separacion)):
            separacion[j] = normalize(separacion[j])
            limpieza2 = re.sub('[^a-zA-Z \n.]', ' ', separacion[j])  # Eliminacion de caracteres especiales

            n2 = limpieza2.lower()  # Colocar en minuscula
            n3 = n2.split()  # Separacion
            n5 = stopwords.words('spanish')

            for word in n3:  # Ciclo for para remover las palabras
                if word in n5:
                    n3.remove(word)
            stemmer = PorterStemmer()


            for i in n3:  # Acumulacion de la palabras para cada titulo
                acum = acum + " " + str(stemmer.stem(i))
        voca.append(acum)
        acum = ""
    return voca

def jaccard(lista):
    matrizcsv=[]
    #CREAMOS LA MATRIZ DE 398*398
    for i in range(len(lista)):
        matrizcsv.append([0]*len(lista))

    for f in range(0,len(lista)):
        list1 = lista[f].split(" ") #SEPARO EL TITULO 
        s1 = set(list1) #HAGO UNA COLECCION DE LA LISTA1
        for c in range(0,len(lista)):
            list2 = lista[c].split(" ") #SEPARO EL TITULO 
            s2 = set(list2) #HAGO UNA COLECCION DE LA LISTA2

            # REALIZO LA INTERSECCION DE LA L1 Y L2 PARA LUEGO DIVIDIR PARA LA 
            #UNION DE LA L1 Y L2 
            resu = (len(s1.intersection(s2)))/(len(s1.union(s2)))
            
            #GUARDO EL LA MATRIZ SU RESPECTIVO RESULTADO
            matrizcsv[f][c] = resu

    return matrizcsv


diccionario_temas_area = []
diccionario_temas = []
temas_area = []

for i in range(0,len(lista_keys_areas)):
    for j in range(0,len(valores_session)):
        if(valores_session[j] == i):
            temas_area.append(temas[j])
            diccionario_temas.append(temas[j])
    diccionario_temas_area.append(temas_area)
    temas_area=[]


##OBTENEMOS QUE SIMILARES SON LOS TEMAS POR AREA 
jaccard_temas_area = []
for i in range(0,len(lista_keys_areas)):
        temas_area = limpieza(diccionario_temas_area[i])
        jaccard_temas_area.append(np.array(jaccard(temas_area)))
 

matriz_temas_general = np.array(jaccard(diccionario_temas))
       
##APRENDIZAJE NO SUPERVISADO --- CLUSTERING 

def metodo_elbow(col_data_iris):
    val_withins = []
    cont = 1
    while (cont <= 10):
        k_means = KMeans(n_clusters=cont)
        k_means.fit(col_data_iris)
        val_withins.append(k_means.inertia_)
        cont = cont + 1
    return val_withins


print("Valores Withins K-Means")

val_elbow = metodo_elbow(matriz_temas_general)


def graficar_elbow(val_elbow):

    (plt.title('Gráfica del método de Elbow'))
    (plt.xlabel('Número de clusters'))
    (plt.ylabel('Suma de Cuadrados Internos'))
    (plt.plot(range(1, 11), val_elbow))
    plt.savefig("public/assets/img/elbow.png")
    plt.show()
    return plt.show()


grafico = graficar_elbow(val_elbow)


def valores_dhc(X,n_clus,metodo):
    
    Z = linkage(X, 'ward', metric=metodo)
    cluster_dhc = AgglomerativeClustering(n_clusters=n_clus, affinity=metodo, linkage='ward')
    plt.figure(figsize=(10, 8))
    plt.title('Dendograma jerárquico DHC')
    plt.ylabel('Distance')
    max_d = 10
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True
    )
    plt.axhline(y=max_d, c='k')
    plt.savefig("public/assets/img/dendograma.png")
    plt.show()
    
    return [cluster_dhc,cluster_dhc.fit_predict(X)]

val_dhc = valores_dhc(matriz_temas_general,4,"euclidean")

def silueta_dhc(X,val_cluster,valores_dhc):
    range_n_clusters = [val_cluster]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        
        
        clusterer = valores_dhc[0]       #DHC CLUSTER
        cluster_labels = valores_dhc[1]  #ETIQUETAS (Predicciones)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("Para n_clusters =", n_clusters,
          "\nEl promedio de la silhouette_score es :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters): 
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("Valores del coeficiente de la silueta")
        ax1.set_ylabel("Cluster Label (Etiqueta)")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="black", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

        ax2.set_title("Visualización de la data del cluster")
        ax2.set_xlabel("Espacio de características para la 1ra característica")
        ax2.set_ylabel("Espacio de características para la 2da característica")

        plt.suptitle(("Silhouette analysis for DHC clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig("public/assets/img/silueta_clustering.png")
    plt.show()
    return cluster_labels
##Llamada al metodo silueta DHC
silueta_dhc = silueta_dhc(matriz_temas_general,4,val_dhc)

##################
#######################
#CLUSTERING VALIDACION 

cont = 0
clus_val_BDD=0
clus_val_SOFT=0
clus_val_REDES=0
clus_val_ML=0
for i in range(0,len(temas)):
    if(val_dhc[1][i] != valores_session[i]):
        cont=cont+1
    if(val_dhc[1][i]==0):
        clus_val_BDD=clus_val_BDD+1
    if(val_dhc[1][i]==1):
        clus_val_SOFT=clus_val_SOFT+1
    if(val_dhc[1][i]==2):
        clus_val_REDES=clus_val_REDES+1
    if(val_dhc[1][i]==3):
        clus_val_ML=clus_val_ML+1
        
lista_val_area_clus = [clus_val_BDD,clus_val_SOFT,clus_val_REDES,clus_val_ML]
error_clustering = cont/len(temas)


###########################

#IMAGENES CLUSTERING

names = ['BASE DE DATOS', 'SOFTWARE', 'REDES','MACHINE LEARNING']
values = [lista_val_areas[0],lista_val_areas[1],lista_val_areas[2],lista_val_areas[3]]
plt.figure(figsize=(20, 10))
plt.plot(131)
plt.bar(names, values)
plt.savefig("public/assets/img/ground")
plt.show()

plt.figure(figsize=(20, 10))
plt.title('                                                                                      Ground Truth VS Groups Clustering')
plt.plot(132)
values = [lista_val_area_clus[0],lista_val_area_clus[1],lista_val_area_clus[2],lista_val_area_clus[3]]
plt.bar(names, values)
plt.savefig("public/assets/img/clus")
plt.show()

#######

#####################################################

#APRENDIZAJE SUPERVISADO CLASIFICACIÓN

## CREACION DE LAS 4 BOLSAS DE PALABRAS 

corpus_BDD=""
corpus_SOFT=""
corpus_REDES=""
corpus_ML=""

for i in range(0,len(diccionario_temas_area)):
    
    for j in range(0,len(diccionario_temas_area[i])):
        if(i==0):
            corpus_BDD = corpus_BDD + str(nltk.sent_tokenize(diccionario_temas_area[i][j])) 
        if(i==1):
            corpus_SOFT = corpus_SOFT + str(nltk.sent_tokenize(diccionario_temas_area[i][j]))
        if(i==2):
            corpus_REDES = corpus_REDES + str(nltk.sent_tokenize(diccionario_temas_area[i][j]))
        if(i==3):
            corpus_ML = corpus_ML + str(nltk.sent_tokenize(diccionario_temas_area[i][j])) 

    

for i in range(0,len(corpus_BDD)):  
    corpus_BDD  = corpus_BDD.lower()
    corpus_BDD = re.sub(r'\W',' ',corpus_BDD)
    corpus_BDD = re.sub(r'\s+',' ',corpus_BDD )

for i in range(len(corpus_SOFT)): 
    corpus_SOFT  = corpus_SOFT.lower()
    corpus_SOFT = re.sub(r'\W',' ',corpus_SOFT)
    corpus_SOFT = re.sub(r'\s+',' ',corpus_SOFT )
    
for i in range(len(corpus_REDES)):    
    corpus_REDES  = corpus_REDES.lower()
    corpus_REDES  = re.sub(r'\W',' ',corpus_REDES)
    corpus_REDES  = re.sub(r'\s+',' ',corpus_REDES)
    
for i in range(len(corpus_ML)):
    corpus_ML = corpus_ML.lower()
    corpus_ML = re.sub(r'\W',' ',corpus_ML)
    corpus_ML = re.sub(r'\s+',' ',corpus_ML)
    
wordfreq_BDD = {}  
wordfreq_SOFT = {}  
wordfreq_REDES = {}  
wordfreq_ML = {}  

corpus_BDD = [corpus_BDD]
corpus_SOFT = [corpus_SOFT]
corpus_REDES = [corpus_REDES]
corpus_ML = [corpus_ML]

for sentence in corpus_BDD:  
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_BDD.keys():
            wordfreq_BDD[token] = 1
        else:
            wordfreq_BDD[token] += 1

for sentence in corpus_SOFT:  
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_SOFT.keys():
            wordfreq_SOFT[token] = 1
        else:
            wordfreq_SOFT[token] += 1

for sentence in corpus_REDES:  
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_REDES.keys():
            wordfreq_REDES[token] = 1
        else:
            wordfreq_REDES[token] += 1

for sentence in corpus_ML:  
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_ML.keys():
            wordfreq_ML[token] = 1
        else:
            wordfreq_ML[token] += 1
        
bolsa_BDD = ""
bolsa_SOFT = ""
bolsa_REDES = ""
bolsa_ML = ""

wordfreq_BDD = sorted(wordfreq_BDD.items(), key=lambda value: value[1]) 
wordfreq_SOFT = sorted(wordfreq_SOFT.items(), key=lambda value: value[1]) 
wordfreq_REDES = sorted(wordfreq_REDES.items(), key=lambda value: value[1]) 
wordfreq_ML = sorted(wordfreq_ML.items(), key=lambda value: value[1]) 


for i in range(len(wordfreq_BDD)-50,len(wordfreq_BDD)):
  bolsa_BDD=bolsa_BDD+" "+str(wordfreq_BDD[i][0])


for i in range(len(wordfreq_SOFT)-50,len(wordfreq_SOFT)):
  bolsa_SOFT=bolsa_SOFT+" "+str(wordfreq_SOFT[i][0])

for i in range(len(wordfreq_REDES)-50,len(wordfreq_REDES)):
  bolsa_REDES=bolsa_REDES+" "+str(wordfreq_REDES[i][0])


for i in range(len(wordfreq_ML)-50,len(wordfreq_ML)):
  bolsa_ML=bolsa_ML+" "+str(wordfreq_ML[i][0])



bolsa_areas = [bolsa_BDD,bolsa_SOFT,bolsa_REDES,bolsa_ML]
tema_csv = ""
val_jaccard = 0
max_val_jaccard = []
X=[]
for i in range(0,len(temas)):
    tema_csv= temas[i].lower()
    tema_csv= re.sub(r'\W',' ',tema_csv)
    tema_csv= re.sub(r'\W',' ',tema_csv)
    for j in range(0,len(diccionario_temas_area)):
        val_jaccard = jaccard([tema_csv,bolsa_areas[j]])
        max_val_jaccard.append(val_jaccard[0][1])
    X.append((max_val_jaccard))
    max_val_jaccard=[]

### EVALUACION 
    
y = valores_session
X = np.array(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)



algoritmoNB = GaussianNB()

algoritmoNB.fit(X_train,y_train)

y_predNB = algoritmoNB.predict(X_test)


#CALCULO DE LA PRECISION DEL MODELO 
precision_NB = precision_score(y_test, y_predNB,  pos_label='positive',average='micro')*100
print("Precision Score Naive Bayes : ",precision_NB)

###############################################
# KNN VECINOS MAS CERCANOS 

algoritmoKNN = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)

##
algoritmoKNN.fit(X_train,y_train)

y_predKNN = algoritmoKNN.predict(X_test)

#CALCULO DE LA PRESICION DEL MODELO 

precision_KNN = precision_score(y_test, y_predKNN,  pos_label='positive',average='micro')*100
print("Precision Score KNN : ",precision_KNN)


algoritmoRDF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
algoritmoRDF.fit(X_train,y_train)

y_predRDF = algoritmoRDF.predict(X_test)

#CALCULO DE LA PRESICION DEL MODELO 


precision_RDF = precision_score(y_test, y_predRDF,  pos_label='positive',average='micro')*100
print("Precision Score Random Forest : ",precision_RDF)

#####


escalar = StandardScaler()
X_train_log = escalar.fit_transform(X_train)
X_test_log = escalar.transform(X_test)

algoritmoLOG = LogisticRegression()

algoritmoLOG = algoritmoLOG.fit(X_train,y_train)

#Predicción

y_predLOG = algoritmoLOG.predict(X_test_log)
precision_LOG = precision_score(y_test, y_predLOG,  pos_label='positive',average='micro')*100
print("Precision Score Logistic Regression : ",precision_LOG)

names = ['Naive Bayes', 'KNN', 'Random Forest','Logistic Regression']
values = [precision_NB,precision_KNN,precision_RDF,precision_LOG]

plt.figure(figsize=(25, 10))

plt.plot(131)
plt.bar(names, values)
plt.title('Resultados Estadístico 1')
plt.savefig("public/assets/img/clasificacion1")
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(132)
plt.scatter(names, values)
plt.suptitle('Distribución Puntos Algoritmos')
plt.savefig("public/assets/img/clasificacion2")
plt.show()


