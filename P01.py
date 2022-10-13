#################################################################
########################## IMPORTS ##############################
#################################################################

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from P01_utils import lire_donnees, visualiser_donnees

"""
This file is designed to implement a k neighbors classifier. 
Our application focus on classifying whether a subject is a male or female given his height and weight.
We will implement the k neighbors classifier from scratch and with numpy and scipy libraries.
"""

#################################################################
########################## FUNCTIONS ############################
#################################################################

np.random.seed(0)

def make_blobs(n_samples, centers, cluster_std):
    # Version simplifiée de la fonction `make_blobs` de scikit-learn
    centers = np.array(centers)
    n_features = centers.shape[1]
    n_centers = centers.shape[0]

    X = []
    y = []

    n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(np.random.normal(loc=centers[i], scale=std,
                                  size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y


def lire_donnees(n_individus):
    # Moyennes issues de https://liguecontrelobesite.org/actualite/taille-poids-et-tour-de-taille-photographie-2020-des-francais/
    X, y = make_blobs(n_samples=n_individus,
                      centers=[[164, 64], [177, 79]],
                      cluster_std=[[20, 5], [20, 5]])
    y_str = np.empty(y.shape, dtype=str)
    y_str[y == 0] = "F"
    y_str[y == 1] = "H"
    return X, y_str


def visualiser_donnees(X, y, X_test=None, nom_fichier=None):
    plt.figure()
    for sexe in ["F", "H"]:
        plt.scatter(X[y == sexe, 0], X[y == sexe, 1], label=sexe)
    if X_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", color="k")
    plt.xlabel("Taille")
    plt.legend(loc="upper left")    
    plt.show()

# This function compute the euclidean distance between 2 points
# Formula can be found here : https://www.normalesup.org/~simonet/teaching/caml-prepa/tp-caml-2001-02.pdf
# @Inputs : 
#  - X_i, Y_i : coordinates of first point
#  - X_j, Y_j : coordinates of second point
# @Outputs 
#  - dist : euclidean distance between the two point

def dist(X_i,Y_i, X_j, Y_j) :
    dist = math.sqrt(pow(X_i-X_j,2) + pow(Y_i-Y_j,2))
    return dist

# This function return the k_classifier prediction of a test point from a training dataset
# @Inputs :
#  - X_test : the coordinates of the point to predict. A tuple (x,y)
#  - X_train : training dataset. A numpy array of dimensions (number_of_training_samples, 2)
#  - Y_train : classes of the training dataset. A numpy array of dimensions (number_of_training_samples, 1). 
#              Classes are defined as 'H' for male and 'F' for female
#  - k : number of neighbor to take into account
#  @Outputs :
#  - char 'H' or 'F' : the prediction of the k_classifier

def k_classifier(X_test, X_train, Y_train, k) :
    distance = []
    cnt_M = 0
    cnt_F = 0
    # compute the distance between test point and training dataset
    for i in range(0, X_train.shape[0]) : 
        distance.append(dist(X_train[i,0], X_train[i,1], X_test[0], X_test[1]))
    # return an array containing the indexs sorted by min distance
    res = np.argsort(np.array(distance)) 
    # for k-closest points, count if male or female 
    for i in range(k) :
        if Y_train[res[i]] == 'H' : 
            cnt_M += 1
        else : 
            cnt_F += 1
    # return prediction by comparing counters 
    if cnt_M > cnt_F :
        return 'H'
    else :
        return 'F'

# This function returns the prediction of a k_classifier for a list of inputs.
# @Inputs :
#  - X_train : training dataset. A numpy array of dimensions (number_of_training_samples, 2)
#  - Y_train : classes of the training dataset. A numpy array of dimensions (number_of_training_samples, 1). 
#              Classes are defined as 'H' for male and 'F' for female
#  - X_test : the coordinates of the point to predict. A tuple (x,y)
#  - k : number of neighbor to take into account
# @Outputs :
#  - pred : a list of prediction from the k_classifier for each test points

def k_plus_proche_voisin_liste(X_train,Y_train,X_test, k) :
    pred = []
    for i in range(0,X_test.shape[0]) :
        pred.append(k_classifier(X_test[i,:],X_train,Y_train, k))
    return pred

# This function returns the prediction of a k_classifier for a list of inputs with numpy library.
# @Inputs :
#  - X_train : training dataset. A numpy array of dimensions (number_of_training_samples, 2)
#  - Y_train : classes of the training dataset. A numpy array of dimensions (number_of_training_samples, 1). 
#              Classes are defined as 'H' for male and 'F' for female
#  - X_test : the coordinates of the point to predict. A tuple (x,y)
#  - k : number of neighbor to take into account
# @Outputs :
#  - 'H' or 'F' : a prediction from the k_classifier

def np_k_classifier(X_test, X_train, Y_train, k) :
    pred = []
    H_list = []
    # compute distance between training dataset and test point
    distance = cdist(X_test, X_train, metric='euclidean')
    # return an array containing the indexs sorted by min distance
    res = np.argsort(np.array(distance))
    # keep the k min distance indexs
    res = res[:,0:k]
    for i in range(res.shape[0]) :
        for j in range(k) :
            H_list.append(Y_train[res[i,j]] == 'H')
        if np.sum(H_list) > k/2 :
            pred.append('H')
        else :
            pred.append('F')
        H_list = []
    return pred

# This function is designed to test the prediction of a k_classifier for a list of inputs. It print out accuracy of the k-classifier.
# @Inputs :
#  - X_train : training dataset. A numpy array of dimensions (number_of_training_samples, 2)
#  - Y_train : classes of the training dataset. A numpy array of dimensions (number_of_training_samples, 1). 
#              Classes are defined as 'H' for male and 'F' for female
#  - X_test : the coordinates of the point to predict. A tuple (x,y)
#  - k : number of neighbor to take into account

def test_k_plus_proche_voisin_liste(X_train,Y_train,X_test, Y_test, k) :
    cnt = 0
    pred  = k_plus_proche_voisin_liste(X_train, Y_train, X_test, k)
    for i in range(len(pred)) :
        if pred[i] == Y_test[i] : 
            cnt+=1
    acc = (cnt/len(pred))*100
    print("Accuracy of the", k, "classifier is : ", acc, " %\n")

# This function is designed to test the prediction of a numpy k_classifier for a list of inputs. It print out accuracy of the k-classifier.
# @Inputs :
#  - X_train : training dataset. A numpy array of dimensions (number_of_training_samples, 2)
#  - Y_train : classes of the training dataset. A numpy array of dimensions (number_of_training_samples, 1). 
#              Classes are defined as 'H' for male and 'F' for female
#  - X_test : the coordinates of the point to predict. A tuple (x,y)
#  - k : number of neighbor to take into account

def test_np_k_plus_proche_voisin_liste(X_train,Y_train,X_test, Y_test, k) :
    cnt = 0
    pred  = np_k_classifier(X_test, X_train, Y_train, k)
    for i in range(len(pred)) :
        if pred[i] == Y_test[i] : 
            cnt+=1
    acc = (cnt/len(pred))*100
    print("Accuracy of the numpy", k, "classifier is : ", acc, " %\n")

#################################################################
##########################   MAIN    ############################
#################################################################

X_train, Y_train = lire_donnees(100)
X_test, Y_test = lire_donnees(10)
test = (195,90)
test1 = (165,58)
# single_pred = k_classifier(test, X_train, Y_train, k=3)
# list_of_pred = k_plus_proche_voisin_liste(X_train, Y_train, X_test, k=90)
# print(single_pred)
# print(list_of_pred)
test_k_plus_proche_voisin_liste(X_train,Y_train,X_test, Y_test, 100)
test_np_k_plus_proche_voisin_liste(X_train,Y_train,X_test, Y_test, 100)
# visualiser_donnees(X_train,Y_train, X_test, Y_test)
