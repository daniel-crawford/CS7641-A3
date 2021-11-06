#Unsupervised Learning and Dimensionality Reduction
#Daniel Crawford

import sklearn
from sklearn import datasets
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from sklearn import random_projection
from sklearn.neural_network import MLPClassifier

import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

import yellowbrick
from yellowbrick.cluster import SilhouetteVisualizer

#Load in data sets
iris = datasets.load_iris()
diabetes = datasets.load_breast_cancer()


N_RUNS = 100
#1. Run the clustering algorithms on the datasets and describe what you see.
iris_kmeans = KMeans(
    n_clusters = 3,
    init = 'random',
    n_init = N_RUNS,
    max_iter = 1000).fit(iris.data)


iris_em_labels = GaussianMixture(
    n_components = 3,
    covariance_type = 'full',
    tol = 0.001,
    n_init = N_RUNS,
    init_params = 'random').fit_predict(iris.data)




#iris_kmeans_translated = np.array([1 if i == 2 else 2 if i == 1 else 0 for i in iris_kmeans.labels_])

plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_kmeans.labels_])
plt, axs = plt.subplots(2,3)
axs[0,0].scatter(iris.data[:,0], iris.data[:,1],c = plot_colors)
axs[0,0].set(xlabel = "Sepal Length", ylabel = "Sepal Width")
axs[0,1].scatter(iris.data[:,0], iris.data[:,2],c = plot_colors)
axs[0,1].set(xlabel = "Sepal Length", ylabel = "Petal Length")
axs[0,2].scatter(iris.data[:,0], iris.data[:,3],c = plot_colors)
axs[0,2].set(xlabel = "Sepal Length", ylabel = "Petal Width")
axs[1,0].scatter(iris.data[:,1], iris.data[:,2],c = plot_colors)
axs[1,0].set(xlabel = "Petal Length", ylabel = "Sepal Width")
axs[1,1].scatter(iris.data[:,2], iris.data[:,3],c = plot_colors)
axs[1,1].set(xlabel = "Petal Length", ylabel = "Petal Width")
axs[1,2].scatter(iris.data[:,2], iris.data[:,3],c = plot_colors)
axs[1,2].set(xlabel = "Sepal Width", ylabel = "Petal Width")
plt.suptitle("KM Clustering of Iris Data Set")
plt.show()

plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_em_labels])
plt, axs = plt.subplots(2,3)
axs[0,0].scatter(iris.data[:,0], iris.data[:,1],c = plot_colors)
axs[0,0].set(xlabel = "Sepal Length", ylabel = "Sepal Width")
axs[0,1].scatter(iris.data[:,0], iris.data[:,2],c = plot_colors)
axs[0,1].set(xlabel = "Sepal Length", ylabel = "Petal Length")
axs[0,2].scatter(iris.data[:,0], iris.data[:,3],c = plot_colors)
axs[0,2].set(xlabel = "Sepal Length", ylabel = "Petal Width")
axs[1,0].scatter(iris.data[:,1], iris.data[:,2],c = plot_colors)
axs[1,0].set(xlabel = "Petal Length", ylabel = "Sepal Width")
axs[1,1].scatter(iris.data[:,2], iris.data[:,3],c = plot_colors)
axs[1,1].set(xlabel = "Petal Length", ylabel = "Petal Width")
axs[1,2].scatter(iris.data[:,2], iris.data[:,3],c = plot_colors)
axs[1,2].set(xlabel = "Sepal Width", ylabel = "Petal Width")
plt.suptitle("EM Clustering of Iris Data Set")
plt.show()

diabetes_kmeans = KMeans(
    n_clusters = 2,
    init = 'random',
    n_init = N_RUNS,
    max_iter = 1000).fit(diabetes.data)

diabetes_em = GaussianMixture(
    n_components = 2,
    covariance_type = 'full',
    tol = 0.001,
    n_init = N_RUNS,
    init_params = 'random').fit(diabetes.data)

a = []
b = []
for n in range(1,31):
    d= GaussianMixture(
    n_components = 2,
    covariance_type = 'full',
    tol = 0.001,
    n_init = N_RUNS,
    init_params = 'random').fit(diabetes.data)

    a.append(d.aic(diabetes.data))
    b.append(d.bic(diabetes.data))


plt.plot(a,label = "AIC")
plt.plot(b, label = "BIC")
plt.legend()

print('done')

#Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
IRIS_N_COMPONENTS = 2
iris_PCA_data = PCA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_ICA_data = FastICA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_KernalPCA_data = KernelPCA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_IncrementalPCA_data = IncrementalPCA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_RP_data = random_projection.GaussianRandomProjection(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
#EPS!


plt.scatter(iris_PCA_data[:,0], iris_PCA_data[:,1])
plt.title("Iris PCA")
plt.savefig("irisPCA.png")
plt.close()

plt.scatter(iris_ICA_data[:,0], iris_ICA_data[:,1])
plt.title("Iris ICA")
plt.savefig("irisICA.png")
plt.close()

plt.scatter(iris_KernalPCA_data[:,0], iris_KernalPCA_data[:,1])
plt.title("Iris KernelPCA")
plt.savefig("irisKPCA.png")
plt.close()

plt.scatter(iris_IncrementalPCA_data[:,0], iris_IncrementalPCA_data[:,1])
plt.title("Iris Incremental PCA")
plt.savefig("irisIPCA.png")
plt.close()

plt.scatter(iris_RP_data[:,0], iris_RP_data[:,1])
plt.title("Iris Randomized Projection")
plt.savefig("irisRP.png")
plt.show()


DIABETES_N_COMPONENTS = 2
diabetes_PCA_data = PCA(n_components = DIABETES_N_COMPONENTS).fit_transform(diabetes.data)
diabetes_ICA_data = FastICA(n_components = DIABETES_N_COMPONENTS).fit_transform(diabetes.data)
diabetes_KernalPCA_data = KernelPCA(n_components = DIABETES_N_COMPONENTS).fit_transform(diabetes.data)
diabetes_IncrementalPCA_data = IncrementalPCA(n_components = DIABETES_N_COMPONENTS).fit_transform(diabetes.data)
diabetes_RP_data = random_projection.GaussianRandomProjection(n_components = DIABETES_N_COMPONENTS).fit_transform(diabetes.data)


plt.scatter(diabetes_PCA_data[:,0], diabetes_PCA_data[:,1])
plt.title("Cancer PCA")
plt.savefig("diaPCA.png")
plt.close()

plt.scatter(diabetes_ICA_data[:,0], diabetes_ICA_data[:,1])
plt.title("Cancer ICA")
plt.savefig("diaICA.png")
plt.close()

plt.scatter(diabetes_KernalPCA_data[:,0], diabetes_KernalPCA_data[:,1])
plt.title("Cancer KernelPCA")
plt.savefig("diaKPCA.png")
plt.close()

plt.scatter(diabetes_IncrementalPCA_data[:,0], diabetes_IncrementalPCA_data[:,1])
plt.title("Cancer Incremental PCA")
plt.savefig("diaIPCA.png")
plt.close()

plt.scatter(diabetes_RP_data[:,0], diabetes_RP_data[:,1])
plt.title("Cancer Randomized Projection")
plt.savefig("diaRP.png")
plt.close()


#3. Reproduce your clustering experiments, but on the data after you've
#run dimensionality reduction on it. Yes, that’s 16 combinations of datasets,
#dimensionality reduction, and clustering method. You should look at all of them,
#but focus on the more interesting findings in your report.

N_INIT = 100
MAX_ITER = 1000
iris_kmeans = KMeans( n_clusters = 3, init = 'random', n_init = N_INIT, max_iter = MAX_ITER).fit(iris_PCA_data)
iris_em_labels = GaussianMixture(n_components =3, covariance_type = 'full', tol = 0.001, n_init = 10, init_params = 'random').fit_predict(iris_PCA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_kmeans.labels_])
plt.scatter(iris_PCA_data[:,0],iris_PCA_data[:,1], c = plot_colors)
plt.title("Iris PCA KMeans")
plt.savefig("IrisPCA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_em_labels])
plt.scatter(iris_PCA_data[:,0],iris_PCA_data[:,1], c = plot_colors)
plt.title("Iris PCA EM")
plt.savefig("IrisPCA_EM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris.target])
plt.scatter(iris_PCA_data[:,0],iris_PCA_data[:,1], c = plot_colors)
plt.title("Iris PCA Target")
plt.savefig("IrisPCA_Target.png")
plt.close()

iris_kmeans = KMeans(n_clusters = 3, init = 'random', n_init = N_INIT, max_iter = MAX_ITER).fit(iris_ICA_data)
iris_em_labels = GaussianMixture( n_components =3,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(iris_ICA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_kmeans.labels_])
plt.scatter(iris_ICA_data[:,0],iris_ICA_data[:,1], c = plot_colors)
plt.title("IrisICA KMeans")
plt.savefig("IrisICA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_em_labels])
plt.scatter(iris_ICA_data[:,0],iris_ICA_data[:,1], c = plot_colors)
plt.title("Iris ICA EM")
plt.savefig("IrisICA_EM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris.target])
plt.scatter(iris_ICA_data[:,0],iris_ICA_data[:,1], c = plot_colors)
plt.title("Iris ICA Target")
plt.savefig("IrisICA_Target.png")
plt.close()

iris_kmeans = KMeans(n_clusters = 3,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(iris_KernalPCA_data)
iris_em_labels = GaussianMixture(n_components =3,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(iris_KernalPCA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_kmeans.labels_])
plt.scatter(iris_KernalPCA_data[:,0],iris_KernalPCA_data[:,1], c = plot_colors)
plt.title("Iris KernelPCA KMeans")
plt.savefig("IrisKerPCA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_em_labels])
plt.scatter(iris_KernalPCA_data[:,0],iris_KernalPCA_data[:,1], c = plot_colors)
plt.title("Iris KernelPCA EM")
plt.savefig("IrisKerPCA_EM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris.target])
plt.scatter(iris_KernalPCA_data[:,0],iris_KernalPCA_data[:,1], c = plot_colors)
plt.title("Iris KernalPCA Target")
plt.savefig("IrisKerPCA_Target.png")
plt.close()

iris_kmeans = KMeans(n_clusters = 3,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(iris_IncrementalPCA_data)
iris_em_labels = GaussianMixture(n_components =3,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(iris_IncrementalPCA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_kmeans.labels_])
plt.scatter(iris_IncrementalPCA_data[:,0],iris_IncrementalPCA_data[:,1], c = plot_colors)
plt.title("Iris IncPCA KMeans")
plt.savefig("IrisIncPCA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_em_labels])
plt.scatter(iris_IncrementalPCA_data[:,0],iris_IncrementalPCA_data[:,1], c = plot_colors)
plt.title("Iris IncPCA EM")
plt.savefig("IrisIncPCA_EM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris.target])
plt.scatter(iris_IncrementalPCA_data[:,0],iris_IncrementalPCA_data[:,1], c = plot_colors)
plt.title("Iris IncPCA Target")
plt.savefig("IrisInPCA_Target.png")
plt.close()

iris_kmeans = KMeans(n_clusters = 3,init = 'random',n_init = N_INIT, max_iter = MAX_ITER).fit(iris_RP_data)
iris_em_labels = GaussianMixture(n_components = 3,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(iris_RP_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_kmeans.labels_])
plt.scatter(iris_RP_data[:,0],iris_RP_data[:,1], c = plot_colors)
plt.title("Iris RP KMeans")
plt.savefig("IrisRP_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris_em_labels])
plt.scatter(iris_RP_data[:,0],iris_RP_data[:,1], c = plot_colors)
plt.title("Iris RP EM")
plt.savefig("IrisRP_EM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in iris.target])
plt.scatter(iris_RP_data[:,0],iris_RP_data[:,1], c = plot_colors)
plt.title("Iris RP Target")
plt.savefig("IrisRP_Target2.png")
plt.close()


N_INIT = 100
MAX_ITER = 1000

diabetes_kmeans = KMeans(n_clusters = 2,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(diabetes_PCA_data)
diabetes_em = GaussianMixture(n_components = 2,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(diabetes_PCA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_kmeans.labels_])
plt.scatter(diabetes_PCA_data[:,0],diabetes_PCA_data[:,1], c = plot_colors)
plt.title("Cancer PCA KMeans")
plt.savefig("DiaPCA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_em])
plt.scatter(diabetes_PCA_data[:,0],diabetes_PCA_data[:,1], c = plot_colors)
plt.title("Cancer PCA EM")
plt.savefig("DiaPCA_EM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes.target])
plt.scatter(diabetes_PCA_data[:,0],diabetes_PCA_data[:,1], c = plot_colors)
plt.title("Cancer PCA Target")
plt.savefig("DiaPCA_Target.png")
plt.close()

diabetes_kmeans = KMeans(n_clusters = 2,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(diabetes_ICA_data)
diabetes_em = GaussianMixture(n_components = 2,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(diabetes_ICA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_kmeans.labels_])
plt.scatter(diabetes_ICA_data[:,0],diabetes_ICA_data[:,1], c = plot_colors)
plt.title("Cancer ICA KMeans")
plt.savefig("DiaICA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_em])
plt.scatter(diabetes_ICA_data[:,0],diabetes_ICA_data[:,1], c = plot_colors)
plt.title("Cancer ICA EM")
plt.savefig("DiaICA_RM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes.target])
plt.scatter(diabetes_ICA_data[:,0],diabetes_ICA_data[:,1], c = plot_colors)
plt.title("Cancer ICA Target")
plt.savefig("DiaICA_Target.png")
plt.close()

diabetes_kmeans = KMeans(n_clusters = 2,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(diabetes_KernalPCA_data)
diabetes_em = GaussianMixture(n_components = 2,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(diabetes_KernalPCA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_kmeans.labels_])
plt.scatter(diabetes_KernalPCA_data[:,0],diabetes_KernalPCA_data[:,1], c = plot_colors)
plt.title("Cancer Kernel PCA KMeans")
plt.savefig("DiaKerPCA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_em])
plt.scatter(diabetes_KernalPCA_data[:,0],diabetes_KernalPCA_data[:,1], c = plot_colors)
plt.title("Cancer Kernel PCA EM")
plt.savefig("DiaKerPCA_RM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes.target])
plt.scatter(diabetes_KernalPCA_data[:,0],diabetes_KernalPCA_data[:,1], c = plot_colors)
plt.title("Cancer Kernel PCA Target")
plt.savefig("DiaKerPCA_Target.png")
plt.close()

diabetes_kmeans = KMeans(n_clusters = 2,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(diabetes_IncrementalPCA_data)
diabetes_em = GaussianMixture(n_components = 2,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(diabetes_IncrementalPCA_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_kmeans.labels_])
plt.scatter(diabetes_IncrementalPCA_data[:,0],diabetes_IncrementalPCA_data[:,1], c = plot_colors)
plt.title("Cancer Incremental PCA KMeans")
plt.savefig("DiaIncPCA_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_em])
plt.scatter(diabetes_IncrementalPCA_data[:,0],diabetes_IncrementalPCA_data[:,1], c = plot_colors)
plt.title("Cancer Incremental PCA EM")
plt.savefig("DiaIncPCA_RM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes.target])
plt.scatter(diabetes_IncrementalPCA_data[:,0],diabetes_IncrementalPCA_data[:,1], c = plot_colors)
plt.title("Cancer Incremental PCA Target")
plt.savefig("DiaIncPCA_Target.png")
plt.close()

diabetes_kmeans = KMeans(n_clusters = 2,init = 'random',n_init = N_INIT,max_iter = MAX_ITER).fit(diabetes_RP_data)
diabetes_em = GaussianMixture(n_components = 2,covariance_type = 'full',tol = 0.001,n_init = 10,init_params = 'random').fit_predict(diabetes_RP_data)
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_kmeans.labels_])
plt.scatter(diabetes_RP_data[:,0],diabetes_RP_data[:,1], c = plot_colors)
plt.title("Cancer RP KMeans")
plt.savefig("DiaIncRP_KM.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes_em])
plt.scatter(diabetes_RP_data[:,0],diabetes_RP_data[:,1], c = plot_colors)
plt.title("Cancer RP EM")
plt.savefig("DiaRP_RP.png")
plt.close()
plot_colors = np.array(["r" if i == 0 else "g" if i == 1 else "b" for i in diabetes.target])
plt.scatter(diabetes_RP_data[:,0],diabetes_RP_data[:,1], c = plot_colors)
plt.title("Cancer RP Target")
plt.savefig("DiaRP_Target.png")
plt.close()


#Apply the dimensionality reduction algorithms to one of your datasets from assignment
#1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this)
#and rerun your neural network learner on the newly projected data.

IRIS_N_COMPONENTS = 2
iris_PCA_data = PCA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_ICA_data = FastICA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_KernalPCA_data = KernelPCA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_IncrementalPCA_data = IncrementalPCA(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)
iris_RP_data = random_projection.GaussianRandomProjection(n_components = IRIS_N_COMPONENTS).fit_transform(iris.data)


splits = [0.5,0.6,0.7,0.8,0.9]
pca, ica, kpca, ipca, rp = [],[],[],[],[]
for s in splits:
    pca_score, ica_score, kerpca_score, incpca_score, rp_score = [],[],[],[],[]
    NUM_RUNS = 100
    for i in range(NUM_RUNS):
        a = 150 * s
        tr_ids = random.sample(range(150), int(a))
        ts_ids = list(set(range(150))-set(tr_ids))


        nn_pca = MLPClassifier(hidden_layer_sizes = [2,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_PCA_data[tr_ids],iris.target[tr_ids])
        nn_ica = MLPClassifier(hidden_layer_sizes = [2,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_ICA_data[tr_ids],iris.target[tr_ids])
        nn_kerpca = MLPClassifier(hidden_layer_sizes = [2,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_KernalPCA_data[tr_ids],iris.target[tr_ids])
        nn_incpca = MLPClassifier(hidden_layer_sizes = [2,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_IncrementalPCA_data[tr_ids],iris.target[tr_ids])
        nn_rp = MLPClassifier(hidden_layer_sizes = [2,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_RP_data[tr_ids],iris.target[tr_ids])


        pca_score.append(nn_pca.score(iris_PCA_data[ts_ids],iris.target[ts_ids]))
        ica_score.append(nn_ica.score(iris_ICA_data[ts_ids],iris.target[ts_ids]))
        kerpca_score.append(nn_kerpca.score(iris_KernalPCA_data[ts_ids],iris.target[ts_ids]))
        incpca_score.append(nn_incpca.score(iris_IncrementalPCA_data[ts_ids],iris.target[ts_ids]))
        rp_score.append(nn_rp.score(iris_RP_data[ts_ids],iris.target[ts_ids]))

    pca.append(statistics.mean(pca_score))
    ica.append(statistics.mean(ica_score))
    kpca.append(statistics.mean(kerpca_score))
    ipca.append(statistics.mean(incpca_score))
    rp.append(statistics.mean(rp_score))

plt.plot(splits, pca, color = "red", label = "PCA")
plt.plot(splits, ica, color = "orange", label = "ICA")
plt.plot(splits, kpca, color = "yellow", label = "KerPCA")
plt.plot(splits, ipca, color = "green", label = "IncPCA")
plt.plot(splits, rp, color = "blue", label = "RP")
plt.legend()
plt.xlabel("Training Percent")
plt.ylabel("Accuracy")
plt.title("Iris Data Set Accuracy after DR (H1 = 2, H2 = 3)")
plt.show()


#Apply the clustering algorithms to the same dataset to which you just applied the
#dimensionality reduction algorithms (you've probably already done this),
#treating the clusters as if they were new features. In other words,
#treat the clustering algorithms as if they were dimensionality reduction algorithms.
#Again, rerun your neural network learner on the newly projected data.
#iris_kmeans
#iris_em_labels

#add cluster
iris_km_data = np.array([list(iris.data[i]) + [iris_kmeans.labels_[i]-1] for i in range(len(iris.data))])
iris_em_data = np.array([list(iris.data[i]) + [iris_em_labels[i]-1] for i in range(len(iris.data))])

splits = [0.5,0.6,0.7,0.8,0.9]
km, em = [],[]


NUM_RUNS = 100
for s in splits:
    km_score, em_score = [],[]
    
    for i in range(NUM_RUNS):
        tr_ids = random.sample(range(150),int(s*150))
        ts_ids = list(set(range(150))-set(tr_ids))


        nn_km = MLPClassifier(hidden_layer_sizes = [5,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_km_data[tr_ids],iris.target[tr_ids])
        nn_em = MLPClassifier(hidden_layer_sizes = [5,3],activation = 'relu', solver = 'adam',learning_rate_init = 0.001,
            max_iter = 100000).fit(iris_em_data[tr_ids],iris.target[tr_ids])
        
        km_score.append(nn_km.score(iris_km_data[ts_ids],iris.target[ts_ids]))
        em_score.append(nn_em.score(iris_em_data[ts_ids],iris.target[ts_ids]))

    km.append(statistics.mean(km_score))
    em.append(statistics.mean(em_score))

plt.plot(splits, km, color = 'red', label = "KM")
plt.plot(splits, em, color = 'blue', label = "EM")
plt.legend()
plt.xlabel("Training Percent")
plt.ylabel("Accuracy")
plt.title("Iris Data Set Accuracy after Adding Cluster Feature (H1 = 4, H2 = 3)")
plt.show()

N_RUNS = 100
#Silhouette
for i in range(3,6):
    diabetes_kmeans = KMeans(
    n_clusters = i,
    init = 'random',
    n_init = N_RUNS,
    max_iter = 1000).fit(diabetes.data)

    vis = SilhouetteVisualizer(diabetes_kmeans)
    vis.fit(diabetes.data)
    vis.show()


