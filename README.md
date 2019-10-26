## README

DMAKit, is an open access library implemented in Python programming language that facilitate the analysis of different kind of data, data mining and pattern recognition techniques, facilitating the implementation of classification, prediction or clustering models, the statistical evaluations and feature analysis of different attributes in dataset.

## Inputs, outputs and workflows in DMAKit

DMAKit processes all types of files in Comma Separated Value (csv) or Tabulated Separated Values (tsv) format, allowing attributes of the discrete and continuous type. Depending on the type of analysis that is selected, a transformation of the discrete variables to continuous values is made by means of the frequency of appearance of its elements, obtaining values between [0,1] where 0 indicates no existence of the value and 1 implies the totality of elements. In addition, it is possible to select the way in which data normalization is performed. DMAKit has four types of data normalization forms. Normal scale allows standardization based on the values of the mean and the standard deviation of the distribution. Min Max Scaler normalizes according to the maximum and minimum values that exist in the distribution. Log scaler and Log normal scaler apply a logarithmic function to distribution of data, if there are values less than zero, when applying the transformation they are considered as zero. In the same way, the results depend on the type of process selected. However, in general, every module generates a file in JSON format with the summary of the process and different pictures or csv files are created.

## Design and implementation

DMAKit was designed under the Object Oriented Programming paradigm, taking advantage of this, the advantages of this paradigm, generating the encapsulation and modularization necessary for this type of development. Its implementation is based on a set of modules written in Python programming language, version 2.7. All modules for generating supervised and unsupervised learning models are based on  scikit-learn library. The data set management is through Pandas library and the graphics using Matplotlib. Finally, scripts have been generated that allow the installation of the modules and be enabled to be imported from any python script, using Disutils to fulfill this objective. Each component was tested using general and public knowledge data sets. The source code and test sets are hosted in the github repository for free access and non-commercial use and license under GNU OpenGL 3.0.

## DMAKit modules

DMAKit is composed of four principal modules, which allow to evaluate characteristics, to develop statistical analysis of the data, to search patterns by means of clustering algorithms and to train classification or regression models through supervised learning algorithms. In addition, it has an exploratory model tool for both types of learning, which allows to evaluate different algorithms and parameters for the same dataset and to evaluate distributions of performance measures associated with the results of the generated models.

## 1. Feature Analysis

The feature analysis module allows the evaluation of relations between  different attributes based on the analysis of the correlation matrix and mutual information techniques. It also implements different dimensionality reduction algorithms, based on linear models, such as Principal Component Analysis (PCA) and its variants, PCA Kernel and Incremental PCA. Additionally, it allows to evaluate the relevance of attributes in the training of supervised learning models using the Random Forest algorithm, both for classification systems and for the  prediction of continuous variables (only available for data sets with response in their features).

For execute Feature analysis module you should exec the script launcherFeatureAnalysis.py, if you wont see all options this script, please exec with option -h:

```
python launcherFeatureAnalysis.py -d DATASET -o OPTION -p PATHRESULT -a PROCESS [-r ATTRIBUTE] [-k KIND_DATA]
```

- OPTION: Option to Normalize data set:

1. Normal Scale
2. Min Max Scaler
3. Log scale
4. Log normal scale.

- PROCESS: Option analyze features:

1. Correlation
2. Spatial Deformation
3. PCA
4. Mutual Information
5. Kernel PCA
6. Incremental PCA

- KIND_DATA: Kind of dataSet:

1. CLASS
2. RESPONSE

## 2. Statistical Analysis and Statistical Test

Different statistical analyzes can be performed on both continuous and discrete type attributes. Distribution analysis, dispersion and frequency evaluation, as well as visualizations of continuous variables from discrete variables such as scatter plot matrix (SPLOM) and parallel coordinates are implemented. In addition, statistical tests of different interests are included. Test to evaluate the normality of the distribution of the data: Shapiro-Wilk and Kolmogorov Smirnov Tests. Test to compare two distributions: Mann Whitney Test, and tests to evaluate correlation of to distributions: Pearson Coefficient, Spearman's rank  and Kendall's tau.

For execute Statistical analysis module you should exec the script launcherStatisticalAnalysis.py, if you wont see all options this script, please exec with option -h:

```
python launcherStatisticalAnalysis.py -d DATASET -o OPTION -p PATHRESULT [-a KEY]
```

- OPTION: Option to process:

1. View Continuos Data.
2. Dispersion View.
3. Histogram
4. Frequence.
5. Parallel Coordinates.
6. SPLOM
7. Summary Statistical

- Key: to evaluate in dataSet if you select dispersion or statistical summary, it is not necesarie.


## 3. Unsupervised learning models for pattern recognition

Different unsupervised learning algorithms for the search of groups or patterns in dataset have been implemented. The list of algorithms implements in DMAKit are: k-Means, Birch, DBScan, Mean Shift, Affinity Propagation and agglomerative and hierarchical algorithms. Each model or partition generated is evaluated by Calinski-Harabasz index and the silhouettes coefficient. The module allows to generate graphs of the distribution of elements by groups to evaluate the imbalance of classes, also the module export input dataset adding the labels associated to the group that was assigned.

For execute Clustering module you should exec the script launcherClustering.py, if you wont see all options this script, please exec with option -h:

```
python launcherClustering.py -d DATASET -o OPTION -p PATHRESULT -a ALGORITHM [-i PARAMS]
```

- OPTION: Option to Normalize data set:

1. Normal Scale
2. Min Max Scaler
3. Log scale
4. Log normal scale.

- ALGORITHM: Algorithm to process clustering:

1. K-means
2. Birch
3. Agglomerative
4. DBSCAN
5. MeanShift
6. Affinity Propagation

- PARAMS: Params to exec algorithm, pleas add in this form: param1-param2-param3, for more details please check launcherClustering.py file.

## 4. Supervised learning models

DMAKit has implemented different supervised learning algorithms, both for use in classification and prediction models. The validation method of each model can vary between cross validation with different \textit{k} values or Leave One Out. Performance vary according to the type of model. In the case in which classification models are developed, Precision, Recall, Accuracy and F1 score is used. While that for continuous variable prediction  Pearson Coefficient, Spearman's rank, Kendall's $\tau$ and R score are used by the system as performance. Algorithms based on distances as k Nearest Neighbour, kernel transformations and evaluation of hyper planes as Support Vector Machine (SVM) and nuSVM, methods of feature evaluation as Decision Tree, methods of assemble and exploration of features as Random Forest, Bagging, Gradient Boosting, AdaBoost, methods based on probability as Naive Bayes and use of neural networks as Multi-Layer Perceptron, are available in this module.

Two options to process supervised learning depending if you use a classification module or regression module.

For classification models, you need execute the script launcherSupervisedClf.py in other case ypu should exec launcherSupervisedPrediction.py.

The command line for both case is:

```
python launcherSupervisedClf.py -d DATASET -o OPTION -p PATHRESULT -r RESPONSECLASS -a ALGORITHM -v VALIDATION [-i PARAMS]
```
- OPTION: Option to Normalize data set:

1. Normal Scale
2. Min Max Scaler
3. Log scale
4. Log normal scale.

- RESPONSECLASS: Name of attribute with response class

- ALGORITHM: Algorithm to process training model:

1. AdaBoostClassifier
2. BaggingClassifier
3. BernoulliNB
4. DecisionTree
5. GaussianNB
6. GradientBoostingClassifier
7. KNeighborsClassifier
8. MLPClassifier
9. NuSVC
10. RandomForest
11. SVC

SVC is the default case.

- VALIDATION: Cross validation value. If you wont use a Leave One Out, input -1.

- PARAMS: Params to exec algorithm, pleas add in this form: param1-param2-param3 for more detail, checks de user manual. If you add Default, it will user the Default params.

For regression models:

```
python launcherSupervisedPrediction.py -d DATASET -o OPTION -p PATHRESULT -r RESPONSECLASS -a ALGORITHM [-i PARAMS]
```
- OPTION: Option to Normalize data set:

1. Normal Scale
2. Min Max Scaler
3. Log scale
4. Log normal scale.

- RESPONSECLASS: Name of attribute with response class.

- ALGORITHM: Algorithm to process training model:

1. AdaBoostRegressor
2. BaggingRegressor
3. DecisionTree
4. GaussianNB
5. GradientBoostingRegressor
6. KNeighborsRegressor
7. MLPRegressor
8. NuSVC
9. RandomForest
10. SVC (Default SVC)

- PARAMS: Params to exec algorithm, pleas add in this form: param1-param2-param3 for more detail, checks de user manual. If you add Default, it will user the Default params.


## 5. Exploration Tool Models

DMAKit has a module exploration tool, that is, the instantaneous execution of different algorithms to the same dataset is possible, in order to evaluate different algorithms and parameters for the same element. This option is enabled for both supervised learning models, both prediction models and classification models, and for the use of clustering. At the end, histograms are generated by performance and a ranking of the best models per measure is proposed, as well as a statistical summary per measure for all execute in the process. The main advantage of this tool is the ease of evaluating different algorithms for the same set of data, which allows knowing the panorama of what the best models may be and in the case of developing classification and prediction systems, it may be a sign to consider different algorithms and parameters in a Meta Learning system and thus improve the performance obtained.

Three differents scripts you can use for create a scanning models: clustering, classifier and regression.

```
python launcherClusteringScan.py -d DATASET -o OPTION -p PATHRESULT
```

- OPTION: Option to Normalize data set:

1. Normal Scale
2. Min Max Scaler
3. Log scale
4. Log normal scale.

```
python launcherScanClassification.py -d DATASET -p PATHRESULT
```

```
python launcherScanPrediction.py -d DATASET -p PATHRESULT
```
## NOTES

If you want send comments, opinion or you find a bug in library, please notify to via email: david.medina@cebib.cl
