# ML project 1 - Higgs Boson
We implement six standard versions and one optimized version of a linear egressor. Subsequently, the Higgs boson data set for binary classification from 30 numerical features is used for testing our implementations. We perform various preprocessing techniques and tune the parameters of our regressors to maximize in first priority the F1-score and in second priority the accuracy of the predictions.

The features in the data set record numerical measurements that coincide with the observation of either a decay signature that was caused by an event involving a Higgs boson ('s' for signal) or one that is not related to Higgs boson like decays ('b' for background). Based on 250'000 samples of training data, our goal is to predict the unknown labels for the test data.

[![Status](https://img.shields.io/badge/status-active-success.svg)]()


## 📝 Table of Contents
- [⛏️ Quick start](#️-quick-start)
- [🔁 Reproduce results](#️-reproduce-results)
- [📂 File structure](#️-file-structure)
- [📚 References](#️-references)
- [✍️ Authors](#️-authors)


## ⛏️ Quick start
To get a minimal working example of the regressors, perform the following steps:
1. Open your terminal, and use the command `git clone https://github.com/FMatti/ML_project1` to clone our repository on your machine.
2. Unzip the `data.zip` archive to obtain the [train.csv](https://github.com/epfml/ML_course/blob/master/projects/project1/data/train.csv.zip) and [test.csv](https://github.com/epfml/ML_course/blob/master/projects/project1/data/test.csv.zip) files.
3. For a minimal working example of our implementations, simply run the following lines of code:

    from proj1_helpers import *
    from implementations import *
    y_train, tX_train, _ = load_csv_data('data/train.csv')
    tX_train = standardize(tX_train)
    print(least_squares_GD(y_train, tX_train))
    print(least_squares_SGD(y_train, tX_train))
    print(least_squares(y_train, tX_train))
    print(ridge_regression(y_train, tX_train))
    print(logistic_regression(y_train, tX_train))
    print(reg_logistic_regression(y_train, tX_train))


## 🔁 Reproduce results
To reproduce the results we have shown in Table 2 of our report, follow the instructions in [⛏️ Quick start](#️-quick-start) and then execute the run.py script with the following command: `python run.py`. The predictions for each of the regressor configurations will then be stored in the directory `data/submission_[NAME OF THE REGRESSOR]`.


## 📂 File structure

```
ML_project1 
│   README.md               (The file you are reading right now)
│   proj1_helpers.py        (Support functions that are provided by the lecturers)
|   project1.ipynb          (Central notebook where one can execute the whole pipeline (preprocessing, regressors, scores) and see visualizations of the data set)
│   implementations.py      (Contains all the implemented regressors)
│   run.py                  (To be executed to get the results) 
│
└───data.zip
│   │   test.csv	        (Test data)
│   │   train.csv           (Training data)
│   
└───report
    │   main.tex            (Main typesetting file)
    │   style.sty           (Stylesheet)
    |   biblio.bib          (Bibliography file)
```

## 📚 References
[1] Adam-Bourdariosa C., Cowanb G., Germain C.,I. Guyond B. Kégl, Rousseau D.Learning to discover: [The Higgs boson machine learning challenge](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf). 2014. 

[2] Jaggi M., Urbanke R., Khan M. E.Machine Learning (CS-433): [Lecture notes](https://github.com/epfml/ML_course). 2021.

## ✍️ Authors
- Fabio Matti
- Ivan Bioli
- Olivier Staehli
