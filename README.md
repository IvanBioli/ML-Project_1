# ML project 1

## Explanation

TODO.


## Schedule

07.10.2021 - All functions implemented

14.10.2021 - Optimized regressor

21.10.2021 - First draft of report

28.10.2021 - Finished version of code / report

01.11.2021 - Finishing touches

## Optimization ideas
- [ ] Preprocessing
  - [ ] Remove features that are not relevant
  - [ ] Boost features that really matter
  - [ ] Check for values that are automatically set at some values (they might be outliers, but be careful because there could be outliers in the test dataset)
  - [ ] Do some research on preprocessing
  - [ ] Understand which features affect most our predictions

- [ ] Iterative Optimization
  - [ ] Ideal parameters for GD, SGD and other algorithms
  - [ ] Try different loss functions (MSE is a lot affected by outliers)
  - [ ] Change the step size every iteration
  - [ ] Stopping criterion

- [ ] New approach
  - [ ] Ensemble regressor


## File structure

```
ML_project1 
│   README.md               (The file you are reading right now)
│   proj1_helpers.py        (Support functions)
|   project1.ipynb          (Notebook for testing and optimizing purposes)
│   implementations.py      (Contains all the implemented regressors and functions) 
│   run.py                  (To be executed to get the results) 
│
└───data
│   │   test.csv	          (Test data)
│   │   train.csv	          (Training data)
│   │   submission.csv      (Submitted predictions)
│   
└───report
    │   main.tex            (Main typesetting file)
    │   style.sty           (Stylesheet)
    |   biblio.bib          (Bibliography file) 
```

## TODO list

### Regressors
- [ ] Implement `least_squares_GD()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [x] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `least_squares_SGD()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [x] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `least_squares()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [x] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `ridge_regression()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [x] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `logistic_regression()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [ ] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `reg_logistic_regression()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [ ] Validate regressor
  - [ ] Optimize regressor

### Handing in
- [ ] Automate regressor calling in `run.py`
- [ ] Maybe input hyperparameters with keyword arguments?

### Competition
- [x] Create a team
- [ ] Create an improved regressor
- [ ] Submit final predictions

### Report
- [x] Create a layout
- [ ] Create a structure
- [ ] Write the report
- [ ] Illustrate our results

## References

TODO.
