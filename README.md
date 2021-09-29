# ML project 1

## Explanation

What's this project all about.


## Schedule

07.10.2021 - All functions implemented

14.10.2021 - First draft of report

21.10.2021 - Optimized regressor

28.10.2021 - Finished version of code / report

01.11.2021 - Finishing touches


## File structure

```
ML_project1 
│   README.md 
│   implementations.py      (Contains all the functions) 
│   run.py                  (To be executed to get the results) 
│   proj1_helpers.py        (Helping functions)
│
└───data
│   │   test.csv	    (Test data)
│   │   train.csv	    (Training data)
│   │   submission.csv      (Submitted predictions)
│   
└───report
    │   file021.txt
    │   file022.txt
```

## TODO list

### Regressors
- [ ] Implement `least_squares_GD()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [ ] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `least_squares_SGD()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [ ] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `least_squares()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [ ] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `ridge_regression()` regressor
  - [x] Write the source code
  - [x] Create documentation
  - [x] Include it in `run.py`
  - [ ] Validate regressor
  - [ ] Optimize regressor
- [ ] Implement `logistic_regression()` regressor
  - [ ] Write the source code
  - [ ] Validate regressor
  - [ ] Create documentation
  - [ ] Include it in `run.py`
- [ ] Implement `reg_logistic_regression()` regressor
  - [ ] Write the source code
  - [ ] Validate regressor
  - [ ] Create documentation
  - [ ] Include it in `run.py`

### `run.py`
- [ ] Automate regressor calling
- [ ] Maybe input hyperparameters with keyword arguments?

### Competition
- [ ] Create a team
- [ ] Submit predictions
- [ ] Create an improved regressor
  - [ ] Only use the most indicative features
  - [ ] Find best hyperparameters
  - [ ] Emulate a randomforest? (pool multiple randomized gradient descents)

### Report
- [ ] Create a layout and the structure
- [ ] Write the report
- [ ] Illustrate our results

## References
