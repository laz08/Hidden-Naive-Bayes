# README

## Authors

* Carles G.
* Laura C.

# Files

The files tree included in this report is the following:
```
.
├── README.html
├── README.md
├── report.pdf
│
├── datasets
│   ├── titanic_ds.txt
│   └── vehicles_dataset.csv
│
└── src
    ├── HNB_preprocessing.R
    ├── hnb.py
    └── nb.py


```

The report.pdf file contains our report on this delivery with full explanations in all of our work.

The datasets folder contains the datasets we have used for our tests.
The doc folder contains the paper proposal of Hidden Naive Bayes in which we have based our study.
The src folder contains all the preprocessing made in R (HNB_preprocessing.R), as well as both implementations of Naive Bayes and Hidden Naive Bayes.

## Execution

In order to execute Hidden naive bayes (src/hnb.py) or naive bayes (src/nb.py), the following statements can be used:

```
$ python src/nb.py dataset_name column_class_index [split_ratio]
$ src/hnb.py dataset_name column_class_index [split_ratio]
```	
Please not that **python3.6** is the version of python used and using python2.7 or lower will not give the same results, for we have used some functions that can only be used on python 3.

## Params

* dataset_name should contain a valid path (either relative or absolute) to a file. For example:

⋅⋅* For the titanic dataset: 					datasets/titanic_ds.txt
⋅⋅* For our own vehicles accidents dataset:	datasets/vehicles_dataset.csv

* column_class_index should cointain a valid attribute column index (0 to columns(dataset) -1).
    The attribute corresponding to this column will be used as class.

* If specified (being) split_ratio should be a float between 0.0 and 1.0, specifing the amount of instances in the dataset that are going to be used to train (and the remaining to test). The default value of split ratio is 0.8

## Execution examples

If we are inside the root folder, we can execute Naive Bayes and Hidden Naive Bayes with the titanic dataset and predict the age of the passenger (column 2) with a 0.75 ratio by:

$ python src/nb.py datasets/titanic_ds.txt 2 0.75
$ python src/hnb.py datasets/titanic_ds.txt 2 0.75

Or if we want to execute it with our vehicles' dataset and predict accident severity (column 0), we can do it so:

```
$ python src/nb.py datasets/vehicles_dataset.csv 0 0.8
$ python src/hnb.py datasets/vehicles_dataset.csv 0 0.8
```