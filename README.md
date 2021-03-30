# AML-DS-2021

[![Build Status](https://github.com/Gci04/AML-DS-2021/actions/workflows/setup.yml/badge.svg)](https://github.com/Gci04/AML-DS-2021/actions/workflows/setup.yml)
[![NN Model Test](https://github.com/Gci04/AML-DS-2021/actions/workflows/neuralNet.yml/badge.svg)](https://github.com/Gci04/AML-DS-2021/actions/workflows/neuralNet.yml)

This is a project for Advanced Machine Learning Course at innopolis university. It contains Seminars coding examples, homework exercises and Course project code.


## Prerequisites

* Keras >= 2.0.8
* TensorFlow >= 2.0
* Numpy >= 1.13.3
* Matplotlib >= 2.0.2
* Seaborn >= 0.7.1
* [Catboost](https://tech.yandex.com/catboost/)
* PyTorch

**All the libraries can be pip installed** using `pip install -r requirements.txt`


## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. Navigate to repository folder
1. Install dependencies which are specified in requirements.txt. use `pip install -r requirements.txt` or `pip3 install -r requirements.txt`
1. Raw Data is being kept [here](data) within this repo.

1. Data processing/transformation scripts are being kept [here](scripts)

1. To run the repository main code nevigate to [scr](src) `cd src` then run `python main.py`. Or execute the .ipynb file [here](notebooks)


#### Setup using
```
cd AML-DS-2021
python -m venv dst-env
```

#### Activate environment
Max / Linux
```
source dst-env/bin/activate
```

Windows
```
dst-env\Scripts\activate
```

#### Install Dependencies
```
pip install -r requirements.txt
```

#### Setting up
```
python setup.py
```


#### Testing
To run tests, install pytest and unittest using pip or conda and then from the repository root run

    pytest tests
    #or
    python -m unittest discover -s tests/ -p '*_test.py' -v

## Repository Structure

```
├── .gitignore               <- Files that should be ignored by git.
│                               
├── conda_env.yml            <- Conda environment definition
├── LICENSE
├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
│                               generated with `pip freeze > requirements.txt`. Might not be needed if using conda.
├── setup.py                 <- Setup script
│
├── data                     <- Data files directory
│   └── Data1                <- Dataset 1 directory
│
├── notebooks                <- Notebooks for analysis and testing
│   ├── eda                  <- EDA Notebooks directory for
│   │   └── eda1.ipynb       <- Example python notebook
│   ├── features             <- Notebooks for generating and analysing features (1 per feature)
│   └── preprocessing        <- Notebooks for Preprocessing

├── scripts                  <- Standalone scripts
│   └── dataExtract.py       <- Data Extraction script
│
├── src                      <- Code for use in this project.
│   ├── train.py             <- train script
│   └── test.py              <- test script
│
└── tests                    <- Test cases (named after module)
    ├── test_notebook.py     <- Test that Jupyter notebooks run without errors
    ├── test1package         <- test1package tests
        ├── test1module      <- examplemodule tests (1 file per method tested)
        ├── features         <- features tests
        ├── io               <- io tests
        └── pipeline         <- pipeline tests
```

## Contributing to This Repository
Contributions to this repository are greatly appreciated and encouraged.<br>
To contribute an update simply:
* Submit an issue describing your proposed change to the repo in question.
* The repo owner will respond to your issue promptly.
* Fork the desired repo, develop and test your code changes.
* Edit this document and the template README.md if needed to describe new files or other important information.
* Submit a pull request.


## References


## Contact
If you would like to get in touch, please contact:
