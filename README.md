# Data Factory

------------

A suite of tools for generating high quality training and validation data for AI applications.

## Overview

![data generation diagram](images/diagram.png)

## Usage

### Requirements

* python 3.6
* Ubuntu

### How to set up an environment

```
conda create --name dataFactoryEnv python=3.6
source activate dataFactoryEnv
pip install -r requirements.txt
```

------------

## DATA AUGMENTATION

### Identify Similar Columns

- For each column C, a set of columns identified which tend to confuse with the column C.

- How to run
```
python main.py --analyze_confusion
```

### Identify Representative Aliases 

- For each column C, find several 3-5 most represenative aliases. 

- How to run
```
python main.py --get_rep_aliases 
```

### Paraphrase (Backtranslation)

- We are using the back-tranlsation to generate paraphrases. Back translation is the process of bringing a previously translated text back to its original language without reference to the original source. It is originally employed to provide additional quality assurance for the most sensitive material.

```
 +-------------+    MT   +---------------------------+    BT    +-------------+    
 |   English   |   ===>  |   Intermediate Lanauage   |   ===>	|   English   |    
 +-------------+     	 +---------------------------+      	+-------------+     
```

- How to run
```
python main.py --paraphrase 
```

-  Sample Input

Your input is a text file that contains sentences to be paraphrased, delimited by new lines. For example:

```
How many deals we created each month?
What industry is our maximum revenue coming from?
Display the deals by their closing year.
```

-  Sample Output

Your output is a dictionary, dumped as json. For example:

```
{
  "How many deals we created each month?": [
                                              "How many deals have we made every month?",
                                              "How much transaction did you do every month?",
                                              "How many offers have we created each month?",
                                              "How many transactions we have created each month?"
                                            ],
  "What industry is our maximum revenue coming from?": [
                                                          "From which industry does our maximum income come from?",
                                                          "From which industry do our maximum revenues come from?",
                                                          "Which industry is the largest income source?",
                                                          "Which industry is our maximum revenue?",
                                                          "From which sector do our maximum income come from?"
                                                        ],
  "Display the deals by their closing year.": [
                                                "View offers within the year of closure.",
                                                "We display transactions by the end date of trading.",
                                                "Show the deals at the end of their year.",
                                                "Show offers by their year of closure."
                                              ]
  }      
```


### Weak Supervision

-

### Contextual data augmentation

-

### RISD

- **Synonym Replacement (SR):** Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
- **Random Insertion (RI):** Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.
- **Random Swap (RS):** Randomly choose two words in the sentence and swap their positions. Do this *n* times.
- **Random Deletion (RD):** For each word in the sentence, randomly remove it with probability *p*.

------------

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

