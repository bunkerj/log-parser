# Log-parser

The goal of this project is to provide a set of implementations and experiments for the rapid prototyping of log parsing solutions.

### Prerequisites

The project dependencies can be installed by using the following command:

```
pip install -r requirements.txt
```

## Project Organization
    .
    ├── data                # Datasets and helper scripts used to process data  
    ├── exp                 # Standalone scripts that each run a standalone experiment
    ├── graphs              # Scripts used to visualize data dumped from experiments in exp/
    ├── papers              # Collection of various relevant papers
    ├── src                 # Log parser implementations and auxiliary support classes/functions
    ├── tests               # Unit tests for key implementations
    ├── constants.py        # Constants used throughout the entire project
    ├── requirements.txt    # Required libraries to run experiments in this project
    └── README.md