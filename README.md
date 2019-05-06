
### Welcome!

This is the experimental directory for the question extraction model. Here you will find all of the experiments run to find the model that best serves to extract question sequences from input sequences that may or may not contain questions.


Research on finding a suitable architecture is still under way and there are many more things to try:

e.g. CNN architecture, tuned/different attention mechanisms, 


## Installation

You may find it useful to install this repo. You have two options:

Run:
1. `python setup.py develop`  

2. `pip install -e .`  


# Usage

This area is set up to house *architectural* variations (under the directory name 'models'). Within each folder, you will find a model that has an architecture different from the models found in other folders. Within each architectural folder, you find many models trained - these are experiments where hyperparameters were modified to find the optimal set of hyperparameters. Each model architecture has its own readme where results are reported. 

The bayesian optimization package should be used to find optimal hyperparameters. More details on that later.

## Data Creation

Data pickles cary the relevant data and metadata necessary to train a graph. There are two types:

1. ### data.generator.pkl  
Data generator pickles contain the relevant metadata for a given pickle as well as a list of questions and a list of non questions. The input generator function for the estimator (`train_generator_input_fn_v2`) uses these lists to prepare data samples on the fly.  

The `GeneratorInputV2` class will produce N number of processes to fill a queue with new samples while the estimator is busy training and prefetching samples from this queue. With 3-4 processes running, it should be possible to keep the queue filled so that the tensorflow dataset API can keep the GPUs fed. I typically see constant GPU usage while running with 3-6 processes.  

This provides a significant speed up (3x) in training speed. This is highly preferable to a static dataset, since data generation code can be iterated on more rapidly (there is no downtime between editing code and testing it in model training).

For this dataset, no prebuilt data is required.


2. ### data.static.pkl  
Along with everything carried by the generator type pickle, static pickles carry a prebuilt fixed size dataset. These types are not preferred since they take time to build (as apposed to the generator which will start producing samples as soon as training begins).

The `train_static_input_fn` input function reads the `data.features` and `data.labels` attributes from the `data.static.pkl` object once its loaded for training.

## Training models

In order to train a model, you may create a new directory, and produce the two following files:

 - run_model.py - this contains the model architecture.
 - data_container.type.pkl - this contains the dataset used to generate the data during training

The run_model.py should contain only the `model_fn` code, and a short run script at the bottom (see existing models for details).

The models themselves can be editted however youd like, however there in a config.py with some parameters that can be set. See `backend/config.py` for specifics`.  


See below for data container options.



## Accessing the models for prediction

In order to access the model, you need only call the access model function as so:
An example can be found in `notebooks/Prediction_test_notebook.ipynb`

The current best performing model is `Single_Layer_Bidir_LSTM/experiment4`  
For experiment results, see `models/EXPERIMENT_README.md`.  

```
$ from backend.predict_helpers import Predictions
$ from backend.container import ArgContainer
$ from run_model import model_fn

$ data_pickle = 'path/to/pickle'
$ model_dir = 'path/to/model'
$ make_preds = Predictions(model_fn, data_pickle, model_dir)

$ test_string = """
this is a seriously flawed string. Is there any way to fix it? Lets see if we can do it!
"""

$ results = make_preds.predict_string(test_string)
$ print(results)

    is there any way to fix it


$ list_of_strings = [
    'this is a good first string but is it possible that its a little too short?',
    'your second string is always your best string. But how do you know if its a real string or not?'
    ]
$ results = make_preds.predict_string(list_of_strings)
$ print(results)
    
    ['is it possible that its a little too short', 'how do you know if its a real string or not']

```

### Introspecting data containers

To see whats inside of a particular data container, load its pickle and call:

    `data.display_metadata()`

    --DATASET METADATA --
    description: None
    embedding_dims: 50
    max_elements: 3
    max_index: 20310
    max_num_questions: 1
    max_seq_length: 30
    vocab_size: 20311



# References
 https://stackoverflow.com/questions/44937105/how-to-use-attentionmechanism-with-multirnncell-and-dynamic-decode

