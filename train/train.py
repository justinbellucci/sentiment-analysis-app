# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 08_09_2020                                  
# REVISED DATE: 

import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

# required by SageMaker to load model artifacts
def model_fn(model_dir):
    """ Required by SageMaker, this function loads in the PyTorch model from 
        the model directory (model_dir). Function name and argument specified 
        by SageMaker.

        Arguments:
        - model_dir: path where the model artifacts are written to in S3

        Returns:
        - model: loaded model
    """

    print("Loading model...")

    # load parameters used to create the model
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'],
                           model_info['hidden_dim'],
                           model_info['vocab_size'])

    # load stored model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    # load saved word dictionary mappings
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()
    print("Done loading model.")

    return model