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

# create PyTorch dataloaders
def _get_train_data_loader(batch_size, training_dir):
    """ Create PyTorch dataloader from training directory csv file
        uploaded to S3. Previously concatenated with each row in dataset 
        of form - label, length, review. 

        Arguments:
        - batch_size: (int) 
        - training_dir: path to csv file in S3

        Return:
        - train_loader: PyTorch Dataloader
    """
    print("Creating dataloader...")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    return train_loader

# main training function
def train(model, batch_size, train_loader, epochs, optimizer, criterion, device):
    """ This is the main training method that is called by the training script.

        Arguments:
        - model: PyTorch model to train
        - train_loader: PyTorch Dataloader
        - epochs: 
        - optimizer: loss function
        - device: gpu or cpu
    """
    for e in range(epochs):
        model.train() # put model in training mode
        h = model.init_hidden(batch_size, device) # initialize hidden dimension parameters

        for batch in train_loader:
            batch_X, batch_y = batch

            # move to GPU if available
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            h = tuple([each.data for each in h])

            optimizer.zero_grad() # zero gradients
            out, h = model.forward(batch_X, h)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()