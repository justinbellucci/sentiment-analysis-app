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
import torch.nn as nn
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False)

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
        total_loss = 0
#         h = model.init_hidden(batch_size) # initialize hidden dimension parameters
        
        for batch in train_loader:
            batch_X, batch_y = batch

            # move to GPU if available
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

#             h = tuple([each.data for each in h])

            optimizer.zero_grad() # zero gradients
            out = model.forward(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(e+1, total_loss/len(train_loader)))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() # initialize argument parser object

    # training parameters
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='Input batch size for training (default=512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of epochs to train (default=10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default=1)')

    # model parameters
    parser.add_argument('--embedding_dim', type=int, default=42, metavar='N',
                        help='Size of word embeddings (default=42)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='Size of hidden dimension (default=100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='Random seed (default=1)')

    # required SageMaker parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    # ---------------------------------------------------
    # begin main program
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} to train...".format(device))

    torch.manual_seed(args.seed)

    # load in training data
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # build the model and move to gpu or cpu
    model = LSTMClassifier(args.vocab_size, args.embedding_dim, args.hidden_dim).to(device)

    # load word dictionary
    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim: {}, hidden_dim: {}, vocab_size: {}".format(args.embedding_dim,
                                                                                       args.hidden_dim, 
                                                                                       args.vocab_size))
    # ------------ train the model --------------
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    train(model, args.batch_size, train_loader, args.epochs, optimizer, criterion, device)

    # save the model hyperparameters
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {'embedding_dim': args.embedding_dim,
                      'hidden_dim': args.hidden_dim,
                      'vocab_size': args.vocab_size}
        torch.save(model_info, f)
    
    # save the word dictionary
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

    # save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)