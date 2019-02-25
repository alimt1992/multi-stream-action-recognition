import re
import os
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.models as models
use_gpu = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if (cuda_available and use_gpu) else "cpu")


from utils import to_var
from model_utils import *

def predict_proba(model, dataloader):
    """ Predict probabilty of classes.
    
        Inputs:
            - model: a pytorch model (eg., a neural network).
            - dataloader: a pytorch dataloder to load data and provide batches to the model.
            - report_every: how often this function should print out statistics.
            
        Output:
            - predictions: a numpy array containing predicted probabilities for all of the data.
    
    """
    model.train(False)
    predictions = []
    y_true = []

    for inputs, targets in tqdm(dataloader):
        inputs = to_var(inputs, volatile=True)
        #bs, ncrops, c, h, w = inputs.size()
        #outputs = model(inputs.view(-1, c, h, w))
        #outputs = outputs.view(bs, ncrops, -1).mean(1)
        outputs = model(inputs)
        scores = F.softmax(outputs, dim=1)
        predictions += [scores.data.cpu().numpy()]
        y_true += [targets.cpu().numpy()]
    
    predictions = np.concatenate(predictions)
    y_true = np.concatenate(y_true)
    return predictions, y_true


def TTA(model, dl, tta_dl, is_test=False, n=5):
    probs = predict_proba(model, dl)
    probs = np.stack([probs] + [predict_proba(model, tta_dl) for i in range(n)]).mean(axis=0)
    y_true = np.concatenate([labels.cpu().numpy() for _, labels in tta_dl]) if not is_test else None
    return probs, y_true


def predict_proba_five_ten_crop(model, dataloader):
    """ Predict probabilty of classes.
    
        Inputs:
            - model: a pytorch model (eg., a neural network).
            - dataloader: a pytorch dataloder to load data and provide batches to the model.
            - report_every: how often this function should print out statistics.
            
        Output:
            - predictions: a numpy array containing predicted probabilities for all of the data.
    
    """
    model.train(False)
    predictions = []

    for inputs, _ in tqdm(dataloader):
        inputs = to_var(inputs, volatile=True)
        bs, ncrops, c, h, w = inputs.size()
        outputs = model(inputs.view(-1, c, h, w))
        outputs = outputs.view(bs, ncrops, -1).mean(1)
        scores = F.softmax(outputs, dim=1)
        predictions += [scores.data.cpu().numpy()]
    
    predictions = np.concatenate(predictions)
    return predictions


def predict_class(model, dataloader):
    """ Predict probabilities for the given model and dataset
    """
    model.train(False)
    result = []
    y = []
    
    for inputs, targets in tqdm(dataloader):
        for i, inp in enumerate(inputs):
            inputs[i] = to_var(inputs[i], True)
        scores = model(inputs[0],inputs[1],inputs[2])
        _, preds = torch.max(scores.data, 1)
        result += [preds.cpu().numpy()]
        y += [targets.cpu().numpy()]
        
    result = np.concatenate(result)
    y = np.concatenate(y)
    return result, y


def predict_class_names(model, dataloader, class_names):
    """ Predict probabilities for the given model and dataset
    
        Inputs:
            - model: a pytorch model
            - dataloader a torch.utils.data.DataLoader object
            - class_names: a list of class names
            
        Output:
            - result: Predicted class name for each input as a python list
    """
    model.train(False)
    result = []
    y = []
    
    for inputs, labels in tqdm(dataloader):
        for i, inp in enumerate(inputs):
            inputs[i] = to_var(inputs[i], True)
        scores = model(inputs[0],inputs[1],inputs[2])
        _, preds = torch.max(scores.data, 1)
        result += [preds.cpu().numpy()]
        y += [labels.cpu().numpy()]
    
    result = np.concatenate(result)
    y = np.concatenate(y)
    
    pred_class_names = [class_names[i] for i in result]
    return pred_class_names, y


def predict_probs_for_models_from_folder(dataloader, models_dir="./models", num_classes=10):
    model_fnames = glob.glob(models_dir + '/*.pth') + glob.glob(models_dir + '/*.h5')
    for i in range(1):
        for model_fname in model_fnames:
            basename = os.path.basename(model_fname)[:-3]
            first_dash_idx = basename.find('-')
            second_dash_idx = basename.find('-', first_dash_idx + 1)
            model_name = basename[first_dash_idx + 1: second_dash_idx].lower()
            
            print('Predicting ouputs for %s...' % model_name)
            
            if model_name == 'resnet18':
                model = load_pretrained_resnet18(model_fname, num_classes)
            elif model_name == 'resnet34':
                model = load_pretrained_resnet34(model_fname, num_classes)
            elif model_name == 'resnet50':
                model = load_pretrained_resnet50(model_fname, num_classes)
            elif model_name == 'densenet121':
                model = load_pretrained_densenet121(model_fname, num_classes)
            elif model_name == 'densenet161':
                model = load_pretrained_densenet161(model_fname, num_classes)
            elif model_name == 'densenet169':
                model = load_pretrained_densenet169(model_fname, num_classes)
            elif model_name == 'densenet201':
                model = load_pretrained_densenet201(model_fname, num_classes)
            else:
                raise NameError('Inavalid model name: {}'.format(model_name))
            
            # predic and save probabilities for this model in the models_dir
            #probs = predict_proba(model, dataloader)
            probs = predict_proba_five_ten_crop(model, dataloader)
            probs_fname = os.path.join(models_dir, "{}-{}.npy".format(basename, i))
            np.save(probs_fname, probs)
