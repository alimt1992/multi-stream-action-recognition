import os
import sys
import time
import torch
import torch.nn as nn
from utils import to_var
use_gpu = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if (cuda_available and use_gpu) else "cpu")


def train_one_epoch(model, dataloder, criterion, optimizer, scheduler):
    if scheduler is not None:
        scheduler.step()
    
    model.train(True)
    
    steps = len(dataloder.dataset) // dataloder.batch_size
    
    running_loss = 0.0
    running_corrects = 0
    
    for i, (inputs, labels) in enumerate(dataloder):
        for j, inp in enumerate(inputs):
            inputs[j] = to_var(inputs[j])
        
        labels = to_var(labels)
        
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs[0],inputs[1],inputs[2])
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        # backward
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        # statistics
        running_loss  = (running_loss * i + loss.item()) / (i + 1)
        running_corrects += torch.sum(preds == labels.data).item()
        
        # report
        sys.stdout.flush()
        sys.stdout.write("\r  Step %d/%d | Loss: %.5f" % (i, steps, loss.item()))
        
    epoch_loss = running_loss
    epoch_acc = running_corrects / len(dataloder.dataset)
    
    sys.stdout.flush()
    print('\r{} Loss: {:.5f} Acc: {:.5f}'.format('  train', epoch_loss, epoch_acc))
    
    return model, epoch_loss, epoch_acc

    
def validate_model(model, dataloder, criterion):
    model.train(False)
    
    steps = len(dataloder.dataset) // dataloder.batch_size
    
    running_loss = 0.0
    running_corrects = 0
    
    for i, (inputs, labels) in enumerate(dataloder):
        for j, inp in enumerate(inputs):
            inputs[j] = to_var(inputs[j], True)
        
        labels = to_var(labels, True)
              
        # forward
        outputs = model(inputs[0],inputs[1],inputs[2])
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
            
        # statistics
        running_loss  = (running_loss * i + loss.item()) / (i + 1)
        running_corrects += torch.sum(preds == labels.data).item()
        
        # report
        sys.stdout.flush()
        sys.stdout.write("\r  Step %d/%d | Loss: %.5f" % (i, steps, loss.item()))
        
    epoch_loss = running_loss
    epoch_acc = running_corrects / len(dataloder.dataset)
    
    sys.stdout.flush()
    print('\r{} Loss: {:.5f} Acc: {:.5f}'.format('  valid', epoch_loss, epoch_acc))
    
    return epoch_loss, epoch_acc


def train_model(model, train_dl, valid_dl, criterion, optimizer,
                scheduler=None, num_epochs=10):

    if not os.path.exists('models3'):
        os.mkdir('models3')
    
    since = time.time()
       
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = float('Inf')
    trn_loss_hist, val_loss_hist = [], []
    trn_acc_hist, val_acc_hist = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        ## train and validate
        model, trn_loss, trn_acc = train_one_epoch(model, train_dl, criterion, optimizer, scheduler)
        val_loss, val_acc = validate_model(model, valid_dl, criterion)
        
        trn_loss_hist.append(trn_loss)
        val_loss_hist.append(val_loss)
        
        trn_acc_hist.append(trn_acc)
        val_acc_hist.append(val_acc)
        
        # deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
        #    best_model_wts = model.state_dict()
        #    torch.save(best_model_wts, "./models3/resnet50-3concat-epoch-{}-acc-{:.5f}.pth".format(epoch, best_acc))
        
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, "./models3/resnet50-3stream-epoch-{}-loss-{:.5f}.pth".format(epoch, best_loss))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    print('Best val Loss: {:.4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, trn_loss_hist, val_loss_hist, trn_acc_hist, val_acc_hist
