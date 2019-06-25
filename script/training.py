import torch
from torch import nn
from torchvision import transforms
from CSVImageDataset import CSVImageDataset
import numpy as np
import PIL

# Import Libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Normalizzazione

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
])

# Carichiamo le immagini

from torch.utils.data import DataLoader
train_set = CSVImageDataset('..\\','..\\csv\\train.csv', transform = train_transform)
valid_set = CSVImageDataset('..\\','..\\csv\\val.csv', transform = test_transform)
test_set = CSVImageDataset('..\\','..\\csv\\test.csv', transform = test_transform)
train_loader = DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, num_workers=4)

# Training

from torch.optim import SGD
from torchnet.meter import AverageValueMeter
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from sklearn.metrics import accuracy_score

def trainval_classifier(model, train_loader, valid_loader, exp_name='experiment', lr=0.01, epochs=10, momentum=0.99):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #plotters
    loss_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Loss', 'legend':['train','valid']})
    acc_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Accuracy','legend':['train','valid']})
    visdom_saver = VisdomSaver(envs=[exp_name])
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'valid' : valid_loader
    }
    #definiamo una funzione per salvare il checkpoint sul disco
    def save_checkpoint(model, epoch):
        torch.save({
        'state_dict' : model.state_dict(),
        'epoch' : epoch
    }, "{}_{}.pth".format(exp_name, 'checkpoint'))
    for e in range(epochs):
        #iteriamo tra due modalità: train e validation
        for mode in ['train','valid'] :
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                for i, batch in enumerate(loader[mode]):
                    x=batch[0].to(device) #"portiamoli sul device corretto"
                    y=batch[1].to(device)
                    output = model(x) #h theta di x
                    l = criterion(output,y) # loss

                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    n = batch[0].shape[0] #numero di elementi nel batch
                    loss_meter.add(l.item()*n,n)
                    acc_meter.add(acc*n,n)
                    if mode=='train':
                        loss_logger.log(e+(i+1)/len(loader[mode]), loss_meter.value()[0], name=mode)
                        acc_logger.log(e+(i+1)/len(loader[mode]), acc_meter.value()[0], name=mode)
            loss_logger.log(e+1, loss_meter.value()[0], name=mode)
            acc_logger.log(e+1, acc_meter.value()[0], name=mode)
            #salviamo il modello corrente
            #conserviamo solo il corrente, sovrascrivendo il passato
        print(e)
        save_checkpoint(model, e )
        

    return model

# Definizione Modello
from torch import nn
from torchvision.models import squeezenet1_0
def get_model(num_class=4):
    model = squeezenet1_0(pretrained=True)
    num_class = 4
    model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_class
    return model

squeezenet = get_model()
squeezenet_finetuned = trainval_classifier(squeezenet, train_loader, valid_loader, exp_name='squeezenet_finetuning', lr =0.001, epochs = 20) # Training
# squeezenet_finetuned.load_state_dict(torch.load('squeezenet_finetuning_checkpoint.pth')['state_dict'])
# Testing
def test_classifier(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        #print(x.shape)
        output = model(x)
        preds = output.to('cpu').max(1)[1].numpy()
        labs = y.to('cpu').numpy()
        predictions.extend(list(preds))
        labels.extend(list(labs))
    return np.array(predictions), np.array(labels)

#predizioni di test del modello all'ultima epoch
squeezenet_finetuned_predictions_test, labels_test = test_classifier(squeezenet_finetuned, test_loader)
print ("Accuracy di Squeezenet ultimo modello: %0.2f%%" % (accuracy_score(labels_test, squeezenet_finetuned_predictions_test)*100,))
device = "cuda" if torch.cuda.is_available() else "cpu"

im = Image.open('fotoeh.jpg')
im = test_transform(im)
pred = DataLoader([(im,0)], batch_size=1, num_workers=0)
for batch in pred:
    x = batch[0].to(device)
    out_predict = squeezenet_finetuned(x)

print(out_predict)