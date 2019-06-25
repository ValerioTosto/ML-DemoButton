from loader import load_dataset
from loader import get_transform
from model import get_model_squeezenet, get_model_vgg
from tester import test_classifier
from trainer import trainval_classifier
from sklearn.metrics import accuracy_score
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
import torch

def training(modelName, pretrainedFlag, epochsNumber):

    # Caricamento del dataset
    train_loader,valid_loader,test_loader = load_dataset()

    # Definizione del modello
    num_class = 4 

    if modelName == 'SqueezeNet':
        model = squeezenet1_0(pretained=pretrainedFlag)
        model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_class
    else:
        model = vgg16(pretrained=pretrainedFlag)
        model.classifier[6] = nn.Linear(4096, num_class)

    model = trainval_classifier(model, train_loader, valid_loader, lr=0.001, exp_name=modelName, momentum=0.99, epochs=epochsNumber)

    # Fase di test
    predictions_test, labels_test = test_classifier(model, test_loader)

    # Predizioni di test del modello all'ultima epoch
    accuracy = accuracy_score(labels_test, predictions_test)*100
    print ("Accuracy di ", modelName, ": ", accuracy)

    return accuracy, model
    
def execute(modelName, fileName):
    # Creo un modello per caricare il checkpoint
    if modelName == 'SqueezeNet':
        model = squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = 4
    else:
        model = vgg16()
        model.classifier[6] = nn.Linear(4096, 4)
    model.load_state_dict('checkpoint\\' + torch.load(modelName + 'checkpoint.pth')['state_dict'])

    # Predico la classe dell'input 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    im = Image.open(fileName)
    im = get_transform(im)
    pred = DataLoader([(im,0)], batch_size=1, num_workers=0)
    for batch in pred:
        x = batch[0].to(device)
        out_predict = model(x)

    print(out_predict)

if __name__ == '__main__':
    training("VGG16", True, 2)