from net.loader import load_dataset
from net.model import get_model_squeezenet, get_model_vgg
from net.tester import test_classifier
from net.trainer import trainval_classifier
from sklearn.metrics import accuracy_score
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16
from torch import nn

def training(modelName, pretrainedFlag, epochsNumber):
    # caricamento del dataset
    train_loader,valid_loader,test_loader = load_dataset()

    # definizione del modello
    choose = 0
    # model = get_model_squeezenet(pretrainedFlag) if choose == 0 else get_model_vgg(pretrainedFlag)
    # model = vgg16(pretrained=pretrainedFlag)
    model = squeezenet1_0(pretrainedFlag)
    num_class = 4
    # model.classifier[6] = nn.Linear(4096, num_class)
    model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_class
    # fase di training 
    net = "Squeezenet model" if choose == 0 else "VGG16"
    model = trainval_classifier(model, train_loader, valid_loader, lr=0.001, exp_name=net, momentum=0.99, epochs=epochsNumber)

    # fase di test
    predictions_test, labels_test = test_classifier(model, test_loader)

    #predizioni di test del modello all'ultima epoch
    accuracy = accuracy_score(labels_test, predictions_test)*100
    print ("Accuracy di ", net, ": ", accuracy)
    return accuracy


def execute(fileName):
    # demo
    predicted = 0
    return predicted