import torch
from torch import nn
from torch.optim import SGD
from torchnet.meter import AverageValueMeter
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from sklearn.metrics import accuracy_score

def trainval_classifier(model, train_loader, valid_loader, exp_name='experiment', lr=0.01, epochs=50, momentum=0.99):
    # Funzione di Loss
    criterion = nn.CrossEntropyLoss()
    # Stochastic gradient descent
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    # Metriche di valutazione
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    # Plot su Visdom
    loss_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Loss', 'legend':['train','valid']})
    acc_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Accuracy','legend':['train','valid']})
    visdom_saver = VisdomSaver(envs=[exp_name])
    # Se Cuda è presente, usalo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # definiamo un dizionario contenente i loader di training e testdefiniamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    # Funzione per salvare il checkpoint
    def save_checkpoint(model, epoch):
        torch.save({
            'state_dict' : model.state_dict(),
            'epoch' : epoch
        }, "{}_{}.pth".format(exp_name, 'checkpoint'))

    # Ciclo principale di training
    for e in range(epochs):
        # Iteriamo tra due modalità: train e validation
        for mode in ['train','valid'] :
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            #abilitiamo i gradienti solo in training
            with torch.set_grad_enabled(mode=='train'): 
                for i, batch in enumerate(loader[mode]):
                    # carichiamo sample x ed etichetta y, 32 alla volta
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    # our theta di x
                    output = model(x)
                    # calcoliamo la loss su Htheta(x) e y
                    l = criterion(output, y)

                    if mode=='train':
                        # calcoliamo le derivate
                        l.backward()
                        # lanciamo uno step della discesa del gradiente
                        optimizer.step()
                        # azzeriamo il gradiente per non accumularlo
                        optimizer.zero_grad()

                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    # numero di elementi nel batch
                    n = batch[0].shape[0] 
                    loss_meter.add(l.item()*n,n)
                    acc_meter.add(acc*n,n)
                    if mode=='train':
                        loss_logger.log(e+(i+1)/len(loader[mode]), loss_meter.value()[0], name=mode)
                        acc_logger.log(e+(i+1)/len(loader[mode]), acc_meter.value()[0], name=mode)
            # log su visdom di loss e accuracy
            loss_logger.log(e+1, loss_meter.value()[0], name=mode)
            acc_logger.log(e+1, acc_meter.value()[0], name=mode)
            #salviamo solo il corrente, sovrascrivendo il passato
        print(e)
        save_checkpoint(model, e )

    return model
                    




























