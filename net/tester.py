import torch
import numpy as np


def test_classifier(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # creiamo due liste, una per le nostre predizioni e
    # una per le etichette reali
    predictions, labels = [], []
    for batch in loader:
        # insieme dei nostri samples
        x = batch[0].to(device)
        # insieme delle nostre etichette
        y = batch[1].to(device)
        # diamo in pasto i samples al modello e calcoliamo Htheta
        output = model(x)
        # ritorna la classe con la maggiore probabilità
        preds = output.to('cpu').max(1)[1].numpy()
        # ritorna l'etichetta reale
        labs = y.to('cpu').numpy()
        # aggiunge tutte le predizioni appena fatte alla lista predictions
        predictions.extend(list(preds))
        # aggiunge tutte le etichette reali appena scorse alla lista labels
        labels.extend(list(labs))
    #ritorna predictions e labels nome array numpy
    return np.array(predictions), np.array(labels)