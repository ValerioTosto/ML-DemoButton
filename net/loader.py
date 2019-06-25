from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from CSVImageDataset import CSVImageDataset

def load_dataset():
    # Normalizzazione DA CAMBIARE
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
    ])

    # Caricamento immagine dai csv
    train_set = CSVImageDataset('..','..\\csv\\train.csv', transform = train_transform)
    valid_set = CSVImageDataset('..','..\\csv\\val.csv', transform = train_transform)
    test_set = CSVImageDataset('..','..\\csv\\test.csv', transform = train_transform)
    train_loader = DataLoader(train_set, batch_size=16, num_workers=6, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=6)

    return train_loader,valid_loader,test_loader

def get_transform(im):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
    ])

    return train_transform(im)
