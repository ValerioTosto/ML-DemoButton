from glob import glob
import pandas as pd
import numpy as np
import random

image_paths = glob('Subdataset\\*\\*')

def class_from_path(path):
    _,cl,_ = path.split('\\')
    return cl

labels = [class_from_path(im) for im in image_paths]

dataset = pd.DataFrame({'path':image_paths, 'label': labels})

# CREAZIONE TRAINING, CROSS-VALIDATION, TEST
from sklearn.model_selection import train_test_split
def split_train_val_test(dataset,perc=[0.6,0.1,0.3]):
    train, testval = train_test_split(dataset, test_size=perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2]))
    return train, val, test

random.seed(1395)
np.random.seed(1359)
train, val, test = split_train_val_test(dataset)
print(len(train), len(val), len(test))

train.to_csv('csv\\train.csv', index=None)
val.to_csv('csv\\val.csv', index=None)
test.to_csv('csv\\test.csv', index=None)
