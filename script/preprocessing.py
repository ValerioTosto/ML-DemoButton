import cv2
from glob import glob 
from os.path import basename
img_paths = glob('.\\dataset\\3\\*')
for path in img_paths:
    img = cv2.imread(path)
    newimg = cv2.resize(img,(224,224))
    cv2.imwrite(path,newimg)
    print(basename(path))