import re
import cv2
# Lettura file e slicing
lines = []

filename_elan = 'elantxt/0087.txt'
filename_video = 'video/0087.mp4'
# Estrazione righe
with open(filename_elan) as f:  
    while(True):
        line = f.readline()
        # print(line)
        if line == '':
            break
        lines.append(line)

# Creazione struttura start-end slice
slices = []
for line in lines:
    splitter = re.split('\t|\n', line)
    start = float(splitter[3])
    end = float(splitter[5])
    label = splitter[8]
    slices.append([start,end,label])
    
# Apertura video ed estrazione frames
cap = cv2.VideoCapture(filename_video)

c_zero = 2864 # Modificare
c_class = 2153 # Modificare

print("--- Inizio estrazione ", filename_video, " - ", filename_elan," ---")
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret!=True):
        break
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    classified = False
    # Controlla se il frame si trova in un intervallo classificato come classe X
    for s in slices:
        if (timestamp > s[0]) & (timestamp < s[1]):
            filename = './dataset/' + str(s[2]) + '/C' + str(s[2]) + '_' + str(c_class) + ".jpg"
            classified = True
            c_class += 1

    # Se non è stato classificato come una classe positiva allora viene classificato nella classe negativa
    if not classified:
        filename = './dataset/0/C0_' + str(c_zero) + ".jpg"
        c_zero += 1
    
    cv2.imwrite(filename, frame)

cap.release()
print("--- Finish ---")
