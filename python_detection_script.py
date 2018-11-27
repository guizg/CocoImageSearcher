from gluoncv import model_zoo, data, utils
import os
import time
import signal
import sys
import pickle

SAVED_FILE_PATH = '/Users/gmore/Desktop/labels.pickle'

# import model
net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

# open labels file
pickle_out = open(SAVED_FILE_PATH, "wb")

# choose images path
DATASET_PATH = '/Volumes/Ext SSD/unlabeled2017'
os.chdir(DATASET_PATH)
files = os.listdir('./')


LEN_FILES = len(files)
i = 0
dic = {}
tic = time.time()

# signal handler for SIGINT (control C)
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    print('Saving file ...')
    try:
        pickle.dump(dic, pickle_out)
    except Exception as e:
        print('Could not save file')
        print(e)
        sys.exit(1)
    print(f'File saved at {SAVED_FILE_PATH}')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

for n, file in enumerate(files):
    print(f'{n} of {LEN_FILES}')
    if(i > LEN_FILES):
        break
    if(i % 500 == 0 and i > 0):
        print('Saving file ...')
        try:
            pickle.dump(dic, pickle_out)
        except Exception as e:
            print('Could not save file')
            print(e)
            sys.exit(1)
        print(f'File saved at {SAVED_FILE_PATH}')
    try:
        x, img = data.transforms.presets.yolo.load_test(file, short=512)
        class_IDs, scores, bounding_boxs = net(x)
        scores = scores.asnumpy()
        class_IDs = class_IDs.asnumpy()
        bounding_boxs = bounding_boxs.asnumpy()
        for j in range(len(class_IDs[0])):
        if(scores[0][j] > 0.6):
            index = int(class_IDs[0][j][0])
            if(net.classes[index] not in dic):
                dic[net.classes[index]] = []
            if(file not in dic[net.classes[index]]):
                dic[net.classes[index]].append(file)
    except Exception as e:
        print(f'Could not open image {file}')
        print(e)
    i+=1

toc = time.time()
pickle.dump(dic, pickle_out)
print(f'Time of execution: {toc - tic}')
