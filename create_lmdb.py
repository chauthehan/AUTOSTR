import os 
import lmdb
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def _is_difficult(word):
    assert isinstance(word, str)
    return not re.match('^[\w]+$', word)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=12000000000)
    cache = {}
    cnt = 1

    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if len(label) == 0:
            continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%04d'%cnt
        labelKey = 'label-%04d'%cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()        
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))

        cnt += 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)



data_dir = 'data/'
lmdb_output_path = 'lmdb_test/'
gt_file = 'data/test_line_annotation.txt'

with open(gt_file, 'r') as f:
    lines = [line.strip('\n') for line in f.readlines()]

imagePathList, labelList = [], []
for i, line in enumerate(lines):
    splits = line.split('\t')
    image_name = splits[0]
    gt_text = splits[1]
    #print(image_name, gt_text)
    imagePathList.append(os.path.join(data_dir, image_name))
    labelList.append(gt_text)

createDataset(lmdb_output_path, imagePathList, labelList)   
