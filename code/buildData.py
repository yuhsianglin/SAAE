import numpy as np
import os

image_id_file = '../data/CUB_200_2011/images.txt'
image_class_file = '../data/CUB_200_2011/image_class_labels.txt'
train_test_file = '../data/CUB_200_2011/train_test_split.txt'
image_features_dir = '../data/preprocessed/'
training_data_file = '../data/preprocessed/training.txt'
testing_data_file = '../data/preprocessed/testing.txt'

def readImage2IdFile(filepath):
    image2Id = {}
    with open(filepath, 'r') as f:
        for line in f:
            _id, filename = line.split()
            image2Id[filename] = _id
    
    return image2Id

def readImage2ClassFile(filepath):
    image2Class = {}
    with open(filepath, 'r') as f:
        for line in f:
            imageId, classId = line.split()
            image2Class[imageId] = int(classId) - 1
    
    return image2Class

def readTrainTestSplitFile(filepath):
    trainTestSplit = {}
    with open(filepath, 'r') as f:
        for line in f:
            imageId, isTrain = line.split()
            trainTestSplit[imageId] = bool(int(isTrain))
    
    return trainTestSplit

def readAndPreprocessImageFeatures(fileDir, image2Id, image2Class, trainTestSplit):
    trainingData, testingData = [], []
    for folder in os.listdir(fileDir):
        if not os.path.isdir(os.path.join(fileDir, folder)):
            continue
        
        print(folder)
        for filename in os.listdir(os.path.join(fileDir, folder)):
            features = np.load(os.path.join(fileDir, folder, filename)).squeeze().tolist()
            name = folder + '/' + filename.rstrip('.npy')
            imgId = image2Id[name]
            _class = image2Class[imgId]
            features.append(_class)
            if trainTestSplit[imgId]:
                trainingData.append(features)
            else:
                testingData.append(features)
    
    print(len(trainingData), len(testingData))
    np.savetxt(training_data_file, np.array(trainingData))
    np.savetxt(testing_data_file, np.array(testingData))
    
    return

if __name__ == "__main__":
    image2Id = readImage2IdFile(image_id_file)
    image2Class = readImage2ClassFile(image_class_file)
    trainTestSplit = readTrainTestSplitFile(train_test_file)
    readAndPreprocessImageFeatures(image_features_dir, image2Id, image2Class, trainTestSplit)

    