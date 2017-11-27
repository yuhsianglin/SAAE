from collections import defaultdict
import os

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle

annotationFile = "../data/CUB_200_2011/attributes/image_attribute_labels.txt"
classFile = "../data/CUB_200_2011/image_class_labels.txt"
outputFile = "../data/preprocessed/attribute_embeddings"
birdTextDir = "../data/BirdsText"
birdTextOutput = "../data/preprocessed/birdsText.txt"

def buildAttributeSemanticSpace():
    img2Class = {}
    classCount = defaultdict(int)
    with open(classFile, 'r') as f:
        for line in f:
            imgId, classId = line.split()
            img2Class[imgId] = int(classId) - 1
            classCount[int(classId) - 1] += 1

    result = np.zeros((200, 312))
    with open(annotationFile, 'r') as f:
        for line in f:
            if len(line.split()) != 5:
                continue
            imgId, featureId, exist, _1, _2 = line.split()
            result[img2Class[imgId]][int(featureId) - 1] += int(exist)
    
    for i in range(200):
        result[i] = result[i] / classCount[i]
    
    np.save(outputFile, result)
    return result

def bird2Text():
    documents = []
    for filename in os.listdir(birdTextDir):
        className = int(filename.split('.')[0]) - 1
        fileContent = open(os.path.join(birdTextDir, filename), 'r', encoding='utf-8', errors='ignore').read()
        tokenList = fileContent.split()
        doc = TaggedDocument(tokenList, [className])
        documents.append(doc)
    model = Doc2Vec(size = 300, min_count = 5, window = 8)

    model.build_vocab(documents)
    for epoch in range(20):
        model.train(documents)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    result = []
    for i in range(200):
        vector = model.docvecs[i]
        result.append(vector)
    
    np.savetxt(birdTextOutput, np.array(result))

if __name__ == "__main__":
    # buildAttributeSemanticSpace()
    bird2Text()
