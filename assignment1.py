import numpy as np
import time
import cPickle as Pickle

from covertype_loader import *
from DecisionTree import *

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data

if __name__ == '__main__':
    
    print 'Data Loading...'
    
    st = time.time()
    data, label, mask = loadFromFile()
    print data.shape, label.shape, mask.shape
    print time.time() - st
    
    st = time.time()
    print 'K-Fold-Splitting'
    kFoldData = K_Fold_data(data, label)
    print time.time() - st
    
    Index = 0
    for fold_i in kFoldData:
        Index += 1
        saveToPKL('fold_'+str(Index), fold_i)
        print 'Fold :', Index
        trainIndex, testIndex = fold_i[0], fold_i[1]
        
        print 'Training...'
        st = time.time()
        model = DecisionTree(data, label, mask)
        model.BuildTree(trainIndex)
        saveToPKL('model_'+str(Index), model)
        print time.time() -st
        
        print 'Testing...'
        
        labels, Mid, Cid, Eid, Mr, Cr, Er = model.Test(trainIndex)
        print 'Train Set'
        print 'Missing Rate:', Mr
        print 'Correct Rate:', Cr
        print 'Error   Rate:', Er 
        
        st = time.time()
        labels, Mid, Cid, Eid, Mr, Cr, Er = model.Test(testIndex)
        print 'Test Set'
        print 'Missing Rate:', Mr
        print 'Correct Rate:', Cr
        print 'Error   Rate:', Er 
        print time.time() - st
        