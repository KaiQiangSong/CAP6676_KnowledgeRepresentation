import numpy as np
from sklearn.model_selection import KFold

def loadFromFile(fName = 'covtype.data'):
    f = open(fName, 'r')
    data, label, Index = [], [], 0
    for l in f:
        Index += 1
        
        line = eval('['+l.strip()+']')
        
        if (len(line) != 55):
            print 'Length Error', Index
            continue
        
        data_i_1 = np.asarray(line[:10], dtype = np.float32)
        data_i_2 = np.argwhere(np.asarray(line[10:14], dtype = np.float32) > 0).astype(float).flatten()
        data_i_3 = np.argwhere(np.asarray(line[14:54], dtype = np.float32) > 0).astype(float).flatten()
        data_i = np.concatenate([data_i_1, data_i_2, data_i_3])
        
        data.append(data_i)
        label.append(line[-1])
    
    mask = np.asarray([True] * 10 + [False] * 2, dtype = np.int)
            
    return np.stack(data), np.asarray(label, dtype = np.int), mask

def K_Fold_data(data, label, k = 5):
    kf = KFold(n_splits=k, shuffle=True)
    kfold = []
    for train_index, test_index in kf.split(data, label):
        kfold.append((train_index, test_index))
    return kfold
    