import numpy as np
from numpy import histogram
from copy import deepcopy

class TreeNode(object):
    def __init__(self, attr = None, threshold = None, children = None, prob = None, category = None):
        self.attr = attr
        self.threshold = threshold
        self.children = children
        self.category = category
        self.prob = prob
        
        self.nodeType = 'Empty'
        if (category is not None):
            self.nodeType = 'Leaf'
        elif (children is not None):
            if (type(threshold) == list):
                self.nodeType = 'discrete'
            else:
                self.nodeType = 'continuous'


class DecisionTree(object):
    EPS = 1e-8
    LOG = open('log.txt','a+')
    
    def __init__(self, data, label, mask, method_eval = 'error_rate', method_attr = 'rand', stopValue = 0.3, maxDeepth = 12, maxUsage = 1):
        self.data = data
        self.label = label
        self.mask = mask
        self.maxUsage = mask.astype(int) * (maxUsage - 1) + 1
        
        self.method_eval = method_eval
        self.method_attr = method_attr
        self.stopValue = stopValue
        self.maxDeepth = maxDeepth
        self.Root = None
        print >> self.LOG, 'A New Start:'
    
    def evaluate(self, index):
        if (index.shape[0] == 0):
            return 0
        
        count = np.bincount(self.label[index])
        prob = count * 1.0 /index.shape[0]
        
        if self.method_eval == 'error_rate':
            category = np.argmax(prob)
            return (1 - prob[category])
        elif self.method_eval == 'GINI':
            return  1- np.sum(prob * prob)
        elif self.method_eval == 'entropy':
            return - np.sum(prob * np.log(prob + self.EPS))
        return 1
        
    def gain(self, score, scores, prob):
        return score - np.inner(scores, prob) 
    
    def Devide(self, index, attr):
        scoreThis = self.evaluate(index)
        dataThis = self.data[index, attr].flatten()
        if (not self.mask[attr]):
            values = np.unique(self.data[index,attr])
            scores = []
            indexes = []
            
            for value in values.tolist():
                newIndex = index[np.where(dataThis == value)]
                score = self.evaluate(newIndex)
                
                indexes.append(newIndex)
                scores.append(score)
            scores = np.asarray(scores, dtype = np.float32)
            prob = np.asarray([id.shape[0] * 1.0 / index.shape[0] for id in indexes], dtype = np.float32)
            benifit = self.gain(scoreThis, scores, prob)
            return benifit, values.tolist(), indexes, prob
        
        values = np.unique(self.data[index,attr])
        bestBenifit, Threshold, bestIndexes, bestProb = None, None, None, None
        for value in values.tolist():     
            subA = index[np.where(dataThis < value)]
            subB = index[np.where(dataThis >= value)]
            
            indexes = [subA, subB]
            scores = np.asarray([self.evaluate(subA), self.evaluate(subB)], dtype = np.float32)
            prob = np.asarray([subA.shape[0] * 1.0 / index.shape[0], subB.shape[0] * 1.0 / index.shape[0]], dtype = np.float32)
            benifit = self.gain(scoreThis, scores, prob)
                
            if (bestBenifit == None) or (benifit > bestBenifit):
                bestBenifit, Threshold, bestIndexes, bestProb = benifit, value, indexes, prob
        return bestBenifit, Threshold, bestIndexes, bestProb
        
    def Build(self, index, usage):
        print >> self.LOG, 'Build :', index.shape, usage
        usageThis = deepcopy(usage)
        attrList = np.argwhere((self.maxUsage - usage) > 0)
        if index.shape[0] == 0:
            print >> self.LOG, 'Empty'
            return TreeNode()
        
        if attrList.flatten().shape[0] == 0:
            print >> self.LOG, 'Run out of usage'
            return TreeNode(category = np.argmax(np.bincount(self.label[index])))
                  
        scoreThis =  self.evaluate(index)
        if (scoreThis <= self.EPS):
            print >> self.LOG, 'One Category Left'
            return TreeNode(category = np.argmax(np.bincount(self.label[index])))
        
        if scoreThis < self.stopValue:
            print >> self.LOG, 'Achieve Lower Bound', scoreThis
            return TreeNode(category = np.argmax(np.bincount(self.label[index])))
        
        if self.method_attr == 'rand':
            attrThis = np.random.choice(attrList.flatten(), 1)
            print >> self.LOG, 'Select', attrThis
            usageThis[attrThis] += 1
            benifit, threshold, indexes, prob = self.Devide(index, attrThis)
            children = []
            for id in indexes:
                print >> self.LOG, id.shape,'from', index.shape
                chNode = self.Build(id, usageThis)
                children.append(chNode)
            return TreeNode(attrThis, threshold, children, prob)
        
        elif self.method_attr == 'best':
            bestBenifit, bestThreshold, bestIndexes, bestAttr = None, None, None, None
            for attrThis in attrList:
                benifit, threshold, indexes, prob = self.Devide(index, attrThis)
                if (bestBenifit == None) or (benifit > bestBenifit):
                    bestBenifit, bestThreshold, bestIndexes, bestAttr = benifit, threshold, indexes, attrThis
            usageThis[bestAttr] += 1
            children = []
            for id in bestIndexes:
                chNode = self.Build(id, usageThis)
                children.append(chNode)
            return TreeNode(bestAttr, bestThreshold, children, prob)
        return TreeNode()
    
    def BuildTree(self, index):
        self.Root = self.Build(index, np.zeros_like(self.mask, dtype = np.int))
    
    def testNode(self, Node, index):
        if (Node.nodeType == 'Empty'):
            return np.zeros_like(index, dtype = np.int)
        
        if (Node.nodeType == 'Leaf'):
            return np.full_like(index, Node.category, dtype = np.int)
        
        labels = np.zeros_like(index, dtype = np.int)
        dataThis = self.data[index, Node.attr]
        
        if Node.nodeType == 'discrete':
            size = len(Node.threshold)
            for i in range(size):
                value = Node.threshold[i]
                ch = Node.children[i]
                newIndex = np.where(dataThis == value)
                labels[newIndex] = self.testNode(ch, index[newIndex])
        else:
            T = Node.threshold
            subA = np.where(dataThis < T)
            labels[subA] = self.testNode(Node.children[0], index[subA])
            subB = np.where(dataThis >= T)
            labels[subB] = self.testNode(Node.children[1], index[subB])
        return labels
        
    def Test(self, index):
        
        labels = self.testNode(self.Root, index).flatten()
        MissingIndex = np.where(labels == 0)[0]
        CorrectIndex = np.where(labels == self.label[index])[0]
        ErrorIndex = np.where((labels != self.label[index]) & (labels != 0))[0]
        MRate = MissingIndex.shape[0] * 1.0 / index.shape[0]
        CRate = CorrectIndex.shape[0] * 1.0 / index.shape[0]
        ERate = ErrorIndex.shape[0] * 1.0 / index.shape[0]
        
        return labels, MissingIndex, CorrectIndex, ErrorIndex, MRate, CRate, ERate
    
    def eval(self, label_pred, index):
        label_target = self.label[index]
        EM = np.zeros((8,8), dtype = np.float32)
        size = index.shape[0]
        for i in range(size):
            EM[label_target[i], label_pred[i]] += 1
        precision, recall = [], []
        for i in range(1,8):
            recall.append(EM[i,i] / (EM[i,:].sum() + self.EPS))
            precision.append(EM[i,i] / (EM[:,i].sum() + self.EPS))
        
        recall = np.asarray(recall, dtype = np.float32)
        precision = np.asarray(precision, dtype = np.float32)
        f1_score = 2 * recall * precision / (recall + precision + self.EPS)
        
        return recall, precision, f1_score, f1_score.mean()