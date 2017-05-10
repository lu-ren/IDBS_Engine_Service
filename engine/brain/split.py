import numpy


class DataSplit(object):
    
    def __init__(self, data, seed=1234):
        self.data = data
        self.randGen = numpy.random.RandomState(seed)
        self.randSamp = self.randGen.choice
        
        self.labeled_set = set([])
        self.unlabeled_set = set(range(self.data.size))
        self.validation_set = set([])
        self.train_set = set(range(self.data.size))
        
        self.labeled_set_idx_array = numpy.asarray ([], dtype='int32')
        
    def initBinaryQuery(self, num_pos, num_neg):
        trainSet = numpy.asarray (list (self.train_set))

        posSet = numpy.where(self.data.gtruth.inmemory[trainSet] == 1)[0]
        idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
        idx = trainSet [idx_idx]
        self.labeled_set |= set (idx)

        negSet = numpy.where(self.data.gtruth.inmemory[trainSet] == 0)[0]
        idx_idx = self.randSamp (negSet, size=num_neg, replace=False)
        idx = trainSet [idx_idx]
        self.labeled_set |= set (idx)                           
        
        self.unlabeled_set -= self.labeled_set

        self.labeled_set_idx_array = numpy.asarray(list(self.labeled_set), dtype='int32')
        self.randGen.shuffle (self.labeled_set_idx_array)
        
    def initMulticlassQuery(self, num_pos):
        trainSet = numpy.asarray (list (self.train_set))
        for cat in self.data.gtruth.idx2label:
            posSet = numpy.where(self.data.gtruth.inmemory[trainSet] == cat)[0]
            idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
            idx = trainSet [idx_idx]
            self.labeled_set |= set (idx)
        
        self.unlabeled_set -= self.labeled_set

        self.labeled_set_idx_array = numpy.asarray(list(self.labeled_set), dtype='int32')
        self.randGen.shuffle (self.labeled_set_idx_array)
                
    def initMultilabelQuery(self, num_pos, num_neg):
        trainSet = numpy.asarray (list (self.train_set))
        for cat in range(self.data.gtruth.inmemory.shape[1]):
            posSet = numpy.where(self.data.gtruth.inmemory[trainSet,cat] == 1)[0]
            idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
            idx = trainSet [idx_idx]
            self.labeled_set |= set (idx)
        
            negSet = numpy.where(self.data.gtruth.inmemory[trainSet,cat] == 0)[0]
            idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
            idx = trainSet [idx_idx]
            self.labeled_set |= set (idx)
        
        self.unlabeled_set -= self.labeled_set

        self.labeled_set_idx_array = numpy.asarray(list(self.labeled_set), dtype='int32')
        self.randGen.shuffle (self.labeled_set_idx_array)
        
    def initBinaryValidationSet(self, num_pos, num_neg):
        unlabeledSet = numpy.asarray (list (self.unlabeled_set))
        
        posSet = numpy.where(self.data.gtruth.inmemory[unlabeledSet] == 1)[0]
        idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
        idx = unlabeledSet [idx_idx]
        self.validation_set |= set (idx)

        negSet = numpy.where(self.data.gtruth.inmemory[unlabeledSet] == 0)[0]
        idx_idx = self.randSamp (negSet, size=num_neg, replace=False)
        idx = unlabeledSet [idx_idx]
        self.validation_set |= set (idx)
        
        self.train_set -= self.validation_set

        
    def initMulticlassValidationSet(self, num_pos):
        unlabeledSet = numpy.asarray (list (self.unlabeled_set))
        for cat in self.data.gtruth.idx2label:
            posSet = numpy.where(self.data.gtruth.inmemory[unlabeledSet] == cat)[0]
            idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
            idx = unlabeledSet [idx_idx]
            self.validation_set |= set (idx)
        
        self.train_set -= self.validation_set

    def initMultilabelValidationSet(self, num_pos, num_neg):
        unlabeledSet = numpy.asarray (list (self.unlabeled_set))
        for cat in range(self.data.gtruth.inmemory.shape[1]):
            posSet = numpy.where(self.data.gtruth.inmemory[unlabeledSet,cat] == 1)[0]
            idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
            idx = unlabeledSet [idx_idx]
            self.validation_set |= set (idx)
        
            negSet = numpy.where(self.data.gtruth.inmemory[unlabeledSet,cat] == 0)[0]
            idx_idx = self.randSamp (posSet, size=num_pos, replace=False)
            idx = unlabeledSet [idx_idx]
            self.validation_set |= set (idx)
        
        self.train_set -= self.validation_set
        
        
        
    def initBinaryQueryFromUser(self, idxList, neg_ratio = 1.): 
        
        pos = []
        neg = []
        
        for idx, label in idxList:
            if label == 0:
                neg.append (idx)
            elif label == 1: 
                pos.append (idx)
            else:
                raise Exception('In binary query mode, label should be either 0 or 1.')

        print pos, neg        
        
        num_neg_required = int(numpy.ceil (len(pos) * neg_ratio))
                
        if len(neg) < num_neg_required:
            self.labeled_set |= set(pos)
            self.labeled_set |= set(neg)
            self.unlabeled_set -= self.labeled_set

            unlabeledSet = numpy.asarray (list (self.unlabeled_set))
            idx = self.randSamp (unlabeledSet, size= num_neg_required-len(neg), replace=False)
            neg += list(idx)

#        elif len(neg) > num_neg_required:
#            idx = self.randSamp (neg, size= num_neg_required, replace=False)
#            neg = list(idx)
        
        print neg
    
        self.labeled_set |= set(pos)
        self.labeled_set |= set(neg)
        self.unlabeled_set -= self.labeled_set
        
        self.data.gtruth.inmemory.fill (numpy.nan)
        self.data.gtruth.inmemory [pos] = 1
        self.data.gtruth.inmemory [neg] = 0

        self.labeled_set_idx_array = numpy.asarray(list(self.labeled_set), dtype='int32')
        self.randGen.shuffle (self.labeled_set_idx_array)
        
        
        
        
        
        
        
        
