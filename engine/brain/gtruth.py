import numpy
import hdf5storage
import theano



class GroundTruth:

    def __init__(self, mode):
                
        if mode == 'multilabel':
            self.mode = 'multilabel'
        elif mode == 'multiclass':
            self.mode = 'multiclass'
        else:
            raise NotImplementedError
        
    
    def load_mat(self, fname, key):
        
        f = hdf5storage.loadmat(fname)
        self.inmemory = f[key].astype('int32')
        if self.mode == 'multiclass':
            self.inmemory = self.inmemory.flatten()
            self.idx2label = numpy.unique(self.inmemory)
            self.idx2label.sort()
            self.label2idx = {j:i for i,j in enumerate (self.idx2label)}
            
        self.ingpu = theano.shared(self.inmemory, borrow=True)
        self.inmem_copy = numpy.copy(self.inmemory)
        
    
    
    def setClassOfInterest(self, cl):
        if self.mode == 'multiclass':
            numpy.copyto (self.inmemory, (self.inmem_copy==cl).astype('int32'))
        elif self.mode == 'multilabel':
            if len(self.inmemory.shape)==1:
                numpy.copyto (self.inmemory, self.inmem_copy[:,cl-1].astype('int32'))
            else:
                self.inmemory = self.inmem_copy[:,cl-1].astype('int32').flatten()
                self.ingpu = theano.shared(self.inmemory, borrow=True)
                print "WARNING: changing the pointer for gtruth.ingpu. Previously compiled theano functions will not update"

    def reset(self):
        if self.mode == 'multiclass':
            numpy.copyto (self.inmemory, self.inmem_copy)
        elif self.mode == 'multilabel':
            if len(self.inmemory.shape)==2:
                numpy.copyto (self.inmemory, self.inmem_copy)
            else:
                self.inmemory = numpy.copy(self.inmem_copy)
                self.ingpu = theano.shared(self.inmemory, borrow=True)
                print "WARNING: changing the pointer for gtruth.ingpu. Previously compiled theano functions will not update"

    
            