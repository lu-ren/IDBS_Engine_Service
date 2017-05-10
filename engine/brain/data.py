import numpy
import hdf5storage
import hickle
import os
import theano
import lmdb


class DataSet:
    
    def create_batch (self, key, batch_size):
        raise NotImplementedError
        
    def update_batch (self, key, idx):
        raise NotImplementedError
        
    def get_batch (self, key):
        raise NotImplementedError
        
        

class InGPUDataSet (DataSet):
    
    def __init__(self, datum_shape):
        
        self.datum_shape = datum_shape
        self.ingpu = theano.shared([])
        self.gtruth = None
        self.batch = {}
    
    def load_mat(self, fname, key, key_gtruth=None):
        
        f = hdf5storage.loadmat(fname)
        self.inmemory = f[key]
        self.size = self.inmemory.shape[0] 

#        self.inmemory_index = numpy.arange (self.size)
#        self.inmemory_indexmap = {j:i for i,j in enumerate(self.inmemory_index)}
    
#        if key_gtruth is not None:
#            self.gtruth = f[key_gtruth].astype('int32')
#            self.gtruth_ingpu = theano.shared(self.gtruth, borrow=True)
#            self.gtruth_copy = numpy.copy(self.gtruth)

        self.ingpu = theano.shared(self.inmemory, borrow=True)

    class DataBatch:

        def __init__(self, ingpu):
            self.inmemory = None
            
            self.idx_inmemory = numpy.asarray([], dtype = 'int32')
            self.idx_ingpu = theano.shared(self.idx_inmemory, borrow=True)
            
            self.ingpu = ingpu[self.idx_ingpu]
    
    def create_batch (self, key):
        self.batch[key] = self.DataBatch (self.ingpu)
        
    def allocate_batch (self, key, batch_size):   
        self.batch[key].size = batch_size
        self.batch[key].idx_inmemory = numpy.zeros ((batch_size,), dtype='int32')
        self.batch[key].idx_ingpu.set_value (self.batch[key].idx_inmemory, borrow=True)
        
    def update_batch (self, key, idx):
        self.batch[key].inmemory = self.inmemory [idx]
        numpy.copyto (self.batch[key].idx_inmemory, idx)
        
    def get_batch (self, key):
        return self.batch[key].ingpu

    
    
    
    
class InMemoryDataSet (DataSet):
        
    def __init__(self, datum_shape):
        
        self.datum_shape = datum_shape
        self.ingpu = theano.shared([])
        self.gtruth = None
        self.batch = {}
    
    def load_mat(self, fname, key, key_gtruth=None):
        
        f = hdf5storage.loadmat(fname)
        self.inmemory = f[key]
        self.size = self.inmemory.shape[0] 

    class DataBatch:

        def __init__(self, datum_shape):
            self.inmemory = numpy.array([], ndmin=1+len(datum_shape), dtype = 'float32')
            self.ingpu = theano.shared (self.inmemory, borrow=True)
    
    def create_batch (self, key):
        self.batch[key] = self.DataBatch (self.datum_shape)
        
    def allocate_batch (self, key, batch_size):   
        self.batch[key].size = batch_size
        
    def update_batch (self, key, idx):
        self.batch[key].inmemory = self.inmemory [idx]
        self.batch[key].ingpu.set_value (self.batch[key].inmemory, borrow=True)
        
    def get_batch (self, key):
        return self.batch[key].ingpu
    
    
    
    
    
    
    
class PickleFilesDataSet (DataSet):
    
    def __init__(self, datum_shape):
        
        self.datum_shape = datum_shape
        self.ingpu = theano.shared([])
        self.gtruth = None
        self.batch = {}
    
    def load_pickle_list(self, fname, basedir):
        
        self.pickle_basedir = basedir
        
        files = []
        with open(fname) as f:
            for row in f:
                row = row.split('/')[-1]
                #if 'v_HandStandPushups' in row:
                #    row = 'v_HandstandPushups' + row [18:]
                row = row.split(' ')
                files.append(row[0])

        self.size = len(files)
        self.pickle_list = numpy.asarray(files)
    

    class DataBatch:

        def __init__(self, datum_shape):
            self.inmemory = numpy.array([], ndmin=1+len(datum_shape), dtype = 'float32')
            self.ingpu = theano.shared (self.inmemory, borrow=True)
    
    def create_batch (self, key):
        self.batch[key] = self.DataBatch (self.datum_shape)
               
    def allocate_batch (self, key, batch_size):   
        self.batch[key].size = batch_size
        self.batch[key].inmemory = numpy.zeros ((batch_size,)+self.datum_shape, dtype='float32')
        self.batch[key].ingpu.set_value (self.batch[key].inmemory, borrow=True)
        
    def update_batch(self, key, idx):
        assert (len (idx) == self.batch[key].size)
        for j,i in enumerate(idx):
            video = hickle.load(os.path.join(self.pickle_basedir, self.pickle_list[i]))
            self.batch[key].inmemory[j] = video
        
    def get_batch (self, key):
        return self.batch[key].ingpu
        
        
        

        
class LMDBDataSet (DataSet):
    
    
    def __init__(self, datum_shape):
        
        self.datum_shape = datum_shape
        self.ingpu = theano.shared([])
        self.gtruth = None
        self.batch = {}
    
    def load_lmdb(self, lmdb_path):
        self.env = lmdb.open(lmdb_path)
        self.txn = self.env.begin()
        self.size = self.env.stat()['entries']
        
        
    class DataBatch:

        def __init__(self, datum_shape):
            self.inmemory = numpy.array([], ndmin=1+len(datum_shape), dtype = 'float32')
            self.ingpu = theano.shared (self.inmemory, borrow=True)
    
    def create_batch (self, key):
        self.batch[key] = self.DataBatch (self.datum_shape)
               
    def allocate_batch (self, key, batch_size):   
        self.batch[key].size = batch_size
        self.batch[key].inmemory = numpy.zeros ((batch_size,)+self.datum_shape, dtype='float32')
        self.batch[key].ingpu.set_value (self.batch[key].inmemory, borrow=True)
        
    def update_batch(self, key, idx):
        assert (len (idx) == self.batch[key].size)
        for j,i in enumerate(idx):
            bytes = self.txn.get ('{:06}'.format(i))
            video = numpy.fromstring(bytes, 'float32').reshape(self.datum_shape)
            self.batch[key].inmemory[j] = video
        
    def get_batch (self, key):
        return self.batch[key].ingpu
    
    
    

    
    
    