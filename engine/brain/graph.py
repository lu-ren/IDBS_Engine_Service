import numpy
import pickle
import theano
import theano.tensor as T



class TheanoGraph (object):
    
    def __init__(self, data):
        self.data = data 

    def constructKNNexp (self, k=7, sigma=3., uploadToGPU = False):
        # The following way causes memory shortage
        # diff = self.nodes.dimshuffle (['x',0,1]) - self.nodes.dimshuffle ([0,'x',1])
        # self.W = T.sum (diff ** L, axis=2) ** (1./L)
        # Wval = self.W.eval()

        self.nodes = self.data.ingpu
        self.k = k
        self.sigma = sigma
        
        result, updates = theano.scan (fn = lambda dim,dist : dist + (dim.dimshuffle([0,'x']) - dim.dimshuffle(['x',0])) ** 2,
                                       outputs_info = T.zeros((self.nodes.shape[0],self.nodes.shape[0])),
                                       sequences = self.nodes.dimshuffle([1,0])
                                      )
        
        kernel = T.exp (result[-1] / (-2*(sigma**2)))
        
        f = theano.function (inputs=[], outputs=kernel, updates=updates)
        
        self.Wval = f()
        Wtemp = self.Wval.copy()
        self.Aval = numpy.zeros(Wtemp.shape, dtype='int32')
        self.WAlist = [];
        
        for i in range(Wtemp.shape[0]):            
            self.WAlist.append([])
            
        for i in range(Wtemp.shape[0]):
            Wtemp [i, i] = 0.
            for j in range(k):
                NN = Wtemp [i].argmax()
                Wtemp [i, NN] = 0.
                self.Aval [i, NN] = 1
                self.Aval [NN, i] = 1
                self.WAlist [i].append ((self.Wval[i,NN], NN))
                self.WAlist [NN].append ((self.Wval[NN,i], i))
                
        #self.save_mtx ()
        self.save_list ()

        if (uploadToGPU):
            self.A = theano.shared (self.Aval, borrow=True)       
            self.W = theano.shared (self.Wval, borrow=True)       
        
    def constructKNNfromW (self, k, fname, uploadToGPU = False):
        self.Wval = hdf5storage.loadmat(fname)['W']
        Wtemp = numpy.copy (self.Wval)        
        self.WAlist = [];
        
        for i in range(Wtemp.shape[0]):            
            self.WAlist.append([])
            
        for i in range(Wtemp.shape[0]):
            Wtemp [i, i] = 0.
            for j in range(k):
                NN = Wtemp [i].argmax()
                Wtemp [i, NN] = 0.
                self.WAlist [i].append ((self.Wval[i,NN], NN))
                self.WAlist [NN].append ((self.Wval[NN,i], i))
                
        #self.save_list ()

        if (uploadToGPU):
            self.W = theano.shared (self.Wval, borrow=True)       
        
        
    def constructGTruthGraph (self):
        self.WAlist = [];
        numnodes = self.data ['data_num']
        
        for i in range(numnodes):            
            self.WAlist.append([])

        for i in range(numnodes):
            for j in range(i):
                if self.data['yval_original'][i] == self.data['yval_original'][j]:
                    self.WAlist[i].append((1.,j))
                    self.WAlist[j].append((1.,i))
                
        
    def save_mtx (self):
        with open ('./data/ucf101_graph_KNNexp_k=%d_sigma=%.2f_seed=%d'%(self.k,self.sigma,self.dataseed), 'w') as fout:
            pickle.dump (self.Aval, fout)
            pickle.dump (self.Wval, fout)
        hdf5storage.savemat('./data/ucf101_graph_KNNexp_k=%d_sigma=%.2f_seed=%d.mat'%(self.k,self.sigma,self.dataseed),
                         {"kernel":self.Wval,
                          "adjacency":self.Aval,
                         })

    def load_mtx (self, k,sigma , uploadToGPU = False):
        self.k = k
        self.sigma = sigma
        
        with open ('./data/ucf101_graph_KNNexp_k=%d_sigma=%.2f_seed=%d'%(self.k,self.sigma,self.dataseed), 'r') as fin:
            self.Aval = pickle.load (fin)
            self.Wval = pickle.load (fin)
        if (uploadToGPU):
            self.A = theano.shared (self.Aval, borrow=True)       
            self.W = theano.shared (self.Wval, borrow=True)       
            
    def save_list (self, fname=None):
        if fname is None:
            fname = './data/ucf101_graph_KNNexp_WAlist_k=%d_sigma=%.2f_seed=%d'%(self.k,self.sigma,self.dataseed)
        with open (fname, 'w') as fout:
            pickle.dump (self.WAlist, fout)
            
    def load_list (self, k,sigma):
        self.k = k
        self.sigma = sigma
        
        with open ('./data/ucf101_graph_KNNexp_WAlist_k=%d_sigma=%.2f_seed=%d'%(self.k,self.sigma,self.dataseed), 'r') as fin:
            self.WAlist = pickle.load (fin)
        
    def load_mtx_fromfile (self, filename, uploadToGPU = False):
        with open (filename, 'r') as fin:
            self.Aval = pickle.load (fin)
            self.Wval = pickle.load (fin)
        if (uploadToGPU):
            self.A = theano.shared (self.Aval, borrow=True)       
            self.W = theano.shared (self.Wval, borrow=True)       
 
    def load_list_fromfile (self, filename):
        with open (filename, 'r') as fin:
            self.WAlist = pickle.load (fin)

