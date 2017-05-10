import timeit
import pickle

import numpy
import theano
import theano.tensor as T


from nn import *

class LinearClassifier(object):
    """Baseline Model

    This Baseline is just a simple softmax classifier
    """

    def __init__(self, 
                 input, gtruth, # symbolic variables
                 n_out, dim_feat, mode,
                 **kwargs
                ):
               
        self.hasSupervised = True
        self.hasUnsupervised = False
        if mode == 'binary':
            self.mode = 'binary'
            LogisticRegressionClass = BinaryLogisticRegression
        elif mode == 'multilabel':
            self.mode = 'multilabel'
            LogisticRegressionClass = MultiLabelLogisticRegression
        elif mode == 'multiclass':
            self.mode = 'multiclass'
            LogisticRegressionClass = MultiClassLogisticRegression
        else:
            raise NotImplementedError
        
        
        self.embedding=input

        self.logRegressionLayer = LogisticRegressionClass(
            input=input,
            gtruth=gtruth,
            n_in=dim_feat,
            n_out=n_out
        )

        self.nll_classify =  self.logRegressionLayer.negLogLikelihood        

        self.cost_classify = self.nll_classify
                
        self.params_classify = self.logRegressionLayer.params
                    
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]

        
class MLP(object):
    """Baseline Model

    This Baseline is just a simple softmax classifier
    """

    def __init__(self, 
                 input, gtruth, # symbolic variables
                 n_out, dim_feat, dim_hid, n_hid,
                 mode, 
                 seed=1234, L1_reg=0., L2_reg=0., activation='relu',
                 **kwargs
                ):
               
        self.hasSupervised = True
        self.hasUnsupervised = False
        if mode == 'binary':
            self.mode = 'binary'
            LogisticRegressionClass = BinaryLogisticRegression
        elif mode == 'multilabel':
            self.mode = 'multilabel'
            LogisticRegressionClass = MultiLabelLogisticRegression
        elif mode == 'multiclass':
            self.mode = 'multiclass'
            LogisticRegressionClass = MultiClassLogisticRegression
        else:
            raise NotImplementedError
        
        randGen = numpy.random.RandomState(seed)

        if hasattr(dim_hid, '__iter__'):
             assert(len(dim_hid) == n_hid)
        else:
             dim_hid = (dim_hid,)*n_hid

        if activation == 'relu':
            h_act = T.nnet.relu
        elif activation == 'tanh':
            h_act = T.tanh
        elif activation == 'sigmoid':
            h_act = T.nnet.sigmoid        
        else:
            raise NotImplementedError
                
        self.hiddenLayers = []
        for i in xrange(n_hid):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = dim_feat if i == 0 else dim_hid[i-1]
            self.hiddenLayers.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid[i],
                    activation=h_act
            ))

        self.embedding=self.hiddenLayers[-1].output
            
        self.logRegressionLayer = LogisticRegressionClass(
            input=self.hiddenLayers[-1].output,
            gtruth=gtruth,
            n_in=dim_hid[-1],
            n_out=n_out
        )
       
        self.nll_classify =  self.logRegressionLayer.negLogLikelihood        

        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.cost_classify = (self.nll_classify
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
                        
        self.params_classify = (sum([x.params for x in self.hiddenLayers], [])
                              + self.logRegressionLayer.params
        )
                    
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]

    def loadParams (self, filename):
        with open(filename) as f:
            params = pickle.load(f)
        for i in range (len (self.hiddenLayers)):
            assert (self.hiddenLayers[i].W.get_value().shape == params[i*2].shape)
            assert (self.hiddenLayers[i].b.get_value().shape == params[i*2+1].shape)
            self.hiddenLayers[i].W.set_value(params[i*2])
            self.hiddenLayers[i].b.set_value(params[i*2+1])
        n = len (self.hiddenLayers)
        if len(params)!=2*n+2:
            print 'model file does not contain logistic regression weights'
            return
        if self.logRegressionLayer.W.get_value().shape != params[n*2].shape:
            print 'the logistic regression layer does not have consistant dimentionality with the model file'
            return
        if self.logRegressionLayer.b.get_value().shape != params[n*2+1].shape:
            print 'the logistic regression layer does not have consistant dimentionality with the model file'
            return
        self.logRegressionLayer.W.set_value(params[n*2])
        self.logRegressionLayer.b.set_value(params[n*2+1])

        
    def saveParams (self, filename):
        params = []
        for i in range (len (self.hiddenLayers)):
            params.append (self.hiddenLayers[i].W.get_value())
            params.append (self.hiddenLayers[i].b.get_value())
        params.append (self.logRegressionLayer.W.get_value())    
        params.append (self.logRegressionLayer.b.get_value())    
        with open(filename, 'w') as f:
            pickle.dump(params, f)
        