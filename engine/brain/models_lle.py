import numpy
import theano
import theano.tensor as T

import pickle

from nn import *
        
class SharedFullyConsLLE(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, gtruth, context, weights,# symbolic variables
                 n_out, dim_feat, dim_hid, n_hid, knn,# hyperparameters
                 mode,
                 seed=1234, L1_reg=0., L2_reg=0., zeromean_reg=0., cov_reg=0., # hyperparameters
                 **kwargs
                ):
        """Initialize the parameters

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        .....................
        .
        ..
        ...
        ....
        """

        self.hasSupervised = True
        self.hasUnsupervised = True
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
               
        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        if hasattr(dim_hid, '__iter__'):
             assert(len(dim_hid) == n_hid)
        else:
             dim_hid = (dim_hid,)*n_hid

        self.hiddenLayers = []
        for i in xrange(n_hid):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = dim_feat if i == 0 else dim_hid[i-1]
            h_act = T.nnet.relu if i == n_hid-1 else T.nnet.relu
            self.hiddenLayers.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid[i],
                    activation=h_act
            ))

        self.embedding=self.hiddenLayers[-1].output

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionClass(
            input=self.hiddenLayers[-1].output,
            gtruth=gtruth,
            n_in=dim_hid[-1],
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.nll_classify =  self.logRegressionLayer.negLogLikelihood        

        self.cost_classify = (self.nll_classify
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        batch_size = context.shape[0]/knn
        
        self.contextBranch = []
        for branch in range (knn):
            net = []
            for i in xrange(n_hid):
                h_input = context [branch*batch_size:(branch+1)*batch_size] if i == 0 else net[i-1].output
                net.append (self.hiddenLayers[i].replicate (h_input))
            self.contextBranch.append(net)
        
        reconstruction = 0
        for branch in range (knn):
            reconstruction = reconstruction + T.tile(
                weights[branch*batch_size:(branch+1)*batch_size].dimshuffle([0,'x']),
                (1,dim_hid[-1])) * self.contextBranch[branch][-1].output
        
        err = T.mean ((self.embedding - reconstruction)**2, axis=1)
        
        self.reconst_loss = T.mean(err)

        self.zeromean_loss = T.mean(T.mean(self.embedding, axis=0)**2)
#        self.zeromean_loss = T.mean(T.abs_(T.mean(self.embedding, axis=0)))
        
        self.cov_loss = T.mean((T.dot(self.embedding.T, self.embedding) - T.eye(dim_hid[-1]))**2)
#        self.cov_loss = T.mean(T.abs_(T.dot(self.embedding.T, self.embedding) - T.eye(dim_hid[-1])))
                        
        
        self.L1_ctx = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
        )

        self.L2_sqr_ctx = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
        )
                
        self.cost_predictContext = (self.reconst_loss
            + zeromean_reg * self.zeromean_loss                   
            + cov_reg * self.cov_loss                       
            + L1_reg * self.L1_ctx
            + L2_reg * self.L2_sqr_ctx
        )
        
        self.debug_info = [self.reconst_loss, self.zeromean_loss, self.cov_loss]
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params_classify = (sum([x.params for x in self.hiddenLayers], [])
                              + self.logRegressionLayer.params
        )

        self.params_predictContext = (sum([x.params for x in self.hiddenLayers], [])
        )
                    
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]

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
