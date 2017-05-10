import numpy
import theano
import theano.tensor as T

import pickle

from nn import *
        
class SharedFullyCons(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, gtruth, context, gamma, # symbolic variables
                 n_out, dim_feat, dim_hid, n_hid, # hyperparameters
                 mode,
                 seed=1234, L1_reg=0., L2_reg=0., # hyperparameters
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
            h_act = T.tanh if i == n_hid-1 else T.tanh#T.nnet.relu
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
        
  
        self.imagHidLayersForCntx = []
        for i in xrange(n_hid):
            h_input = context if i == 0 else self.imagHidLayersForCntx[i-1].output
            self.imagHidLayersForCntx.append (self.hiddenLayers[i].replicate (h_input))
        
        self.contextPrediction = T.nnet.sigmoid(
                gamma * T.diag(T.dot(self.hiddenLayers[-1].output, self.imagHidLayersForCntx[-1].output.T))
        )
        
        
        self.L1_ctx = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
        )

        self.L2_sqr_ctx = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
        )
        
        
        self.nll_predictContext = -T.mean(T.log(self.contextPrediction))

        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1_ctx
            + L2_reg * self.L2_sqr_ctx
        )

        self.debug_info = []
        
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
                
        
class SplittedFullyCons(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, gtruth, context, gamma, # symbolic variables
                 n_out, dim_feat, dim_hid, n_hid, # hyperparameters
                 mode,
                 seed=1234, L1_reg=0., L2_reg=0., # hyperparameters
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
            h_act = T.tanh if i == n_hid-1 else T.tanh#T.nnet.relu
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
                
        self.CntxHidLayers = []
        for i in xrange(n_hid):
            h_input = context if i == 0 else self.CntxHidLayers[i-1].output
            h_in = dim_feat if i == 0 else dim_hid[i-1]
            h_act = T.tanh if i == n_hid-1 else T.tanh#T.nnet.relu
            self.CntxHidLayers.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid[i],
                    activation=h_act
            ))

        self.L1_ctx = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + sum([abs(x.W).sum() for x in self.CntxHidLayers])
        )

        self.L2_sqr_ctx = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + sum([(x.W ** 2).sum() for x in self.CntxHidLayers])
        )

            
        self.contextPrediction = T.nnet.sigmoid(
                gamma * T.diag(T.dot(self.hiddenLayers[-1].output, self.CntxHidLayers[-1].output.T))
        )
        
        self.nll_predictContext = -T.mean(T.log(self.contextPrediction))
#        self.nll_predictContext = -T.mean(self.contextPrediction)

        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1_ctx
            + L2_reg * self.L2_sqr_ctx
        )

        self.debug_info = []
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params_classify = (sum([x.params for x in self.hiddenLayers], [])
                              + self.logRegressionLayer.params
        )

        self.params_predictContext = (sum([x.params for x in self.hiddenLayers], [])
                                    + sum([x.params for x in self.CntxHidLayers], [])
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
        for i in range (len (self.CntxHidLayers)):
            j = i + len (self.hiddenLayers)
            assert (self.CntxHidLayers[i].W.get_value().shape == params[j*2].shape)
            assert (self.CntxHidLayers[i].b.get_value().shape == params[j*2+1].shape)
            self.CntxHidLayers[i].W.set_value(params[j*2])
            self.CntxHidLayers[i].b.set_value(params[j*2+1])
        n = len (self.hiddenLayers) + len (self.CntxHidLayers)
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
        for i in range (len (self.CntxHidLayers)):
            params.append (self.CntxHidLayers[i].W.get_value())
            params.append (self.CntxHidLayers[i].b.get_value())
        params.append (self.logRegressionLayer.W.get_value())    
        params.append (self.logRegressionLayer.b.get_value())    
        with open(filename, 'w') as f:
            pickle.dump(params, f)
                
        

        
