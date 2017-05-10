import numpy
import theano
import theano.tensor as T

from nn import *
       
class PlanetoidTransductiveModel(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, gtruth, idx_input, idx_context, gamma, # symbolic variables
                 dict_size, n_out, dim_feat, dim_hid, dim_emb, n_hid1, n_hid2, # hyperparameters
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
        
        # keep track of model input
        self.input = input
        
        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        # if hasattr(n_hidden, '__iter__'):
        #     assert(len(n_hidden) == n_hiddenLayers)
        # else:
        #     n_hidden = (n_hidden,)*n_hiddenLayers

        self.hiddenLayers1 = []
        for i in xrange(n_hid1):
            h_input = self.input if i == 0 else self.hiddenLayers1[i-1].output
            h_in = dim_feat if i == 0 else dim_hid
            self.hiddenLayers1.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid,
                    activation=T.nnet.relu
            ))

        emb_values = numpy.asarray(
            randGen.uniform(
                low=-numpy.sqrt(1. / (dict_size + dim_emb)),
                high=numpy.sqrt(1. / (dict_size + dim_emb)),
                size=(dict_size, dim_emb)
            ),
            dtype=theano.config.floatX
        )
        self.embdict = theano.shared(
            emb_values*1.,
#            value=numpy.zeros(
#                (dict_size, dim_emb),
#                dtype=theano.config.floatX
#            ),
            name='embdict',
            borrow=True
        )
        self.embedding = self.embdict [idx_input]
        
        self.hiddenLayers2 = []
        for i in xrange(n_hid2):
            h_input = self.embedding if i == 0 else self.hiddenLayers2[i-1].output
            h_in = dim_emb if i == 0 else dim_hid
            self.hiddenLayers2.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid,
                    activation=T.nnet.relu
            ))        

        self.softmax_input = T.concatenate ([self.hiddenLayers1[-1].output, self.hiddenLayers2[-1].output], axis=1)    

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionClass(
            input=self.softmax_input,
            gtruth=gtruth,
            n_in=2*dim_hid,
            n_out=n_out
        )

        self.skipgramLayer = SkipgramLayer (
            rng=randGen,
            embedding=self.embedding, 
            idx_context=idx_context, 
            gamma=gamma, 
            dim_emb=dim_emb, 
            dict_size=dict_size
        )
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers1])
            + sum([abs(x.W).sum() for x in self.hiddenLayers2])
            + abs(self.logRegressionLayer.W).sum()
            + abs(self.skipgramLayer.W).sum()
            + abs(self.embdict).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers1])
            + sum([(x.W ** 2).sum() for x in self.hiddenLayers2])
            + (self.logRegressionLayer.W ** 2).sum()
            + (self.skipgramLayer.W ** 2).sum()
            + (self.embdict ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.nll_classify =  self.logRegressionLayer.negLogLikelihood        

        self.cost_classify = (self.nll_classify
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        # same holds for the function computing the number of errors
        self.errorRate = self.logRegressionLayer.errorRate
        self.precision = self.logRegressionLayer.precision
        self.recall = self.logRegressionLayer.recall
        
        self.nll_predictContext = self.skipgramLayer.negLogLikelihood        

#        self.cost_predictContext = (self.nll_predictContext
#            + L1_reg * self.L1
#            + L2_reg * self.L2_sqr
#        )

        self.cost_predictContext = self.nll_predictContext + L2_reg * self.skipgramLayer.regul
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params_classify = (sum([x.params for x in self.hiddenLayers1], [])
                              + sum([x.params for x in self.hiddenLayers2], [])
#                              + [self.embdict]
                              + self.logRegressionLayer.params
        )

        self.params_predictContext = ([self.embdict]
                                    + self.skipgramLayer.params
        )
                    
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]

        
        
        
class PlanetoidInductiveModel(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, gtruth, idx_input, idx_context, gamma, # symbolic variables
                 dict_size, n_out, dim_feat, dim_hid, n_hid1, n_hid2, n_hid3, # hyperparameters
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
        
        # keep track of model input
        self.input = input
        
        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        # if hasattr(n_hidden, '__iter__'):
        #     assert(len(n_hidden) == n_hiddenLayers)
        # else:
        #     n_hidden = (n_hidden,)*n_hiddenLayers

        self.hiddenLayers1 = []
        for i in xrange(n_hid1):
            h_input = self.input if i == 0 else self.hiddenLayers1[i-1].output
            h_in = dim_feat if i == 0 else dim_hid
            self.hiddenLayers1.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid,
                    activation=T.nnet.relu
            ))

        self.hiddenLayers3 = []
        for i in xrange(n_hid3):
            h_input = self.input if i == 0 else self.hiddenLayers3[i-1].output
            h_in = dim_feat if i == 0 else dim_hid
            self.hiddenLayers3.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid,
                    activation=T.nnet.relu
            ))

        self.embedding = self.hiddenLayers3[-1].output
        
        self.hiddenLayers2 = []
        for i in xrange(n_hid2):
            h_input = self.embedding if i == 0 else self.hiddenLayers2[i-1].output
            h_in = dim_hid if i == 0 else dim_hid
            self.hiddenLayers2.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid,
                    activation=T.nnet.relu
            ))        

        self.softmax_input = T.concatenate ([self.hiddenLayers1[-1].output, self.hiddenLayers2[-1].output], axis=1)    

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionClass(
            input=self.softmax_input,
            gtruth=gtruth,
            n_in=2*dim_hid,
            n_out=n_out
        )

        self.skipgramLayer = SkipgramLayer (
            rng=randGen,
            embedding=self.embedding, 
            idx_context=idx_context, 
            gamma=gamma, 
            dim_emb=dim_hid, 
            dict_size=dict_size
        )
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers1])
            + sum([abs(x.W).sum() for x in self.hiddenLayers2])
            + sum([abs(x.W).sum() for x in self.hiddenLayers3])
            + abs(self.logRegressionLayer.W).sum()
            + abs(self.skipgramLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers1])
            + sum([(x.W ** 2).sum() for x in self.hiddenLayers2])
            + sum([(x.W ** 2).sum() for x in self.hiddenLayers3])
            + (self.logRegressionLayer.W ** 2).sum()
            + (self.skipgramLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.nll_classify =  self.logRegressionLayer.negLogLikelihood        

        self.cost_classify = (self.nll_classify
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        # same holds for the function computing the number of errors
        self.errorRate = self.logRegressionLayer.errorRate
        self.precision = self.logRegressionLayer.precision
        self.recall = self.logRegressionLayer.recall
        
        self.nll_predictContext = self.skipgramLayer.negLogLikelihood        

        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params_classify = (sum([x.params for x in self.hiddenLayers1], [])
                              + sum([x.params for x in self.hiddenLayers2], [])
                              + sum([x.params for x in self.hiddenLayers3], [])
                              + self.logRegressionLayer.params
        )

        self.params_predictContext = (sum([x.params for x in self.hiddenLayers3], [])
                                    + self.skipgramLayer.params
        )
                    
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]

        
        
        
class MyPlanetoidModel1(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, gtruth, idx_input, idx_context, gamma, # symbolic variables
                 dict_size, n_out, dim_feat, dim_hid, n_hid, # hyperparameters
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
        
        # keep track of model input
        self.input = input
        
        # If n_hidden is a list (or tuple), check its length is equal to the
        # number of hidden layers. If n_hidden is a scalar, we set up every
        # hidden layers with same number of units.
        # if hasattr(n_hidden, '__iter__'):
        #     assert(len(n_hidden) == n_hiddenLayers)
        # else:
        #     n_hidden = (n_hidden,)*n_hiddenLayers

        self.hiddenLayers = []
        for i in xrange(n_hid):
            h_input = self.input if i == 0 else self.hiddenLayers[i-1].output
            h_in = dim_feat if i == 0 else dim_hid
            self.hiddenLayers.append(
                HiddenLayer(
                    rng=randGen,
                    input=h_input,
                    n_in=h_in,
                    n_out=dim_hid,
                    activation=T.nnet.relu
            ))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionClass(
            input=self.hiddenLayers[-1].output,
            gtruth=gtruth,
            n_in=dim_hid,
            n_out=n_out
        )

        self.skipgramLayer = SkipgramLayer (
            rng=randGen,
            embedding=self.hiddenLayers[-1].output, 
            idx_context=idx_context, 
            gamma=gamma, 
            dim_emb=dim_hid, 
            dict_size=dict_size
        )
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.logRegressionLayer.W).sum()
            + abs(self.skipgramLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.logRegressionLayer.W ** 2).sum()
            + (self.skipgramLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.nll_classify =  self.logRegressionLayer.negLogLikelihood        

        self.cost_classify = (self.nll_classify
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        # same holds for the function computing the number of errors
        self.errorRate = self.logRegressionLayer.errorRate
        self.precision = self.logRegressionLayer.precision
        self.recall = self.logRegressionLayer.recall
        
        self.nll_predictContext = self.skipgramLayer.negLogLikelihood        

        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params_classify = (sum([x.params for x in self.hiddenLayers], [])
                              + self.logRegressionLayer.params
        )

        self.params_predictContext = (sum([x.params for x in self.hiddenLayers], [])
                                    + self.skipgramLayer.params
        )
                    
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]
        
        
        
        
class SkipgramLayer(object):
    """SkipgramLayer Class

    The ...
    """

    def __init__(self, rng, embedding, idx_context, gamma, dim_emb, dict_size):
        """ Initialize the parameters of the logistic regression

        :type n_in: int
        :param n_in:

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (dict_size + dim_emb)),
                high=numpy.sqrt(6. / (dict_size + dim_emb)),
                size=(dict_size, dim_emb)
            ),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(
            W_values*0.,
#            value=numpy.zeros(
#                (dict_size, dim_emb),
#                dtype=theano.config.floatX
#            ),
            name='SoftmaxW',
            borrow=True
        )
        
        self.params = [self.W]
               
        self.prediction = T.nnet.sigmoid(
                gamma * T.diag(T.dot(embedding, self.W[idx_context].T))
        )
        
        self.regul = T.mean (T.diag(T.dot(embedding, embedding.T) + T.dot(self.W[idx_context], self.W[idx_context].T)))
    
        self.negLogLikelihood = -T.mean(T.log(self.prediction))
        