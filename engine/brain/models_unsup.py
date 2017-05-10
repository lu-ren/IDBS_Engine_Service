import numpy
import theano
import theano.tensor as T

from nn import *

        
class ShallowGraphModelParametric(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, context, idx_input, idx_context, gamma, # symbolic variables
                 n_classes, dim_feat, dim_emb, # hyperparameters
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

        self.hasSupervised = False
        self.hasUnsupervised = True
        
        randGen = numpy.random.RandomState(seed)
               
        self.emb1 = HiddenLayer(
                rng=randGen,
                input=input,
                n_in=dim_feat,
                n_out=dim_emb,
                activation=T.tanh
        )
        
        self.embedding=self.emb1.output

        self.emb2 = HiddenLayer(
                rng=randGen,
                input=context,
                n_in=dim_feat,
                n_out=dim_emb,
                activation=T.tanh
        )
                
        self.contextPrediction = T.nnet.sigmoid(
                gamma * T.diag(T.dot(self.emb1.output, self.emb2.output.T))
        )
        
        self.nll_predictContext = -T.mean(T.log(self.contextPrediction))

        self.L1 = abs(self.emb1.W).sum() + abs(self.emb2.W).sum() 
        self.L2_sqr = (self.emb1.W **2).sum() + (self.emb2.W **2).sum() 

        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        self.params_predictContext = self.emb1.params + self.emb2.params
                    
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]

class ShallowGraphModelNonParametric(object):
    """PLANETOID Class

    A PLANETOID is a ...
    """

    def __init__(self, 
                 input, context, idx_input, idx_context, gamma, # symbolic variables
                 n_classes, dim_emb, dict_size, # hyperparameters
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

        self.hasSupervised = False
        self.hasUnsupervised = True
        
        randGen = numpy.random.RandomState(seed)

        emb_values1 = numpy.asarray(
            randGen.uniform(
                low=-numpy.sqrt(1. / (dict_size + dim_emb)),
                high=numpy.sqrt(1. / (dict_size + dim_emb)),
                size=(dict_size, dim_emb)
            ),
            dtype=theano.config.floatX
        )
        
        
        self.embdict1 = theano.shared(
            emb_values1*1.,
#            value=numpy.zeros(
#                (dict_size, dim_emb),
#                dtype=theano.config.floatX
#            ),
            name='embdict1',
            borrow=True
        )
        self.emb1 = self.embdict1 [idx_input]
        
        self.embedding=self.emb1
        
        emb_values2 = numpy.asarray(
            randGen.uniform(
                low=-numpy.sqrt(1. / (dict_size + dim_emb)),
                high=numpy.sqrt(1. / (dict_size + dim_emb)),
                size=(dict_size, dim_emb)
            ),
            dtype=theano.config.floatX
        )
        self.embdict2 = theano.shared(
            emb_values2*1.,
#            value=numpy.zeros(
#                (dict_size, dim_emb),
#                dtype=theano.config.floatX
#            ),
            name='embdict2',
            borrow=True
        )
        self.emb2 = self.embdict2 [idx_context]
                
        self.contextPrediction = T.nnet.sigmoid(
                gamma * T.diag(T.dot(self.emb1, self.emb2.T))
        )
        
        self.nll_predictContext = -T.mean(T.log(self.contextPrediction))

        self.L1 = abs(self.emb1).sum() + abs(self.emb2).sum() 
        self.L2_sqr = (self.emb1 **2).sum() + (self.emb2 **2).sum() 

        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )
        
        self.params_predictContext = [self.embdict1, self.embdict2]
                    
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]

