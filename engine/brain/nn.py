import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
downsample = pool


class BinaryLogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, gtruth, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, 1),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (1,),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        self.linearOutput = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.sigmoid (self.linearOutput.flatten())

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = self.p_y_given_x > 0.5

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        self.negLogLikelihood = -T.mean(gtruth * T.log(self.p_y_given_x) + (1-gtruth) * T.log (1-self.p_y_given_x))
        self.debug1 = self.p_y_given_x
        self.debug2 = gtruth

        self.accuracy = T.mean(T.eq(self.y_pred, gtruth))
        self.errorRate = T.mean(T.neq(self.y_pred, gtruth))

        truepos = T.sum(T.and_(T.eq(self.y_pred,1),T.eq(gtruth,1)))
        truepos = T.cast (truepos, 'float32')
        self.precision = truepos/T.sum(T.eq(self.y_pred,1))
        self.recall = truepos/T.sum(T.eq(gtruth,1))

        self.ranking = (-self.p_y_given_x).argsort()
        self.ranking_gt = gtruth[self.ranking]
       
        d_recal = T.eq (self.ranking_gt, 1).nonzero()[0]

        res,upd = theano.scan (fn = lambda i, j, sp: sp + i.__truediv__(j),
                               outputs_info = T.cast(0.,'float64'),
                               sequences = [T.arange(d_recal.shape[0])+1, d_recal+1]
                              )
        self.AP = res [-1]/d_recal.shape[0]
        self.AP_updates = upd


        

class MultiClassLogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, gtruth, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        self.linearOutput = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(self.linearOutput)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        self.negLogLikelihood = -T.mean(T.log(self.p_y_given_x)[T.arange(gtruth.shape[0]), gtruth])
        self.debug1 = self.p_y_given_x
        self.debug2 = gtruth
        self.debug3 = T.log(self.p_y_given_x)[T.arange(gtruth.shape[0]), gtruth]

        self.accuracy = T.mean(T.eq(self.y_pred, gtruth))
        self.errorRate = T.mean(T.neq(self.y_pred, gtruth))

        # these only work in binary classification mode (n_classes = 2)
        truepos = T.sum(T.and_(T.eq(self.y_pred,1),T.eq(gtruth,1)))
        truepos = T.cast (truepos, 'float32')
        self.precision = truepos/T.sum(T.eq(self.y_pred,1))
        self.recall = truepos/T.sum(T.eq(gtruth,1))
        self.ranking = (-self.p_y_given_x[:,1]).argsort()
        self.ranking_gt = gtruth[self.ranking]

        d_recal = T.eq (self.ranking_gt, 1).nonzero()[0]

        res,upd = theano.scan (fn = lambda i, j, sp: sp + i.__truediv__(j),
                               outputs_info = T.cast(0.,'float64'),
                               sequences = [T.arange(d_recal.shape[0])+1, d_recal+1]
                              )
        self.AP = res [-1]/d_recal.shape[0]
        self.AP_updates = upd


class MultiLabelLogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, gtruth, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        self.linearOutput = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.sigmoid (self.linearOutput)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = self.p_y_given_x > 0.5

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        self.negLogLikelihood = -T.mean(gtruth * T.log(self.p_y_given_x) + (1-gtruth) * T.log (1-self.p_y_given_x))
        self.debug1 = self.p_y_given_x
        self.debug2 = gtruth

        self.accuracy = T.mean(T.eq(self.y_pred, gtruth))
        self.errorRate = T.mean(T.neq(self.y_pred, gtruth))

        truepos = T.sum(T.and_(T.eq(self.y_pred,1),T.eq(gtruth,1)))
        truepos = T.cast (truepos, 'float32')
        self.precision = truepos/T.sum(T.eq(self.y_pred,1))
        self.recall = truepos/T.sum(T.eq(gtruth,1))

        self.ranking = (-self.p_y_given_x).argsort(axis=0)
        j1,j2 = T.mgrid [0:self.ranking.shape[0], 0:self.ranking.shape[1]]

        self.ranking_gt = gtruth[self.ranking,j2]
       
        res,upd = theano.scan (fn = lambda i, g, prec, num: (prec + g.flatten() * ((num + g.flatten()).__truediv__(i)), num + g.flatten()),
                               outputs_info = [T.zeros ((n_out,), dtype='float64'), T.zeros ((n_out,), dtype='int32')],
                               sequences = [T.arange(input.shape[0])+1, self.ranking_gt]
                              )
        self.AP = res [0][-1] / T.sum (gtruth, axis=0)
        self.AP_updates = upd
        
        
        
class FullLogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, gtruth, n_in, n_label, n_class):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_label, n_class),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_label, n_class),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        self.linearOutput = T.tensordot(input, self.W, ([1],[0])) + self.b.dimshuffle('x',0,1)
        self.exp = T.exp(self.linearOutput)
        self.p_y_given_x = self.exp / T.sum(self.exp, axis=2).dimshuffle(0,1,'x')

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=2)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        i1,i2 = T.mgrid [0:n_in, 0:n_label]
        j1 = i1.flatten()
        j2 = i2.flatten()
        self.negLogLikelihood = -T.mean(T.log(self.p_y_given_x)[j1, j2, gtruth[j1,j2]])
        self.debug1 = self.p_y_given_x
        self.debug2 = gtruth

        self.accuracy = T.mean(T.eq(self.y_pred, gtruth))
        self.errorRate = T.mean(T.neq(self.y_pred, gtruth))

        # these only work in binary classification mode (n_classes = 2)
        truepos = T.sum(T.and_(T.eq(self.y_pred,1),T.eq(gtruth,1)))
        truepos = T.cast (truepos, 'float32')
        self.precision = truepos/T.sum(T.eq(self.y_pred,1))
        self.recall = truepos/T.sum(T.eq(gtruth,1))

        self.ranking = (-self.p_y_given_x[:,:,1]).argsort(axis=0)
        self.ranking_gt = gtruth[self.ranking]

        res,upd = theano.scan (fn = lambda i, g, prec, num: (prec + g.flatten() * ((num + g.flatten()).__truediv__(i)), num + g.flatten()),
                               outputs_info = [T.zeros ((n_out,), dtype='float64'), T.zeros ((n_out,), dtype='int32')],
                               sequences = [T.arange(input.shape[0])+1, self.ranking_gt]
                              )
        self.AP = res [0][-1] / T.sum (gtruth, axis=0)
        self.AP_updates = upd
        
        

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.activation = activation

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            if activation == T.nnet.sigmoid:
                mult = 1.
            elif activation == T.tanh:
                mult = 1.
            elif activation == T.nnet.relu:
                mult = 2.
                
            mult *= 0.05**2. #This line is only in case we are changing relu to tanh
                
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(mult * 6. / (n_in + n_out)),
                    high=numpy.sqrt(mult * 6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
#            print "creating a weight matrix with shape %s and activation %s" %(W_values.shape, activation)
#            print "The weight matrix has norm %f"%numpy.linalg.norm(W_values,'fro')
            W = theano.shared(value=W_values, borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b

        self.lin_output = T.dot(input, self.W) + self.b
        self.output = (
            self.lin_output if activation is None
            else activation(self.lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def replicate (self, input):
        return HiddenLayer (input=input, 
                            W=self.W, 
                            b=self.b,
                            activation=self.activation,
                            rng=None,
                            n_in=None, 
                            n_out=None
                           )
