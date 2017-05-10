import numpy
import theano
import theano.tensor as T
import c3d
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers.shape import PadLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import softmax

import pickle

from nn import *        
    
    
class C3D_GraphModel(object):
    """C3d model
    """

    def __init__(self, 
                 input, gtruth, context, gamma, # symbolic variables
                 n_out, num_layers_scratch, num_layers_finetune, num_layers_unshared, rectified, # hyperparameters
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
        

        num_train = num_layers_scratch + num_layers_finetune

        randGen = numpy.random.RandomState(seed)

        # ------------------------- CLASSIFICATION PART ----------------------------
        # Create c3d model for classification input_var = input
        self.c3d_model = c3d.C3DModel(input, rectified_fc_layers = rectified)

        self.embedding = lasagne.layers.get_output(self.c3d_model.net['fc7-1'])
#        self.embedding = lasagne.layers.get_output(self.c3d_model.net['pool5']).flatten(ndim=2)
        
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionClass(
            input=lasagne.layers.get_output(self.c3d_model.net['fc7-1']),
            gtruth=gtruth,
            n_in=lasagne.layers.get_output_shape(self.c3d_model.net['fc7-1'])[1],
            n_out=n_out
        )        

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        layers_W = []
        
        if num_train >= 1:
            layers_W.append ('fc7-1')
        if num_train >= 2:
            layers_W.append ('fc6-1')
        if num_train >= 3:
            layers_W.append ('conv5b')
        if num_train >= 4:
            layers_W.append ('conv5a')
        if num_train >= 5:
            layers_W.append ('conv4b')
        if num_train >= 6:
            layers_W.append ('conv4a')
        if num_train >= 7:
            layers_W.append ('conv3b')
        if num_train >= 8:
            layers_W.append ('conv3a')
        if num_train >= 9:
            layers_W.append ('conv2a')
        if num_train >= 10:
            layers_W.append ('conv2a')
        
        self.L1 = (
            sum([abs(self.c3d_model.net[key].W).sum() for key in layers_W])
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(self.c3d_model.net[key].W ** 2).sum() for key in layers_W])
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of classification is given by the negative
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
        
        
        
        # ------------------------- CONTEXT PREDICTION PART ----------------------------
        # Create c3d model for predicting graph context input_var = context
        self.c3d_model_ctxt = self.c3d_model.replicate_model(context, num_layers_unshared = num_layers_unshared)         
    
        self.c3d_model.load_weights('data/c3d_model.pkl', num_layes_not_to_load=num_layers_scratch)
        self.c3d_model_ctxt.load_weights('data/c3d_model.pkl', num_layes_not_to_load=num_layers_scratch)
        
        
        self.contextPrediction = T.nnet.sigmoid(
                gamma * T.diag(T.dot(lasagne.layers.get_output(self.c3d_model.net['fc7-1']), 
                                     lasagne.layers.get_output(self.c3d_model_ctxt.net['fc7-1']).T))
        )
        
        self.nll_predictContext = -T.mean(T.log(self.contextPrediction))

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        
        self.L1_ctxt = (
            sum([abs(self.c3d_model_ctxt.net[key].W).sum() for key in layers_W])
            + sum([abs(self.c3d_model.net[key].W).sum() for key in layers_W])
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr_ctxt = (
            sum([(self.c3d_model_ctxt.net[key].W ** 2).sum() for key in layers_W])
            + sum([(self.c3d_model.net[key].W ** 2).sum() for key in layers_W])
        )
        
        self.cost_predictContext = (self.nll_predictContext
            + L1_reg * self.L1_ctxt
            + L2_reg * self.L2_sqr_ctxt
        )
        
        
        self.params_classify = self.logRegressionLayer.params
        self.params_predictContext = []
        
        if num_train >= 1:
            self.params_classify += self.c3d_model.net['fc7-1'].get_params()
            self.params_predictContext += self.c3d_model.net['fc7-1'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['fc7-1'].get_params()
        if num_train >= 2:
            self.params_classify += self.c3d_model.net['fc6-1'].get_params()
            self.params_predictContext += self.c3d_model.net['fc6-1'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['fc6-1'].get_params()
        if num_train >= 3:
            self.params_classify += self.c3d_model.net['conv5b'].get_params()
            self.params_predictContext += self.c3d_model.net['conv5b'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv5b'].get_params()
        if num_train >= 4:
            self.params_classify += self.c3d_model.net['conv5a'].get_params()
            self.params_predictContext += self.c3d_model.net['conv5a'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv5a'].get_params()
        if num_train >= 5:
            self.params_classify += self.c3d_model.net['conv4b'].get_params()
            self.params_predictContext += self.c3d_model.net['conv4b'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv4b'].get_params()
        if num_train >= 6:
            self.params_classify += self.c3d_model.net['conv4a'].get_params()
            self.params_predictContext += self.c3d_model.net['conv4a'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv4a'].get_params()
        if num_train >= 7:
            self.params_classify += self.c3d_model.net['conv3b'].get_params()
            self.params_predictContext += self.c3d_model.net['conv3b'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv3b'].get_params()
        if num_train >= 8:
            self.params_classify += self.c3d_model.net['conv3a'].get_params()
            self.params_predictContext += self.c3d_model.net['conv3a'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv3a'].get_params()
        if num_train >= 9:
            self.params_classify += self.c3d_model.net['conv2a'].get_params()
            self.params_predictContext += self.c3d_model.net['conv2a'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv2a'].get_params()
        if num_train >= 10:
            self.params_classify += self.c3d_model.net['conv1a'].get_params()
            self.params_predictContext += self.c3d_model.net['conv1a'].get_params()
            self.params_predictContext += self.c3d_model_ctxt.net['conv1a'].get_params()

        self.params_classify = list (set (self.params_classify))
        self.params_predictContext = list (set (self.params_predictContext))
            
        self.grad_params_classify = [T.grad(self.cost_classify, param) for param in self.params_classify]
        self.grad_params_predictContext = [T.grad(self.cost_predictContext, param) for param in self.params_predictContext]

        

    def loadParams (self, filename):
        self.c3d_model.load_weights(filename+'_branch_1.p')
        self.c3d_model_ctxt.load_weights(filename+'_branch_2.p')
        # todo: also save the weights of the logregressionlayer
        
    def saveParams (self, filename):
        self.c3d_model.save_weights(filename+'_branch_1.p')
        self.c3d_model_ctxt.save_weights(filename+'_branch_2.p')
        # todo: also save the weights of the logregressionlayer
        
        
class C3D_Original(object):
    """C3d model
    """

    def __init__(self, 
                 input, 
                 emb_layer='fc7-1',
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
        self.hasUnsupervised = False
        
        self.net = {}
        
        self.net['input'] = InputLayer((None, 3, 16, 112, 112), input_var = input)

        # ----------- 1st layer group ---------------
        self.net['conv1a'] = Conv3DDNNLayer(self.net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
        self.net['pool1']  = MaxPool3DDNNLayer(self.net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

        # ------------- 2nd layer group --------------
        self.net['conv2a'] = Conv3DDNNLayer(self.net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['pool2']  = MaxPool3DDNNLayer(self.net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 3rd layer group --------------
        self.net['conv3a'] = Conv3DDNNLayer(self.net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['conv3b'] = Conv3DDNNLayer(self.net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['pool3']  = MaxPool3DDNNLayer(self.net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 4th layer group --------------
        self.net['conv4a'] = Conv3DDNNLayer(self.net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['conv4b'] = Conv3DDNNLayer(self.net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['pool4']  = MaxPool3DDNNLayer(self.net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

        # ----------------- 5th layer group --------------
        self.net['conv5a'] = Conv3DDNNLayer(self.net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['conv5b'] = Conv3DDNNLayer(self.net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
        self.net['pad']    = PadLayer(self.net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
        self.net['pool5']  = MaxPool3DDNNLayer(self.net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
        self.net['fc6-1']  = DenseLayer(self.net['pool5'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['fc7-1']  = DenseLayer(self.net['fc6-1'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
        self.net['fc8-1']  = DenseLayer(self.net['fc7-1'], num_units=487, nonlinearity=None)
        self.net['prob']  = NonlinearityLayer(self.net['fc8-1'], softmax)


        self.embedding = lasagne.layers.get_output(self.net[emb_layer]).flatten(ndim=2)
        
        with open('data/c3d_model.pkl') as f:
            model = pickle.load(f)
        lasagne.layers.set_all_param_values(self.net['prob'], model, trainable=True)
