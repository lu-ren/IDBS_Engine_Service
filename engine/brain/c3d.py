# C3D, AlexNet type model with 3D convolutions (for video processing).
# From "Learning Spatiotemporal Features with 3D Convolutional Networks"
#
# Pretrained weights from https://data.vision.ee.ethz.ch/gyglim/C3D/c3d_model.pkl
# and the snipplet mean from
# https://data.vision.ee.ethz.ch/gyglim/C3D/snipplet_mean.npy
#
# License: Not specified
    # Author: Michael Gygli, https://github.com/gyglim
#

import numpy
import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import softmax

import theano
import numpy as np
import skimage.transform
from skimage import color
import pickle


class C3DModel:

    def __init__(self, input_var=None, empty=False, rectified_fc_layers = False):
        '''
        Builds C3D model

        Returns
        -------
        dict
            A dictionary containing the network layers, where the output layer is at key 'prob'
        '''
        self.net = {}
        
        if empty:
            return
        
        self.net['input'] = InputLayer((None, 3, 16, 112, 112), input_var = input_var)

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
        
        self.fc_activation = lasagne.nonlinearities.rectify if rectified_fc_layers else lasagne.nonlinearities.tanh
        
        self.net['fc6-1']  = DenseLayer(self.net['pool5'], num_units=4096,nonlinearity=self.fc_activation
                                   , W=lasagne.init.GlorotUniform(gain=0.05))
        self.net['fc7-1']  = DenseLayer(self.net['fc6-1'], num_units=4096,nonlinearity=self.fc_activation
                                   , W=lasagne.init.GlorotUniform(gain=0.05))
        print "FC6 has norm %f"%numpy.linalg.norm(self.net['fc6-1'].W.get_value(),'fro')
        print "FC7 has norm %f"%numpy.linalg.norm(self.net['fc7-1'].W.get_value(),'fro')
    #    self.net['fc8-1']  = DenseLayer(self.net['fc7-1'], num_units=487, nonlinearity=None)
    #    self.net['prob']  = NonlinearityLayer(self.net['fc8-1'], softmax)


    def replicate_model(self, input_var=None, num_layers_unshared = 0):
        '''
        Builds C3D model

        num_layers_unshared = 0 means all layers are shared
        num_layers_unshared = 1 means fc7 is not shared
        num_layers_unshared = 2 means fc7 and fc6 are not shared
        ... and so on on so forth

        Returns
        -------
        dict
            A dictionary containing the network layers, where the output layer is at key 'prob'
        '''
        
        out = C3DModel(empty=True)
        
        out.net['input'] = InputLayer((None, 3, 16, 112, 112), input_var = input_var)

        # ----------- 1st layer group ---------------
        if num_layers_unshared >= 10:
            out.net['conv1a'] = Conv3DDNNLayer(out.net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
        else:    
            out.net['conv1a'] = Conv3DDNNLayer(out.net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False
                                           , W = self.net['conv1a'].W, b = self.net['conv1a'].b)

        out.net['pool1']  = MaxPool3DDNNLayer(out.net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

        # ------------- 2nd layer group --------------
        if num_layers_unshared >= 9:
            out.net['conv2a'] = Conv3DDNNLayer(out.net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv2a'] = Conv3DDNNLayer(out.net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv2a'].W, b = self.net['conv2a'].b)

        out.net['pool2']  = MaxPool3DDNNLayer(out.net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))


        # ----------------- 3rd layer group --------------
        if num_layers_unshared >= 8:
            out.net['conv3a'] = Conv3DDNNLayer(out.net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv3a'] = Conv3DDNNLayer(out.net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv3a'].W, b = self.net['conv3a'].b)

        if num_layers_unshared >= 7:
            out.net['conv3b'] = Conv3DDNNLayer(out.net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv3b'] = Conv3DDNNLayer(out.net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv3b'].W, b = self.net['conv3b'].b)

        out.net['pool3']  = MaxPool3DDNNLayer(out.net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))


        # ----------------- 4th layer group --------------
        if num_layers_unshared >= 6:
            out.net['conv4a'] = Conv3DDNNLayer(out.net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv4a'] = Conv3DDNNLayer(out.net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv4a'].W, b = self.net['conv4a'].b)

        if num_layers_unshared >= 5:
            out.net['conv4b'] = Conv3DDNNLayer(out.net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv4b'] = Conv3DDNNLayer(out.net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv4b'].W, b = self.net['conv4b'].b)

        out.net['pool4']  = MaxPool3DDNNLayer(out.net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))


        # ----------------- 5th layer group --------------
        if num_layers_unshared >= 4:
            out.net['conv5a'] = Conv3DDNNLayer(out.net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv5a'] = Conv3DDNNLayer(out.net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv5a'].W, b = self.net['conv5a'].b)

        if num_layers_unshared >= 3:
            out.net['conv5b'] = Conv3DDNNLayer(out.net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
        else:    
            out.net['conv5b'] = Conv3DDNNLayer(out.net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify
                                           , W = self.net['conv5b'].W, b = self.net['conv5b'].b)

        # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
        out.net['pad']    = PadLayer(out.net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
        out.net['pool5']  = MaxPool3DDNNLayer(out.net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))


        # ----------------- Fully Connected Layers ------------------    
        if num_layers_unshared >= 2:
            out.net['fc6-1']  = DenseLayer(out.net['pool5'], num_units=4096,nonlinearity=self.fc_activation
                                           , W=lasagne.init.GlorotUniform(gain=0.05))
            print "FC6 has norm %f"%numpy.linalg.norm(out.net['fc6-1'].W.get_value(),'fro')
        else:    
            out.net['fc6-1']  = DenseLayer(out.net['pool5'], num_units=4096,nonlinearity=self.fc_activation
                                           , W = self.net['fc6-1'].W, b = self.net['fc6-1'].b)

        if num_layers_unshared >= 1:    
            out.net['fc7-1']  = DenseLayer(out.net['fc6-1'], num_units=4096,nonlinearity=self.fc_activation
                                      , W=lasagne.init.GlorotUniform(gain=0.05))
            print "FC7 has norm %f"%numpy.linalg.norm(out.net['fc7-1'].W.get_value(),'fro')
        else:
            out.net['fc7-1']  = DenseLayer(out.net['fc6-1'], num_units=4096,nonlinearity=self.fc_activation
                                       , W = self.net['fc7-1'].W, b = self.net['fc7-1'].b)


    #    if num_layers_unshared >= 1:    
    #        out.net['fc8-1']  = DenseLayer(out.net['fc7-1'], num_units=487, nonlinearity=None)
    #    else:
    #        out.net['fc8-1']  = DenseLayer(out.net['fc7-1'], num_units=487, nonlinearity=None
    #                                   , W = self.net['fc8-1'].W, b = self.net['fc8-1'].b)        
    #    out.net['prob']  = NonlinearityLayer(out.net['fc8-1'], softmax)

        return out


    def load_weights(self, model_file, num_layes_not_to_load=0):
        '''
        Sets the parameters of the model using the weights stored in model_file
        Parameters
        ----------
        net: a Lasagne model (dictionary of layers)

        model_file: string
            path to the model that containes the weights

        num_layes_not_to_load: int
            between 0 and 9

        Returns
        -------
        None

        '''
        with open(model_file) as f:
            print('Load pretrained weights from %s...' % model_file)
            model = pickle.load(f)
        print('Set the weights...')

        model = model [:20]
        getfrom = model [:len(model)-2*num_layes_not_to_load]
        nth = nth_last_layer (num_layes_not_to_load)
        if nth is not None:
            setto = self.net[nth]
            lasagne.layers.set_all_param_values(setto, getfrom, trainable=True)

        
    def save_weights(self, model_file):
        
        paramarray = lasagne.layers.get_all_param_values(self.net[nth_last_layer (0)])
        
        with open(model_file, 'w') as f:
            pickle.dump(paramarray, f)

        
        
######## Below, there are several helper functions to transform (lists of) images into the right format  ######

def nth_last_layer (n):
    if n == 0:
        return 'fc7-1'
    elif n == 1:
        return 'fc6-1'
    elif n == 2:
        return 'conv5b'
    elif n == 3:
        return 'conv5a'
    elif n == 4:
        return 'conv4b'
    elif n == 5:
        return 'conv4a'
    elif n == 6:
        return 'conv3b'
    elif n == 7:
        return 'conv3a'
    elif n == 8:
        return 'conv2a'
    elif n == 9:
        return 'conv1a'

def get_snips(images,image_mean=None,start=0, with_mirrored=False):
    '''
    Converts a list of images to a 5d tensor that serves as input to C3D
    Parameters
    ----------
    images: 4d numpy array or list of 3d numpy arrays
        RGB images

    image_mean: 4d numpy array
        snipplet mean (given by C3D)

    start: int
        first frame to use from the list of images

    with_mirrored: bool
        return the snipplet and its mirrored version (horizontal flip)

    Returns
    -------
    caffe format 5D numpy array (serves as input to C3D)

    '''
    assert len(images) >= start+16, "Not enough frames to fill a snipplet of 16 frames"
    
    # Convert images to caffe format and stack them
    caffe_imgs=map(lambda x: rgb2caffe(x).reshape(1,3,128,171),images[start:start+16])
    snip=np.vstack(caffe_imgs).swapaxes(0,1)

    # Remove the mean
    #snip-= image_mean

    # Get the center crop
    snip=snip[:,:,8:120,29:141]
    snip=snip.reshape(1,3,16,112,112)

    if with_mirrored: # Return nromal and flipped version
        return np.vstack((snip,snip[:,:,:,:,::-1]))
    else:
        return snip

def rgb2caffe(im, out_size=(128, 171)):
    '''
    Converts an RGB image to caffe format and downscales it as needed by C3D

    Parameters
    ----------
    im numpy array
        an RGB image
    downscale

    Returns
    -------
    a caffe image (channel,height, width) in BGR format

    '''
    im=np.copy(im)
    if len(im.shape)==2: # Make sure the image has 3 channels
        im = color.gray2rgb(im)

    h, w, _ = im.shape
    im = skimage.transform.resize(im, out_size, preserve_range=True)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    return np.array(im,theano.config.floatX)


def convert_back(raw_im, image_mean=None,idx=0):
    '''
    Converts a Caffe format image back to the standard format, so that it can be plotted.

    Parameters
    ----------
    raw_im numpy array
        a bgr caffe image; format (channel,height, width)
    add_mean boolean
        Add the C3D mean?
    idx integer (default: 0)
        position in the snipplet (used for mean addtion, but differences are very small)

    Returns
    -------
    a RGB image; format (w,h,channel)
    '''

    raw_im=np.copy(raw_im)
    if image_mean is not None:
        raw_im += image_mean[idx,:,8:120,29:141].squeeze()

    # Convert to RGB
    raw_im = raw_im[::-1, :, :]

    # Back in (y,w,channel) order
    im = np.array(np.swapaxes(np.swapaxes(raw_im, 1, 0), 2, 1),np.uint8)
    return im
