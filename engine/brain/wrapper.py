import timeit
from Queue import Empty
from multiprocessing import Process,Queue
import sys
import numpy
import theano
import theano.tensor as T

import h5py
import hdf5storage

import lasagne

from randwalk import *
from nn import *

NUM_THREADS = 20

class DeepGraphWrapper(object):

    def __init__(self, modelClass, modelOptions, train_data, test_data, split, graph,
                 learning_rate_sup=0.01, learning_rate_usup=0.01
                ):
        
        self.train_data = train_data
        self.test_data = test_data
        self.split = split
        self.graph = graph
        
        numdims = 1+len(train_data.datum_shape)
        tensor = T.TensorType(theano.config.floatX, [False] * numdims)
        
        idx_input = T.ivector('idx_input') 
        idx_context = T.ivector('idx_context') 
        input = tensor('input') 
        context = tensor('context')

        if len(self.train_data.gtruth.inmemory.shape) == 2:
            gtruth = T.imatrix('gtruth') 
        elif len(self.train_data.gtruth.inmemory.shape) == 1:
            gtruth = T.ivector('gtruth') 
        gamma = T.ivector('gamma') 

        self.model = modelClass (input=input,
                                 gtruth=gtruth,
                                 context=context,
                                 gamma=gamma,
                                 **modelOptions
                                )

        self.learning_rate_sup = theano.shared(value=numpy.asarray(learning_rate_sup, 
                                                                        dtype=theano.config.floatX),
                                                    name='learning_rate_sup', borrow=True)        
        self.learning_rate_usup = theano.shared(value=numpy.asarray(learning_rate_usup,
                                                                              dtype=theano.config.floatX),
                                                          name='learning_rate_usup', borrow=True)        
        
        self.train_data.create_batch ('input')

        self.func_embed_batch = theano.function(
            inputs=[],
            outputs=self.model.embedding,
            givens={
                input: self.train_data.get_batch('input'),
            }
        )
        
        if self.test_data is not None:
            self.test_data.create_batch ('input')

            self.func_embed_batch_test = theano.function(
                inputs=[],
                outputs=self.model.embedding,
                givens={
                    input: self.test_data.get_batch('input'),
                }
            )
        
        if self.model.hasSupervised:   
            
            self.train_data.create_batch ('labeled')
            self.train_data.create_batch ('validation')
            
            self.updates_classify = [
                (param, param - self.learning_rate_sup * gparam)
                for param, gparam in zip(self.model.params_classify, self.model.grad_params_classify)
            ]

            self.func_train_classify = theano.function(
                inputs=[idx_input],
                outputs=self.model.cost_classify,
    #            outputs=self.model.cost_classify,
                updates=self.updates_classify,
                givens={
                    input: self.train_data.get_batch('labeled'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }           
            )
            self.func_predict_batch_conf = theano.function(
                inputs=[],
                outputs=self.model.logRegressionLayer.p_y_given_x,
                givens={
                    input: self.train_data.get_batch('input'),
                }
            )
            self.func_predict_batch = theano.function(
                inputs=[],
                outputs=self.model.logRegressionLayer.y_pred,
                givens={
                    input: self.train_data.get_batch('input'),
                }
            )
                        
            self.func_test_precrec_valid = theano.function(
                inputs=[idx_input],
                outputs=[self.model.logRegressionLayer.precision,
                         self.model.logRegressionLayer.recall],
                givens={
                    input: self.train_data.get_batch('validation'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }
            )
            self.func_test_acc_valid = theano.function(
                inputs=[idx_input],
                outputs=self.model.logRegressionLayer.accuracy,
                givens={
                    input: self.train_data.get_batch('validation'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }
            )
            
            self.func_test_AP_valid = theano.function(
                inputs=[idx_input],
                outputs=self.model.logRegressionLayer.AP,
                updates=self.model.logRegressionLayer.AP_updates,
                givens={
                    input: self.train_data.get_batch('validation'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }
            )
            
            self.func_test_precrec = theano.function(
                inputs=[idx_input],
                outputs=[self.model.logRegressionLayer.precision,
                         self.model.logRegressionLayer.recall],
                givens={
                    input: self.train_data.get_batch('input'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }
            )
            self.func_test_acc = theano.function(
                inputs=[idx_input],
                outputs=self.model.logRegressionLayer.accuracy,
                givens={
                    input: self.train_data.get_batch('input'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }
            )
            self.func_test_AP = theano.function(
                inputs=[idx_input],
                outputs=self.model.logRegressionLayer.AP,
                updates=self.model.logRegressionLayer.AP_updates,
                givens={
                    input: self.train_data.get_batch('input'),
                    gtruth: self.train_data.gtruth.ingpu[idx_input]
                }
            )
            
            if self.test_data is not None:
                self.func_test_precrec_test = theano.function(
                    inputs=[idx_input],
                    outputs=[self.model.logRegressionLayer.precision,
                             self.model.logRegressionLayer.recall],
                    givens={
                        input: self.test_data.get_batch('input'),
                        gtruth: self.test_data.gtruth.ingpu[idx_input]
                    }
                )
                self.func_test_acc_test = theano.function(
                    inputs=[idx_input],
                    outputs=self.model.logRegressionLayer.accuracy,
                    givens={
                        input: self.test_data.get_batch('input'),
                        gtruth: self.test_data.gtruth.ingpu[idx_input]
                    }
                )
                self.func_test_AP_test = theano.function(
                    inputs=[idx_input],
                    outputs=self.model.logRegressionLayer.AP,
                    updates=self.model.logRegressionLayer.AP_updates,
                    givens={
                        input: self.test_data.get_batch('input'),
                        gtruth: self.test_data.gtruth.ingpu[idx_input]
                    }
                )
            
                self.func_predict_batch_conf_test = theano.function(
                    inputs=[],
                    outputs=self.model.logRegressionLayer.p_y_given_x,
                    givens={
                        input: self.test_data.get_batch('input'),
                    }
                )
                self.func_predict_batch_test = theano.function(
                    inputs=[],
                    outputs=self.model.logRegressionLayer.y_pred,
                    givens={
                        input: self.test_data.get_batch('input'),
                    }
                )
                

        if self.model.hasUnsupervised:   
            
            self.train_data.create_batch ('context')
            
            self.updates_predictContext = [
                (param, param - self.learning_rate_usup * gparam)
                for param, gparam in zip(self.model.params_predictContext, self.model.grad_params_predictContext)
            ]
        
            self.func_train_predictContext = theano.function(
                inputs=[gamma],
                outputs=self.model.cost_predictContext,
                updates=self.updates_predictContext,
                givens={
                    input: self.train_data.get_batch('input'),
                    context: self.train_data.get_batch('context')
                },
                #on_unused_input='warn' 
            )


        
    def __del__(self):
        print 'deleting the model...'
        try:
            for param in self.model.params_classify:
                param.set_value(numpy.array([0],ndmin=len(param.get_value().shape)))
        except:
            pass
        try:
            for param in self.model.params_predictContext:
                param.set_value(numpy.array([0],ndmin=len(param.get_value().shape)))
        except:
            pass
    
    def set_learning_rate (self, learning_rate_sup=None, learning_rate_usup=None):
        if learning_rate_sup is not None:
            self.learning_rate_sup.set_value (numpy.asarray(learning_rate_sup, 
                                                             dtype=theano.config.floatX))
        if learning_rate_usup is not None:
            self.learning_rate_usup.set_value (numpy.asarray(learning_rate_usup, 
                                                                   dtype=theano.config.floatX))
    
    
    
    
    
    
    
    
    
    
    
    
    
    def train_fast(self, seed=1234, 
                   max_outer_iter=100, max_inner_iter_sup=1, max_inner_iter_usup=1,                    
                   batch_size_sup=128, batch_size_usup=128,
                   r1=0.5, r2=0.0, d=3, alpha=1.0,
                   learning_rate_sup = 0.01, learning_rate_usup = 0.01,
                   verbose=False, verboseRate=1, 
                   pids=None, logfile=True
             ): 
        
        randGen = numpy.random.RandomState(seed)
        
        if not isinstance(max_outer_iter,list):
            max_outer_iter = [max_outer_iter]
        if not isinstance(learning_rate_sup,list):
            learning_rate_sup = [learning_rate_sup]
        if not isinstance(learning_rate_usup,list):
            learning_rate_usup = [learning_rate_usup]
        if not isinstance(max_inner_iter_sup,list):
            max_inner_iter_sup = [max_inner_iter_sup]
        if not isinstance(max_inner_iter_usup,list):
            max_inner_iter_usup = [max_inner_iter_usup]
        
        if self.model.hasSupervised:
            if self.train_data.__class__.__name__ == 'InGPUDataSet' or len(self.split.validation_set) <= batch_size_sup:
                single_batch_valid = True
                self.train_data.allocate_batch ('validation', len(self.split.validation_set))
                self.train_data.update_batch ('validation', list(self.split.validation_set))
            else:
                single_batch_valid = False
                self.train_data.allocate_batch ('validation', batch_size_sup)
                
            if self.train_data.__class__.__name__ == 'InGPUDataSet' or len(self.split.labeled_set) <= batch_size_sup:
                single_batch_labeled = True
                self.train_data.allocate_batch ('labeled', len(self.split.labeled_set))
                self.train_data.update_batch ('labeled', self.split.labeled_set_idx_array)
            else:
                single_batch_labeled = False
                self.train_data.allocate_batch ('labeled', batch_size_sup)
                
        if self.model.hasUnsupervised:        
            self.train_data.allocate_batch ('input', batch_size_sup)
            self.train_data.allocate_batch ('context', batch_size_usup)
        
        if self.model.hasSupervised:        
            max_global_iter_sup = sum([a*b for a,b in zip (max_outer_iter, max_inner_iter_sup)])
        if self.model.hasUnsupervised:        
            max_global_iter_usup = sum([a*b for a,b in zip (max_outer_iter, max_inner_iter_usup)])
        
        if self.model.hasSupervised:        
            cost_classify = [0.]*max_global_iter_sup
            time_classify = [0.]*max_global_iter_sup
            test_result = [0.]*max_global_iter_sup
            
        if self.model.hasUnsupervised:        
            cost_predictContext = [0.]*max_global_iter_usup
            time_predictContext = [0.]*max_global_iter_usup
        
        if self.model.hasSupervised:        
            labeled_set = self.split.labeled_set_idx_array
            num_batch_classify = -(-labeled_set.shape[0] // batch_size_sup)
        
        if self.model.hasUnsupervised:
            randWalk = RandomWalker (self.train_data, self.graph, self.split, seed)
            
            
            
        if self.model.hasUnsupervised:        

            que = Queue ()

            numRandWalksNeeded = max_global_iter_usup * batch_size_usup        
            assert (numRandWalksNeeded % NUM_THREADS == 0)

            def randomWalkThread(thread_id):            
                randWalk = RandomWalker (self.train_data, self.graph, self.split, seed+thread_id*7)
                for ii in range (numRandWalksNeeded // NUM_THREADS):
                    idx_input, idx_context, gamma = randWalk.step_my1 (1, r1, r2, d)
                    que.put ((idx_input, idx_context, gamma))

                    
            threads = []

            for ii in range (NUM_THREADS):
                threads.append (Process (target = randomWalkThread, args=(ii,)))
            for thread in threads:
                thread.daemon = True
                thread.start()
                if pids is not None:
                    pids.append(thread.pid)
        
            
            
            
            
        t1 = 0.
        t2 = 0.
        t3 = 0.
        t4 = 0.
        t5 = 0.
                
        start_time = timeit.default_timer()
        
        for seqnum in range(len(max_outer_iter)):

            if self.model.hasSupervised:        
                if verbose:
                    print ('setting classification learning rate to %.3f' %learning_rate_sup[seqnum])
                self.set_learning_rate (learning_rate_sup=learning_rate_sup[seqnum])
            if self.model.hasUnsupervised:        
                if verbose:
                    print ('setting predict context learning rate to %.3f' %learning_rate_usup[seqnum])
                self.set_learning_rate (learning_rate_usup=learning_rate_usup[seqnum])
            
            for outer_iter in range(max_outer_iter[seqnum]):

                if self.model.hasUnsupervised:        
                    for inner_iter in range(max_inner_iter_usup[seqnum]):
                        ## Context Prediction phase

                        global_iter = ( sum([a*b for a,b in zip (max_outer_iter[:seqnum], max_inner_iter_usup[:seqnum])]) 
                                      + outer_iter * max_inner_iter_usup[seqnum] + inner_iter)

                        # Random walk
                        flag_time = timeit.default_timer()

                        batch_idx_input = numpy.zeros (shape = (batch_size_usup,), dtype='int32')
                        batch_idx_context = numpy.zeros (shape = (batch_size_usup,), dtype='int32')
                        batch_gamma = numpy.zeros (shape = (batch_size_usup,), dtype='int32')

                        for ii in range (batch_size_usup):
                            while True:
                                try:
                                    temp = que.get (block=False)
                                    break
                                except Empty:
                                    pass
                            batch_idx_input [ii], batch_idx_context [ii], batch_gamma [ii] = (temp[0][0], temp[1][0], temp[2][0])
                        
                        t3 += timeit.default_timer() - flag_time

                        # Load data
                        flag_time = timeit.default_timer()
                    
                        self.train_data.update_batch ('input', batch_idx_input)
                        self.train_data.update_batch ('context', batch_idx_context)
                        
                        t4 += timeit.default_timer() - flag_time

                        flag_time = timeit.default_timer()

                        cost_predictContext[global_iter] = self.func_train_predictContext(batch_gamma)

                        time_predictContext[global_iter] = timeit.default_timer() - start_time

                        t5 += timeit.default_timer() - flag_time



                        if logfile is not None and global_iter%verboseRate==(-1)%verboseRate:
                            logfile.write ('graph context iteration %i, cost = %f, average cost = %f\n' % (
                                    global_iter,
                                    cost_predictContext[global_iter],
                                    numpy.mean (cost_predictContext[max (0, global_iter-verboseRate) : global_iter])
                            ))

                        if verbose and global_iter%verboseRate==(-1)%verboseRate:
                            print ('graph context iteration %i, cost = %f, average cost = %f' % (
                                    global_iter+1,
                                    cost_predictContext[global_iter],
                                    numpy.mean (cost_predictContext[max (0, global_iter-verboseRate) : global_iter])
                                ))       
                
                if self.model.hasSupervised:        
                    for inner_iter in range(max_inner_iter_sup[seqnum]):            
                        ## Classification phase

                        global_iter = ( sum([a*b for a,b in zip (max_outer_iter[:seqnum], max_inner_iter_sup[:seqnum])]) 
                                      + outer_iter * max_inner_iter_sup[seqnum] + inner_iter )

                        if single_batch_labeled:
                             idx_input = labeled_set
                        else:
                            # Load batch data
                            flag_time = timeit.default_timer()
                            idx_idx_input = range ((global_iter % num_batch_classify) * batch_size_sup, 
                                                   min ((global_iter % num_batch_classify + 1) * batch_size_sup, 
                                                        labeled_set.shape[0]))

                            idx_input = labeled_set [idx_idx_input]

                            if len (idx_input) != batch_size_sup:
                                self.train_data.allocate_batch ('labeled', len (idx_input))

                            self.train_data.update_batch ('labeled', idx_input)

                            t1 += timeit.default_timer() - flag_time

                        # Backprop
                        flag_time = timeit.default_timer()

                        cost_classify[global_iter] = self.func_train_classify (idx_input)

                        time_classify[global_iter] = timeit.default_timer() - start_time
                        t2 += timeit.default_timer() - flag_time
                        #print predictions, errorRate, ground_truth,  p_y_given_x

                        if logfile is not None and global_iter%verboseRate==(-1)%verboseRate:
                            logfile.write ('classification iteration %i, minibatch %i/%i, cost = %f\n' % (global_iter,
                                                                                        global_iter % num_batch_classify,
                                                                                        num_batch_classify,
                                                                                        cost_classify[global_iter]
                                                                                       ))

                        if verbose and global_iter%verboseRate==(-1)%verboseRate:
                            print ('classification iteration %i, minibatch %i/%i, cost = %f' % (global_iter+1,
                                                                                        global_iter % num_batch_classify,
                                                                                        num_batch_classify,
                                                                                        cost_classify[global_iter]
                                                                                       ))
                            
                            if len (self.split.validation_set) > 0:
                                if single_batch_valid:
                                    if self.model.mode == 'binary' or self.model.mode == 'multilabel':
                                        test_result [global_iter] = self.test_singlebatch(validation=True,
                                                                                          AP=True)
                                    elif self.model.mode == 'multiclass':
                                        test_result [global_iter] = self.test_singlebatch(validation=True,
                                                                                          accuracy=True)
                                else:
                                    if self.model.mode == 'binary' or self.model.mode == 'multilabel':
                                        test_result [global_iter] = self.test_multibatch(
                                            batch_size=batch_size_sup,
                                            validation=True,
                                            AP=True
                                        )
                                    elif self.model.mode == 'multiclass':
                                        test_result [global_iter] = self.test_multibatch(
                                            batch_size=batch_size_sup,
                                            validation=True,
                                            accuracy=True
                                        )
                                
                            

                # log.append ('t1 = %.4f, t2 = %.4f, t3 = %.4f, t4 = %.4f, t5 = %.4f\n'  %(t1,t2,t3,t4,t5))      
            
        results = {}
        if self.model.hasSupervised:        
            results ['cost_classify'] = cost_classify
            results ['time_classify'] = time_classify
            results ['test_result'] = test_result
        if self.model.hasUnsupervised:        
            results ['cost_predictContext'] = cost_predictContext
            results ['time_predictContext'] = time_predictContext
    
        return results
        
        
####### Single Thread Version of train ()

    def train(self, seed=1234, 
                   max_outer_iter=100, max_inner_iter_sup=1, max_inner_iter_usup=1,                    
                   batch_size_sup=128, batch_size_usup=128,
                   r1=0.5, r2=0.0, d=3, alpha=1.0,
                   learning_rate_sup = 0.01, learning_rate_usup = 0.01,
                   verbose=False, verboseRate=1, 
                   pids=None, logfile=True
             ): 
        
        randGen = numpy.random.RandomState(seed)
        
        if not isinstance(max_outer_iter,list):
            max_outer_iter = [max_outer_iter]
        if not isinstance(learning_rate_sup,list):
            learning_rate_sup = [learning_rate_sup]
        if not isinstance(learning_rate_usup,list):
            learning_rate_usup = [learning_rate_usup]
        if not isinstance(max_inner_iter_sup,list):
            max_inner_iter_sup = [max_inner_iter_sup]
        if not isinstance(max_inner_iter_usup,list):
            max_inner_iter_usup = [max_inner_iter_usup]
        
        if self.model.hasSupervised:
            if self.train_data.__class__.__name__ == 'InGPUDataSet' or len(self.split.validation_set) <= batch_size_sup:
                single_batch_valid = True
                self.train_data.allocate_batch ('validation', len(self.split.validation_set))
                self.train_data.update_batch ('validation', list(self.split.validation_set))
            else:
                single_batch_valid = False
                self.train_data.allocate_batch ('validation', batch_size_sup)
                
            if self.train_data.__class__.__name__ == 'InGPUDataSet' or len(self.split.labeled_set) <= batch_size_sup:
                single_batch_labeled = True
                self.train_data.allocate_batch ('labeled', len(self.split.labeled_set))
                self.train_data.update_batch ('labeled', self.split.labeled_set_idx_array)
            else:
                single_batch_labeled = False
                self.train_data.allocate_batch ('labeled', batch_size_sup)
                
                
                
        if self.model.hasUnsupervised:        
            self.train_data.allocate_batch ('input', batch_size_sup)
            self.train_data.allocate_batch ('context', batch_size_usup)
        
        if self.model.hasSupervised:        
            max_global_iter_sup = sum([a*b for a,b in zip (max_outer_iter, max_inner_iter_sup)])
        if self.model.hasUnsupervised:        
            max_global_iter_usup = sum([a*b for a,b in zip (max_outer_iter, max_inner_iter_usup)])
        
        if self.model.hasSupervised:        
            cost_classify = [0.]*max_global_iter_sup
            time_classify = [0.]*max_global_iter_sup
            test_result = [0.]*max_global_iter_sup
            
        if self.model.hasUnsupervised:        
            cost_predictContext = [0.]*max_global_iter_usup
            time_predictContext = [0.]*max_global_iter_usup
        
        if self.model.hasSupervised:        
            labeled_set = self.split.labeled_set_idx_array
            num_batch_classify = -(-labeled_set.shape[0] // batch_size_sup)
        
        if self.model.hasUnsupervised:
            randWalk = RandomWalker (self.train_data, self.graph, self.split, seed)
            
        t1 = 0.
        t2 = 0.
        t3 = 0.
        t4 = 0.
        t5 = 0.
                
        start_time = timeit.default_timer()
        
        for seqnum in range(len(max_outer_iter)):

            if self.model.hasSupervised:        
                if verbose:
                    print ('setting classification learning rate to %.3f' %learning_rate_sup[seqnum])
                self.set_learning_rate (learning_rate_sup=learning_rate_sup[seqnum])
            if self.model.hasUnsupervised:        
                if verbose:
                    print ('setting predict context learning rate to %.3f' %learning_rate_usup[seqnum])
                self.set_learning_rate (learning_rate_usup=learning_rate_usup[seqnum])
            
            for outer_iter in range(max_outer_iter[seqnum]):

                if self.model.hasUnsupervised:        
                    for inner_iter in range(max_inner_iter_usup[seqnum]):
                        ## Context Prediction phase

                        global_iter = ( sum([a*b for a,b in zip (max_outer_iter[:seqnum], max_inner_iter_usup[:seqnum])]) 
                                      + outer_iter * max_inner_iter_usup[seqnum] + inner_iter)

                        # Random walk
                        flag_time = timeit.default_timer()

#                        batch_idx_input, batch_idx_context, batch_gamma = randWalk.step_PARW (batch_size_usup, r1, alpha)
                        batch_idx_input, batch_idx_context, batch_gamma = randWalk.step_my1 (batch_size_usup, r1, r2, d)
                        
                        t3 += timeit.default_timer() - flag_time

                        # Load data
                        flag_time = timeit.default_timer()
                    
                        self.train_data.update_batch ('input', batch_idx_input)
                        self.train_data.update_batch ('context', batch_idx_context)
                        
                        t4 += timeit.default_timer() - flag_time

                        flag_time = timeit.default_timer()

                        cost_predictContext[global_iter] = self.func_train_predictContext(batch_gamma)

                        time_predictContext[global_iter] = timeit.default_timer() - start_time

                        t5 += timeit.default_timer() - flag_time



                        if logfile is not None and global_iter%verboseRate==(-1)%verboseRate:
                            logfile.write ('graph context iteration %i, cost = %f, average cost = %f\n' % (
                                    global_iter,
                                    cost_predictContext[global_iter],
                                    numpy.mean (cost_predictContext[max (0, global_iter-verboseRate) : global_iter])
                            ))

                        if verbose and global_iter%verboseRate==(-1)%verboseRate:
                            print ('graph context iteration %i, cost = %f, average cost = %f' % (
                                    global_iter+1,
                                    cost_predictContext[global_iter],
                                    numpy.mean (cost_predictContext[max (0, global_iter-verboseRate) : global_iter])
                                ))       
                
                if self.model.hasSupervised:        
                    for inner_iter in range(max_inner_iter_sup[seqnum]):            
                        ## Classification phase

                        global_iter = ( sum([a*b for a,b in zip (max_outer_iter[:seqnum], max_inner_iter_sup[:seqnum])]) 
                                      + outer_iter * max_inner_iter_sup[seqnum] + inner_iter )

                        if single_batch_labeled:
                             idx_input = labeled_set
                        else:
                            # Load batch data
                            flag_time = timeit.default_timer()
                            idx_idx_input = range ((global_iter % num_batch_classify) * batch_size_sup, 
                                                   min ((global_iter % num_batch_classify + 1) * batch_size_sup, 
                                                        labeled_set.shape[0]))

                            idx_input = labeled_set [idx_idx_input]

                            if len (idx_input) != batch_size_sup:
                                self.train_data.allocate_batch ('labeled', len (idx_input))

                            self.train_data.update_batch ('labeled', idx_input)

                            t1 += timeit.default_timer() - flag_time

                        # Backprop
                        flag_time = timeit.default_timer()

                        cost_classify[global_iter] = self.func_train_classify (idx_input)

                        
                        time_classify[global_iter] = timeit.default_timer() - start_time
                        t2 += timeit.default_timer() - flag_time
                        #print predictions, errorRate, ground_truth,  p_y_given_x

                        if logfile is not None and global_iter%verboseRate==(-1)%verboseRate:
                            logfile.write ('classification iteration %i, minibatch %i/%i, cost = %f\n' % (global_iter,
                                                                                        global_iter % num_batch_classify,
                                                                                        num_batch_classify,
                                                                                        cost_classify[global_iter]
                                                                                       ))

                        if verbose and global_iter%verboseRate==(-1)%verboseRate:
                            print ('classification iteration %i, minibatch %i/%i, cost = %f' % (global_iter+1,
                                                                                        global_iter % num_batch_classify,
                                                                                        num_batch_classify,
                                                                                        cost_classify[global_iter]
                                                                                       ))
                            
                            if len (self.split.validation_set) > 0:
                                if single_batch_valid:
                                    if self.model.mode == 'binary' or self.model.mode == 'multilabel':
                                        test_result [global_iter] = self.test_singlebatch(validation=True,
                                                                                          AP=True)
                                    elif self.model.mode == 'multiclass':
                                        test_result [global_iter] = self.test_singlebatch(validation=True,
                                                                                          accuracy=True)
                                else:
                                    if self.model.mode == 'binary' or self.model.mode == 'multilabel':
                                        test_result [global_iter] = self.test_multibatch(
                                            batch_size=batch_size_sup,
                                            validation=True,
                                            AP=True
                                        )
                                    elif self.model.mode == 'multiclass':
                                        test_result [global_iter] = self.test_multibatch(
                                            batch_size=batch_size_sup,
                                            validation=True,
                                            accuracy=True
                                        )

                            

                # log.append ('t1 = %.4f, t2 = %.4f, t3 = %.4f, t4 = %.4f, t5 = %.4f\n'  %(t1,t2,t3,t4,t5))      
            
        results = {}
        if self.model.hasSupervised:        
            results ['cost_classify'] = cost_classify
            results ['time_classify'] = time_classify
            results ['test_result'] = test_result
        if self.model.hasUnsupervised:        
            results ['cost_predictContext'] = cost_predictContext
            results ['time_predictContext'] = time_predictContext
    
        return results

        
        
        
        
        
        
        
        
        
        
        
        
        
        
       
        
        
    def saveParams (self, filename):
        self.model.saveParams (filename)

    def loadParams (self, filename):
        self.model.loadParams (filename)

    
    def saveEmb (self, filename, batch_size = 100, additional_info = None):
        self.emb = []
        num_batch = -(-self.train_data.size // batch_size)
        
        self.train_data.allocate_batch ('input', batch_size)
        
        for global_iter in range (num_batch):            
            idx_input = range ((global_iter % num_batch) * batch_size, 
                                   min ((global_iter % num_batch + 1) * batch_size, 
                                        self.train_data.size))
            
            self.train_data.update_batch ('input', idx_input)

            emb_batch = self.func_embed_batch ()
                        
            self.emb.append (emb_batch)
                        
        self.emb = numpy.concatenate(self.emb)
        hdf5storage.savemat(filename, 
                         {'emb': self.emb,
                         })   
        
    def saveEmbTest (self, filename, batch_size = 100, additional_info = None):
        self.emb = []
        num_batch = -(-self.test_data.size // batch_size)
        
        self.test_data.allocate_batch ('input', batch_size)
        
        for global_iter in range (num_batch):
            idx_input = range ((global_iter % num_batch) * batch_size, 
                                   min ((global_iter % num_batch + 1) * batch_size, 
                                        self.test_data.size))
            
            self.test_data.update_batch ('input', idx_input)

            emb_batch = self.func_embed_batch_test ()
                        
            self.emb.append (emb_batch)

        self.emb = numpy.concatenate(self.emb)
        hdf5storage.savemat(filename, 
                         {'emb': self.emb,
                         })   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def test_multibatch (self, batch_size,
              validation=False, inductive=False, transductive=False,
              accuracy=False, precision_recall=False, AP=False, ranking=False,
              rank_and_AP=False,
             ):
        
        result = {}

        if precision_recall:
            if validation:
                num_batch = -(-len(self.split.validation_set) // batch_size)
        
                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = numpy.arange ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.validation_set)))

                    idx_input = numpy.asarray(list(self.split.validation_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    pred_batch = self.func_predict_batch ()

                    pred = pred_batch if global_iter==0 else numpy.concatenate ((pred, pred_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.validation_set)]
                truepos = numpy.sum(numpy.logical_and(numpy.equal(pred,1),numpy.equal(gt,1)))
                prec = truepos/numpy.sum(numpy.equal(pred,1))
                rec = truepos/numpy.sum(numpy.equal(gt,1))

                print ('performance on validation set: precision=%f, recall=%f, F1=%f' 
                       % (prec, rec, 2*prec*rec/(prec+rec)))

                result ['validation_precision'] = prec
                result ['validation_recall'] = rec

            if transductive:
                num_batch = -(-len(self.split.unlabeled_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.unlabeled_set)))

                    idx_input = numpy.asarray(list(self.split.unlabeled_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    pred_batch = self.func_predict_batch ()

                    pred = pred_batch if global_iter==0 else numpy.concatenate ((pred, pred_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.unlabeled_set)]
                truepos = numpy.sum(numpy.logical_and(numpy.equal(pred,1),numpy.equal(gt,1)))
                prec = truepos/numpy.sum(numpy.equal(pred,1))
                rec = truepos/numpy.sum(numpy.equal(gt,1))

                print ('performance in transductive test: precision=%f, recall=%f, F1=%f' 
                       % (prec, rec, 2*prec*rec/(prec+rec)))

                result ['transductive_precision'] = prec
                result ['transductive_recall'] = rec

            if inductive:
                num_batch = -(-self.test_data.size // batch_size)

                self.test_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                self.test_data.size))

                    if len (idx_input) != batch_size:
                        self.test_data.allocate_batch ('input', len (idx_input))
                        
                    self.test_data.update_batch ('input', idx_input)

                    pred_batch = self.func_predict_batch_test ()

                    pred = pred_batch if global_iter==0 else numpy.concatenate ((pred, pred_batch))

                gt = self.test_data.gtruth.inmemory
                truepos = numpy.sum(numpy.logical_and(numpy.equal(pred,1),numpy.equal(gt,1)))
                prec = truepos/numpy.sum(numpy.equal(pred,1))
                rec = truepos/numpy.sum(numpy.equal(gt,1))

                print ('performance in inductive test: precision=%f, recall=%f, F1=%f' 
                       % (prec, rec, 2*prec*rec/(prec+rec)))

                result ['inductive_precision'] = prec
                result ['inductive_recall'] = rec


        if accuracy:
            if validation:
                num_batch = -(-len(self.split.validation_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.validation_set)))

                    idx_input = numpy.asarray(list(self.split.validation_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    pred_batch = self.func_predict_batch ()

                    pred = pred_batch if global_iter==0 else numpy.concatenate ((pred, pred_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.validation_set)]
                acc = numpy.mean(numpy.equal(pred,gt))
                
                print ('performance on validation set: accuracy=%f' % acc)
                result ['validation_accuracy'] = acc

            if transductive:
                num_batch = -(-len(self.split.unlabeled_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.unlabeled_set)))

                    idx_input = numpy.asarray(list(self.split.unlabeled_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    pred_batch = self.func_predict_batch ()

                    pred = pred_batch if global_iter==0 else numpy.concatenate ((pred, pred_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.unlabeled_set)]
                acc = numpy.mean(numpy.equal(pred,gt))
                
                print ('performance in transductive test: accuracy=%f' % acc)
                result ['transductive_accuracy'] = acc

            if inductive:
                num_batch = -(-self.test_data.size // batch_size)

                self.test_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                self.test_data.size))

                    if len (idx_input) != batch_size:
                        self.test_data.allocate_batch ('input', len (idx_input))
                        
                    self.test_data.update_batch ('input', idx_input)

                    pred_batch = self.func_predict_batch_test ()

                    pred = pred_batch if global_iter==0 else numpy.concatenate ((pred, pred_batch))

                gt = self.test_data.gtruth.inmemory
                acc = numpy.mean(numpy.equal(pred,gt))

                print ('performance in inductive test: accuracy=%f' % acc)
                result ['inductive_accuracy'] = acc


        if AP:
            if validation:
                num_batch = -(-len(self.split.validation_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.validation_set)))

                    idx_input = numpy.asarray(list(self.split.validation_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    conf_batch = self.func_predict_batch_conf ()

                    conf = conf_batch if global_iter==0 else numpy.concatenate ((conf, conf_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.validation_set)]

                
                if self.model.mode == 'multilabel':
                    
                    sortidx = numpy.argsort (-conf, axis=0)
                    j1,j2 = numpy.meshgrid(range(sortidx.shape[1]),range(sortidx.shape[0]))                    
                    gt_sorted = gt [sortidx, j1]
                    
                    prec = numpy.zeros ((gt.shape[1]), dtype='float32')
                    num = numpy.zeros ((gt.shape[1]), dtype='int32')
                    for i in range (gt.shape[0]):
                        g = gt_sorted [i]
                        num = num + g
                        prec = prec + g * num / float(i+1) 
                    prec /= numpy.sum (gt, axis=0)
                    mAP = numpy.mean(prec)

                    print ('performance on validation set: Mean Average Precision=%f' % mAP)
                    result ['validation_AP'] = prec
                    result ['validation_mAP'] = mAP

                elif self.model.mode == 'binary':
                    
                    sortidx = numpy.argsort (-conf)
                    gt_sorted = gt [sortidx]

                    d_recall = numpy.nonzero (gt_sorted)[0]
                    ap = 0.
                    for i,j in enumerate (d_recall):
                        ap += float(i+1)/float(j+1)
                    ap /= len (d_recall)

                    print ('performance on validation set: Average Precision=%f' % ap)
                    result ['validation_AP'] = ap

            if transductive:
                num_batch = -(-len(self.split.unlabeled_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.unlabeled_set)))

                    idx_input = numpy.asarray(list(self.split.unlabeled_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    conf_batch = self.func_predict_batch_conf ()

                    conf = conf_batch if global_iter==0 else numpy.concatenate ((conf, conf_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.unlabeled_set)]

                if self.model.mode == 'multilabel':
                    
                    sortidx = numpy.argsort (-conf, axis=0)
                    j1,j2 = numpy.meshgrid(range(sortidx.shape[1]),range(sortidx.shape[0]))                    
                    gt_sorted = gt [sortidx, j1]
                    
                    prec = numpy.zeros ((gt.shape[1]), dtype='float32')
                    num = numpy.zeros ((gt.shape[1]), dtype='int32')
                    for i in range (gt.shape[0]):
                        g = gt_sorted [i]
                        num = num + g
                        prec = prec + g * num / float(i+1) 
                    prec /= numpy.sum (gt, axis=0)
                    mAP = numpy.mean(prec)

                    print ('performance in transductive set: Mean Average Precision=%f' % mAP)
                    result ['transductive_AP'] = prec
                    result ['transductive_mAP'] = mAP

                elif self.model.mode == 'binary':
                    
                    sortidx = numpy.argsort (-conf)
                    gt_sorted = gt [sortidx]

                    d_recall = numpy.nonzero (gt_sorted)[0]
                    ap = 0.
                    for i,j in enumerate (d_recall):
                        ap += float(i+1)/float(j+1)
                    ap /= len (d_recall)

                    print ('performance in transductive set: Average Precision=%f' % ap)
                    result ['transductive_AP'] = ap


            if inductive:
                num_batch = -(-self.test_data.size // batch_size)

                self.test_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                self.test_data.size))

                    if len (idx_input) != batch_size:
                        self.test_data.allocate_batch ('input', len (idx_input))
                        
                    self.test_data.update_batch ('input', idx_input)

                    conf_batch = self.func_predict_batch_conf_test ()

                    conf = conf_batch if global_iter==0 else numpy.concatenate ((conf, conf_batch))

                gt = self.test_data.gtruth.inmemory

                if self.model.mode == 'multilabel':
                    
                    sortidx = numpy.argsort (-conf, axis=0)
                    j1,j2 = numpy.meshgrid(range(sortidx.shape[1]),range(sortidx.shape[0]))                    
                    gt_sorted = gt [sortidx, j1]
                    
                    prec = numpy.zeros ((gt.shape[1]), dtype='float32')
                    num = numpy.zeros ((gt.shape[1]), dtype='int32')
                    for i in range (gt.shape[0]):
                        g = gt_sorted [i]
                        num = num + g
                        prec = prec + g * num / float(i+1) 
                    prec /= numpy.sum (gt, axis=0)
                    mAP = numpy.mean(prec)

                    print ('performance in inductive set: Mean Average Precision=%f' % mAP)
                    result ['inductive_AP'] = prec
                    result ['inductive_mAP'] = mAP

                elif self.model.mode == 'binary':
                    
                    sortidx = numpy.argsort (-conf)
                    gt_sorted = gt [sortidx]

                    d_recall = numpy.nonzero (gt_sorted)[0]
                    ap = 0.
                    for i,j in enumerate (d_recall):
                        ap += float(i+1)/float(j+1)
                    ap /= len (d_recall)

                    print ('performance in inductive set: Average Precision=%f' % ap)
                    result ['inductive_AP'] = ap

                    
        if ranking:
            if validation:
                num_batch = -(-len(self.split.validation_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.validation_set)))

                    idx_input = numpy.asarray(list(self.split.validation_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    conf_batch = self.func_predict_batch_conf ()

                    conf = conf_batch if global_iter==0 else numpy.concatenate ((conf, conf_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.validation_set)]
                
                if self.model.mode == 'binary':                    
                    sortidx = numpy.argsort (-conf)
                    result ['validation_ranking'] = sortidx                   
                    
                    if rank_and_AP:
                        gt_sorted = gt [sortidx]

                        d_recall = numpy.nonzero (gt_sorted)[0]
                        ap = 0.
                        for i,j in enumerate (d_recall):
                            ap += float(i+1)/float(j+1)
                        ap /= len (d_recall)
                        result ['validation_AP'] = ap
                        
                else:
                    raise NotImplementedError
                    

            if transductive:
                num_batch = -(-len(self.split.unlabeled_set) // batch_size)

                self.train_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                len(self.split.unlabeled_set)))

                    idx_input = numpy.asarray(list(self.split.unlabeled_set), dtype='int32')[idx_idx_input]

                    if len (idx_input) != batch_size:
                        self.train_data.allocate_batch ('input', len (idx_input))
                        
                    self.train_data.update_batch ('input', idx_input)

                    conf_batch = self.func_predict_batch_conf ()

                    conf = conf_batch if global_iter==0 else numpy.concatenate ((conf, conf_batch))

                gt = self.train_data.gtruth.inmemory[list(self.split.unlabeled_set)]

                if self.model.mode == 'binary':                    
                    sortidx = numpy.argsort (-conf)
                    result ['transductive_ranking'] = sortidx                   
                    
                    if rank_and_AP:
                        gt_sorted = gt [sortidx]

                        d_recall = numpy.nonzero (gt_sorted)[0]
                        ap = 0.
                        for i,j in enumerate (d_recall):
                            ap += float(i+1)/float(j+1)
                        ap /= len (d_recall)
                        result ['transductive_AP'] = ap
                        
                else:
                    raise NotImplementedError


            if inductive:
                num_batch = -(-self.test_data.size // batch_size)

                self.test_data.allocate_batch ('input', batch_size)

                for global_iter in range (num_batch):
                    idx_input = range ((global_iter % num_batch) * batch_size, 
                                           min ((global_iter % num_batch + 1) * batch_size, 
                                                self.test_data.size))

                    if len (idx_input) != batch_size:
                        self.test_data.allocate_batch ('input', len (idx_input))
                        
                    self.test_data.update_batch ('input', idx_input)

                    conf_batch = self.func_predict_batch_conf_test ()

                    conf = conf_batch if global_iter==0 else numpy.concatenate ((conf, conf_batch))

                gt = self.test_data.gtruth.inmemory

                if self.model.mode == 'binary':                    
                    sortidx = numpy.argsort (-conf)
                    result ['inductive_ranking'] = sortidx                   
                    
                    if rank_and_AP:
                        gt_sorted = gt [sortidx]

                        d_recall = numpy.nonzero (gt_sorted)[0]
                        ap = 0.
                        for i,j in enumerate (d_recall):
                            ap += float(i+1)/float(j+1)
                        ap /= len (d_recall)
                        result ['inductive_AP'] = ap
                        
                else:
                    raise NotImplementedError
                    

        return result
                
                
                
    def test_singlebatch (self, 
              validation=False, inductive=False, transductive=False,
              accuracy=False, precision_recall=False, AP=False, ranking=False,
             ):
        
        result = {}

        if precision_recall:
            if validation:
                [prec, rec] = self.func_test_precrec_valid (list(self.split.validation_set))

                print ('performance on validation set: precision=%f, recall=%f, F1=%f' 
                       % (prec, rec, 2*prec*rec/(prec+rec)))

                result ['validation_precision'] = prec
                result ['validation_recall'] = rec

            if transductive:
                self.train_data.allocate_batch ('input', len(self.split.unlabeled_set))
                self.train_data.update_batch ('input', list(self.split.unlabeled_set))
                
                [prec, rec] = self.func_test_precrec (list(self.split.unlabeled_set))

                print ('performance on transductive test: precision=%f, recall=%f, F1=%f' 
                       % (prec, rec, 2*prec*rec/(prec+rec)))

                result ['transductive_precision'] = prec
                result ['transductive_recall'] = rec
                
            if inductive:
                self.test_data.allocate_batch ('input', self.test_data.size)
                self.test_data.update_batch ('input', range(self.test_data.size))
                
                [prec, rec] = self.func_test_precrec_test (range(self.test_data.size))

                print ('performance on inductive test: precision=%f, recall=%f, F1=%f' 
                       % (prec, rec, 2*prec*rec/(prec+rec)))

                result ['inductive_precision'] = prec
                result ['inductive_recall'] = rec
                

        if accuracy:
            if validation:
                acc = self.func_test_acc_valid (list(self.split.validation_set))
                print ('performance on validation set: accuracy=%f' % acc)
                result ['validation_accuracy'] = acc

            if transductive:
                self.train_data.allocate_batch ('input', len(self.split.unlabeled_set))
                self.train_data.update_batch ('input', list(self.split.unlabeled_set))
                
                acc = self.func_test_acc (list(self.split.unlabeled_set))
                print ('performance on transductive test: accuracy=%f' % acc)
                result ['transductive_accuracy'] = acc
                
            if inductive:
                self.test_data.allocate_batch ('input', self.test_data.size)
                self.test_data.update_batch ('input', range(self.test_data.size))

                acc = self.func_test_acc_test (range(self.test_data.size))
                print ('performance on inductive test: accuracy=%f' % acc)
                result ['inductive_accuracy'] = acc

        if AP:
            if validation:
                
                ap = self.func_test_AP_valid (list(self.split.validation_set))
                print ('performance on validation set: Mean Average Precision=%f' % ap.mean())
                result ['validation_AP'] = ap
                result ['validation_mAP'] = ap.mean()

            if transductive:
                self.train_data.allocate_batch ('input', len(self.split.unlabeled_set))
                self.train_data.update_batch ('input', list(self.split.unlabeled_set))
                
                ap = self.func_test_AP (list(self.split.unlabeled_set))
                print ('performance on transductive test: Mean Average Precision=%f' % ap.mean())
                result ['transductive_AP'] = ap
                result ['transductive_mAP'] = ap.mean()
                
            if inductive:
                self.test_data.allocate_batch ('input', self.test_data.size)
                self.test_data.update_batch ('input', range(self.test_data.size))

                ap = self.func_test_AP_test (range(self.test_data.size))
                print ('performance on inductive test: Mean Average Precision=%f' % ap.mean())
                result ['inductive_AP'] = ap
                result ['inductive_mAP'] = ap.mean()
                
        if ranking:
            
            raise NotImplementedError
            
            if validation:                
                sortidx = self.func_test_rank_valid (list(self.split.validation_set))
                result ['validation_ranking'] = sortidx                   

            if transductive:
                self.train_data.allocate_batch ('input', len(self.split.unlabeled_set))
                self.train_data.update_batch ('input', list(self.split.unlabeled_set))
                
                sortidx = self.func_test_rank (list(self.split.unlabeled_set))
                result ['transductive_ranking'] = sortidx                   
                
            if inductive:
                self.test_data.allocate_batch ('input', self.test_data.size)
                self.test_data.update_batch ('input', range(self.test_data.size))

                sortidx = self.func_test_rank_test (range(self.test_data.size))
                result ['inductive_ranking'] = sortidx                   
                
        return result
            
            
