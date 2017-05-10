import timeit
import numpy
import os
from datetime import datetime
import json
#import pdb

from brain.data import *
from brain.split import *
from brain.gtruth import *
from brain.graph import *

from brain.wrapper import *

from brain.models_fc_new import *
from brain.models_sup import *
#from brain.models_c3d import *
#from brain.models_fc import *
#from brain.models_unsup import *

class Engine:
    
    def __init__(self, engine_path):        
        os.chdir (engine_path)
        
        self.load_config ()
        self.init_log ()
        self.load_data ()
        self.load_graph ()
        self.query = None    
        self.pids = []

        
    def processQuery(self, idxList, newQuery=True):
        self.log ('processQuery() was called')
        
        if newQuery:
            self.init_query (idxList)
            self.create_model ()
            self.load_model ()
            self.train ()
            output = self.retrieve ()
            return output
            
        else:
            raise NotImplementedError

            

    def load_config(self):
        with open ('config.json', 'r') as fin:
            self.options = json.load (fin)
         
        if self.options.get ('run_gpumem') is not None:
            os.environ['THEANO_FLAGS'] = 'lib.cnmem={}'.format(self.options['run_gpumem'])
        
            
            
    def init_log(self):
        self.logfile = open ('./log/exp_%d.txt' % self.options['exp_number'], 'a', 1)        
        self.logfile.write ("\n\n\n\n==================== %s ====================\n\n" % datetime.now())

        
        
    def load_data(self):
        self.log("loading data ...")
        start_time = timeit.default_timer()

        if self.options ['data_type'] == 'InGPU':
            self.train_data = InGPUDataSet (self.options ['data_shape'])
            self.train_data.load_mat (self.options ['data_file'], 'emb')
            if self.options.get('data_file_test') is None:
                self.test_data = None
            else:
                self.test_data = InGPUDataSet (self.options ['data_shape'])
                self.test_data.load_mat (self.options ['data_file_test'], 'emb')

        elif self.options ['data_type'] == 'InMemory':
            self.train_data = InMemoryDataSet (self.options ['data_shape'])
            self.train_data.load_mat (self.options ['data_file'], 'emb')
            if self.options.get('data_file_test') is None:
                self.test_data = None
            else:
                self.test_data = InMemoryDataSet (self.options ['data_shape'])
                self.test_data.load_mat (self.options ['data_file_test'], 'emb')

        elif self.options ['data_type'] == 'PickleFiles':
            self.train_data = PickleFilesDataSet (self.options ['data_shape'])
            self.train_data.load_pickle_list (self.options ['pickle_list_file'], self.options ['data_path'])
            if self.options.get('data_path_test') is None:
                self.test_data = None
            else:
                self.test_data = PickleFilesDataSet (self.options ['data_shape'])
                self.test_data.load_pickle_list (self.options ['pickle_list_file_test'], self.options ['data_path_test'])

        elif self.options ['data_type'] == 'LMDB':
            self.train_data = LMDBDataSet (self.options ['data_shape'])
            self.train_data.load_lmdb (self.options ['data_path'])
            if self.options.get('data_path_test') is None:
                self.test_data = None
            else:
                self.test_data = LMDBDataSet (self.options ['data_shape'])
                self.test_data.load_lmdb (self.options ['data_path_test'])


        self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))

        gtruth = GroundTruth (self.options.get('gtruth_type'))
        gtruth.load_mat (self.options['gtruth_file'], 'label')
        self.train_data.gtruth = gtruth

        if self.options.get('gtruth_file_test') is not None:
            gtruth_test = GroundTruth (options.get('gtruth_type'))
            gtruth_test.load_mat (self.options['gtruth_file_test'], 'label')
            self.test_data.gtruth = gtruth_test

            

    def load_graph(self):
        self.graph = TheanoGraph (self.train_data)
        if self.options['graph_type'] == 'load':
            self.graph.load_list_fromfile (self.options['graph_file'])
        elif self.options['graph_type'] == 'gtruth':
            self.log ("constructing ground truth graph ...")
            self.graph.constructGTruthGraph ()
        elif self.options['graph_type'] == 'KNN':
            if self.options['graph_rebuild']:
                self.log ("constructing graph ...")
                start_time = timeit.default_timer()
                self.graph.constructKNNexp (k=self.options['graph_k'], 
                                       sigma=self.options['graph_sigma'])
                self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))
            else:
                self.log("loading graph ...")
                start_time = timeit.default_timer()
                self.graph.load_list (self.options['graph_k'], self.options['graph_sigma']) # can change it to load_mtx
                self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))
        
        
        
    def log (self, s):
        if not self.options.get('run_silent') == True:
            print s
        self.logfile.write('%s\n'%s)    
    
    def forcelog (self, s):
        print s
        self.logfile.write('%s\n'%s)    
        
        
        
    def init_query(self, idxList):
        self.log("creating the query ...")
        self.query = DataSplit (self.train_data, seed=self.options['query_seed'])

        if self.options.get('query_type') == 'binary':
            self.query.initBinaryQueryFromUser (idxList,
                                                self.options['query_negratio']
                                               )
        else:
            raise NotImplementedError
        
        
        
    def create_model(self):
        self.log ("creating a model ...")
        start_time = timeit.default_timer()

        ModelClass = globals()[self.options['model_type']]
        modelOptions = {}
        for key in self.options:
            if 'model' in key:# and 'type' not in key and '_mode' not in key:
                modelOptions[key[6:]] = self.options[key]

        if self.options['model_mode'] == 'binary':       
            modelOptions ['n_out'] = 1
        elif self.options['model_mode'] == 'multilabel':       
            modelOptions ['n_out'] = self.train_data.gtruth.inmemory.shape[1]
        elif self.options['model_mode'] == 'multiclass':       
            modelOptions ['n_out'] = self.train_data.gtruth.idx2label.shape[0]
        #modelOptions ['dim_feat'] = data['data_dim']

        #WrapperClass = globals()[options['wrapper_type']]
        WrapperClass = DeepGraphWrapper
        self.modelWrapper = WrapperClass (ModelClass, 
                                          modelOptions, 
                                          self.train_data, 
                                          self.test_data, 
                                          self.query, 
                                          self.graph
                                         )

        self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))
        
        
        
    def load_model(self):
        if self.options.get('model_filename') is None:
            self.log ("No model file to load.")
        else:
            self.log ("loading the model ...")
            start_time = timeit.default_timer()

            self.modelWrapper.loadParams (filename = self.options['model_filename'])

            self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))
    
        
        
        
    def train(self):
        self.log ("training the model ...")
        start_time = timeit.default_timer()

        trainFunc = self.modelWrapper.train_fast if self.options['train_fast'] else self.modelWrapper.train    

        trainOptions = {}
        for key in self.options:
            if 'train' in key and 'fast' not in key:
                trainOptions[key[6:]] = self.options[key]
        trainOptions ['pids'] = self.pids
        trainOptions ['logfile'] = self.logfile

        train_result = trainFunc (**trainOptions)

        self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))
        
        
        
    def retrieve(self):
        self.log ("retrieving results ...")
        start_time = timeit.default_timer()

        if self.options['query_type'] == 'binary':
            if self.options ['data_type'] == 'InGPU' and self.options.get('test_batchsize') is None:
                test_result = self.modelWrapper.test_singlebatch (
                    transductive = True,
                    ranking = True
                )
            else:
                test_result = self.modelWrapper.test_multibatch (
                    batch_size = self.options.get('test_batchsize'),
                    transductive = True,
                    ranking=True
                )
        else:
            raise NotImplementedError

        self.log ("finished %d seconds." % int(timeit.default_timer()-start_time))
        
        return test_result ['transductive_ranking']
        
        
        
        
        
        
        
        
        
        