import numpy

class RandomWalker:
    
    def __init__ (self, data, graph, split, seed):
        self.randGen = numpy.random.RandomState(seed)
        self.randSamp = self.randGen.choice
        self.data = data
        self.graph = graph
        self.split = split
        
    def step_PARW (self, batch_size, r1, alpha):
        batch_idx_input = [0.]*batch_size
        batch_idx_context = [0.]*batch_size
        batch_gamma = [0.]*batch_size

        for ii in range (batch_size):
            idx = self.randSamp (self.data.size)                
            if self.randSamp (2, p=[1-r1,r1]) == 1:
                gamma = 1
                idx_ctx = idx
                while True:
                    # using graph in adj list form
                    p = numpy.asarray ( [float(tup[0]) for tup in self.graph.WAlist[idx_ctx]] )
                    NN = numpy.asarray ( [tup[1] for tup in self.graph.WAlist[idx_ctx]] )
                    # These lines are in case graph is only in matrix form
                    # NN = numpy.nonzero(self.graph.Aval[idx_ctx])[0]
                    # p = self.graph.Wval[idx_ctx][NN]

                    NN = numpy.append (NN, [-1])
                    p = numpy.append (p, [alpha])
                    p = p/p.sum()
                    temp = self.randSamp (NN, p=p)
                    if temp == -1 and idx_ctx != idx:
                        break
                    else:
                        idx_ctx = temp
            else:
                idx_ctx = self.randSamp (self.data.size)  
                gamma = -1

            batch_idx_input [ii] = idx    
            batch_idx_context [ii] = idx_ctx   
            batch_gamma [ii] = gamma  

            if False:#self.randSamp (2, p=[0.999,0.001]) == 1:
                print (gamma,
                       int(self.data.gtruth.inmemory[idx]),
                       int(self.data.gtruth.inmemory[idx_ctx]),
                       numpy.sum((self.data.inmemory[idx] - self.data.inmemory[idx_ctx])**2),
                       'This only works for inmemory and ingpu datasets'
                      )

        return batch_idx_input, batch_idx_context, batch_gamma
    
    
    
    def step_planetoid (self, batch_size, r1, r2, d):
        pass
    
    def step_my1 (self, batch_size, r1, r2, d):
        batch_idx_input = [0.]*batch_size
        batch_idx_context = [0.]*batch_size
        batch_gamma = [0.]*batch_size

        for ii in range (batch_size):
            if self.randSamp (2, p=[1-r2,r2]) == 1:
                labeledSet = list (self.split.labeled_set)
                posSet = numpy.where (self.data.gtruth.inmemory[labeledSet] == 1)[0]
                idx_idx = self.randSamp (posSet)
                idx = labeledSet [idx_idx]
            else:
                idx = self.randSamp (self.data.size)                
            if self.randSamp (2, p=[1-r1,r1]) == 1:
                gamma = 1
                idx_ctx = idx
                seqLen = self.randSamp(d)+1
                for step in range(seqLen):
                    # using graph in adj list form
                    p = numpy.asarray ( [float(tup[0]) for tup in self.graph.WAlist[idx_ctx]] )
                    NN = numpy.asarray ( [tup[1] for tup in self.graph.WAlist[idx_ctx]] )
                    # These lines are in case graph is only in matrix form
                    # NN = numpy.nonzero(self.graph.Aval[idx_ctx])[0]
                    # p = self.graph.Wval[idx_ctx][NN]
                    if p.sum() > 0:
                        p = p/p.sum()
                        idx_ctx = self.randSamp (NN, p=p)
            else:
                idx_ctx = self.randSamp (self.data.size)  
                gamma = -1

            batch_idx_input [ii] = idx    
            batch_idx_context [ii] = idx_ctx   
            batch_gamma [ii] = gamma  

            if False:#self.randSamp (2, p=[0.999,0.001]) == 1:
                print (gamma,
                       int(self.data.gtruth.inmemory[idx]),
                       int(self.data.gtruth.inmemory[idx_ctx]),
                       numpy.sum((self.data.inmemory[idx] - self.data.inmemory[idx_ctx])**2),
                       'This only works for inmemory and ingpu datasets'
                      )

        return batch_idx_input, batch_idx_context, batch_gamma
    
    
    def step_lle (self, batch_size, knn):
        batch_idx_input = [0.]*batch_size
        batch_idx_context = [0.]*(batch_size*knn)
        batch_weights = [0.]*(batch_size*knn)

        for ii in range (batch_size):
            idx = self.randSamp (self.data.size)                
            # using graph in adj list form
            p = numpy.asarray ( [float(tup[0]) for tup in self.graph.WAlist[idx]] )
            NN = numpy.asarray ( [tup[1] for tup in self.graph.WAlist[idx]] )
            # These lines are in case graph is only in matrix form
            # NN = numpy.nonzero(self.graph.Aval[idx])[0]
            # p = self.graph.Wval[idx][NN]
            
            p_sorted = -numpy.sort(-p)
            NN_sorted = NN [numpy.argsort(-p)]
            
            p_knn = p_sorted [:knn]
            NN_knn = NN_sorted [:knn]
            
            p_normal = p_knn / p_knn.sum()
            NN_normal = NN_knn / NN_knn.sum()
            
            batch_idx_input [ii] = idx
            for jj in range (knn):
                batch_idx_context [jj*batch_size + ii] = NN_normal [jj]
                batch_weights [jj*batch_size + ii] = p_normal [jj]

        return batch_idx_input, batch_idx_context, batch_weights    