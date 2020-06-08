import numpy as np
import random
from sklearn.neighbors import KDTree

import heapq

class LSH:
  def __init__(self,n_dims,k,L,query_mode,R=10,n_candidates=2,seed=10):
    self.n_dims = n_dims
    self.k = k
    self.L = L
    self.hashfn = []
    self.R = R
    self.n_candidates = n_candidates
    self.query_mode = query_mode
    self.hash_calls = 0

    np.random.seed(seed)
    for i in range(self.L):
      self.hashfn.append(self.sample_hashfn())

  # data: [N, D]
  def setup(self,data):
    assert(data.shape[1] == (self.n_dims))
    assert(len(data.shape) == 2)
    self.data = data
    self.N = self.data.shape[0]


    # max_norm = np.max(np.sum(self.data**2,axis=1)**0.5)
    # for i in range(self.N):
    #   self.data[i] = self.data[i] / max_norm


    self.hash_table = [{} for _ in range(self.L)]

    for i in range(self.N):
      data_point = self.preprocess(data[i])

      for j in range(self.L):
        key = self.hashfn[j](data_point)  
      
        if key in self.hash_table[j].keys():
          self.hash_table[j][key].append(i)
        else: self.hash_table[j][key] = [i]

    self.bucket_data_store = {}
    self.bucket_key_store = {}
    if self.query_mode == 'kdtree':
      
      self.kd_table = []
      for j in range(self.L):
        kd_dict = {}
        for key in self.hash_table[j].keys():
          store_key = '{}_{}'.format(j,key)
          self.bucket_data_store[store_key] = []
          self.bucket_key_store[store_key] = []
          for idx in self.hash_table[j][key]:
            self.bucket_data_store[store_key].append(self.data[idx])
            self.bucket_key_store[store_key].append(idx)
          self.bucket_data_store[store_key] = np.array(self.bucket_data_store[store_key])
          kd_dict[key] = KDTree(self.bucket_data_store[store_key],leaf_size=self.n_candidates)
        
        self.kd_table.append(kd_dict)

  def query(self,x,top_k):

    x = self.preprocess(x)
    best_match = [0, np.inf]
    matches = []
    matches_set = set()

    search_counts = []

    # Scan through all L hash-tables
    for j in range(self.L):
      
      search_count = 0
      ########### 
      # Step 1: Select buckets/keys that you want to search in the current hash-table
      keys_to_search = [self.hashfn[j](x)]
      
      ###########

      # Search through each of the buckets/keys
      for key in keys_to_search:
        
        # Ignore if bucket is empty
        if not key in self.hash_table[j].keys():
          continue

        ########### 
        # Step 2: Search Process Within A Bucket
        

        # Random candidate picking
        if self.query_mode == 'random':
          qxs = np.random.choice(self.hash_table[j][key],min(len(self.hash_table[j][key]),self.n_candidates),replace=False)
          for qx in qxs:
            distance = self.get_distance(self.data[qx],x)
            search_count += 1

            if not qx in matches_set: 
              matches_set.add(qx)
              heapq.heappush(matches,(-distance,qx))
              if len(matches) > top_k:
                dx = heapq.heappop(matches)[1]
                matches_set.remove(dx)


            if distance < best_match[1]:
              best_match = [qx,distance]
        
        # Linear Scan
        elif self.query_mode == 'linear':
          for qx in self.hash_table[j][key]:
            distance = self.get_distance(self.data[qx],x)
            search_count += 1

            if not qx in matches_set: 
              matches_set.add(qx)
              heapq.heappush(matches,(-distance,qx))
              if len(matches) > top_k:
                dx = heapq.heappop(matches)[1]
                matches_set.remove(dx)

            
            if distance < best_match[1]:
              best_match = [qx,distance]

        # KD-Tree Query
        elif self.query_mode == 'kdtree':
          self.kd_table[j][key].reset_n_calls()
          distances, qxs = self.kd_table[j][key].query(x.reshape((1,)+x.shape),min(len(self.hash_table[j][key]),self.n_candidates))
          search_count += self.kd_table[j][key].get_n_calls()
          
          for distance,qx in zip(distances[0],qxs[0]):
            try:
              qx = self.bucket_key_store['{}_{}'.format(j,key)][qx]
            except:
              print('error')
              continue
            if not qx in matches_set: 
              matches_set.add(qx)
              heapq.heappush(matches,(-distance,qx))
              if len(matches) > top_k:
                dx = heapq.heappop(matches)[1]
                matches_set.remove(dx)
            if distance < best_match[1]:
              best_match = [qx,distance]

      search_counts.append(search_count)  
        ########### 
    matches.sort()
    matches = [i for d,i in matches]

    return self.data[best_match[0]],matches,search_counts

  def sample_hashfn(self):
    w_list = [np.random.randn(self.n_dims) for _ in range(self.k)]
    b_list = [np.random.random() for _ in range(self.k)]

    def f(x):
      self.hash_calls += 1
      buckets = [ int(np.floor((np.dot(w,x) + b)/self.R)) for w,b in zip(w_list,b_list) ]
      return self.get_key_from_buckets(buckets)
    return f

  def reset_hash_calls(self):
    self.hash_calls = 0
    
  def get_key_from_buckets(self,buckets):
    key = ''
    for bucket in buckets:
      key = key + '#{}'.format(bucket)
    return key

  def get_distance(self,x,y):
    if(len(x) != len(y)):
      print('Length mismatch while computing distance!')
      return None
    distance = 0
    for i in range(len(x)):
      distance = distance + (x[i]-y[i])**2
    return distance

  def preprocess(self,x):
    return x# / np.linalg.norm(x)

  def exact_nn(self,x,top_k):
    best_match = [0, np.inf]
    matches = []
    
    for i in range(self.N):
      distance = self.get_distance(self.data[i],x)
      
      heapq.heappush(matches,(-distance,i))
      if len(matches) > top_k:
        heapq.heappop(matches)
      
      if distance < best_match[1]:
        best_match = [i,distance]
    
    matches.sort()
    matches = [i for d,i in matches]
    return self.data[best_match[0]],matches



