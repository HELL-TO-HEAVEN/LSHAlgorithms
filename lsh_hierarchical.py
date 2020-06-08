import numpy as np
import random
from scipy import spatial
import heapq

class LSH:
  def __init__(self,n_dims,k,L,R=10,n_candidates=2,seed=10):
    self.n_dims = n_dims
    self.k = k
    self.L = L
    self.hashfn = []
    self.n_candidates = n_candidates
    self.R = R
    self.max_depth = 8
    self.max_bucket_size = 40
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

    # Get a dictionary and a hash function
    indices = list(range(self.N))
    
    self.hash_table = [self.construct_tree(indices,0) for _ in range(self.L)]
    
  def construct_tree(self,indices,depth):

    if depth > self.max_depth or len(indices) <= self.max_bucket_size:
      return indices, None

    hashfn = self.sample_hashfn()
    hashtable = {}
    for i in indices:
      data_point = self.preprocess(self.data[i])
      key = hashfn(data_point)  
      if key in hashtable.keys():
          hashtable[key].append(i)
      else: hashtable[key] = [i]

    for key in hashtable.keys():
      hashtable[key] = self.construct_tree(hashtable[key],depth+1)
    return hashtable, hashfn


  def query_tree(self,x,hashtable,hashfn):
    if hashfn is None:
      assert(type(hashtable) == list)
      return hashtable
    key = hashfn(x)
    if key not in hashtable.keys():
      return []
    return self.query_tree(x,hashtable[key][0],hashtable[key][1])


  def query(self,x,top_k):

    x = self.preprocess(x)
    best_match = [0, np.inf]

    matches = []
    matches_set = set()
    search_counts = []

    # Scan through all L hash-tables
    for j in range(self.L):
      search_count = 0

      q_list = self.query_tree(x,self.hash_table[j][0],self.hash_table[j][1])
      #print(len(q_list))
      for qx in q_list:
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
      search_counts.append(search_count)

    matches.sort()
    matches = [i for d,i in matches]
    
    return self.data[best_match[0]],matches,search_counts

  def sample_hashfn(self):
    w_list = [np.random.random(self.n_dims) for _ in range(self.k)]
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

  def exact_nn(self,x):
    best_match = [0, np.inf]
    for i in range(self.N):
      distance = self.get_distance(self.data[i],x)
      if distance < best_match[1]:
        best_match = [i,distance]
    return self.data[best_match[0]]



