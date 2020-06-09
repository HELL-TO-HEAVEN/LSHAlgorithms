# Locality Sensitive Hashing Algorithms

Implementation of locality sensitive hashing algorithms for approximate nearest neighbour search.

<ul>
  <li><a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Naive LSH</a></li>
  <li><a href="https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf">Multi-Probe LSH</a></li>
  <li><a href="https://www.researchgate.net/profile/Wei_Tsang_Ooi/publication/224755947_Hierarchical_non-uniform_locality_sensitive_hashing_and_its_application_to_video_identification/links/00b7d51c2b9d125b7c000000/Hierarchical-non-uniform-locality-sensitive-hashing-and-its-application-to-video-identification.pdf">Hierarchical LSH</a></li>
</ul>

## Usage

```
import lsh_naive, lsh_hierarchical, lsh_multiprobe

# Define hash-table parameters
n_dims = 32
k = 8
L = 201
n_candidates = 10
R = 0.40
```
```
# Setup hash-table
ht = lsh_naive.LSH(n_dims=n_dims,k=k,L=L,R=R,n_candidates=n_candidates,query_mode='kdtree')
ht.setup(data)
```

```
# Query Approximate NN
best_match, top_k_data_indices, _ = ht.query(q, top_k = 10)
```
