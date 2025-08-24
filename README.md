# DBSCAN_ArrayFire
OpenCL ArrayFire implementation of the DBSCAN clustering algorithm in C++. Designed for clustering of unit vectors based on their Cosine Distances. <br><br>

Requires the ArrayFire Library: https://github.com/arrayfire/arrayfire <br><br>

### Example use
```C++
#include "DBSCAN.h"

af::setBackend(AF_BACKEND_OPENCL);

af::array X = af::randn(10000, 16); // Matrix of unit-vectors (Num of vectors x vector dimensions)

DBSCAN db(0.7f, 50); // DBSCAN instance, minimum dot product and minimum points to form a cluster

std::vector<int> labels = db.fit(X); // cluster data
```
