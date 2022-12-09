# polycheck

A short and sweet library to check whether a point is contained within an arbitrary polygon.  Created as an excuse to work with GPU/CUDA 
code, and because Shapely is too slow...

## Results (preliminary)

Here's a snapshot of the current execution time for a simple square polygon, with a grid of 
1000x1000 cells.  Three methods are tested:
* Shapely -- create a polygon and test each individual point
* Polycheck -- use the GPU to implement a parallel version of the Winding Number algorithm 
  published by [Dan Sunday](https://en.wikipedia.org/wiki/Point_in_polygon).
* Local -- same algorithm as Polycheck, but implemented in Python as a benchmark of sorts.

```
(conda) reggie$ python test_poly.py
Shapely total time: 7.612478733062744
Polycheck total time: 0.3291971683502197
Local check total time: 13.260296821594238
```

## Prerequisites

* Cuda installed in /usr/local/cuda 
* Python 3.6 or greater 
* Cmake 3.6 or greater 

## To build 

```source install.bash``` 

Test it with

```python3 test_poly.py``` 
 
