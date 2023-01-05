import polycheck

import numpy as np
import time
import matplotlib.pyplot as plt



if __name__ == "__main__":

    t0 = time.time()
    grid = np.zeros( (30,30))
    result = np.zeros_like(grid)
    start = np.array( [5,5]).reshape(1,2)
    polycheck.visibility( grid, start, result)
    print(f'visibility total time: {time.time()-t0}')

    plt.figure()
    plt.imshow(result)
    plt.title('Empty')

    t0 = time.time()
    grid[ 20:23, 5:21 ] = 1
    result = np.zeros_like(grid)
    start = np.array( [5,5]).reshape(1,2)
    polycheck.visibility( grid, start, result)
    print(f'visibility total time: {time.time()-t0}')

    plt.figure()
    plt.imshow(result)
    plt.title('Obstructed')

    grid_size = (300,300)
    t0 = time.time()
    grid = np.zeros( grid_size )
    grid[ 20:23, 205:221 ] = 1
    grid[ 50:70, 50:70] = np.random.random( [20,20])
    grid[ 180:200, 250:275] = np.random.random( [20,25])  / 10.0
    grid[ 160:260, 25:30] = np.random.random( [100,5])
    result = np.zeros_like(grid)
    start = np.array( [150,150]).reshape(1,2)
    polycheck.visibility( grid, start, result)
    print(f'big visibility total time: {time.time()-t0}')

    plt.figure()
    plt.imshow(result)
    plt.title(f'big ({grid_size})')

    grid_size = (300,300)
    t0 = time.time()
    result = np.zeros_like(grid)
    xx, yy = np.meshgrid( np.arange( 200, 300, 1 ), np.arange( 150,250,1))
    ends = np.array([[x,y] for x,y in zip( xx.flatten(), yy.flatten() )])
    start = np.array( [150,150]).reshape(1,2)
    polycheck.region_visibility( grid, start, ends, result)
    print(f'region visibility total time: {time.time()-t0}')

    plt.figure()
    plt.imshow(result)
    plt.title(f'big ({grid_size})')



    plt.show(block=True)
