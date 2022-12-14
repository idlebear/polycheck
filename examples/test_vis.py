import polycheck

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image


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
    grid[ 20:70, 40:90] = 1
    grid[ 180:200, 250:275] = np.random.random( [20,25])  / 10.0
    grid[ 160:260, 25:30] = np.random.random( [100,5])
    grid[ 175:220, 90:155] = 0.25
    result = np.zeros_like(grid)
    start = np.array( [150,150]).reshape(1,2)
    polycheck.visibility( grid, start, result)
    print(f'big visibility total time: {time.time()-t0}')

    plt.figure()
    plt.imshow(result)
    plt.title(f'big ({grid_size})')

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

    fig, ax = plt.subplots( 2, 5, figsize=(15,8) )

    for i in range(10):
        t0 = time.time()
        result = np.zeros_like(grid)
        xx, yy = np.meshgrid( np.arange( 200, 300, 1 ), np.arange( 150,250,1))
        start = np.array( ( np.random.random((1,2))*grid_size[0]).astype(int)).reshape(1,2)
        polycheck.visibility( grid, start, result)
        print(f'region visibility total time: {time.time()-t0}')

        result[start[0,1], start[0,0]] = 5

        ax[int(i/5), int(i%5)].imshow(result)

    t0 = time.time()
    sxs, sys = 5, 5
    sxe, sye = 295, 295
    xx, yy = np.meshgrid( np.arange( sxs, sxe, 1 ), np.arange( sys, sye, 1))
    starts = np.array([[x,y] for x,y in zip( xx.flatten(), yy.flatten() )])

    # starts = np.array( [
    #     [100, 100],
    #     [150, 100],
    #     [100, 150],
    #     [150, 150]
    # ])
    exs, eys = 5, 10
    exe, eye = 35, 100
    grid[ eys:eye, exs:exe ] = 0.1  # add some uncertainty
    xx, yy = np.meshgrid( np.arange( exs, exe, 1 ), np.arange( eys, eye, 1))
    ends = np.array([[x,y] for x,y in zip( xx.flatten(), yy.flatten() )])
    result = np.zeros( (starts.shape[0], ends.shape[0]))

    polycheck.visibility_from_region( grid, starts, ends, result)

    print(f'region visibility total time: {time.time()-t0}')

    # rows = 2 
    # cols = 2 
    rows = sye - sys
    cols = sxe - sxs

    observe_dx = exe - exs
    observe_dy = eye - eys

    composite_image = np.zeros((rows*observe_dy,cols*observe_dx))
    for r in range( rows ):
        for c in range( cols ):
            oy = r * observe_dy
            ox = c * observe_dx
            composite_image[ oy:oy+observe_dy, ox:ox+observe_dx] = result[r*cols +c].reshape( observe_dy, observe_dx )

    plt.figure()
    plt.imshow( composite_image )

    log_result = -np.log(result+0.00000001) * result
    ig = np.sum( log_result, axis=1 ).reshape( int(sye-sys), int(sxe-sxs) )
    normalized_ig = ig - np.min(ig)
    normalized_ig = normalized_ig / np.max(normalized_ig)

    print( f"max ig: {np.max(ig)}, min ig: {np.min(ig)}")
    print( f"max normalized ig: {np.max(normalized_ig)}, min norm ig: {np.min(normalized_ig)}")

    composite_image = np.zeros((rows*observe_dy,cols*observe_dx))
    for r in range( rows ):
        for c in range( cols ):
            oy = r * observe_dy
            ox = c * observe_dx
            composite_image[ oy:oy+observe_dy, ox:ox+observe_dx] = log_result[r*cols +c].reshape( observe_dy, observe_dx )

    plt.figure()
    plt.imshow( composite_image )

    plt.figure()
    plt.imshow( normalized_ig  )

    plt.figure()

    image = np.zeros( (*grid_size, 3) )
    image[:,:,0] = grid
    image[ sys:sye, sxs:sxe, 2] = normalized_ig

    # image[ eys:eye, exs:exe, :] += 0.5
    image[ eys:eye, exs:exe, 1] += 0.5

    image = Image.fromarray((image * 255.0).astype(np.uint8))
    plt.imshow(image)

    plt.title(f'big ({grid_size})')




    plt.show(block=True)
