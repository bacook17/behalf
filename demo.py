import schwimmbad
from octree import octree, bbox
import sys
import numpy as np

N = 5000 # number of points
points = (np.random.uniform(0, 5, size=(N,3))) # initialize randomly
masses = np.ones(N)
sim_box = bbox([[np.min(points), np.max(points)], [np.min(points), np.max(points)], [np.min(points), np.max(points)]]) # make bounding box
tree = octree(points, masses, sim_box, np.zeros_like(masses))

GRAVITATIONAL_CONSTANT = 1
THETA = 0.5

def worker(task):
        return tree.force(THETA, task, GRAVITATIONAL_CONSTANT)

def main(pool):
        tasks = np.arange(N)
        results = pool.map(worker, tasks)
        pool.close()
        return results

if __name__ == "__main__":
        pool = schwimmbad.MPIPool()

        if not pool.is_master():
                pool.wait()
                sys.exit(0)
                
        import time
        t = time.time()
        results = main(pool)
        print('Time taken: %f' % (time.time() - t))