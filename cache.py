import numpy as np
import time

# https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

N = 500 #300 #100
x = np.zeros((1024, 1024), dtype = np.float32)

# write 1000 copies
for i in range(N):
    fname = f'zeros_{i}.npy'
    np.save(fname, x)


print("starting read test")
# read each file within the loop
start = time.time()
for i in range(N):
    np.load(f'zeros_{i}.npy')
    y = x * x
end = time.time()
print("Elapsed", round(end - start, 3))



# write N points to 1 mmap
map_fname = 'zeros_mapped.npy'
x_mapped = np.memmap(map_fname, dtype = 'float32', mode = 'w+', shape = (N, 1024,1024))
for i in range(N):
    x_mapped[i, :, :] = x
x_mapped.flush()
del x_mapped


print("starting mmap test")
# load the mmmap
x_mapped = np.memmap(map_fname, mode = 'r', dtype = np.float32, shape = (N, 1024, 1024))
start = time.time()
for i in range(N):
    #x = np.array(x_mapped[i])
    y = x[i] * x[i]
end = time.time()
print("Elapsed", round(end - start, 3))


# could also try to see what happens with a list of N mmaps instead of one N-long mmap
