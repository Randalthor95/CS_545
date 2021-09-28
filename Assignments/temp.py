import pandas
import numpy as np

T = [[3],[4],[5],[6],]
quality = [[3,0], [4,1], [5,2], [6,3], [7,4], [8,5], [9,6]]
for val in T:
    for q in quality:
        if val[0] == q[0]:
            val[0] = q[1]
            break

print(T)