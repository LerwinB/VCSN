import pandas as pd
import random
import glob
import os
import se3lib
import numpy as np
import csv
with open('gt.csv', 'w') as file:
    for m in range(1, 4):
        for i in range(1, 17):
            roll = 22.5*(i-1)
            for j in range(1, 17):
                pitch = 22.5*(j-1)
                for k in range(1, 17):
                    yaw = 22.5*(k-1)
                    q = se3lib.euler2quatt(-roll, -pitch, yaw)
                    q = np.array(q)
                    print(q[0][0])
                    if m == 1:
                        z = 6
                    else:
                        z = 7
                    x = 0
                    y = 0
                    #q= ["x", "y", "z", q]
                    x1 = x
                    y1 = y
                    z1 = z
                    q0 = q[0][0]
                    q1 = q[1][0]
                    q2 = q[2][0]
                    q3 = q[3][0]
                    data = [x1, y1, z1, q0, q1, q2, q3]
                    writer = csv.writer(file)
                    writer.writerow(data)
                    #writer.writerow('\n')
