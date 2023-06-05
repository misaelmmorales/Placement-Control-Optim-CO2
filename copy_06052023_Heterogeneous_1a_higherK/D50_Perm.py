import sys,os
from matk import matk, pest_io 
from subprocess import call
from itertools import product
import numpy as np

# 8.669634441084135e-15  9.290011899336757e-15  1.814568998309092e-16      1.298435968399734
# Model function

pb = np.genfromtxt("perm_base.txt")
pbmods = pb*1.298435968399734
with open('perm_base_R.dat','w') as fh:
    fh.write('perm\n')
    fh.write('    1    0	0	1e-16	1e-16	1e-16\n')
    fh.write("    %d %d %d %e %e %e\n"%(-12, 0, 0, 1e-19, 1e-19, 1e-19))
    fh.write("    %d %d %d %e %e %e\n"%(-13, 0, 0, 1e-13, 1e-13, 1e-13))
    fh.write("    %d %d %d %e %e %e\n"%(26531, 49940, 2601, 8.669634441084135e-15, 8.669634441084135e-15, 8.669634441084135e-15))
    fh.write("    %d %d %d %e %e %e\n"%(26551, 49960, 2601, 9.290011899336757e-15, 9.290011899336757e-15, 9.290011899336757e-15))
    fh.write("    %d %d %d %e %e %e\n"%(27561, 50970, 2601, 1.814568998309092e-16, 1.814568998309092e-16, 1.814568998309092e-16))
    for i,pbmod in enumerate(pbmods):
        fh.write("    %d %d %d %e %e %e\n"%(i+1, i+1, 1, pbmod[0], pbmod[1], pbmod[2]))
    fh.write("\n")
