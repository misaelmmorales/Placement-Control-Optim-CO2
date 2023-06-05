import sys,os
from matk import matk, pest_io 
from subprocess import call
from itertools import product
import numpy as np

# Model function
def fehm(p):
    # Create simulator input file
    os.symlink('../fehmn.files','fehmn.files')
    pest_io.tpl_write(p, '../run.tpl', 'run.dat')
    # Call simulator
    ierr = call('xfehm ../fehmn.files', shell=True)

p = matk(model=fehm)
p.add_par('perm1',min=-14,max=-12)
p.add_par('perm2',min=-20,max=-18)
p.add_par('perm3',min=-14,max=-12)
p.add_par('perm4',min=-12,max=-10)
#p.add_par('leak1',min=4071,max=5000)
p.add_par('q_co2',min=3.17,max=31.7)

### Create your sample from scratch
#p1 = [-13]
#p2 = [-17]
#p3 = [-13]
##l1 = np.arange(4071,5000, 10)
##l1 = [4071,5000]
#l1=[]
#for i in range(1,5):
#    for j in range(1,5):
#        leak = 2601*10+51*10*i+10*(j-1)+11
#        l1.append(leak)
#q_co2 = [0.005,0.01]
##q_co2 = np.arange(0.005,0.01, 10)

#s = p.create_sampleset(list(product(*[p1,p2,p3,l1,q_co2])))


# Or use auto parstudy method
#s = p.parstudy(nvals=[2,1,2,2])

#Create LHS sample
s = p.lhs(siz=500, seed=1000)

#out = s.samples.hist(ncols=2,title='Parameter Histograms by Counts')

s.savetxt('sample.txt')

# Run model with parameter samples
s.run(cpus=20, workdir_base='workdir', outfile='results.dat', logfile='log.dat',verbose=False,reuse_dirs=True)
