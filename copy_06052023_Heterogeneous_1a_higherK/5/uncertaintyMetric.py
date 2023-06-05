import numpy as np

def uncertaintyMetric(samples):
    y1 = np.percentile(samples,10)
    y2 = np.percentile(samples,90)
    p90mp10 = y2 -y1
    return p90mp10
