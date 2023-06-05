import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as o

import numpy as np

dpoints = np.array([
           ['M1', 'K_L1', 0.12123026],
           ['M1', 'K_L2', 0.29652571],
           ['M1', 'K_L3', 0.9230106],
           ['M1', 'K_Reservoir', -0.18939909],
           ['M2', 'K_L1', 0.05293998],
           ['M2', 'K_L2', 0.19402502],
           ['M2', 'K_L3', 0.95556927],
           ['M2', 'K_Reservoir', -0.19692776]])

fig = plt.figure()
ax = fig.add_subplot(111)

def barplot(ax, dpoints):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.

    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''

    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,0])]
    #print conditions
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,1])]

    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    #conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    #print conditions
    #categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]
    #print categories
    conditions = ['M1', 'M2']
    categories = ['K_L1', 'K_L2', 'K_L3','K_Reservoir']
    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))
    #print dpoints
    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond, 
               color=cm.Accent(float(i) / n))

    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(categories,fontsize=14)
    plt.setp(plt.xticks()[1], rotation=0)

    # Add the axis labels
    ax.set_ylabel("Correlation Coefficient",fontsize=16,fontweight="bold")
    ax.set_xlabel("Uncertain Parameters",fontsize=16,fontweight="bold")

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',fontsize=14)

barplot(ax, dpoints)
base=0
plt.axhline(base, color='black')
plt.yticks(fontsize=14)
plt.savefig("sensitivity.png",bbox_inches='tight')
plt.show()