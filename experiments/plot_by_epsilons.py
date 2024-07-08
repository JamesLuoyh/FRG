import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import style

#####
# load data from files
#####





#####
# make plots
#####
savename = "plot.png"
save_format = "png"
figsize = (10, 4.5)
fig = plt.figure(figsize=figsize)
n_rows = 1
n_cols = 5
ax_performance = fig.add_subplot(n_rows, n_cols, 1)

########################
### PERFORMANCE PLOT ###
########################



# ax_performance = fig.add_subplot(n_rows, n_cols, 2)



plt.tight_layout()
plt.savefig(savename, format=save_format)
print(f"Saved {savename}")
