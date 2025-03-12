#!/usr/bin/env python
# coding: utf-8

# # Generating Results on Datasets

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist


# ## Plots for OTB, NFS and UAV

# In[ ]:


trackers = []
trackers.extend(trackerlist('dimp', 'dimp50', range(0,1), 'DIMP'))


dataset = get_dataset('otb')
plot_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec'), 
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

