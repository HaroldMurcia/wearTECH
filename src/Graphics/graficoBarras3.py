#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:39:01 2019

@author: juanitatriana
"""

import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 7
means_Full = (0.8,	1,	0.85714286,	0.8,	1,	1,	0.85714286)
means_Best = (0.5,	1,	0.66666667,	0.85714286,	1,	1,	0.85714286)
means_Hierarchical=(1,1,1,1,1,1,1)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.9
 
rects1 = plt.bar(index, means_Full, bar_width,
alpha=opacity,
color='blue',
label='Full Features')
 
rects2 = plt.bar(index + bar_width, means_Best, bar_width,
alpha=opacity,
color='green',
label='Best Features')

rects2 = plt.bar(index + bar_width+bar_width, means_Hierarchical, bar_width,
alpha=opacity,
color='red',
label='Hierarchical')
 
plt.xlabel('Activity')
plt.ylabel('F1 Scores')
plt.xticks(index + bar_width, ('Up', 'Down', 'Static', 'Walking','Running','Cycling','Car'))
plt.legend()
 
plt.tight_layout()
plt.show()