"""
File: synt.py
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import process_datasets as pd
import putils as pu
import time
import os.path
import metrics
from stats import Stats


##############################################
############## Main script code ##############
##############################################


tm = time.time()

refinement = 'indeg_guided'
ksize = 23
imbalanced = False
num_c = 4
internoiselvl = 0
intranoiselvl = 0

G, GT, labeling = pd.custom_cluster_matrix(1000, [250]*4, 0.2, 1, 0, 0)

s = Stats("/tmp/test.csv")
c = Codec(0, 0.5, 20)

k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
tcompression = time.time() - tm

sze = c.decompress(G, 0, classes, k, reg_list)
tdecompression = time.time() - tm

fsze = c.post_decompression(sze, ksize)
tpostdecompression = time.time() - tm

#red = c.reduced_matrix(k, epsilon, classes, reg_list)

telapsed = [tcompression, tdecompression-tcompression, tpostdecompression-tdecompression]

s.compute_stats(imbalanced, num_c, internoiselvl, intranoiselvl, k, epsilon, sze_idx, nirr, refinement, G, GT, labeling, sze, fsze, telapsed, write=True, plot=True, pp=True)

