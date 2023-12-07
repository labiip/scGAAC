from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import minmax_scale
import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd



def read_csv(filename, take_scale):
    """ Read  data of a dataset saved in csv format:
    row--genes;column--cells

    Format of the data.csv:
    first column: gene symbols
    rest  column: data express(float)
    Format of the label.csv:
    first row: title group
    rest  row: cell  labels
        Args:
        filename: name of the csv file
        take_scale: whether do scale [0,1] on input data (feature scale)
    Returns:
        row--cells;column--genes;
        dataset: a dict with keys 'data', 'label', 'gene'.
    """
    data_path = "../Dataset/" + filename + "/data.csv"
    lab_path  = "../Dataset/" + filename + "/label.csv"
    dataset = {}
    dat = pd.read_csv(data_path, header = None)
    dat = dat.values
    data = dat[:,1:].astype(float)
    gene = dat[:,1]

    label = pd.read_csv(lab_path, header = 0)
    lab = label.values

    if take_scale:
        data = data.transpose()
        data = minmax_scale(data, feature_range=(0, 1), axis=1, copy=False)
        data = data.astype(np.float32)
    
    dataset['label'] = lab.flatten()
    dataset['data']  = data
    dataset['gene']  = gene

    return dataset
