from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from LVQ import LVQ
from GLVQ import GLVQ
from Preprocessing import Preprocessing
from Menu import Menu
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
import operator
