# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:18:43 2024

@author: Daniel Pessoa
"""


import pandas as pd
import seaborn as sns
import numpy as np
from io import BytesIO
import os
import csv
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib as mpl
#import tkinter as tk
#from tkinter import ttk
from math import sqrt
import soccerplots
from soccerplots.radar_chart import Radar  
from sklearn.model_selection import train_test_split
import numpy as np
import math
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from mplsoccer import PyPizza, add_image,FontManager
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats

from mplsoccer import PyPizza, add_image,FontManager
import matplotlib.pyplot as plt

    

def calculate_params(df_plyr,dataframe_tot,params):
    df_plyr = df_plyr[params]
    dataframe_tot = dataframe_tot[params]
    player = list(df_plyr.loc[0])
    values = []
   # names = []
    dict_params = {}
    dict_values = {}
    

    for x in range(len(params)):
       # if player[x] > 0:
        parametro = params[x]
              #  names.append(parametro)
        dict_params[parametro] = player[x]
        dict_values[parametro] = math.floor(stats.percentileofscore(dataframe_tot[params[x]],player[x]))
        values.append(math.floor(stats.percentileofscore(dataframe_tot[params[x]],player[x])))

    return dict_values#,names





def create_radar (dataframe,player_1,tipo_nome = 'amigavel'):
    
    values = []
    
    dataframe['nome_amigavel'] = [f"{nome} - {clube}" for nome, clube in zip(dataframe['Player'], dataframe['Club'])]
    
    URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-Regular.ttf')
    font_normal = FontManager(URL1)
    URL2 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
    font_italic = FontManager(URL2)
    URL3 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
        'RobotoSlab%5Bwght%5D.ttf')
    font_bold = FontManager(URL3)
    
    if tipo_nome == 'amigavel':
        df_plyr_1 = dataframe[(dataframe['nome_amigavel'] == player_1)].reset_index().drop(columns = ['index']) 
        comp_plyr_1 = df_plyr_1['Competition'][0]
        pos_plyr_1 = df_plyr_1['Position'][0]
        club_plyr_1 = df_plyr_1['Club'][0]
        plyr_real_name = df_plyr_1['Player'][0]
        
        df_all =  dataframe[(dataframe['nome_amigavel'] != player_1) & (dataframe['Position'] == pos_plyr_1) & (dataframe['Competition'] == comp_plyr_1)].reset_index().drop(columns = 'index')
    else:
        df_plyr_1 = dataframe[(dataframe['Player'] == player_1)].reset_index().drop(columns = ['index']) 
        comp_plyr_1 = df_plyr_1['Competition'][0]
        pos_plyr_1 = df_plyr_1['Position'][0]
        club_plyr_1 = df_plyr_1['Club'][0]
        plyr_real_name = df_plyr_1['Player'][0]
        
        df_all =  dataframe[(dataframe['Player'] != player_1) & (dataframe['Position'] == pos_plyr_1) & (dataframe['Competition'] == comp_plyr_1)].reset_index().drop(columns = 'index')

    
    params = ['Performance Ast',
              'Blocks Blocks',
              'Challenges Tkl%',
              'Clr',
              'Int',
              'Performance Int',
              'Progression PrgP',
              'Tackles Tkl',
              'Total Att',
              'Total Cmp%',
              'GCA GCA90',
              'Per 90 Minutes Ast',
              'Per 90 Minutes G+A',
              'Per 90 Minutes npxG',
              'Per 90 Minutes npxG+xAG',
              'SCA SCA90',
              'Standard Sh/90']

    params_alias = ['Assits',
              'Blocks',
              'Challenges Tkl%',
              'Clearences',
              'Int',
              'Performance Int',
              'Progression PrgP',
              'Tackles',
              'Total Att',
              'Total Cmp%',
              'GCA GCA90',
              'Assits/90 min',
              'G+A/90 min',
              'npxG/90 min',
              'npxG+xAG/90 min',
              'SCA/90 min',
              'Standard Sh/90']
    
 #   df_all,params = calculate_variance(df_all)
    
    values_radar= calculate_params(df_plyr_1,df_all,params)
    
    for x in range(len(params)):
        param = params[x]
        values.append(values_radar[param])
    
    
    baker = PyPizza(
        params=params_alias,                  # list of parameters
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=1,               # linewidth of last circle
        other_circle_lw=1,              # linewidth for other circles
        other_circle_ls="-."            # linestyle for other circles
        )
    
    # plot pizza
    fig,ax = baker.make_pizza(
        values,              # list of values
        figsize=(8, 8),      # adjust figsize according to your need
        param_location=110,  # where the parameters will be added
        kwargs_slices=dict(
        facecolor="cornflowerblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting slices
    kwargs_params=dict(
        color="#000000", fontsize=10,
        fontproperties=font_normal.prop, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=10,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
            )
        )  
                    # values to be used when adding parameter-values
    )
    
    
    # add title
    fig.text(
        0.515, 0.97, "{}- {}".format(plyr_real_name,club_plyr_1), size=18,
        ha="center", fontproperties=font_bold.prop, color="#000000"
    )
    
    # add subtitle
    fig.text(
        0.515, 0.942,
        "Percentile Rank vs League",
        size=15,
        ha="center", fontproperties=font_bold.prop, color="#000000"
    )
    
    # add credits
    CREDIT_1 = "data: footystats"
    
    fig.text(
        0.99, 0.005, f"{CREDIT_1}", size=9,
        fontproperties=font_italic.prop, color="#000000",
        ha="right"
    )
    

    
    return fig


