# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:36:53 2021

@author: HP
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

# import modules
import pandas as pd
import numpy as np

# read in full data frame
df = pd.read_csv('deGrom_data_full.csv')

# list of initial columns to delete
toDelete = ['release_speed',
            'release_pos_x',
            'release_pos_z',
            'player_name',
            'pitcher',
            'events',
            'description',
            'spin_dir',
            'spin_rate_deprecated',
            'break_angle_deprecated',
            'break_length_deprecated',
            'zone',
            'des',
            'game_type',
            'p_throws',
            'type',
            'hit_location',
            'bb_type',
            'game_year',
            'pfx_x',
            'pfx_z',
            'plate_x',
            'plate_z',
            'hc_x',
            'hc_y',
            'tfs_deprecated',
            'tfs_zulu_deprecated',
            'umpire',
            'sv_id',
            'vx0',
            'vy0',
            'vz0',
            'ax',
            'ay',
            'az',
            'sz_top',
            'sz_bot',
            'hit_distance_sc',
            'launch_speed',
            'launch_angle',
            'effective_speed',
            'release_spin_rate',
            'release_extension',
            'game_pk',
            'pitcher.1',
            'fielder_2.1',
            'release_pos_y',
            'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle',
            'woba_value',
            'woba_denom',
            'babip_value',
            'iso_value',
            'launch_speed_angle',
            'pitch_name',
            'post_away_score',
            'post_home_score',
            'post_bat_score',
            'post_fld_score',
            'spin_axis',
            'delta_home_win_exp',
            'delta_run_exp']

# delete the columns
df.drop(toDelete, axis=1, inplace=True)

# dictionary to match the pitch type to a class number
pitchDict = {'SL':1,'FF':2,'CH':3,'CU':4,'FT':5,'IN':6}

# build array of the pitch classes
slider      = pitchDict['SL'] * np.array(df['pitch_type']=='SL')
fastball4   = pitchDict['FF'] * np.array(df['pitch_type']=='FF')
changeup    = pitchDict['CH'] * np.array(df['pitch_type']=='CH')
curveball   = pitchDict['CU'] * np.array(df['pitch_type']=='CU')
fastball2   = pitchDict['FT'] * np.array(df['pitch_type']=='FT')
intent_ball = pitchDict['IN'] * np.array(df['pitch_type']=='IN')
pitch_class = slider + fastball4 + changeup + curveball + fastball2 + intent_ball

# add pitch class array to data frame
df['pitch_class']=pitch_class

# delete rows where pitch class is unknown
df.drop(df.loc[df['pitch_class']==0].index, axis=0, inplace=True)

# delete pitch_type column, it is not needed anymore
df.drop(['pitch_type'], axis=1, inplace=True)

# dictionary to match stance of batter to a class number
standDict = {'R':1, 'L':2}

# build arrary of stance classes
right_stand = standDict['R'] * np.array(df['stand']=='R')
left_stand  = standDict['L'] * np.array(df['stand']=='L')
stand_class = right_stand + left_stand

# add stand class array to data frame
df['stand_class']=stand_class

# delete stand column, it is not needed anymore
df.drop(['stand'], axis=1, inplace=True)

# create column to indicate if Mets are home (1) or away (0)
home = 1*(df['home_team']=='NYM')

# add home to data frame
df['home']=home

# delete home_team and away_team columns, they are not needed anymore
df.drop(['home_team','away_team'], axis=1, inplace=True)

# array to indicate if it is the top (1) or bottom (0) of the inning
top_bot = 1*(df['inning_topbot']=='Top')

# calculate the half inning for every entry
half_inning = 2*df['inning'] - top_bot

# add the half inning class to the data frame
df['half_inning']=half_inning

# delete inning and inning_topbot columns, they are not needed anymore
df.drop(['inning','inning_topbot'], axis=1, inplace=True)

# parse the date in to separate columns and format as integers
df['year']  = 0
df['month'] = 0
df['day']   = 0
for k in df.index:
    df.at[k,'year']  = int(df.at[k,'game_date'][0:4])
    df.at[k,'month'] = int(df.at[k,'game_date'][5:7])
    df.at[k,'day']   = int(df.at[k,'game_date'][8:10])

# delete game_date column, it is not needed anymore
df.drop(['game_date'], axis=1, inplace=True)

# create new column base_runner_class to indicate where base runners are
df['base_runner_class']=0
df['1b']=1*(df['on_1b'].notnull())
df['2b']=1*(df['on_2b'].notnull())
df['3b']=1*(df['on_3b'].notnull())
for k in df.index:
    if df.at[k,'1b']==0 and df.at[k,'2b']==0 and df.at[k,'3b']==0:
        df.at[k,'base_runner_class']=1
        
    elif df.at[k,'1b']==1 and df.at[k,'2b']==0 and df.at[k,'3b']==0:
        df.at[k,'base_runner_class']=2
        
    elif df.at[k,'1b']==0 and df.at[k,'2b']==1 and df.at[k,'3b']==0:
        df.at[k,'base_runner_class']=3
        
    elif df.at[k,'1b']==0 and df.at[k,'2b']==0 and df.at[k,'3b']==1:
        df.at[k,'base_runner_class']=4
        
    elif df.at[k,'1b']==1 and df.at[k,'2b']==1 and df.at[k,'3b']==0:
        df.at[k,'base_runner_class']=5
        
    elif df.at[k,'1b']==1 and df.at[k,'2b']==0 and df.at[k,'3b']==1:
        df.at[k,'base_runner_class']=6
        
    elif df.at[k,'1b']==0 and df.at[k,'2b']==1 and df.at[k,'3b']==1:
        df.at[k,'base_runner_class']=7
        
    elif df.at[k,'1b']==1 and df.at[k,'2b']==1 and df.at[k,'3b']==1:
        df.at[k,'base_runner_class']=8
        
'''
Base runner class key

On 1st    On 2nd    On 3rd    Class
0         0         0         1
1         0         0         2
0         1         0         3
0         0         1         4
1         1         0         5
1         0         1         6
0         1         1         7
1         1         1         8
'''

# delete base running columns that we don't need anymore
toDelete1 = ['on_1b',
             'on_2b',
             'on_3b',
             '1b',
             '2b',
             '3b']
df.drop(toDelete1, axis=1, inplace=True)

# dictionary to match fielding alignment to class number
alignmentDict = {'Standard':1, 'Strategic':2, 'Infield shift':3}

# build array of infield alignment class and add to dataframe
if_standard      = alignmentDict['Standard']      * np.array(df['if_fielding_alignment']=='Standard')
if_strategic     = alignmentDict['Strategic']     * np.array(df['if_fielding_alignment']=='Strategic')
if_infield_shift = alignmentDict['Infield shift'] * np.array(df['if_fielding_alignment']=='Infield shift')
if_alignment_class = if_standard + if_strategic + if_infield_shift
df['if_alignment_class']=if_alignment_class

# build array of outfield alignment class and add to dataframe
of_standard      = alignmentDict['Standard']  * np.array(df['of_fielding_alignment']=='Standard')
of_strategic     = alignmentDict['Strategic'] * np.array(df['of_fielding_alignment']=='Strategic')
of_alignment_class = of_standard + of_strategic
df['of_alignment_class']=of_alignment_class

# drop if_fielding_alignment and of_fielding_alignment columns, we don't need them anymore
df.drop(['if_fielding_alignment','of_fielding_alignment'], axis=1, inplace=True)

# get rid of any rows where the alignment is unknown
df.drop(df.loc[df['if_alignment_class']==0].index, axis=0, inplace=True)
df.drop(df.loc[df['of_alignment_class']==0].index, axis=0, inplace=True)

# split in to data (X) and class (C)
X = df.drop(['pitch_class'],axis=1).copy()
C = df['pitch_class'].copy()

# save cleaned data and class
X.to_csv('deGrom_data_clean.csv',index=False)
C.to_csv('deGrom_data_class.csv',index=False)

























