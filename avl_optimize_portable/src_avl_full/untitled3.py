# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:38:11 2024

@author: Aspirant2
"""
import os
import pandas as pd

usr_path = os.path.expanduser('~')
gen_path = os.path.join(usr_path, 'Auto_full_0/gen/14.10')
file_path = os.path.join(gen_path, 'Px_37.xlsx')
df = pd.read_excel(file_path)
length = len(df)
a_S = df['a_S']

n_1_1_normal = 0
n_1_2_normal = 0
n_1_3_normal = 0

n_2_1_normal = 0
n_2_2_normal = 0
n_2_3_normal = 0

n_3_1_normal = 0
n_3_2_normal = 0
n_3_3_normal = 0

n_1_x_duck = 0

n_2_x_duck = 0

n_3_x_duck = 0

for i in range(length):
    if a_S[i] <= 0.5:
        if df['scheme_fuse'][i] < 1/3:
            if df['scheme_vertical'][i] < 1/3:
                n_1_1_normal += 1
            elif df['scheme_vertical'][i] > 2/3:
                n_1_3_normal += 1
            else:
                n_1_2_normal += 1
        elif df['scheme_fuse'][i] > 2/3:
            if df['scheme_vertical'][i] < 1/3:
                n_3_1_normal += 1
            elif df['scheme_vertical'][i] > 2/3:
                n_3_3_normal += 1
            else:
                n_3_2_normal += 1
        else:
            if df['scheme_vertical'][i] < 1/3:
                n_2_1_normal += 1
            elif df['scheme_vertical'][i] > 2/3:
                n_2_3_normal += 1
            else:
                n_2_2_normal += 1
    else:
        if df['scheme_fuse'][i] < 1/3:
            n_1_x_duck += 1
        elif df['scheme_fuse'][i] > 2/3:
            n_3_x_duck += 1
        else:
            n_2_x_duck += 1
    
if n_1_1_normal > 0:
    if os.path.exists(gen_path + '/n_1_1_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_1_1_normal')
if n_1_2_normal > 0:
    if os.path.exists(gen_path + '/n_1_2_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_1_2_normal')
if n_1_3_normal > 0:
    if os.path.exists(gen_path + '/n_1_3_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_1_3_normal')
        
if n_2_1_normal > 0:
    if os.path.exists(gen_path + '/n_2_1_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_2_1_normal')
if n_2_2_normal > 0:
    if os.path.exists(gen_path + '/n_2_2_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_2_2_normal')
if n_2_3_normal > 0:
    if os.path.exists(gen_path + '/n_2_3_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_2_3_normal')
        
if n_3_1_normal > 0:
    if os.path.exists(gen_path + '/n_3_1_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_3_1_normal')
if n_3_2_normal > 0:
    if os.path.exists(gen_path + '/n_3_2_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_3_2_normal')
if n_3_3_normal > 0:
    if os.path.exists(gen_path + '/n_3_3_normal'):
        pass
    else:
        os.mkdir(gen_path + '/n_3_3_normal')
        
if n_1_x_duck > 0:
    if os.path.exists(gen_path + '/n_1_x_duck'):
        pass
    else:
        os.mkdir(gen_path + '/n_1_x_duck')
if n_2_x_duck > 0:
    if os.path.exists(gen_path + '/n_2_x_duck'):
        pass
    else:
        os.mkdir(gen_path + '/n_2_x_duck')
if n_3_x_duck > 0:
    if os.path.exists(gen_path + '/n_3_x_duck'):
        pass
    else:
        os.mkdir(gen_path + '/n_3_x_duck')