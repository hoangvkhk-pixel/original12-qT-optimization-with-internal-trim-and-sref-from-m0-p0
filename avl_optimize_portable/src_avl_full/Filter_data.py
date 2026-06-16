# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:37:58 2024

@author: HUNG HOANG
"""

import numpy as np
import pandas as pd
import os

g = 36
usr_path = os.path.expanduser('~')
excel_path = os.path.join(usr_path, 'Auto_full_0/gen/')
file_path = os.path.join(excel_path, f'Px_{g}.xlsx')
file_path1 = os.path.join(excel_path, f'info_aircraft_{g}.xlsx')
df = pd.read_excel(file_path)
df1 = pd.read_excel(file_path1)
n1_1 = []
m1_1 = []
n1_2 = []
m1_2 = []
n1_3 = []
m1_3 = []
n2_1 = []
m2_1 = []
n2_2 = []
m2_2 = []
n2_3 = []
m2_3 = []
n3_1 = []
m3_1 = []
n3_2 = []
m3_2 = []
n3_3 = []
m3_3 = []
u1_x = []
m1_x = []
u2_x = []
m2_x = []
u3_x = []
m3_x = []
length = len(df)


for i in range(length):
    if df['a_S'][i] <= 0.5:
        if df['scheme_fuse'][i] < 1/3:
            if df['scheme_vertical'][i] < 1/3:
                n1_1.append(i)
                m1_1.append(df1['mtow_out'][i])
            elif df['scheme_vertical'][i] > 2/3:
                n1_3.append(i)
                m1_3.append(df1['mtow_out'][i])
            else:
               n1_2.append(i)
               m1_2.append(df1['mtow_out'][i])
        elif df['scheme_fuse'][i] > 2/3:
            if df['scheme_vertical'][i] < 1/3:
                n3_1.append(i)
                m3_1.append(df1['mtow_out'][i])
            elif df['scheme_vertical'][i] > 2/3:
                n3_3.append(i)
                m3_3.append(df1['mtow_out'][i])
            else:
                n3_2.append(i)
                m3_2.append(df1['mtow_out'][i])
        else:
            if df['scheme_vertical'][i] < 1/3:
                n2_1.append(i)
                m2_1.append(df1['mtow_out'][i])
            elif df['scheme_vertical'][i] > 2/3:
                n2_3.append(i)
                m2_3.append(df1['mtow_out'][i])
            else:
                n2_2.append(i)
                m2_2.append(df1['mtow_out'][i])
    else:
        if df['scheme_fuse'][i] < 1/3:
            u1_x.append(i)
            m1_x.append(df1['mtow_out'][i])
        elif df['scheme_fuse'][i] > 2/3:
            u3_x.append(i)
            m3_x.append(df1['mtow_out'][i])
        else:
            u2_x.append(i)
            m2_x.append(df1['mtow_out'][i])

if len(m1_1) > 0:
    min_value = min(m1_1)
    vt1_1 = m1_1.index(min_value)
    vt1_1 = n1_1[vt1_1]
if len(m1_2) > 0:
    min_value = min(m1_2)
    vt1_2 = m1_2.index(min_value)
    vt1_2 = n1_2[vt1_2]
if len(m1_3) > 0:
    min_value = min(m1_3)
    vt1_3 = m1_3.index(min_value) 
    vt1_3 = n1_3[vt1_3]
if len(m2_1) > 0:
    min_value = min(m2_1)
    vt2_1 = m2_1.index(min_value)  
    vt2_1 = n2_1[vt2_1]
if len(m2_2) > 0:
    min_value = min(m2_2)
    vt2_2 = m2_2.index(min_value) 
    vt2_2 = n2_2[vt2_2]
if len(m2_3) > 0:
    min_value = min(m2_3)
    vt2_3 = m2_3.index(min_value)
    vt2_3 = n2_3[vt2_3]
if len(m3_1) > 0:
    min_value = min(m3_1)
    vt3_1 = m3_1.index(min_value)
    vt3_1 = n3_1[vt3_1]
if len(m3_2) > 0:
    min_value = min(m3_2)
    vt3_2 = m3_2.index(min_value)
    vt3_2 = n3_2[vt3_2]
if len(m3_3) > 0:
    min_value = min(m3_3)
    vt3_3 = m3_3.index(min_value) 
    vt3_3 = n3_3[vt3_3]
if len(m1_x) > 0:
    min_value = min(m1_x)
    vt1_x = m1_x.index(min_value)  
    vt1_x = u1_x[vt1_x]
if len(m2_x) > 0:
    min_value = min(m2_x)
    vt2_x = m2_x.index(min_value)
    vt2_x = u2_x[vt2_x]
if len(m3_x) > 0:
    min_value = min(m3_x)
    vt3_x = m3_x.index(min_value)
    vt3_x = u3_x[vt3_x]
    
    