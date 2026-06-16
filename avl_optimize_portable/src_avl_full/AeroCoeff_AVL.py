# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:24:45 2023

@author: Hung_Hoang
"""
import subprocess
import os
import math
import tempfile
import numpy as np
from datetime import datetime
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_FULL_ROOT = os.environ.get(
    'AUTO_FULL_ROOT',
    os.path.abspath(os.path.join(MODULE_DIR, '..', 'runs')),
)

#Input
def inputValuesAtmospheric(V, H=0):
    ro_0 = 1.225
    ro = ro_0*(1-H/44300)**4.256
    q = np.around(ro * V ** 2 / 2, decimals=4)
    return q

def input_flight_cond(V=0, alpha=0, H=0, beta=0):
    V = np.around(V, decimals=3)
    alpha = np.around(alpha, decimals=3)
    beta = np.around(beta, decimals=3)
    H = np.around(H, decimals=3)
    return [V, alpha, H, beta]

def input_lift_surface_data(aspect=0, sweep=0, taper=0, twist=0, dihedral=0, delta=0, area=0, L=0, Nchord=13, Nspan=19):
    return [aspect, sweep, taper, twist, dihedral, delta, area, L, Nchord, Nspan]

def input_body_data(aspect_nose=0, aspect_center=0, aspect_tail=0, height_max=0, width_max=0, x_loc=0, Nchord=10, Nspan=5):
    return [aspect_nose, aspect_center, aspect_tail, height_max, width_max, x_loc, Nchord, Nspan]

def lift_surface_def(lift_surface_geo_char):
    aspect = lift_surface_geo_char[0]
    taper = lift_surface_geo_char[2]
    area = lift_surface_geo_char[6]
    dihedral = lift_surface_geo_char[4]
    if area == 0:
        span = 0
        root_chord = 0
        tip_chord = 0
    else:
        if dihedral == 90: #Vertical tail
            #area = area*np.cos(math.radians(dihedral))
            span = np.sqrt(aspect * area/2)
            tip_chord = 2 * area / span / (1 + taper)
            root_chord = tip_chord * taper 
        else:  # Wing and horizontal
            area = area*np.cos(math.radians(dihedral))
            span = np.sqrt(aspect * area)
            tip_chord = 2 * area / span / (1 + taper)
            root_chord = tip_chord * taper      
    return span, root_chord, tip_chord

def lift_wing_sec(lift_surface_geo_char):
    sweep = lift_surface_geo_char[1]
    twist = lift_surface_geo_char[3]
    dihedral = lift_surface_geo_char[4]
    span, root_chord, tip_chord = lift_surface_def(lift_surface_geo_char)
    
    x_0 = 0
    y_0 = 0
    z_0 = 0
    twist_0 = 0
    
    x_k = span*np.tan(math.radians(sweep))/2
    y_k = span/2
    z_k = span*np.tan(math.radians(dihedral))/2
    twist_k = twist
    
    x = [x_0, x_k]
    y = [y_0, y_k]
    z = [z_0, z_k]
    Ain = [twist_0, twist_k]
    chord = [root_chord, tip_chord]
    return x, y, z, chord, Ain

def lift_v_tail_sec(lift_surface_geo_char):
    sweep = lift_surface_geo_char[1]
    twist = lift_surface_geo_char[3]
    span, root_chord, tip_chord = lift_surface_def(lift_surface_geo_char)
    
    x_0 = 0
    y_0 = 0
    z_0 = 0
    twist_0 = 0
    
    x_k = span*np.tan(math.radians(sweep))
    y_k = 0
    z_k = span
    twist_k = twist
    
    x = [x_0, x_k]
    y = [y_0, y_k]
    z = [z_0, z_k]
    Ain = [twist_0, twist_k]
    chord = [root_chord, tip_chord]
    return x, y, z, chord, Ain

def lift_h_tail_sec(h_tail_geo_char, scheme_fuse):
    h_sweep = h_tail_geo_char[1]
    h_twist = h_tail_geo_char[3]
    h_dihedral = h_tail_geo_char[4]
    h_span, h_root_chord, h_tip_chord = lift_surface_def(h_tail_geo_char)  
        
    if scheme_fuse == 2:
        x_0 = 0
        y_0 = 0
        z_0 = 0
        twist_0 = 0
        
        x_k = h_span*np.tan(math.radians(h_sweep))/2
        y_k = h_span/2
        z_k = h_span*np.tan(math.radians(h_dihedral))/2
        twist_k = h_twist
    else:
        x_0 = 0
        y_0 = 0
        z_0 = h_span*np.tan(math.radians(h_dihedral))/2
        twist_0 = 0
        
        x_k = h_span*np.tan(math.radians(h_sweep))/2
        y_k = h_span/2
        z_k = 0
        twist_k = h_twist
          
    x = [x_0, x_k]
    y = [y_0, y_k]
    z = [z_0, z_k]
    Ain = [twist_0, twist_k]
    chord = [h_root_chord, h_tip_chord]
    return x, y, z, chord, Ain

def body_def(body_geo_char):
    if body_geo_char[0] == 0:
        d_e_body = 0
        l_body = 0
        S_body = 0
        Sxq_body = 0
    else:
        if body_geo_char[3] == body_geo_char[4]:
            d_e_body = body_geo_char[4]
        else:
            d_e_body = 2*np.sqrt(body_geo_char[3]*body_geo_char[4]/math.pi)
        l_nose = body_geo_char[0]*d_e_body
        l_center = body_geo_char[1]*d_e_body
        l_tail = body_geo_char[2]*d_e_body
        l_body = l_nose + l_center + l_tail
        R_e_body = d_e_body/2
        S_body = math.pi*R_e_body**2
        Sxq_nose = math.pi*R_e_body*((R_e_body**2+4*l_nose**2)**(3/2)-R_e_body**3)/(6*l_nose**2)
        Sxq_center = math.pi*d_e_body*l_center
        Sxq_tail = math.pi*R_e_body*((R_e_body**2+4*l_tail**2)**(3/2)-R_e_body**3)/(6*l_tail**2)
        Sxq_body = Sxq_nose+Sxq_center+Sxq_tail
    return d_e_body, l_body, S_body, Sxq_body

def body_H_sec(body_geo_char):
    if body_geo_char[1] == 0:
        body_H_section_origin = 0
        body_H_section_chord = 0
    else:
        semispan_H = body_geo_char[4] * 0.5
        vec1 = [2 , 1.5, 1.2 , 1]
        vec2 = [1  ,1.57 , 3.2,  8]
        x = np.linspace(0,1,4)
        body_definition = body_def(body_geo_char)
    
        body_nose_curvature =  np.interp(np.interp(body_geo_char[0],vec2,x), x , vec1)
        body_tail_curvature =  np.interp(np.interp(body_geo_char[2],vec2,x), x , vec1)

        # Horizontal Sections
        if semispan_H != 0:
            width_array = np.linspace(-semispan_H, semispan_H, num = 3, endpoint = True)
            body_H_section_origin = np.zeros([len(width_array),3])
            body_H_section_chord = np.zeros([len(width_array),1])
            i = 0
            for section_width in width_array:
                body_H_section_cylinder_length  = body_geo_char[1]*body_definition[0]
                body_H_section_nose_length   = ((1 - ((abs(section_width/semispan_H))**body_nose_curvature))**(1/body_nose_curvature))*body_geo_char[0]*body_definition[0]
                body_H_section_tail_length   = ((1 - ((abs(section_width/semispan_H))**body_tail_curvature ))**(1/body_tail_curvature))*body_geo_char[2]*body_definition[0]
                body_H_section_nose_origin   = body_geo_char[0]*body_definition[0] - body_H_section_nose_length
                body_H_section_origin[i]       = [body_H_section_nose_origin, section_width, 0]
                body_H_section_chord[i]        = body_H_section_cylinder_length + body_H_section_nose_length + body_H_section_tail_length
                i += 1
    return body_H_section_origin, body_H_section_chord
   
def body_V_sec(body_geo_char):
    if body_geo_char[1] == 0:
        body_V_section_origin = 0
        body_V_section_chord = 0
    else:
        semispan_V = body_geo_char[3] * 0.5
        vec1 = [2 , 1.5, 1.2 , 1]
        vec2 = [1  ,1.57 , 3.2,  8]
        x = np.linspace(0,1,4)
        body_definition = body_def(body_geo_char)
    
        body_nose_curvature =  np.interp(np.interp(body_geo_char[0],vec2,x), x , vec1)
        body_tail_curvature =  np.interp(np.interp(body_geo_char[2],vec2,x), x , vec1)
    
        # Vertical Sections
        if semispan_V != 0:
            height_array = np.linspace(-semispan_V, semispan_V, num = 3, endpoint = True)
            body_V_section_origin = np.zeros([len(height_array),3])
            body_V_section_chord = np.zeros([len(height_array),1])
            i = 0
            for section_height in height_array:
                body_V_section_cylinder_length  = body_geo_char[1]*body_definition[0]
                body_V_section_nose_length   = ((1 - ((abs(section_height/semispan_V))**body_nose_curvature))**(1/body_nose_curvature))*body_geo_char[0]*body_definition[0]
                body_V_section_nose_origin   = body_geo_char[0]*body_definition[0] - body_V_section_nose_length
                body_V_section_origin[i]        = [body_V_section_nose_origin, 0, section_height]
                if section_height <0:
                    body_V_section_tail_length   = ((1 - ((abs(section_height/semispan_V))**body_tail_curvature ))**(1/body_tail_curvature))*body_geo_char[2]*body_definition[0]     
                    body_V_section_chord[i]         = body_V_section_cylinder_length + body_V_section_nose_length + body_V_section_tail_length
                else:
                    body_V_section_tail_length   =  body_geo_char[2]*body_definition[0]  
                    body_V_section_chord[i]      = body_V_section_cylinder_length + body_V_section_nose_length + body_V_section_tail_length
                i += 1
    return body_V_section_origin, body_V_section_chord

def ref_dim_lift_surface(lift_surface_geo_char):
    taper = lift_surface_geo_char[2]
    
    if taper != 0:
        span, root_chord, tip_chord = lift_surface_def(lift_surface_geo_char)
        mac =  2/3*root_chord*(1+ taper + taper**2)/(1 + taper)/taper
    else:
        mac = 0     
    return mac

def Mach_Reynolds_number(V, mac, H=0):
    k = 1.4
    R = 287
    T0 = 288
    T_H = T0 - 0.0065*H
    a = math.sqrt(k*R*T_H)
    Mach = V/a
    f_H = 2.33*(1 - H/12 + H**2/535)*(10**7)
    Reynolds = Mach*mac*f_H
    return Mach, Reynolds

def avl_file(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, kk, center_mass=0):
    d_e_body, l_body, S_body, Sxq_body = body_def(fuse_geo_char)
    
    f_aspect = f_w_geo_char[0]
    f_teta = f_w_geo_char[5]
    f_S = f_w_geo_char[6]
    
    a_aspect = a_w_geo_char[0]
    a_teta = a_w_geo_char[5]
    a_S = a_w_geo_char[6]
    a_L = a_w_geo_char[7]
    
    v_aspect = v_w_geo_char[0]
    v_teta = v_w_geo_char[5]
    v_L = v_w_geo_char[7]
    
    f_span, f_root_chord, f_tip_chord = lift_surface_def(f_w_geo_char)
    a_span, a_root_chord, a_tip_chord = lift_surface_def(a_w_geo_char)
    v_span, v_root_chord, v_tip_chord = lift_surface_def(v_w_geo_char)
    S_ref = f_S + a_S
    distance_two_fuse = min(a_span, f_span)
    v_w_x, v_w_y, v_w_z, v_w_chord, v_Ain = lift_v_tail_sec(v_w_geo_char)
    
    if f_S >= a_S:
        mac = ref_dim_lift_surface(f_w_geo_char)  
        f_w_x, f_w_y, f_w_z, f_w_chord, f_Ain = lift_wing_sec(f_w_geo_char)
        a_w_x, a_w_y, a_w_z, a_w_chord, a_Ain = lift_h_tail_sec(a_w_geo_char, scheme_fuse)

    else:
        mac = ref_dim_lift_surface(a_w_geo_char)
        f_w_x, f_w_y, f_w_z, f_w_chord, f_Ain = lift_h_tail_sec(f_w_geo_char, scheme_fuse)
        a_w_x, a_w_y, a_w_z, a_w_chord, a_Ain = lift_wing_sec(a_w_geo_char)

    a_w_loc = a_L*mac
    v_w_loc = v_L*mac
    
    fuse_H_section_origin, fuse_H_section_chord = body_H_sec(fuse_geo_char)
    fuse_V_section_origin, fuse_V_section_chord = body_V_sec(fuse_geo_char)

    #%% Write file .avl
    tmp_path = os.path.join(AUTO_FULL_ROOT, f'Auto_full_{kk}', 'tmp')
    os.makedirs(tmp_path, exist_ok=True)
    avl_file = f'Aircraft_{kk}.avl'
    avl_path = os.path.join(tmp_path, avl_file)
    if os.path.exists(avl_path):
        return 0
    with open(avl_path, 'a') as f:
        print('Aircraft', file=f)
        print('', file=f)
        print('0             | Mach', file=f)
        print('0     0     0     | iYsym    iZsym   Zsym', file=f)
        #print(f'{S_ref} {mac} {f_span} |Sref   Cref    Bref    reference area, chord, span', file=f)
        if f_S >= a_S:
            print(f'{S_ref} {mac} {f_span} |Sref   Cref    Bref    reference area, chord, span', file=f)
        else:
            print(f'{S_ref} {mac} {a_span} |Sref   Cref    Bref    reference area, chord, span', file=f)
            
        print(f'{center_mass} 0 0  |Xref   Yref    Zref    moment reference location (arb.)', file=f)
        print('#', file=f)
        print('#===============================================================', file = f)
        print('#', file=f)
        # Fuselage
        if scheme_fuse == 2 or scheme_fuse == 3:
            # H_fuse
            print('SURFACE', file = f)
            print('H-fuselage', file = f)
            print(f'{fuse_geo_char[6]}    1   {fuse_geo_char[7]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    0   0', file = f)
            print('#', file = f)
            for i in range(3):
                print(f'#------------------------- {i+1}st section -------------------------', file = f)
                print('SECTION', file=f)
                print('#Xle   Yle    Zle      Chord   Ainc  Nspanwise  Sspace', file = f)
                print(f'{fuse_H_section_origin[i,0]} {fuse_H_section_origin[i,1]} {fuse_H_section_origin[i,2]} {fuse_H_section_chord[i,0]}     0    1   0', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)
            #V_Fuselage
            print('SURFACE', file = f)
            print('V-fuselage', file = f)
            print(f'{fuse_geo_char[6]}    1   {fuse_geo_char[7]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('NOWAKE', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    0   0', file = f)
            print('#', file = f)
            for i in range(3):
                print(f'#------------------------- {i+1}st section -------------------------', file = f)     
                print('SECTION', file=f)
                print('#Xle   Yle    Zle      Chord   Ainc  Nspanwise  Sspace', file = f)
                print(f'{fuse_V_section_origin[i,0]} {fuse_V_section_origin[i,1]} {fuse_V_section_origin[i,2]} {fuse_V_section_chord[i,0]}      0    1   0', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)   
        else:
            # H_fuse 1
            print('SURFACE', file = f)
            print('H-fuselage 1', file = f)
            print(f'{fuse_geo_char[6]}    1   {fuse_geo_char[7]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {distance_two_fuse/2}   0', file = f)
            print('#', file = f)
            for i in range(3):
                print(f'#------------------------- {i+1}st section -------------------------', file = f)
                print('SECTION', file=f)
                print('#Xle   Yle    Zle      Chord   Ainc  Nspanwise  Sspace', file = f)
                print(f'{fuse_H_section_origin[i,0]} {fuse_H_section_origin[i,1]} {fuse_H_section_origin[i,2]} {fuse_H_section_chord[i,0]}     0    1   0', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)
            # H_fuse 2
            print('SURFACE', file = f)
            print('H-fuselage 2', file = f)
            print(f'{fuse_geo_char[6]}    1   {fuse_geo_char[7]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {-distance_two_fuse/2}   0', file = f)
            print('#', file = f)
            for i in range(3):
                print(f'#------------------------- {i+1}st section -------------------------', file = f)
                print('SECTION', file=f)
                print('#Xle   Yle    Zle      Chord   Ainc  Nspanwise  Sspace', file = f)
                print(f'{fuse_H_section_origin[i,0]} {fuse_H_section_origin[i,1]} {fuse_H_section_origin[i,2]} {fuse_H_section_chord[i,0]}     0    1   0', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)
            #V_Fuselage 1
            print('SURFACE', file = f)
            print('V-fuselage 1', file = f)
            print(f'{fuse_geo_char[6]}    1   {fuse_geo_char[7]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('NOWAKE', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {distance_two_fuse/2}   0', file = f)
            print('#', file = f)
            for i in range(3):
                print(f'#------------------------- {i+1}st section -------------------------', file = f)     
                print('SECTION', file=f)
                print('#Xle   Yle    Zle      Chord   Ainc  Nspanwise  Sspace', file = f)
                print(f'{fuse_V_section_origin[i,0]} {fuse_V_section_origin[i,1]} {fuse_V_section_origin[i,2]} {fuse_V_section_chord[i,0]}      0    1   0', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)
            #V_Fuselage 2
            print('SURFACE', file = f)
            print('V-fuselage 2', file = f)
            print(f'{fuse_geo_char[6]}    1   {fuse_geo_char[7]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('NOWAKE', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {-distance_two_fuse/2}   0', file = f)
            print('#', file = f)
            for i in range(3):
                print(f'#------------------------- {i+1}st section -------------------------', file = f)     
                print('SECTION', file=f)
                print('#Xle   Yle    Zle      Chord   Ainc  Nspanwise  Sspace', file = f)
                print(f'{fuse_V_section_origin[i,0]} {fuse_V_section_origin[i,1]} {fuse_V_section_origin[i,2]} {fuse_V_section_chord[i,0]}      0    1   0', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)    
                       
        #Forward Wing
        if f_aspect == 0:
            pass
        else:
            print('SURFACE', file=f)
            print('Forward Wing', file=f)
            print('#Horseshoes Vortex Distribution', file = f)
            if f_S >= a_S:
                print('13    1   19     -2            | Nchord  Cspace    Nspan Sspace ', file=f)
            else:
                print('7    1   9     -2            | Nchord  Cspace    Nspan Sspace ', file=f)
            print('#', file = f)
            print('# reflect image wing about y=0 plane', file = f)
            print('YDUPLICATE', file = f)
            print('0', file = f)
            print('# twist angle bias for whole surface', file = f)
            print('ANGLE', file=f)
            print(f'{f_teta}', file=f)
            print('#', file = f)
            print('# x,y,z bias for whole surface', file = f)
            print('TRANSLATE', file=f)
            print('0     0     0', file=f)
            print('#', file = f)
            print(f'#------------------------- {1} section -------------------------', file = f)
            print('#   Xle        Yle        Zle         chord       angle', file = f)
            print('SECTION', file=f)
            print(f'{f_w_x[0]} {f_w_y[0]} {f_w_z[0]} {f_w_chord[0]} {f_Ain[0]}', file=f)
            print('NACA', file=f)
            if f_S >= a_S:
                print('2212', file=f)
            else:
                print('0012', file=f)
            print('#', file = f)
            print(f'#------------------------- {2} section -------------------------', file = f)
            print('#   Xle        Yle        Zle         chord       angle', file = f)
            print('SECTION', file=f)
            print(f'{f_w_x[1]} {f_w_y[1]} {f_w_z[1]} {f_w_chord[1]} {f_Ain[1]}', file=f)
            print('NACA', file=f)
            if f_S >= a_S:
                print('2212', file=f)
            else:
                print('0012', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)
            
        #Aftward Wing
        if a_aspect == 0:
            pass
        else:
            print('SURFACE', file = f)
            print('Aftward Wing', file = f)
            if f_S > a_S:
                print('7    1   9     -2            | Nchord  Cspace    Nspan Sspace ', file=f)
            else:
                print('13    1   19     -2            | Nchord  Cspace    Nspan Sspace ', file=f)
            print('# reflect image wing about y=0 plane', file = f)
            print('YDUPLICATE', file = f)
            print('0', file = f)
            print('# twist angle bias for whole surface', file = f)
            print('ANGLE', file=f)
            print(f'{a_teta}', file=f)
            print('#', file = f)
            print('# x,y,z bias for whole surface', file = f)
            print('TRANSLATE', file=f)
            print(f'{a_w_loc}     0     0', file=f)
            print('#', file = f)
            print('#------------------------- 1st section -------------------------', file = f)
            print('#   Xle        Yle        Zle         chord       angle', file = f)
            print('SECTION', file=f)
            print(f'{a_w_x[0]} {a_w_y[0]} {a_w_z[0]} {a_w_chord[0]} {a_Ain[0]}', file=f)
            print('NACA', file=f)
            if f_S > a_S:
                print('0012', file=f)
            else:
                print('2212', file=f)
            print('#', file = f)
            print('#------------------------- 2nd section -------------------------', file = f)
            print('#   Xle        Yle        Zle         chord       angle', file = f)
            print('SECTION', file=f)
            print(f'{a_w_x[1]} {a_w_y[1]} {a_w_z[1]} {a_w_chord[1]} {a_Ain[1]}', file=f)
            print('NACA', file=f)
            if f_S > a_S:
                print('0012', file=f)
            else:
                print('2212', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f) 
            
        # Vertical Wing
        if v_aspect == 0:
            pass
        else:
            print('SURFACE', file = f)
            print('Vertical Wing', file = f)
            print('7    1   9    -2  |Nchord  Cspace    Nspan    Sspace', file = f)
            print('#', file = f)
            print('# twist angle bias for whole surface', file = f)
            print('ANGLE', file=f)
            print(f'{v_teta}', file=f)
            print('#', file = f)
            print('# x,y,z bias for whole surface', file = f)
            print('TRANSLATE', file=f)
            print(f'{v_w_loc}     0     0', file=f)
            print('#', file = f)
            print('#------------------------- 1st section -------------------------', file = f)
            print('#   Xle        Yle        Zle         chord       angle', file = f)
            print('SECTION', file=f)
            print(f'{v_w_x[0]} {v_w_y[0]} {v_w_z[0]} {v_w_chord[0]} {v_Ain[0]}', file=f)
            print('NACA', file=f)
            print('0012', file=f)
            print('#', file = f)
            print('#------------------------- 2nd section -------------------------', file = f)
            print('#   Xle        Yle        Zle         chord       angle', file = f)
            print('SECTION', file=f)
            print(f'{v_w_x[1]} {v_w_y[1]} {v_w_z[1]} {v_w_chord[1]} {v_Ain[1]}', file=f)
            print('NACA', file=f)
            print('0012', file=f)
            print('#', file = f)
            print('#==============================================================', file = f)
            print('#', file = f)          
            
        print(f'#------------ Created by Hoang Van Hung, {datetime.date(datetime.now())} ------------', file = f)  
    return 0

def avl_run(flight_cond, kk):
    V = flight_cond[0]
    alpha = flight_cond[1]
    H = flight_cond[2]
    beta = flight_cond[3]
    
    tmp_path = os.path.join(AUTO_FULL_ROOT, f'Auto_full_{kk}', 'tmp')
    os.makedirs(tmp_path, exist_ok=True)
    avl_path = os.path.join(MODULE_DIR, 'avl.exe')
    if not os.path.exists(avl_path):
        avl_path = os.path.join(MODULE_DIR, 'avl')
    geo_file = f'Aircraft_{kk}.avl'
    out_file = os.path.join(tmp_path, 'calc.txt')
    if os.path.exists(out_file):
        return out_file
    k = 1.4
    R = 287
    T0 = 288
    T_H = T0 - 0.0065*H
    a = math.sqrt(k*R*T_H)
    Mach = V/a
    comm_lines = [
        f'load {geo_file}',
        'oper',
        'm',
        'mn',
        f'{Mach}',
        'v',
        f'{V}',
        '',
        f'a a {alpha}',
        f'b b {beta}',
        'x',
        'ft',
        'calc.txt',
        'o',
        '',
        'quit',
        '',
    ]
    comm_string = '\r\n'.join(comm_lines)
    with open(os.devnull, 'w')  as FNULL:
        try:
            process = subprocess.Popen(avl_path, stdin=subprocess.PIPE, stdout=FNULL, stderr=FNULL, shell=True, cwd=tmp_path)
            process.communicate(bytes(comm_string, encoding='utf8'))
        except subprocess.CalledProcessError:
            print('ERROR')     
    geo_abs = os.path.join(tmp_path, geo_file)
    if os.path.exists(geo_abs):
        os.remove(geo_abs)
    return out_file

#Calculate coefficients
def aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, flight_cond, kk, center_mass=0):
    f_S = f_w_geo_char[6]
    a_S = a_w_geo_char[6]
    
    if center_mass == 0:
        center_mass_abs = 0
    else:
        if f_S >= a_S:
            mac = ref_dim_lift_surface(f_w_geo_char)
        else:
            mac = ref_dim_lift_surface(a_w_geo_char)

        center_mass_abs = center_mass * mac
        
    avl_file(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, kk, center_mass_abs)
    out_file = avl_run(flight_cond, kk)     
    cy = get_value(out_file, 'CLtot')
    cxi = get_value(out_file, 'CDtot')
    mz = get_value(out_file, 'Cmtot')
    mx = get_value(out_file, 'Cltot')
    #mx = -mx
    my = get_value(out_file, 'Cntot')
    my = -my
    cx0 = parasite_drag(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, flight_cond[0])
    cx = cxi + cx0
    os.remove(out_file)
    return cx, cy, mx, my, mz

def parasite_drag_body(fuse_geo_char, V, H=0):
    if fuse_geo_char[1] == 0:
        Cxp_body = 0
    else:
        body_definition = body_def(fuse_geo_char)
        beta_tail = np.arctan(1/2/fuse_geo_char[2])
        Mach, Re = Mach_Reynolds_number(V, body_definition[1], H)
        xt_body = (fuse_geo_char[0]/(fuse_geo_char[0]+fuse_geo_char[1]+fuse_geo_char[2])+1.5/(5+Re*10**-6))*(1+0.15*Mach**(2/3))
        cf_body = 0.087*(1-xt_body)/((np.log10(Re)-1.6)**2)+1.33*np.sqrt(xt_body)/np.sqrt(Re)
        eta_lamda = 1+0.5*(2-xt_body)/(fuse_geo_char[0]+fuse_geo_char[1]+fuse_geo_char[2])+1.5/((fuse_geo_char[0]+fuse_geo_char[1]+fuse_geo_char[2])**2)
        eta_M_body = (1/np.sqrt(1+0.2*Mach**2)+0.055*(xt_body**2)*Mach)*(1+2*Mach*(fuse_geo_char[0]+fuse_geo_char[1]+fuse_geo_char[2])/(1+(fuse_geo_char[0]+fuse_geo_char[1]+fuse_geo_char[2])**2))
        Cxp_TB = cf_body*eta_lamda*eta_M_body*3.8*(fuse_geo_char[0]+fuse_geo_char[1]+fuse_geo_char[2])
        delta_Cxp_body = (0.04/np.sqrt(Cxp_TB))*(np.tan(0.5*beta_tail))**(3/2)
        Cxp_body = Cxp_TB + delta_Cxp_body

    return Cxp_body

def parasite_drag_lift_surface(lift_surface_geo_char, V, H=0):
    aspect = lift_surface_geo_char[0]
    sweep = lift_surface_geo_char[1]
    thickness = 0.12
    x_thickness = 0.3
    
    if aspect == 0:
        Cxp_lift = 0
    else:
        mac = ref_dim_lift_surface(lift_surface_geo_char)
        Mach, Re = Mach_Reynolds_number(V, mac, H)
        x_t0 = thickness * x_thickness/ (thickness + 0.02) + 0.95 / (Re*10**-6 + 2.4)
        k_m = 1 + 0.35 * (Mach ** 0.5)
        k_sweep = (1 - 0.6 * np.sin(np.deg2rad(sweep)) ** 2) * np.cos(np.deg2rad(sweep)) ** 2    
        x_t = x_t0 * k_m * k_sweep
        eta_c = 1 + 2*thickness*np.e**(-2.4*x_t) + 9*thickness**2*np.e**(-4*x_t)
        eta_M = (1/(1 + 0.2*Mach**2)**0.5 + 0.055*x_t**2*Mach)*(1 + 5*thickness*Mach)
        cf = (0.087/(np.log10(Re) - 1.6)**2)*(1 - x_t) + 1.33/Re**0.5*x_t**0.5
        Cxp_lift = 2*cf*eta_c*eta_M
        
    return Cxp_lift
'''
def parasite_drag_add(lift_surface_geo_char, fuse_geo_char):
    if lift_surface_geo_char[0] == 0 or fuse_geo_char[1] == 0:
        delta_Cxp_body = 0
    else:
        k_int = 0.15
        lift_surface_definition = lift_surface_def(lift_surface_geo_char)
        body_defifinition = body_def(fuse_geo_char)
        delta_S = (lift_surface_definition[1]*(2-body_defifinition[0]/lift_surface_definition[0])+lift_surface_definition[2]*body_defifinition[0]/lift_surface_definition[0])*body_defifinition[0]/2
        delta_Cxp_body = k_int*delta_S/lift_surface_geo_char[6]
    return delta_Cxp_body
'''                            
def parasite_drag(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, scheme_fuse, V, H=0):
    f_S = f_w_geo_char[6]
    a_S = a_w_geo_char[6]
    v_S = v_w_geo_char[6]
    S_sum = f_S + a_S
    Cxp_body = parasite_drag_body(fuse_geo_char, V, H)
    Cxp_f_wing = parasite_drag_lift_surface(f_w_geo_char, V, H)
    Cxp_a_wing = parasite_drag_lift_surface(a_w_geo_char, V, H)
    Cxp_v_wing = parasite_drag_lift_surface(v_w_geo_char, V, H)
    '''
    delta_Cxp_body_f_wing1 = parasite_drag_add(f_w_geo_char, fuse_geo_char)
    delta_Cxp_body_f_wing = Cxp_f_wing*delta_Cxp_body_f_wing1
    delta_Cxp_body_a_wing1 = parasite_drag_add(a_w_geo_char, fuse_geo_char)
    delta_Cxp_body_a_wing = Cxp_a_wing*delta_Cxp_body_a_wing1
    '''
    body_definition = body_def(fuse_geo_char)
    # flow stagnation coefficient
    # due to small flow rate
    k_T_f_wing = 1
    k_T_a_wing = 1
    if scheme_fuse == 3 or scheme_fuse == 2:
        Cx0 = Cxp_body*body_definition[2]/S_sum + k_T_f_wing*Cxp_f_wing*f_S/S_sum + k_T_a_wing*Cxp_a_wing*a_S/S_sum + Cxp_v_wing*v_S/S_sum
    else:
        Cx0 = 2*Cxp_body*body_definition[2]/S_sum + k_T_f_wing*Cxp_f_wing*f_S/S_sum + k_T_a_wing*Cxp_a_wing*a_S/S_sum + Cxp_v_wing*v_S/S_sum
    Cx0 = 1.3*Cx0
    return Cx0
                

def get_value(output_file, variable_name):
    var = '{}'.format(variable_name)
    with open(output_file,'r') as ex:
        for line in ex:
            line = line.replace('     ','/')
            line = line.replace('|','/')
            if var in line:
                line_wo_space = line.replace(' ', '')
                coeficient_list = line_wo_space.split('/')
                for data in coeficient_list:
                    if var in data:
                        coef = data.split('=')
                        coef = coef[1].strip('\n')
                        if coef == '**********':
                            return 0
                        return float(coef)
def vol_coeff(f_w_geo_char, a_w_geo_char, center_mass):
    f_sweep = f_w_geo_char[1]
    f_taper = f_w_geo_char[2]
    f_S = f_w_geo_char[6]
    
    a_aspect = a_w_geo_char[0]
    a_sweep = a_w_geo_char[1]
    a_taper = a_w_geo_char[2]
    a_S = a_w_geo_char[6]
    a_L = a_w_geo_char[7]
    if a_aspect == 0:
        A = 0
        L = 0
    else:
        f_mac = ref_dim_lift_surface(f_w_geo_char)
        f_span, f_root_chord, f_tip_chord = lift_surface_def(f_w_geo_char)
        a_mac = ref_dim_lift_surface(a_w_geo_char)
        a_span, a_root_chord, a_tip_chord = lift_surface_def(a_w_geo_char)
        # Create funcion para y_mac y x_mac
        f_y_mac = f_span*(f_taper + 2)/6/(f_taper + 1)
        f_x_mac = round(f_y_mac * np.tan(np.deg2rad(f_sweep)), 4)
        a_y_mac = a_span*(a_taper + 2)/6/(a_taper + 1)
        a_x_mac = round(a_y_mac * np.tan(np.deg2rad(a_sweep)), 4)
        if f_S >= a_S:
            center_mass_abs = center_mass * f_mac
            L = (a_L*f_mac - center_mass_abs + a_x_mac + 0.25*a_mac)/f_mac
            A = L*a_S/f_S
        else:
            center_mass_abs =  center_mass * a_mac
            L = (center_mass_abs - f_x_mac - 0.25*f_mac)/a_mac
            A = L*f_S/a_S
    return A, L

