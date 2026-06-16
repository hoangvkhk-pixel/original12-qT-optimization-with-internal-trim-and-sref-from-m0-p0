# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:05:39 2024

@author: Aspirant2
"""
import subprocess
import os
import math
import numpy as np
from datetime import datetime


#Input
def inputValuesAtmospheric(V, H=0):
    ro_0 = 1.225
    ro = ro_0*(1-H/44300)**4.256
    q = np.around(ro * V ** 2 / 2, decimals=4)
    return q

def input_flight_cond(V=0, alpha=0, beta=0, H=0):
    V = np.around(V, decimals=3)
    alpha = np.around(alpha, decimals=3)
    beta = np.around(beta, decimals=3)
    H = np.around(H, decimals=3)
    return [V, alpha, beta, H]

def input_lift_surface_data(aspect=0, sweep=0, taper=0, twist=0, dihedral=0, delta=0, area=0, L=0, z_loc=0, n = 1, Nchord=13, Nspan=19):
    return [aspect, sweep, taper, twist, dihedral, delta, area, L, z_loc, n, Nchord, Nspan]

def input_body_data(aspect_nose=0, aspect_center=0, aspect_tail=0, height_max=0, width_max=0, x_loc=0, n=1, Nchord=30, Nspan=9,):
    return [aspect_nose, aspect_center, aspect_tail, height_max, width_max, x_loc, n, Nchord, Nspan]

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
            if lift_surface_geo_char[9] == 1:
                span = np.sqrt(aspect * area/2)
                tip_chord = 2 * area / span / (1 + taper)
                root_chord = tip_chord * taper 
            else:
                span = np.sqrt(aspect * area/4)
                tip_chord = area / span / (1 + taper)
                root_chord = tip_chord * taper
        else:  # Wing and horizontal
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

def lift_h_tail_sec(wing_geo_char, h_tail_geo_char, v_tail_geo_char, fuse_geo_char):
    h_sweep = h_tail_geo_char[1]
    h_twist = h_tail_geo_char[3]
    h_dihedral = h_tail_geo_char[4]
    h_span, h_root_chord, h_tip_chord = lift_surface_def(h_tail_geo_char)
    mac = ref_dim_lift_surface(wing_geo_char)  
    L2 = h_tail_geo_char[7]*mac
    L3 = v_tail_geo_char[7]*mac
    d_e_body, l_body, S_body, Sxq_body = body_def(fuse_geo_char)
    if l_body > max(L2, L3) and fuse_geo_char[6] ==1:
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
        z_0 = h_span*np.tan(math.radians(-h_dihedral))/2
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

def avl_file(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, center_mass=0):
    d_e_body, l_body, S_body, Sxq_body = body_def(fuse_geo_char)
    
    f_aspect = f_w_geo_char[0]
    f_teta = f_w_geo_char[5]
    f_S = f_w_geo_char[6]
    f_z_loc = f_w_geo_char[8]*d_e_body
    
    a_aspect = a_w_geo_char[0]
    a_teta = a_w_geo_char[5]
    a_S = a_w_geo_char[6]
    a_L = a_w_geo_char[7]
    a_z_loc = a_w_geo_char[8]*d_e_body
    
    v_aspect = v_w_geo_char[0]
    v_teta = v_w_geo_char[5]
    v_L = v_w_geo_char[7]
    
    f_span, f_root_chord, f_tip_chord = lift_surface_def(f_w_geo_char)
    a_span, a_root_chord, a_tip_chord = lift_surface_def(a_w_geo_char)
    v_span, v_root_chord, v_tip_chord = lift_surface_def(v_w_geo_char)
    S_ref = f_S + a_S
    
    v_w_x, v_w_y, v_w_z, v_w_chord, v_Ain = lift_v_tail_sec(v_w_geo_char)
    
    if f_S >= a_S:
        mac = ref_dim_lift_surface(f_w_geo_char)  
        f_w_x, f_w_y, f_w_z, f_w_chord, f_Ain = lift_wing_sec(f_w_geo_char)
        a_w_x, a_w_y, a_w_z, a_w_chord, a_Ain = lift_h_tail_sec(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char)
        h_span, h_root_chord, h_tip_chord = lift_surface_def(a_w_geo_char)
    else:
        mac = ref_dim_lift_surface(a_w_geo_char)
        f_w_x, f_w_y, f_w_z, f_w_chord, f_Ain = lift_h_tail_sec(a_w_geo_char, f_w_geo_char, v_w_geo_char, fuse_geo_char)
        a_w_x, a_w_y, a_w_z, a_w_chord, a_Ain = lift_wing_sec(a_w_geo_char)
        h_span, h_root_chord, h_tip_chord = lift_surface_def(f_w_geo_char)
    a_w_loc = a_L*mac
    v_w_loc = v_L*mac
    
    fuse_H_section_origin, fuse_H_section_chord = body_H_sec(fuse_geo_char)
    fuse_V_section_origin, fuse_V_section_chord = body_V_sec(fuse_geo_char)

    #%% Write file .avl
    usr_path = os.path.expanduser('~')
    tmp_path = os.path.join(usr_path, 'Creat_png_AVL_0/tmp/')
    avl_file = 'Aircraft.avl'
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
        if fuse_geo_char[6] == 1:
            # H_fuse
            print('SURFACE', file = f)
            print('H-fuselage', file = f)
            print(f'{fuse_geo_char[7]}    1   {fuse_geo_char[8]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
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
            print(f'{fuse_geo_char[7]}    1   {fuse_geo_char[8]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
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
            print(f'{fuse_geo_char[7]}    1   {fuse_geo_char[8]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {h_span/2}   0', file = f)
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
            print(f'{fuse_geo_char[7]}    1   {fuse_geo_char[8]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {-h_span/2}   0', file = f)
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
            print(f'{fuse_geo_char[7]}    1   {fuse_geo_char[8]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('NOWAKE', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {h_span/2}   0', file = f)
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
            print(f'{fuse_geo_char[7]}    1   {fuse_geo_char[8]}     1 |Nchord  Cspace    Nspan      Sspace', file = f)
            print('NOWAKE', file = f)
            print('COMPONENT', file = f)
            print('1', file = f)
            print('TRANSLATE', file = f)
            print(f'{fuse_geo_char[5]}    {-h_span/2}   0', file = f)
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
            print(f'{f_w_geo_char[10]}    1   {f_w_geo_char[11]}     -2            | Nchord  Cspace    Nspan Sspace ', file=f)
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
            print(f'0     0     {f_z_loc}', file=f)
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
            print(f'{a_w_geo_char[10]}    1   {a_w_geo_char[11]}    -2  |Nchord  Cspace    Nspan    Sspace', file = f)
            print('# reflect image wing about y=0 plane', file = f)
            print('YDUPLICATE', file = f)
            print('0', file = f)
            print('# twist angle bias for whole surface', file = f)
            print('ANGLE', file=f)
            print(f'{a_teta}', file=f)
            print('#', file = f)
            print('# x,y,z bias for whole surface', file = f)
            print('TRANSLATE', file=f)
            print(f'{a_w_loc}     0     {a_z_loc}', file=f)
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
        #Vertical Wing
        if v_aspect == 0:
            pass
        else:
            if v_w_geo_char[9] == 1:
                print('SURFACE', file = f)
                print('Vertical Wing', file = f)
                print(f'{v_w_geo_char[10]}    1   {v_w_geo_char[11]}    -2  |Nchord  Cspace    Nspan    Sspace', file = f)
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
            else:
                print('SURFACE', file = f)
                print('Vertical Wing 1', file = f)
                print(f'{v_w_geo_char[10]}    1   {v_w_geo_char[11]}    -2  |Nchord  Cspace    Nspan    Sspace', file = f)
                print('#', file = f)
                print('# twist angle bias for whole surface', file = f)
                print('ANGLE', file=f)
                print(f'{v_teta}', file=f)
                print('#', file = f)
                print('# x,y,z bias for whole surface', file = f)
                print('TRANSLATE', file=f)
                print(f'{v_w_loc}     {h_span/2}     0', file=f)
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
                
                print('SURFACE', file = f)
                print('Vertical Wing 2', file = f)
                print(f'{v_w_geo_char[10]}    1   {v_w_geo_char[11]}    -2  |Nchord  Cspace    Nspan    Sspace', file = f)
                print('#', file = f)
                print('# twist angle bias for whole surface', file = f)
                print('ANGLE', file=f)
                print(f'{v_teta}', file=f)
                print('#', file = f)
                print('# x,y,z bias for whole surface', file = f)
                print('TRANSLATE', file=f)
                print(f'{v_w_loc}     {-h_span/2}     0', file=f)
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

def avl_run(flight_cond):
    V = flight_cond[0]
    alpha = flight_cond[1]
    beta = flight_cond[2]
    H = flight_cond[3]
    
    usr_path = os.path.expanduser('~')
    tmp_path = os.path.join(usr_path, 'Creat_png_AVL_0/tmp/')
    avl_path = os.path.join(usr_path, 'Avl/bin/avl')     
    geo_file = os.path.join(tmp_path, 'Aircraft.avl')
    out_file = os.path.join(tmp_path, 'calc.txt')
    if os.path.exists(out_file):
        return out_file
    k = 1.4
    R = 287
    T0 = 288
    T_H = T0 - 0.0065*H
    a = math.sqrt(k*R*T_H)
    Mach = V/a
    comm_string = f'load {geo_file}\n oper\n m\n mn\n {Mach}\n v\n {V}\n \n a a {alpha}\n b b {beta}\n x\n ft\n{out_file}\n \n \n quit'
    with open(os.devnull, 'w')  as FNULL:
        try:
            process = subprocess.Popen([avl_path], stdin=subprocess.PIPE, stdout = FNULL, shell=True)
            process.communicate(bytes(comm_string, encoding='utf8'))
        except subprocess.CalledProcessError:
            print('ERROR')     
    #os.remove(geo_file)     
    return out_file

#Calculate coefficients
def aero_calc(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, flight_cond, center_mass=0):
    f_sweep = f_w_geo_char[1]
    f_taper = f_w_geo_char[2]
    f_S = f_w_geo_char[6]
    
    a_sweep = a_w_geo_char[1]
    a_taper = a_w_geo_char[2]
    a_S = a_w_geo_char[6]
    a_L = a_w_geo_char[7]
    
    f_span, f_root_chord, f_tip_chord = lift_surface_def(f_w_geo_char)
    a_span, a_root_chord, a_tip_chord = lift_surface_def(a_w_geo_char)
    v_span, v_root_chord, v_tip_chord = lift_surface_def(v_w_geo_char)
    '''
    if f_S >= a_S:
        mac = ref_dim_lift_surface(f_w_geo_char)
        # Create funcion para y_mac y x_mac
        f_y_mac = f_span*(f_taper + 2)/6/(f_taper + 1)
        f_x_mac = round(f_y_mac * np.tan(np.deg2rad(f_sweep)), 4)
        center_mass_abs = f_x_mac + center_mass * mac
    else:
        mac = ref_dim_lift_surface(a_w_geo_char)
        # Create funcion para y_mac y x_mac
        a_y_mac = a_span*(a_taper + 2)/6/(a_taper + 1)
        a_x_mac = round(a_y_mac * np.tan(np.deg2rad(a_sweep)), 4)
        center_mass_abs =  a_L*mac + a_x_mac + center_mass * mac
    '''  
    if center_mass == 0:
        center_mass_abs = 0
    else:
        if f_S >= a_S:
            mac = ref_dim_lift_surface(f_w_geo_char)
            # Create funcion para y_mac y x_mac
            f_y_mac = f_span*(f_taper + 2)/6/(f_taper + 1)
            f_x_mac = round(f_y_mac * np.tan(np.deg2rad(f_sweep)), 4)
            center_mass_abs = f_x_mac + center_mass * mac
        else:
            mac = ref_dim_lift_surface(a_w_geo_char)
            # Create funcion para y_mac y x_mac
            a_y_mac = a_span*(a_taper + 2)/6/(a_taper + 1)
            a_x_mac = round(a_y_mac * np.tan(np.deg2rad(a_sweep)), 4)
            center_mass_abs =  a_L*mac + a_x_mac + center_mass * mac
    
    avl_file(f_w_geo_char, a_w_geo_char, v_w_geo_char, fuse_geo_char, center_mass_abs)
    out_file = avl_run(flight_cond)     
    cy = get_value(out_file, 'CLtot')
    cx = get_value(out_file, 'CDtot')
    mz = get_value(out_file, 'Cmtot')
    my = get_value(out_file, 'Cntot')
    mx = get_value(out_file, 'Cltot')
    os.remove(out_file)
    return cx, cy, mx, my, mz

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