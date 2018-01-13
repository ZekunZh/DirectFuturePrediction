#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:57:09 2018

Add position information to the measurement variable.
Assume that z is always 0, meaning that there is no jump, the map is flat.

@author: mingxing
"""

''' Parameters 

Object names = ['Clip', 'CustomMedikit', 'DoomImp', 'DoomPlayer']
Search for zdoom + name for explaination of the name.

Input : 
    labels : list of Labels (state.labels). Class Label see https://github.com/mwydmuch/ViZDoom/blob/master/doc/Types.md#label for more details
    angle : float (degree). Player's orientation

Output : 
    list of position information. We return the 100/sqrt(nearest distance) of 'Clip', 'CustomMedikit'
 and 'DoomImp', and their relative direction (angle in degree), so in total it's a 6-length list.
 
 The reason to choose 100/sqrt(nearest distance) is :
     1 when object is not present, the default value 0 means nearest distance = inf
     2 since original code normalize every measurement by 100, so 100/sqrt(nearest distance) keep the input order to the network approximately 1.
    
'''
import numpy as np
import math

Object_names = ['Clip', 'CustomMedikit', 'DoomImp', 'DoomImpBall']
def dist_info(labels, angle):
    # player's position
    x, y = labels[-1].object_position_x, labels[-1].object_position_y
    if labels[-1].object_name != 'DoomPlayer':
        raise ValueError('DoomPlayer is not at the end of the label {}'.format(labels[-1].object_name))
    dist_info = {}
    for object_ in Object_names:
        dist_info[object_] = (0, 0) # first element means 100/sqrt(nearest distance), second means relative angle in degree
    
    for label in labels[0:-1]:
        x1, y1 = label.object_position_x, label.object_position_y
        name = label.object_name
        if name not in Object_names:
            #raise ValueError('unknow object {}'.format(name))
            continue
        dist = 10000.0/np.sqrt((x1-x)**2 + (y1-y)**2)
        if dist > dist_info[name][0]:
            dist_ = np.sqrt((x1-x)**2 + (y1-y)**2)
            relative_angle = ((math.acos((x1-x)/dist_) * 180.0 / np.pi) * np.sign(y1-y) + 720 - angle) % 360
            relative_angle = relative_angle if relative_angle<180 else relative_angle-360
            dist_info[name] = (dist, relative_angle)
    result = []
    for object_ in Object_names:
        result += dist_info[object_]
    return result 

        
        
        
        
        
        