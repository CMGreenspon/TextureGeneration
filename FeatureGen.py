'''
    TextureGenerationSuite - CreateFeature

    Contains functions for the parametric creation of individual features.
    These features can be tiled on a strip or placed onto a sphere.
'''

#%% Imports
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#%% Feature creation
def create_cone(height, width, trunc_val, axis):
    radius = width / 2
    if trunc_val is None:
        trunc_dist = 0
    else:
        trunc_dist = trunc_val / 2
    
    feature_mesh = np.zeros((len(axis), len(axis)))
        
    for x in range(len(axis)):
        for y in range(len(axis)):
            distance = np.sqrt(axis[x]**2 + axis[y]**2)
            if distance < trunc_dist:
                feature_mesh[x,y] = np.nan
            elif distance > trunc_dist and distance < radius:
                feature_mesh[x,y] = 1 - ((distance-trunc_dist) / (radius-trunc_dist))
            elif distance > radius*1.1:
                feature_mesh[x,y] = np.nan

    
    feature_mesh = feature_mesh * height
    
    return feature_mesh

def create_dot(size, trunc_val, axis):
    radius = size
    if trunc_val is None:
        trunc_radius = 0
    
    feature_mesh = np.zeros((len(axis), len(axis)))    
    for x in range(len(axis)):
        for y in range(len(axis)):
            distance = np.sqrt(axis[x]**2 + axis[y]**2)
            rd = distance / radius
            if rd < trunc_radius:
                feature_mesh[x,y] = np.nan
            elif rd < 1:
                theta = np.arccos(rd)
                feature_mesh[x,y] = np.sin(theta) * radius
            elif rd < 1.05:
                feature_mesh[x,y] = 0
            else:
                feature_mesh[x,y] = np.nan
        
    if not trunc_val is None:
        h_idx = feature_mesh > trunc_val
        feature_mesh[h_idx] = trunc_val

    return feature_mesh


def create_feature(feature_type, feature_options):
    resolution = feature_options['resolution']
    width = abs(feature_options['width'])
    height = feature_options['height']
    trunc_val = feature_options['trunc_val']
    
    spatial_res = resolution/1000
    
    if feature_type == 'cone':
        extents = (width/2) * 1.05
        axis_length = int(np.ceil(extents / spatial_res))
        axis = np.linspace(-extents, extents, axis_length)
        
        feature_mesh = create_cone(height, width, trunc_val, axis)
        
    elif feature_type == 'dot':
        extents = abs(width) * 1.05
        axis_length = int(np.ceil(extents / spatial_res))
        axis = np.linspace(-extents, extents, axis_length)
        
        feature_mesh = create_dot(width, trunc_val, axis)
        if feature_options['width'] < 0:
            feature_mesh = feature_mesh * -1
            
    elif feature_type == 'gaussian':
        print('not ready')
    
    elif feature_type == 'cross':
        print('not ready')
        
    return feature_mesh, axis


def strip_nan(feature_mesh, axis):
    xx, yy = np.meshgrid(axis, axis)
    feature_points = np.transpose(np.vstack((np.reshape(xx, -1), np.reshape(yy, -1), np.reshape(feature_mesh, -1))))
    feature_nan_idx = np.isnan(np.reshape(feature_mesh, -1))
    feature_points = feature_points[~feature_nan_idx, :]
    
    return feature_points


#%% Example 1 - Make a feature
'''
feature_type = 'dot'
feature_options = {
    'height': 2, # Height of feature
    'width': 3, # The size of the feature along the longest axis (overwrites height for dot)
    'trunc_val': None, # Overrides height command and truncates values according to feature
    'resolution': 25} # microns

feature_mesh, axis = create_feature(feature_type, feature_options)
plot_feature = 1

if plot_feature:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx, yy = np.meshgrid(axis, axis)
    surf = ax.plot_surface(xx, yy, feature_mesh, linewidth=0, antialiased=False)
'''