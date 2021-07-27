# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 07:15:00 2021

@author: somlab
"""

'''
    TextureGenerationSuite - CreateSphere

    Functions for:
        1. creating a sphere
        2. optimizing point location by number
        3. optimizing point location by target distance
'''

import math
import numpy as np
from scipy.optimize import minimize, fmin, minimize_scalar, dual_annealing
import open3d as o3d
import cv2
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as mpl_pp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


#%% Latitude & Longitude to Cartesian
def ThetaPhi2xyz(ThetaPhi,r,method='xyz'):
    xyz = np.zeros((ThetaPhi.shape[0],3))
    
    if method == 'xyz':
        for i in range(ThetaPhi.shape[0]):
            theta = ThetaPhi[i,0]
            phi = ThetaPhi[i,1]
            xp = r*np.sin(theta)*np.cos(phi)
            yp = r*np.sin(theta)*np.sin(phi)
            zp = r*np.cos(theta)
            
            xyz[i,:] = [xp,yp,zp]
            
    elif method == 'latlong': # Rotates around X plane (latitude) and then Z plane (longitude)
        xyz[:,2] = r
        for i in range(ThetaPhi.shape[0]):
            theta = ThetaPhi[i,0]
            phi = ThetaPhi[i,1]
            
            temp = np.transpose(deepcopy(xyz[i,:]))
         
            rot_mat_x = np.array([[1, 0, 0],
                                  [0, np.cos(theta), -np.sin(theta)],
                                  [0, np.sin(theta), np.cos(theta)]])
        
            rot_mat_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                                  [np.sin(phi), np.cos(phi), 0],
                                  [0, 0, 1]])
    
            temp = np.matmul(np.transpose(rot_mat_x), temp)
            temp = np.matmul(np.transpose(rot_mat_z), temp)
    
            xyz[i,:] = np.transpose(temp)
            
    return xyz

#%% Point distances
def xyzSphereDist(xyz, r):
    n_features = xyz.shape[0]
    pdm = np.zeros((n_features,n_features)) # Point distance matrix
    for i in range(n_features):
        for k in range(n_features):
            arc_dist = arc_distance(xyz[i,:], xyz[k,:], r)
            pdm[i,k] = arc_dist
            
    return pdm

def xyzSphereDist_SinglePoint(xyz, r):
    n_features = xyz.shape[0]
    pdv = np.zeros((n_features,1)) # Point distance vector
    
    xyz_ref = xyz[0,:]
    for i in range(n_features):
        arc_dist = arc_distance(xyz_ref, xyz[i,:], r)
        pdv[i] = arc_dist
            
    return pdv[1:]

def arc_distance(xyz1, xyz2, r):
    # Euclidean distacne
    dist = math.sqrt((xyz1[0]-xyz2[0])**2+(xyz1[1]-xyz2[1])**2+(xyz1[2]-xyz2[2])**2)
    sin_dist = dist/2/r # Circumferential distance
    if sin_dist > 1: # Error check
        sin_dist = 1
        
    phi = math.asin(sin_dist) # Angle necessary to produce circumferential distance at radius
    arc_dist = 2*phi*r # Length of arc between two points of the calculated angle
    if arc_dist == 0:
        arc_dist = np.nan
    
    return arc_dist

def get_min_dist(xyz, r):
    pdm = xyzSphereDist(xyz, r)
    min_dist_vec = np.sort(pdm,axis=0)[0,:]
    min_dist_mean = np.mean(min_dist_vec)
    min_dist_std = np.std(min_dist_vec)
    
    return min_dist_mean, min_dist_std

def feature_arc_offset(sphere_radius, feature_width):
    circ = sphere_radius * np.pi * 2
    circ_prop = feature_width / circ
    feature_rad = circ_prop * 2 * math.pi * 2
    arc_offset = 1 - math.cos(feature_rad)
    
    return arc_offset

#%% Point optimization
def nn_ThetaPhi_loss(pdm): # Loss based on nearest neighbor
    sorted_dist_mat = np.sort(pdm,axis=0)
    loss = (1 / np.nanmean(sorted_dist_mat[0,:])) * (pdm.shape[0]**2)
    #loss = (1 / np.percentile(sorted_dist_mat[0,:],5)) * (pdm.shape[0]**2)
        
    return loss

def ThetaPhiLoss(ThetaPhi, r):
    ThetaPhi = np.reshape(ThetaPhi, [int(np.size(ThetaPhi)/2),2])
    xyz = ThetaPhi2xyz(ThetaPhi,r)
    pdm = xyzSphereDist(xyz,r)
    loss = nn_ThetaPhi_loss(pdm)
        
    return loss

def optimize_ThetaPhi(feature_ThetaPhi, sphere_radius, method = 'local'):
    n_features = feature_ThetaPhi.shape[0]
    pi_lims = (0, 2*np.pi)
    ThetaPhi_bounds = ((pi_lims, ) * (n_features*2))
    ThetaPhi_long = np.reshape(feature_ThetaPhi, [np.size(feature_ThetaPhi),1])
    
    if method == 'local':
        minimize_opts={'maxiter': 1e10, 'disp': True}
        res = minimize(ThetaPhiLoss, ThetaPhi_long, bounds=ThetaPhi_bounds, options=minimize_opts, args=(sphere_radius,))
        optimized_ThetaPhi = res.x
        optimized_ThetaPhi = np.reshape(optimized_ThetaPhi, [int(np.size(optimized_ThetaPhi)/2),2])
        
    elif method == 'global':
        res = dual_annealing(ThetaPhiLoss, ThetaPhi_bounds, args=(sphere_radius,))
        
    ThetaPhiLoss(optimized_ThetaPhi, sphere_radius)
    
    return optimized_ThetaPhi

def create_optimized_points(sphere_radius, n_features, verbose=False, method = 'local'):
    original_ThetaPhi, original_points = create_sphere(sphere_radius, n_features, trunc = True)
    optimized_ThetaPhi = optimize_ThetaPhi(original_ThetaPhi, sphere_radius, method = method)
    optimized_points = ThetaPhi2xyz(optimized_ThetaPhi, sphere_radius)
    
    if verbose:
        orig_mean, orig_std = get_min_dist(original_points, sphere_radius)
        optim_mean, optim_std = get_min_dist(optimized_points, sphere_radius)
        
        print('Init distance = ' + str(np.round(orig_mean,3)) + ' +/- ' + str(np.round(orig_std,3)))
        print('Opim distance = ' + str(np.round(optim_mean,3)) + ' +/- ' + str(np.round(optim_std,3)))
    
    return optimized_ThetaPhi, optimized_points
    

#%% Sphere creation
def create_sphere(radius, n_features=1000, trunc=False, method='xyz'):
    #theta_phi = np.zeros((n_features,2))
    theta_phi = []
    
    alpha = 4.0*np.pi/n_features
    d = np.sqrt(alpha)
    m_theta = int(np.ceil(np.pi/d))
    d_theta = np.pi/m_theta
    d_phi = alpha/d_theta
    count = 0
    for m in range (0,m_theta):
        theta = np.pi*(m+0.5)/m_theta # Latitude
        m_phi = int(np.ceil(2*np.pi*np.sin(theta)/d_phi))
        for n in range (0,m_phi):
            phi = 2*np.pi*n/m_phi # Longitude
            theta_phi.append([theta,phi])

    theta_phi = np.vstack(theta_phi)
    
    if trunc: # Randomize which points are deleted to prevent imbalance and make optimization easier
        point_diff = theta_phi.shape[0] - n_features
        idx = np.random.permutation(np.arange(theta_phi.shape[0]))
        del_idx = idx[:point_diff]
        theta_phi = np.delete(theta_phi,del_idx,0)
        
    
    points = ThetaPhi2xyz(theta_phi, radius, method)
    
    return theta_phi, points

def rotate_feature(feature_points_offset, theta, phi):
    temp = np.transpose(deepcopy(feature_points_offset))
    rot_mat_x = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
    
    rot_mat_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                          [np.sin(phi), np.cos(phi), 0],
                          [0, 0, 1]])
    
    temp = np.matmul(np.transpose(rot_mat_x), temp)
    temp = np.matmul(np.transpose(rot_mat_z), temp)
    rotated_feature_points = np.transpose(temp)
    
    return rotated_feature_points

def remove_overlapping_points_from_sphere(base_sphere, sphere_radius, optimized_points, feature_radius, margin=1.05):
    reduced_sphere = deepcopy(base_sphere)
    for f in range(optimized_points.shape[0]):
        point_mat = np.vstack((optimized_points[f,:], reduced_sphere))
        pdv = xyzSphereDist_SinglePoint(point_mat, sphere_radius)
        within_bound_logit = pdv < feature_radius*margin
        
        reduced_sphere = reduced_sphere[~np.reshape(within_bound_logit, -1),:]
    
    return reduced_sphere

