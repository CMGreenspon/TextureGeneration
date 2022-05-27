"""
Texture Generation Script
@author: Charles Greenspon
"""
import os
import numpy as np
from scipy.spatial import Delaunay
from stl import mesh # requires numpy-stl
from copy import deepcopy

#%% Function inputs
# Required inputs
output_dir = r'C:\Users\somlab\Desktop'
output_mode = 'csv' # csv or stl
feature = 'sine_wave' # cone, sine_wave, square_wave, dot
area = (45,23) # Size in millimeters (scanning direction, width)
height = 1 # Size of bumps or ridges above the base or the amplitude of a sine wave
width = 1.5 # Width of the bumps or ridges (no effect for sine wave or dot)
periodicity = 5 # Spacing between ridges or features. If sine wave this is the 1/frequency.
 
# Optional inputs
trunc = .5
'''If not False then the truncation parameter for the feature.
   Cones or dots: this is the maximum diameter (mm)
   Sine_wave: this can only be true and if so removes all negative values
   Square_wave: no effect'''
offset = 0
'''If 0 then the feature is a regular square pattern. If 1 then the feature is offset by half the
     period. If 2 then the hypotenuse is set to be the periodicity.
'''
base = 1  # Thickness of the base in mm
resolution = 50 # Resolution of the print in um

alignment_marker = False # Insert an alignment marker (L-shaped)
alignment_offset = (5.5,5.5) # How far from the top left corner to insert the alignment_markers in mm
marker_ratio = (0.125, 0.166)

# Create filename
if feature != 'sine_wave':
    filename = '{}x{}mm_{}_h{}_w{}_p{}_r{}um.stl'.format(area[0], area[1], feature, height, width,
                                                         periodicity, resolution)
else:
    filename = '{}x{}mm_{}_h{}_p{}_r{}um.stl'.format(area[0], area[1], feature, height,
                                                     periodicity, resolution)

file_path = os.path.join(output_dir, filename)

fs = 1000/resolution
spatial_res = resolution/1000 # convert to um
x_vec = np.arange(0, area[0], spatial_res)
y_vec = np.arange(0, area[1], spatial_res)


#%% Error checks
if not trunc:
    if trunc > width:
        raise ValueError('trunc cannot be greater than feature width.')

if not marker_ratio == 'auto':
    if type(marker_ratio) == int:
        raise ValueError('Marker ratio must be a float or tuple/list of.')
    elif type(marker_ratio) == float:
        marker_ratio = (marker_ratio,marker_ratio)
    elif type(marker_ratio) == list and len(marker_ratio) == 2:
        marker_ratio = tuple(marker_ratio)
    
    if not len(marker_ratio) == 2:
        raise ValueError('Marker ratio must be a float or tuple/list of.') 

#%% Output function
def output_mesh(faces, vertices, file_path):
    mesh_obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_obj.vectors[i][j] = vertices[f[j],:]
    
    mesh_obj.save(file_path)

#%% Base function
def create_base(vertices, faces, x_vec, y_vec, base):
    front = [[x_vec[0], y_vec[0], 0],
             [x_vec[0], y_vec[0], base],
             [x_vec[0], y_vec[-1], 0],
             [x_vec[0], y_vec[-1], 0],
             [x_vec[0], y_vec[0], base],
             [x_vec[0], y_vec[-1], base]] 
    
    back = [[x_vec[-1], y_vec[-1], 0],
             [x_vec[-1], y_vec[-1], base],
             [x_vec[-1], y_vec[0], 0],
             [x_vec[-1], y_vec[0], 0],
             [x_vec[-1], y_vec[-1], base],
             [x_vec[-1], y_vec[0], base]] 
    
    side1 = [[x_vec[-1], y_vec[0], 0],
              [x_vec[-1], y_vec[0], base],
              [x_vec[0], y_vec[0], 0],
              [x_vec[0], y_vec[0], 0],
              [x_vec[-1], y_vec[0], base],
              [x_vec[0], y_vec[0], base]] 
    
    side2 = [[x_vec[0], y_vec[-1], 0],
              [x_vec[0], y_vec[-1], base],
              [x_vec[-1], y_vec[-1], 0],
              [x_vec[-1], y_vec[-1], 0],
              [x_vec[0], y_vec[-1], base],
              [x_vec[-1], y_vec[-1], base]] 
    
    bottom = [[x_vec[0], y_vec[0], 0],
              [x_vec[0], y_vec[-1], 0],
              [x_vec[-1], y_vec[-1], 0],
              [x_vec[-1], y_vec[-1], 0],
              [x_vec[-1], y_vec[0], 0],
              [x_vec[0], y_vec[0], 0]] 
    
    new_faces = np.arange(len(vertices), len(vertices)+30).reshape((10,3))
    
    vertices = np.vstack((vertices, front, back, side1, side2, bottom))
    faces = np.vstack((faces, new_faces))
    
    return vertices, faces

#%% Main

# Make the texture
if feature == 'cone' or feature == 'dot':
    # Create the base matrix
    mesh_size = np.array(np.round(np.array(area) * fs), dtype=int)
    mesh_elements = np.array(mesh_size, dtype=np.float64)
    mesh_elements = mesh_elements[0] * mesh_elements[1]
    ''' The below lines of code apply to all bump like objects and needs to be moved when other 
        features are added'''
        
    if offset < 2 or offset is None:
        periodicity = (periodicity, periodicity)
    if offset == 2:
        hypotenuse = periodicity
        hyp2 = hypotenuse**2
        x_space = np.sqrt(hyp2/2)*2
        y_space = np.sqrt(hyp2 - (x_space/2)**2)
        periodicity = (y_space, x_space)

    bump_size = (int(round(periodicity[0] * fs)), int(round(periodicity[1] * fs)))
    bump = np.zeros((bump_size[0],bump_size[1]))
    bump.fill(np.nan)
    # Assign height values
    if trunc is False:
        min_dist = 0
    else:
        min_dist = trunc/2
    
    x_dist = np.linspace(-(bump_size[0]*spatial_res)/2,(bump_size[0]*spatial_res)/2, bump_size[0])
    y_dist = np.linspace(-(bump_size[1]*spatial_res)/2,(bump_size[1]*spatial_res)/2, bump_size[1])
    
    if feature == 'cone':
        radius = width / 2
        for x in range(len(x_dist)):
            for y in range(len(y_dist)):
                distance = np.sqrt(x_dist[x]**2 + y_dist[y]**2)
                if distance < min_dist and distance > (min_dist - spatial_res):
                    bump[x,y] = height
                elif distance > min_dist and distance < radius:
                    height_ratio = 1 - ((distance-min_dist) / (radius-min_dist))
                    bump[x,y] = height_ratio * height
                    
    elif feature == 'dot':
        radius = height / 2
        for x in range(len(x_dist)):
            for y in range(len(y_dist)):
                distance = np.sqrt(x_dist[x]**2 + y_dist[y]**2)
                rd = distance / radius
                if rd < 1:
                    theta = np.arccos(rd)
                    z = np.sin(theta)  * radius
                    bump[x,y] = z
                        
    # Pad the bump
    for r in range(bump_size[0]):
        nan_idx = ~np.isnan(bump[r,:])
        if any(nan_idx):
            real_idx = np.where(nan_idx)[0]
            bump[r, real_idx[0]-1] = 0
            bump[r, real_idx[-1]+1] = 0
            
    for c in range(bump_size[1]):
        nan_idx = ~np.isnan(bump[:,c])
        if any(nan_idx):
            real_idx = np.where(nan_idx)[0]
            if not bump[real_idx[0],c] == 0:
                bump[real_idx[0]-1,c] = 0
            if not bump[real_idx[-1],c] == 0:
                bump[real_idx[-1]+1, c] = 0

    bump = bump + base
    # Make the matrix
    n_features = np.ceil(mesh_size / bump_size)
    if offset == 0:
        texture_mat = np.tile(bump, (int(n_features[0]), int(n_features[1])))
    else:
        init_row = np.tile(bump, (1, int(n_features[1])))
        offset_row = np.roll(init_row, int(np.round(bump_size[1]/2)))
        # Check if there are partial bumps on either edge
        if all(np.isnan(offset_row[:,0])) is False:
            # Find the extent of the bump
            temp = np.isnan(offset_row)
            nan_idx = np.all(temp,axis=0)
            t_idx = np.where(nan_idx)[0]
            offset_row[:,:t_idx[0]] = np.nan
        if all(np.isnan(offset_row[:,-1])) is False:
            temp = np.isnan(offset_row)
            nan_idx = np.all(temp,axis=0)
            t_idx = np.where(nan_idx)[0]
            offset_row[:,t_idx[-1]:] = np.nan
        
        combi_row = np.vstack((init_row, offset_row))
        combi_row[0,0], combi_row[0,-1], combi_row[-1,0], combi_row[-1,-1] = base,base,base,base
        texture_mat = np.tile(combi_row, (int(np.floor(n_features[0]/2)), 1))
    
    # Ensure correct shape of the texture mat
    if any(np.shape(texture_mat) != mesh_size):
        offsets = np.array(np.floor((np.shape(texture_mat) - mesh_size) / 2), dtype=int)

        if offsets[0] < 0: # If there is a negative offset then append rows
            rows_to_append = np.zeros((abs(offsets[0]), np.shape(texture_mat)[1]))
            rows_to_append.fill(np.nan)
            texture_mat = np.concatenate((rows_to_append,texture_mat,rows_to_append), axis=0)
        elif offsets[0] > 0:
            texture_mat = texture_mat[offsets[0]:(np.shape(texture_mat)[0]-offsets[0]),:]
        # Col idx
        if offsets[1] < 0:
            cols_to_append = np.zeros((abs(offsets[1]), np.shape(texture_mat)[0]))
            cols_to_append.fill(np.nan)
            texture_mat = np.concatenate((cols_to_append,texture_mat,cols_to_append), axis=1)
        elif offsets[1] > 0:
            texture_mat = texture_mat[:, offsets[1]:(np.shape(texture_mat)[1]-offsets[1])]
    # In case of rounding issue
    texture_mat = texture_mat[0:mesh_size[0], 0:mesh_size[1]] 
    texture_mat[0,0], texture_mat[0,-1], texture_mat[-1,0], texture_mat[-1,-1] = base,base,base,base
    
    # Insert the alignment marker
    if alignment_marker:
        f_x_centers = np.linspace((periodicity[1] - x_dist[-1]),
                                  (n_features[1]*periodicity[1])-x_dist[-1],
                                  int(n_features[1]))
        f_y_centers = np.linspace((periodicity[0] - y_dist[-1]),
                                  (n_features[0]*periodicity[0])-y_dist[-1],
                                  int(n_features[0]))
        
        # Find the correct location to replace bumps with the marker
        x_init = (np.abs(f_x_centers - alignment_offset[1])).argmin() + 1
        y_init = (np.abs(f_y_centers - alignment_offset[0])).argmin() + 1
        
        corner_init_loc = (f_x_centers[x_init], f_y_centers[y_init])
        corner_init_idx = [int(np.abs(x_vec - corner_init_loc[0]).argmin()), int(np.abs(y_vec - corner_init_loc[1]).argmin())]
        
        texture_mat_align = deepcopy(texture_mat)
        if marker_ratio == 'auto':
            bar_sizes = periodicity
        else:
            bar_sizes = (marker_ratio[0] * area[0], marker_ratio[1] * area[1])
            
        bar_sizes_idx = np.divide(bar_sizes,(resolution/1000))
        # Remove bumps that would overlap with alignment marker
        num_markers_to_remove = np.ceil((bar_sizes[0] / periodicity[0], bar_sizes[1] / periodicity[1]))
        blank_init = [corner_init_loc[0] - x_dist[-1], corner_init_loc[1] - y_dist[-1]]
        blank_init_idx = [int(np.abs(x_vec - blank_init[0]).argmin()), int(np.abs(y_vec - blank_init[1]).argmin())]

        x_blank_end = f_x_centers[int(x_init + num_markers_to_remove[1])] + x_dist[-1]
        y_blank_end = f_y_centers[int(y_init + num_markers_to_remove[0])] + y_dist[-1]
        
        x_blank_idx = np.abs(x_vec - x_blank_end).argmin()
        y_blank_idx = np.abs(y_vec - y_blank_end).argmin()
        
        # Blanking
        texture_mat_align[blank_init_idx[0]:x_blank_idx,blank_init_idx[1]:blank_init_idx[1]+len(y_dist)] = np.nan
        texture_mat_align[blank_init_idx[0]:blank_init_idx[0]+len(x_dist),blank_init_idx[1]:y_blank_idx] = np.nan
        
        # Insert alignment marker
        bar_width = x_dist[-1] * 0.75
        bar_width_idx = int(np.ceil(bar_width/2 / (resolution/1000)))
        
        align_max = np.nanmax(texture_mat_align)
        
        texture_mat_align[corner_init_idx[0]:corner_init_idx[0]+int(bar_sizes_idx[0]),
                          corner_init_idx[0]:corner_init_idx[0]+bar_width_idx] = align_max
        texture_mat_align[corner_init_idx[1]:corner_init_idx[1]+bar_width_idx,
                  corner_init_idx[1]:corner_init_idx[1]+int(bar_sizes_idx[1])] = align_max
        # Padding
        texture_mat_align[corner_init_idx[0]-1,corner_init_idx[0]-1:corner_init_idx[0]-1+int(bar_sizes_idx[1])+2] = base
        texture_mat_align[corner_init_idx[0]+bar_width_idx,
                          corner_init_idx[0]+bar_width_idx+1:corner_init_idx[1]-1+int(bar_sizes_idx[1])+2] = base
        texture_mat_align[corner_init_idx[0]+int(bar_sizes_idx[1]),
                          corner_init_idx[0]:corner_init_idx[1]+bar_width_idx] = base
        
        texture_mat_align[corner_init_idx[0]-1:corner_init_idx[0]+int(bar_sizes_idx[1])+1,corner_init_idx[1]-1] = base
        texture_mat_align[corner_init_idx[0]+bar_width_idx:corner_init_idx[0]+int(bar_sizes_idx[0])+1,
                          corner_init_idx[0]+bar_width_idx] = base
        texture_mat_align[corner_init_idx[0]:corner_init_idx[0]+bar_width_idx,
                          corner_init_idx[1]+int(bar_sizes_idx[1])] = base
        
        texture_mat = texture_mat_align
        
    # Create and export the STL
    [X,Y] = np.meshgrid(y_vec, x_vec)
    nan_check = ~np.isnan(texture_mat)
    X_real = X[nan_check]
    Y_real = Y[nan_check]
    X_real = X_real.reshape(len(X_real),1)
    Y_real = Y_real.reshape(len(Y_real),1)
    texture_mat_real = texture_mat[nan_check]
    texture_mat_real = texture_mat_real.reshape(len(texture_mat_real),1)
    
    # Make the vertices and faces
    tri_points = np.hstack((X_real, Y_real))
    vertices = np.hstack((tri_points, texture_mat_real))
    faces = Delaunay(tri_points)
    faces = faces.simplices
    # For some reason x and y are reversed here
    vertices, faces = create_base(vertices, faces, y_vec, x_vec, base) 
    output_mesh(faces, vertices, file_path)
    
elif feature == 'square_wave':
    total_length = width + periodicity
    num_ridges = int(np.floor(area[0]/total_length))
    bh = base+height
    if not alignment_marker:
        # This code is a little long but it produces the smallest possible files
        # How much to increment per ridge/valley pair
        x_inc = np.where(x_vec == total_length)[0][0]
        r_inc = np.where(x_vec == width)[0][0] # Ridge increment
        x_init = int(x_inc/4) # Offset for the first half valley
        # Build the grating
        vertices = []
        # Add a half width valley
        vertices.append([[x_vec[0], y_vec[0], base],
                          [x_vec[x_init], y_vec[0], base],
                          [x_vec[0], y_vec[-1], base],
                          [x_vec[0], y_vec[-1], base],
                          [x_vec[x_init], y_vec[0], base],
                          [x_vec[x_init], y_vec[-1], base]])
        
        for r in range(num_ridges):
            x_idx = (x_inc * r) + x_init
            
            x1 = x_idx
            x2 = x_idx + r_inc
            x3 = x_idx + x_inc
        
            ramp_up = [[x_vec[x1], y_vec[0], base],
                       [x_vec[x1], y_vec[0], bh],
                       [x_vec[x1], y_vec[-1], base],
                       [x_vec[x1], y_vec[-1], base],
                       [x_vec[x1], y_vec[0], bh],
                       [x_vec[x1], y_vec[-1], bh]]
            vertices.append(ramp_up)
        
            ridge = [[x_vec[x1], y_vec[0], bh],
                     [x_vec[x2], y_vec[0], bh],
                     [x_vec[x1], y_vec[-1], bh],
                     [x_vec[x1], y_vec[-1], bh],
                     [x_vec[x2], y_vec[0], bh],
                     [x_vec[x2], y_vec[-1], bh]]
            vertices.append(ridge)
        
            ramp_down = [[x_vec[x2], y_vec[-1], base],
                         [x_vec[x2], y_vec[-1], bh],
                         [x_vec[x2], y_vec[0], base],
                         [x_vec[x2], y_vec[0], base],
                         [x_vec[x2], y_vec[-1], bh],
                         [x_vec[x2], y_vec[0], bh]]
            vertices.append(ramp_down)
        
            side1 = [[x_vec[x1], y_vec[0], base],
                      [x_vec[x2], y_vec[0], base],
                      [x_vec[x2], y_vec[0], bh],
                      [x_vec[x1], y_vec[0], base],
                      [x_vec[x2], y_vec[0], bh],
                      [x_vec[x1], y_vec[0], bh]]  
            vertices.append(side1)
        
            side2 = [[x_vec[x1], y_vec[-1], base],
                     [x_vec[x1], y_vec[-1], bh],
                     [x_vec[x2], y_vec[-1], bh],
                     [x_vec[x2], y_vec[-1], bh],
                     [x_vec[x2], y_vec[-1], base],
                     [x_vec[x1], y_vec[-1], base]]  
            vertices.append(side2)
        
            if r < num_ridges-1:
                valley = [[x_vec[x2], y_vec[0], base],
                          [x_vec[x3], y_vec[0], base],
                          [x_vec[x2], y_vec[-1], base],
                          [x_vec[x2], y_vec[-1], base],
                          [x_vec[x3], y_vec[0], base],
                          [x_vec[x3], y_vec[-1], base]]  
                vertices.append(valley)
        # Add the last half valley
        vertices.append([[x_vec[x2], y_vec[0], base],
                          [x_vec[-1], y_vec[0], base],
                          [x_vec[x2], y_vec[-1], base],
                          [x_vec[x2], y_vec[-1], base],
                          [x_vec[-1], y_vec[0], base],
                          [x_vec[-1], y_vec[-1], base]])
         
        # Export
        vertices = np.vstack(vertices)
        faces = np.arange(len(vertices)).reshape((int(len(vertices)/3),3))
        vertices, faces = create_base(vertices, faces, x_vec, y_vec, base)
        output_mesh(faces, vertices, file_path)
        
    else:
        # Revert to less efficient method
        # Create the base ridge
        ppf = np.abs(x_vec - total_length).argmin() # points per feature
        ppr = np.abs(x_vec - width).argmin() # points per ridge
        offset = int(np.round((ppf-ppr) / 4))
        # Make the ridge
        ridge = np.ones((ppf, len(y_vec))) * base
        ridge[offset:offset+ppr] = bh   
        # Tile the ridge to make the whole array
        texture_mat = np.tile(ridge, (num_ridges, 1))
        # Pad and center
        delta_x = len(x_vec) - texture_mat.shape[0]
        texture_mat = np.pad(texture_mat, ((int(np.floor(delta_x)),0),(0,0)),constant_values=base)
        
        # Insert the alignment marker
        texture_mat_align = deepcopy(texture_mat)
        if marker_ratio == 'auto':
            bar_sizes = (periodicity, periodicity)
        else:
            bar_sizes = (marker_ratio[0] * area[0], marker_ratio[1] * area[1])
            
        bar_sizes_idx = np.divide(bar_sizes,(resolution/1000))
        # Remove bumps that would overlap with alignment marker
        num_markers_to_remove = np.ceil((bar_sizes[0] / periodicity, bar_sizes[1] / periodicity))
        corner_init_loc = (alignment_offset[0], alignment_offset[1])
        corner_init_idx = [int(np.abs(x_vec - corner_init_loc[0]).argmin()), int(np.abs(y_vec - corner_init_loc[1]).argmin())]
        
        blank_init_idx = [int(np.abs(x_vec - corner_init_loc[0]).argmin()) - (int(np.round(bar_sizes_idx[1])*0.5)),
                          int(np.abs(y_vec - corner_init_loc[1]).argmin()) - (int(np.round(bar_sizes_idx[1])*0.5))]
        x_blank_idx = corner_init_idx[0] + int(np.round(bar_sizes_idx[0] * 1.5))
        y_blank_idx = corner_init_idx[1] + int(np.round(bar_sizes_idx[1] * 1.5))

        # Blanking
        texture_mat_align[blank_init_idx[0]:x_blank_idx,blank_init_idx[1]:y_blank_idx] = base
        
        # Insert alignment marker
        bar_width = 1
        bar_width_idx = int(np.ceil(bar_width/2 / (resolution/1000)))
        
        align_max = np.nanmax(texture_mat_align)
        
        texture_mat_align[corner_init_idx[0]:corner_init_idx[0]+int(bar_sizes_idx[0]),
                          corner_init_idx[0]:corner_init_idx[0]+bar_width_idx] = align_max
        texture_mat_align[corner_init_idx[1]:corner_init_idx[1]+bar_width_idx,
                  corner_init_idx[1]:corner_init_idx[1]+int(bar_sizes_idx[1])] = align_max
        texture_mat = texture_mat_align
        
        # In case of rounding issue
        mesh_size = np.array(np.round(np.array(area) * fs), dtype=int)
        texture_mat = texture_mat[0:mesh_size[0], 0:mesh_size[1]] 
        # Add edges
        texture_mat[0,:], texture_mat[-1,:], texture_mat[:,0], texture_mat[:,-1] = base,base,base,base
        
         # Create and export the STL
        [X,Y] = np.meshgrid(y_vec, x_vec)
        X = X.reshape(np.prod(X.shape),1)
        Y = Y.reshape(np.prod(Y.shape),1)
        texture_mat = texture_mat.reshape(np.prod(texture_mat.shape),1)
        
        # Make the vertices and faces
        tri_points = np.hstack((X, Y))
        vertices = np.hstack((tri_points, texture_mat))
        faces = Delaunay(tri_points)
        faces = faces.simplices
        # For some reason x and y are reversed here
        vertices, faces = create_base(vertices, faces, y_vec, x_vec, base) 
        output_mesh(faces, vertices, file_path)
        
elif feature == 'sine_wave':
    periodicity = 1/periodicity
    z_vals = np.sin(2*np.pi*x_vec*periodicity) * height 
    if trunc:
        z_vals[z_vals<0] = 0 # who needs to save space anyway
        z_vals = z_vals + base
    else:
        z_vals = z_vals + base + height
    
    if not alignment_marker:
        vertices = []
        # Create the actual sine wave
        for x in range(len(x_vec)-1):
            vertices.append([[x_vec[x], y_vec[0], z_vals[x]],
                             [x_vec[x+1], y_vec[0], z_vals[x+1]],
                             [x_vec[x], y_vec[-1], z_vals[x]],
                             [x_vec[x], y_vec[-1], z_vals[x]],
                             [x_vec[x+1], y_vec[0], z_vals[x+1]],
                             [x_vec[x+1], y_vec[-1], z_vals[x+1]]])
        
        # Add the sides
        for x in range(len(x_vec)-1):
            vertices.append([[x_vec[x], y_vec[0], 0],
                             [x_vec[x+1], y_vec[0], z_vals[x+1]],
                             [x_vec[x], y_vec[0], z_vals[x]],#
                             [x_vec[x], y_vec[0], 0],
                             [x_vec[x+1], y_vec[0], 0],
                             [x_vec[x+1], y_vec[0], z_vals[x+1]], # End of side 1
                             [x_vec[x], y_vec[-1], 0],
                             [x_vec[x], y_vec[-1], z_vals[x]],
                             [x_vec[x+1], y_vec[-1], 0],#
                             [x_vec[x+1], y_vec[-1], 0],
                             [x_vec[x], y_vec[-1], z_vals[x]],
                             [x_vec[x+1], y_vec[-1], z_vals[x+1]]])
        
        # Front & back
        vertices.append([[x_vec[0], y_vec[0], 0],
                          [x_vec[0], y_vec[0], z_vals[0]],
                          [x_vec[0], y_vec[-1], z_vals[0]],
                          [x_vec[0], y_vec[0], 0],
                          [x_vec[0], y_vec[-1], z_vals[0]],
                          [x_vec[0], y_vec[-1], 0], ###
                          [x_vec[-1], y_vec[0], 0],
                          [x_vec[-1], y_vec[-1], 0],
                          [x_vec[-1], y_vec[-1], z_vals[-1]],
                          [x_vec[-1], y_vec[0], 0],
                          [x_vec[-1], y_vec[-1], z_vals[-1]],
                          [x_vec[-1], y_vec[0], z_vals[-1]]])
        
        # Add the bottom
        for x in range(len(x_vec)-1):
            vertices.append([[x_vec[x], y_vec[-1], 0],
                             [x_vec[x+1], y_vec[-1], 0],
                             [x_vec[x], y_vec[0], 0],#
                             [x_vec[x+1], y_vec[-1], 0],
                             [x_vec[x+1], y_vec[0], 0],
                             [x_vec[x], y_vec[0], 0]])
         
        vertices = np.vstack(vertices)            
        faces = np.arange(len(vertices)).reshape((int((len(vertices))/3),3))
        # Add the base and output
        output_mesh(faces, vertices, file_path)
    
    else:
        # Revert to less efficient method
        texture_mat = np.transpose(np.tile(z_vals, (len(y_vec), 1)))
        
        # Insert the alignment marker
        texture_mat_align = deepcopy(texture_mat)
        if marker_ratio == 'auto':
            bar_sizes = (3, 3)
        else:
            bar_sizes = (marker_ratio[0] * area[0], marker_ratio[1] * area[1])
            
        bar_sizes_idx = np.divide(bar_sizes,(resolution/1000))
        
        corner_init_loc = (alignment_offset[0], alignment_offset[1])
        corner_init_idx = [int(np.abs(x_vec - corner_init_loc[0]).argmin()), int(np.abs(y_vec - corner_init_loc[1]).argmin())]
        
        blank_init_idx = [int(np.abs(x_vec - corner_init_loc[0]).argmin()) - (int(np.round(bar_sizes_idx[1])*0.5)),
                          int(np.abs(y_vec - corner_init_loc[1]).argmin()) - (int(np.round(bar_sizes_idx[1])*0.5))]
        x_blank_idx = corner_init_idx[0] + int(np.round(bar_sizes_idx[0] * 1.5))
        y_blank_idx = corner_init_idx[1] + int(np.round(bar_sizes_idx[1] * 1.5))

        # Blanking
        texture_mat_align[blank_init_idx[0]:x_blank_idx,blank_init_idx[1]:y_blank_idx] = base
        
        # Insert alignment marker
        bar_width = 1
        bar_width_idx = int(np.ceil(bar_width/2 / (resolution/1000)))
        
        align_max = np.nanmax(texture_mat_align)
        
        texture_mat_align[corner_init_idx[0]:corner_init_idx[0]+int(bar_sizes_idx[0]),
                          corner_init_idx[0]:corner_init_idx[0]+bar_width_idx] = align_max
        texture_mat_align[corner_init_idx[1]:corner_init_idx[1]+bar_width_idx,
                  corner_init_idx[1]:corner_init_idx[1]+int(bar_sizes_idx[1])] = align_max
        
        texture_mat = texture_mat_align
        
        # In case of rounding issue
        mesh_size = np.array(np.round(np.array(area) * fs), dtype=int)
        texture_mat = texture_mat[0:mesh_size[0], 0:mesh_size[1]] 
        # Add edges
        texture_mat[0,:], texture_mat[-1,:], texture_mat[:,0], texture_mat[:,-1] = base,base,base,base
        
         # Create and export the STL
        [X,Y] = np.meshgrid(y_vec, x_vec)
        X = X.reshape(np.prod(X.shape),1)
        Y = Y.reshape(np.prod(Y.shape),1)
        texture_mat = texture_mat.reshape(np.prod(texture_mat.shape),1)
        
        # Make the vertices and faces
        tri_points = np.hstack((X, Y))
        vertices = np.hstack((tri_points, texture_mat))
        faces = Delaunay(tri_points)
        faces = faces.simplices
        # For some reason x and y are reversed here
        vertices, faces = create_base(vertices, faces, y_vec, x_vec, base) 
        output_mesh(faces, vertices, file_path)
'''
#%% Testing new side script
periodicity = 1/periodicity
z_vals = np.sin(2*np.pi*x_vec*periodicity) * height 
if trunc:
    z_vals[z_vals<0] = 0 # who needs to save space anyway
    z_vals = z_vals + base
else:
    z_vals = z_vals + base + height
    

vertices = []
# Create the actual sine wave
for x in range(len(x_vec)-1):
    vertices.append([[x_vec[x], y_vec[0], z_vals[x]],
                     [x_vec[x+1], y_vec[0], z_vals[x+1]],
                     [x_vec[x], y_vec[-1], z_vals[x]],
                     [x_vec[x], y_vec[-1], z_vals[x]],
                     [x_vec[x+1], y_vec[0], z_vals[x+1]],
                     [x_vec[x+1], y_vec[-1], z_vals[x+1]]])

n_side_vertices = int(np.ceil(x_vec[-1] / (1/periodicity)))
side_points = np.linspace(x_vec[0], x_vec[-1], n_side_vertices)
# Add the sides
for x in range(len(x_vec)-1):
    vertices.append([[x_vec[x], y_vec[0], 0],
                     [x_vec[x+1], y_vec[0], z_vals[x+1]],
                     [x_vec[x], y_vec[0], z_vals[x]],#
                     [x_vec[x], y_vec[0], 0],
                     [x_vec[x+1], y_vec[0], 0],
                     [x_vec[x+1], y_vec[0], z_vals[x+1]], # End of side 1
                     [x_vec[x], y_vec[-1], 0],
                     [x_vec[x], y_vec[-1], z_vals[x]],
                     [x_vec[x+1], y_vec[-1], 0],#
                     [x_vec[x+1], y_vec[-1], 0],
                     [x_vec[x], y_vec[-1], z_vals[x]],
                     [x_vec[x+1], y_vec[-1], z_vals[x+1]]])
        
# Check if the end is open
if z_vals[-1] > base:
    vertices.append([[x_vec[-1], y_vec[0], 0],
                     [x_vec[-2], y_vec[0],z_vals[-2]],
                     [x_vec[-1], y_vec[0],z_vals[-1]],
                     [x_vec[-1], y_vec[-1], 0],
                     [x_vec[-2], y_vec[-1],z_vals[-2]],
                     [x_vec[-1], y_vec[-1],z_vals[-1]],###
                     [x_vec[-1], y_vec[0], 0],
                     [x_vec[-1], y_vec[0],z_vals[-1]],
                     [x_vec[-1], y_vec[-1],z_vals[-1]],
                     [x_vec[-1], y_vec[-1], z_vals[-1]],
                     [x_vec[-1], y_vec[-1],50],
                     [x_vec[-1], y_vec[0],0]])
# Front & back
vertices.append([[x_vec[0], y_vec[0], 0],
                 [x_vec[0], y_vec[0], base],
                 [x_vec[0], y_vec[-1], 0],
                 [x_vec[0], y_vec[-1], 0],
                 [x_vec[0], y_vec[0], base],
                 [x_vec[0], y_vec[-1], base],
                 [x_vec[-1], y_vec[-1], 0],
                 [x_vec[-1], y_vec[-1], base],
                 [x_vec[-1], y_vec[0], 0],
                 [x_vec[-1], y_vec[0], 0],
                 [x_vec[-1], y_vec[-1], base],
                 [x_vec[-1], y_vec[0], base]])

# Add the bottom
for x in range(len(x_vec)-1):
    vertices.append([[x_vec[x], y_vec[-1], 0],
                     [x_vec[x+1], y_vec[-1], 0],
                     [x_vec[x], y_vec[0], 0],#
                     [x_vec[x+1], y_vec[-1], 0],
                     [x_vec[x+1], y_vec[0], 0],
                     [x_vec[x], y_vec[0], 0]])
 
vertices = np.vstack(vertices)            
faces = np.arange(len(vertices)).reshape((int((len(vertices))/3),3))
# Add the base and output
output_mesh(faces, vertices, file_path)
'''