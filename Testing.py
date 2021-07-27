# -*- coding: utf-8 -*-
from FeatureGen import *
from SphereGen import *

#%% Create base sphere and get coordinates for features
view_feature = 0
view_sphere = 1
export_to_stl = 0

path = r'C:\Users\somlab\Desktop'
sphere_radius = 10 # Must declare 'sphere_radius' as global variable
n_features = 10

feature_type = 'dot'
feature_options = {
    'height': 3, # Height of feature 
    'width': 3, # The size of the feature along the longest axis (overwrites height for dot)
    'trunc_val': None, # Truncates objects at height or radius
    'resolution': 25} # microns


if feature_options['height'] < 0:
    sphere_radius = sphere_radius - feature_options['height']

if feature_type == 'dot':
    feature_options['height'] = feature_options['width']
    
filename = '\\' + str(sphere_radius) + 'mm_sphere_' + feature_type + '_' + str(feature_options['height']) + '.stl'

# Create feature
offset_val = abs(feature_options['width'])

feature_mesh, axis = create_feature(feature_type, feature_options)
feature_points = strip_nan(feature_mesh, axis)

if view_feature:
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(feature_points)  
    o3d.geometry.PointCloud.estimate_normals(temp_pcd)
    o3d.visualization.draw_geometries([temp_pcd])

#%% Create the sphere
_, base_sphere = create_sphere(sphere_radius, n_features=50000)
optimized_ThetaPhi, optimized_points = create_optimized_points(sphere_radius, n_features, verbose=True, method = 'local')
#%%
arc_offset = feature_arc_offset(sphere_radius, offset_val)
#arc_offset = 0

feature_points_offset = deepcopy(feature_points)
feature_points_offset[:,2] = feature_points_offset[:,2] + sphere_radius - arc_offset
rotated_feature_points = []
for f in range(n_features):
    theta, phi = optimized_ThetaPhi[f,:]
    temp_rotated_features = rotate_feature(feature_points_offset, theta, phi)
    rotated_feature_points.append(temp_rotated_features)

rotated_feature_points = np.vstack(rotated_feature_points)

reduced_sphere = remove_overlapping_points_from_sphere(
    base_sphere, sphere_radius, optimized_points, offset_val/2, margin=1.05)

composite_sphere = np.vstack((reduced_sphere, rotated_feature_points))

if view_sphere:
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=sphere_radius/2, origin=[0, 0, 0])
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(composite_sphere)  
    o3d.geometry.PointCloud.estimate_normals(temp_pcd)
    o3d.visualization.draw_geometries([temp_pcd, origin_frame])

if export_to_stl:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(composite_sphere)  
    o3d.geometry.PointCloud.estimate_normals(pcd)
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array([0.0, 0.0, 0.0]))
    temp = np.asarray(pcd.normals) * -1
    pcd.normals = o3d.utility.Vector3dVector(temp)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.5, linear_fit=False)[0]
    o3d.geometry.TriangleMesh.compute_triangle_normals(poisson_mesh)
    o3d.io.write_triangle_mesh(path + filename, poisson_mesh)


# #%%
# radii = np.around(np.linspace(30, 70, 5))
# feature_size = 3

# min_dists = np.zeros((5,5))
# for ri, r in enumerate(radii):
#     nf = np.around(np.linspace(r, r*5, 5))
#     for fi,f in enumerate(nf):
#         optimized_ThetaPhi, optimized_points = create_optimized_points(int(r), int(f), verbose=True)
#         temp_md ,md_std = get_min_dist(optimized_points, 30)
#         min_dists[ri,fi] = temp_md

        