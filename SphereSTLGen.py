import math
import numpy as np
from scipy.optimize import minimize, fmin, minimize_scalar

import matplotlib
import matplotlib.pyplot as mpl_pp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

path = r'C:\Users\somlab\Desktop'


#%% XYZ from NuPhi
def NuPhi2xyz(NuPhi,r):
    xyz = np.zeros((NuPhi.shape[0],3))
    for i in range(NuPhi.shape[0]):
        nu = NuPhi[i,0]
        phi = NuPhi[i,1]
        xp = r*np.sin(nu)*np.cos(phi)
        yp = r*np.sin(nu)*np.sin(phi)
        zp = r*np.cos(nu)
        
        xyz[i,:] = [xp,yp,zp]
        
    return xyz

#%% XYZ Point Distance
def xyzSphereDist(xyz,r):
    n_features = xyz.shape[0]
    pdm = np.zeros((n_features,n_features)) # Point distance matrix
    for i in range(n_features):
        for k in range(n_features):
            dist = math.sqrt((xyz[i,0]-xyz[k,0])**2+(xyz[i,1]-xyz[k,1])**2+(xyz[i,2]-xyz[k,2])**2) # Absolute distance
            #print(str(dist) + '' +  str(r))
            sin_dist = dist/2/r
            if sin_dist > 1:
                sin_dist = 1
                
            phi = math.asin(sin_dist) # Angle necessary to produce distance at radius
            arc_dist = 2*phi*r # Length of arc between two points of the calculated angle
            if arc_dist == 0:
                arc_dist = np.nan
            pdm[i,k] = arc_dist
            
    return pdm

#%% Distance loss function
def dist_loss(pdm):
    #per_point_minimum = np.nanmin(pdm, axis=0)
    sorted_dist_mat = np.sort(pdm,axis=0)
    nearest_neighbors = sorted_dist_mat[0,:]
    loss = 1/np.mean(nearest_neighbors)
    
    return loss

#%% Combined fun
def NuPhiLoss(NuPhi):
    r = 10
    NuPhi = np.reshape(NuPhi, [int(np.size(NuPhi)/2),2])
    xyz = NuPhi2xyz(NuPhi,r)
    pdm = xyzSphereDist(xyz,r)
    loss = dist_loss(pdm)
    
    return loss


#%% Generate base sphere
n_features = 10000 # Number of feature
r = 10 # Radius
points = np.zeros((n_features,3))
nu_phi = np.zeros((n_features,2))


alpha = 4.0*np.pi/n_features
d = np.sqrt(alpha)
m_nu = int(np.round(np.pi/d))
d_nu = np.pi/m_nu
d_phi = alpha/d_nu
count = 0
for m in range (0,m_nu):
    nu = np.pi*(m+0.5)/m_nu # Latitude
    m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
    for n in range (0,m_phi):
        phi = 2*np.pi*n/m_phi # Longitude
        nu_phi[count,:] = [nu,phi]
        count = count +1
        if count == n_features: # Due to rounding errors sometimes there is an extra point
            break

points = NuPhi2xyz(nu_phi,r)
#init_dist_matrix = xyzSphereDist(points,r)
#init_loss = dist_loss(init_dist_matrix)
#print('Initial loss: ' + str(np.round(init_loss,3)))

#%% Optimize using initalized params
pi_lims = (0, 2*np.pi)
#nuphi_bounds = tuple(np.vstack(([r-0.01,r-+0.01], np.tile(pi_lims, [np.size(nu_phi),1]))))
nuphi_bounds = ((pi_lims, ) * (n_features*2))
nuphi_long = np.reshape(nu_phi, [np.size(nu_phi),1])

minimize_opts={'maxiter': 1e4, 'disp': True}
res = minimize(NuPhiLoss, nuphi_long, bounds=nuphi_bounds, options=minimize_opts)


#%% Compare outputs
minimized_nuphi = res.x
minimized_nuphi = np.reshape(minimized_nuphi, [int(np.size(minimized_nuphi)/2),2])
points2 = NuPhi2xyz(minimized_nuphi,r)
minimized_dist = xyzSphereDist(points2,r)
term_loss = dist_loss(minimized_dist)
print('Terminal loss: ' + str(np.round(term_loss,3)))

fig = mpl_pp.figure()
fig.canvas.set_window_title('Equidistant Sphere')
ax = fig.gca(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2])
ax.scatter(points2[:,0],points2[:,1],points2[:,2])

#%%
import open3d as o3d

# Generate point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)  
o3d.geometry.PointCloud.estimate_normals(pcd)
o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(pcd, 1)
o3d.visualization.draw_geometries([pcd])


#%%  Poisson mesh
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.5, linear_fit=False)[0]
o3d.geometry.TriangleMesh.compute_triangle_normals(poisson_mesh)
o3d.visualization.draw_geometries([poisson_mesh])

o3d.io.write_triangle_mesh(path + '\poiss.stl', poisson_mesh)


#%%
