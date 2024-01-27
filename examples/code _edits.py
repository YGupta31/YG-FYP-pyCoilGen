
#%%
# System imports
import sys

import os
script_dir = os.path.dirname(__file__)
# Logging
import logging
import matplotlib.pyplot as plt
# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE
from pyCoilGen.sub_functions.data_structures import Mesh
import numpy as np
import warnings
np.warnings = warnings

#%%ARGS

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'z',  # definition of the target field
        'target_gradient_strength': 200,
        'coil_mesh': 'create bi-planar mesh',
        'planar_mesh_parameter_list': [0.25,0.25,30,30,1,0,0,0,0,0,0.2],
        #'coil_mesh_file': 'bi_planer_rectangles_width_1000mm_distance_500mm.stl',
        'target_mesh_file': 'none',
        'b_0_direction': [0,0,1],
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.075,  # in meter
        'target_region_resolution': 10,  # MATLAB 10 is the default
        'use_only_target_mesh_verts': True,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 10,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor':0.25,
        'surface_is_cylinder_flag': False,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.005,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.5,
        'iteration_num_mesh_refinement': 1,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'skip_postprocessing': False,
        'skip_inductance_calculation': True,
        'tikhonov_reg_factor': 15,  # Tikhonov regularization factor for the SF optimization

        'output_directory': 'trial_1',  # [Current directory]
        'project_name': '0.25x0.25, 40x40, 15',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)
#%%
# =============================================================================
# print('mesh vertices')
# print(result.combined_mesh.vertices)
# print('mesh faces using vertex indices')
# print(result.combined_mesh.faces)
# =============================================================================
# =============================================================================
# print('mesh face normals')
# print(result.combined_mesh.n)
# print('boundary of mesh following vertex indicies')
# print(result.combined_mesh.boundary)
# print('b-field')
# print(result.target_field.b[2])
# print('coordinates')
# print(result.target_field.coords)
# print('gradient direction')
# print(result.target_field.target_gradient_dbdxyz)
# =============================================================================
#%%
#Span the verticies and determine which satisfy our conditions;

symmetry = [1,1,1] #xy, xz, yz

xm = [] #index of satisfied verticies in mesh
ym = []
zm = []
#index of satisified verticies in target field
xt = []
yt = []
zt = []


# for xy symmetry
if symmetry[0] != 0:
    for i in range(len(result.combined_mesh.vertices)):
        if result.combined_mesh.vertices[i][2] <=0:# for mesh
            zm = zm + [i]
    for i in range(len(result.target_field.coords[2])):
        if result.target_field.coords[2][i]<0.00000001:# for target field
            zt = zt + [i]
else:
    zm = np.arange(0, len(result.combined_mesh.vertices), 1)
    zt = np.arange(0, len(result.target_field.coords[2]), 1)            
# for xz symmetry

if symmetry[1] != 0:
    for i in range(len(result.combined_mesh.vertices)):
        if result.combined_mesh.vertices[i][1] <=0: # for mesh
            ym = ym + [i]
    for i in range(len(result.target_field.coords[1])):
        if result.target_field.coords[1][i]<0.00000001:# for target field
            yt = yt + [i]
else:
    ym = np.arange(0, len(result.combined_mesh.vertices), 1)
    yt = np.arange(0, len(result.target_field.coords[1]), 1)
    
# for yz symmetry

if symmetry[2] != 0:
    for i in range(len(result.combined_mesh.vertices)):
        if result.combined_mesh.vertices[i][0] <=0: # for mesh
            xm = xm + [i]
    for i in range(len(result.target_field.coords[0])):
        if result.target_field.coords[0][i]<0.00000001:# for target field
            xt = xt + [i]
else:
    xm = np.arange(0, len(result.combined_mesh.vertices), 1)
    xt = np.arange(0, len(result.target_field.coords[0]), 1)

# find corresponding values of satisified verticies
mesh_inds =list(set(xm).intersection(ym, zm))
target_inds = list(set(xt).intersection(yt, zt))

print(mesh_inds)
print(target_inds)


#%%

mesh = os.path.join(script_dir, 'mesh')
os.makedirs(mesh, exist_ok=True)
#plot target field
fig = plt.figure()
ax = plt.axes(projection = '3d')

for i in range(len(result.target_field.coords[0])):
    if i in target_inds:
        ax.scatter(result.target_field.coords[0][i], result.target_field.coords[1][i], result.target_field.coords[2][i], c= 'r')
    else:
        ax.scatter(result.target_field.coords[0][i], result.target_field.coords[1][i], result.target_field.coords[2][i], c= 'b')
    #plt.annotate(i, result.target_field.coords[0][i], result.target_field.coords[1][i], result.target_field.coords[2][i])
ax.set_title('Target Field, Target Resolution = %d'%arg_dict['target_region_resolution'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#fig.savefig(os.path.join(mesh,'Target_Field_Target_Resolution_=_%d'%arg_dict['target_region_resolution']))
plt.show()
plt.close()
#%%
#plot mesh

x1 =[]
y1 =[]
z1 =[]

for i in range(len(result.combined_mesh.vertices)):
    x1 = x1 + [result.combined_mesh.vertices[i][0]]
    y1 = y1 + [result.combined_mesh.vertices[i][1]]
    z1 = z1 + [result.combined_mesh.vertices[i][2]]
    
    
fig1 = plt.figure()
ax = plt.axes(projection = '3d')
#plt.scatter(x1,y1)
for i in range(len(x1)):
    if i in mesh_inds:
        ax.scatter(x1[i], y1[i], z1[i], c= 'r')
        #ax.annotate(i,(x1[i],y1[i]))
    else:
        ax.scatter(x1[i], y1[i], z1[i], c= 'b')
    #plt.annotate(i,(x1[i],y1[i]))


ax.set_title('Coil Mesh, Coil Mesh Resolution = %d x '%(arg_dict['planar_mesh_parameter_list'][2]*arg_dict['iteration_num_mesh_refinement']) + '%d'%(arg_dict['planar_mesh_parameter_list'][3]*arg_dict['iteration_num_mesh_refinement']))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#fig1.savefig(os.path.join(mesh,'Coil_Mesh_Coil_Mesh_Resolution_=_%d'%(arg_dict['planar_mesh_parameter_list'][2]*arg_dict['iteration_num_mesh_refinement']) + '_%d'%(arg_dict['planar_mesh_parameter_list'][3]*arg_dict['iteration_num_mesh_refinement'])))
plt.show()
plt.close
#%%

import numpy as np

face_inds = []
face = []

for i in range(len(result.coil_parts[0].coil_mesh.f)):
    vertices = result.coil_parts[0].coil_mesh.f[i]
    
    # Check if all vertices of the face are in mesh_inds
    if np.all(np.isin(vertices, mesh_inds)):
        face_inds.append(i)
        face.append(vertices)

print(result.coil_parts[0].coil_mesh.f)            
print(face_inds)

# =============================================================================
# face_inds = []
# face=[]
# for i in range(8):
#     if np.any(mesh_inds) == result.coil_parts[0].coil_mesh.f[i][0]:
#         print(result.coil_parts[0].coil_mesh.f[i][0])
#         if np.any(mesh_inds) == result.coil_parts[0].coil_mesh.f[i][1]:
#             print(result.coil_parts[0].coil_mesh.f[i][1])
#             if np.any(mesh_inds) == result.coil_parts[0].coil_mesh.f[i][2]:
#                 face_inds = face_inds  + [i]
#                 face=face+ [result.coil_parts[0].coil_mesh.f[i]]
# print(result.coil_parts[0].coil_mesh.f)            
# print(face)              
# =============================================================================
#%%           
# =============================================================================
# print(result.coil_parts[0].coil_mesh.sym)
# print(result.coil_parts[0].coil_mesh.trimesh_obj.faces)
# print('')
# print(result.coil_parts[0].coil_mesh.normal_rep)
# print('')
# print(result.coil_parts[0].coil_mesh.v)
# print('')
# print(result.coil_parts[0].coil_mesh.f)
# print('')
# print(result.coil_parts[0].coil_mesh.fn)
# print('')
# print(result.coil_parts[0].coil_mesh.n)
# print('')
# print(result.coil_parts[0].coil_mesh.uv)
# print('')
# print(result.coil_parts[0].coil_mesh.unique_vert_inds)
# ==========================================================================
def reduce_matrices_for_boundary_nodes(full_mat, coil_mesh, set_zero_flag):
    """
    Reduce the sensitivity matrix in order to limit the degrees of freedom on
    the boundary nodes and make sure that they have constant sf later for each boundary.

    Args:
        full_mat: Full matrix to be reduced.
        coil_mesh: Coil mesh.
        set_zero_flag: Flag to force the potential on the boundary nodes to zero.

    Returns:
        reduced_mat: Reduced matrix.
        boundary_nodes: Boundary nodes.
        is_not_boundary_node: Non-boundary nodes.
    """

    num_nodes = coil_mesh.vertices.shape[0]
    dim_to_reduce = [x == num_nodes for x in full_mat.shape]
    num_boundaries = len(coil_mesh.boundary)
    num_nodes_per_boundary = np.array([len(np.unique(coil_mesh.boundary[x])) for x in range(num_boundaries)])

    is_not_boundary_node = np.setdiff1d(np.arange(full_mat.shape[1]), np.concatenate(coil_mesh.boundary))
    boundary_nodes = [np.unique(coil_mesh.boundary[x]) for x in range(num_boundaries)]
    reduced_mat = full_mat.copy()

    if np.any(dim_to_reduce):
        for dim_to_reduce_ind in np.where(dim_to_reduce)[0]:
            for boundary_ind in range(num_boundaries):
                if set_zero_flag:
                    reduced_mat[boundary_nodes[boundary_ind], :] = 0
                else:
                    Index1 = [slice(None)] * np.ndim(full_mat)
                    Index1[dim_to_reduce_ind] = boundary_nodes[boundary_ind][0]
                    Index2 = [slice(None)] * np.ndim(full_mat)
                    Index2[dim_to_reduce_ind] = boundary_nodes[boundary_ind]  # [:-1]
                    reduced_mat[tuple(Index1)] = np.sum(reduced_mat[tuple(Index2)], axis=dim_to_reduce_ind)

        boundary_nodes_first_inds = [boundary_nodes[x][0] for x in range(num_boundaries)]

        for dim_to_reduce_ind in np.where(dim_to_reduce)[0]:
            prev_reduced_mat = reduced_mat.copy()
            Index1 = [slice(None)] * np.ndim(full_mat)
            Index1[dim_to_reduce_ind] = slice(0, num_boundaries)  # The first entries are substitutes for the boundaries
            Index2 = [slice(None)] * np.ndim(full_mat)
            Index2[dim_to_reduce_ind] = boundary_nodes_first_inds  # Indices of the first nodes of the boundaries
            Index3 = [slice(None)] * np.ndim(full_mat)
            # Indices of the non-boundary nodes
            Index3[dim_to_reduce_ind] = slice(num_boundaries, len(is_not_boundary_node) + num_boundaries)
            Index4 = [slice(None)] * np.ndim(full_mat)
            # Old non-boundary nodes
            Index4[dim_to_reduce_ind] = is_not_boundary_node
            Index5 = [slice(None)] * np.ndim(full_mat)
            # Entries to be deleted
            Index5[dim_to_reduce_ind] = slice(num_nodes - (sum(num_nodes_per_boundary) - num_boundaries), num_nodes)
            reduced_mat[tuple(Index1)] = prev_reduced_mat[tuple(Index2)]
            reduced_mat[tuple(Index3)] = prev_reduced_mat[tuple(Index4)]
            reduced_mat = np.delete(reduced_mat, Index5[dim_to_reduce_ind], axis=dim_to_reduce_ind)

# =============================================================================
#             if get_level() > DEBUG_VERBOSE:
#                 log.debug(" -- %d reduced_mat.shape: %s", dim_to_reduce_ind, reduced_mat.shape)
# 
# =============================================================================
    return reduced_mat, boundary_nodes, is_not_boundary_node
#%%

A,b,c=reduce_matrices_for_boundary_nodes(result.coil_parts[0].sensitivity_matrix[2],result.combined_mesh, False)
#%%
print(A.shape)
print(result.coil_parts[0].sensitivity_matrix.shape)

#%%
#Sysetem
import numpy as np
#from sympy import symbols, diff, lamdify
import trimesh
from typing import List

#logging
import logging

#local imports
# =============================================================================
# from pyCoilGen.sub_functions.data_structures import DataStruture, CoilPart
# from pyCoilGen.sub_functions.constants import get_level, DEBUG_NONE, DEBUG_VERBOSE
# from pyCoilGen.sub_functions.data_structures import Mesh
# =============================================================================
log = logging.getLogger(__name__)

def symmetry_bounds(coil_parts ,target_field):
    """
    Identify the required symmetry planes and reduce the mesh of the target field and current density appropriately. 
    Identify the boundary nodes of the redcuced meshes and differentiate between true boundaries and boundariesthat lie on symmetrical planes.
    
    Intitialises:
        
        
    Args:
        -target field
        -coil_parts
        -input_args.symmetry_plane (list): list of active symmetry planes
        
    Returns:
        sym_mesh (object): Updated list of coil parts

    """
    
    symcondition = True
    if symcondition == True:
        #Span the verticies and determine which satisfy our conditions;
        
        symplanes = [1,1,1] #xy, xz, yz
        coil_part = coil_parts[0]
        coil_mesh = coil_part.coil_mesh
        xm = [] #index of satisfied verticies in mesh
        ym = []
        zm = []
        #index of satisified verticies in target field
        xt = []
        yt = []
        zt = []
        #reduced mesh nodes
        # for xy symmetry
        if symplanes[0] != 0:
            for i in range(len(coil_mesh.v)):
                if coil_mesh.v[i][2] <=0:# for mesh
                    zm = zm + [i]
            for i in range(len(target_field.coords[2])):
                if target_field.coords[2][i]<0.000000000001:# for target field
                    zt = zt + [i]
        else:
            zm = np.arange(0, len(coil_mesh.v), 1)
            zt = np.arange(0, len(target_field.coords[2]), 1)            
        # for xz symmetry
    
        if symplanes[1] != 0:
            for i in range(len(coil_mesh.v)):
                if coil_mesh.v[i][1] <=0: # for mesh
                    ym = ym + [i]
            for i in range(len(target_field.coords[1])):
                if target_field.coords[1][i]<0.000000000001:# for target field
                    yt = yt + [i]
        else:
            ym = np.arange(0, len(coil_mesh.v), 1)
            yt = np.arange(0, len(target_field.coords[1]), 1)
            
        # for yz symmetry
    
        if symplanes[2] != 0:
            for i in range(len(coil_mesh.v)):
                if coil_mesh.v[i][0] <=0: # for mesh
                    xm = xm + [i]
            for i in range(len(target_field.coords[0])):
                if target_field.coords[0][i]<0.000000000001:# for target field
                    xt = xt + [i]
        else:
            xm = np.arange(0, len(coil_mesh.v), 1)
            xt = np.arange(0, len(target_field.coords[0]), 1)
    
        # find corresponding values of satisified verticies
        mesh_inds =list(set(xm).intersection(ym, zm))
        target_inds = list(set(xt).intersection(yt, zt))
        mesh_verts = []
        for i in range(len(coil_mesh.v)):
            for j in mesh_inds:
                if i==j:
                    mesh_verts = mesh_verts+[coil_mesh.v[i]]
        mesh_face_inds=[]
        mesh_face=[]
        #store faces
        for i in range(len(coil_mesh.f)):
            vertices = coil_mesh.f[i]
            
            # Check if all vertices of the face are in mesh_inds
            if np.all(np.isin(vertices, mesh_inds)):
                mesh_face_inds.append(i)
                mesh_face.append(vertices)
                
    
    
# =============================================================================
#         #boundary nodes
#         sredboundary = mesh_inds.boundary_indicies()   
#         #true boundary nodes
#         t_boundary = list(set(coil_mesh.boundary).intersection(sredboundary))
#         #symmetry plane boundary nodes
#         splane_boundary = []
#         for i in sredboundary:
#             if i not in t_boundary:
#                 splane_boundary = splane_boundary +[i]
# =============================================================================
                
        symmetry=trimesh.Trimesh(verticies = mesh_verts, faces=mesh_face)
# =============================================================================
#         symmetry.sinds = mesh_inds
#         symmetry.sym.boundary = sredboundary
#         symmetry.sym.not_boundary=sred_not_boundary
#         symmetry.sym.t_boundary = t_boundary
#         symmetry.sym.splane_boundary = splane_boundary
# =============================================================================
        return symmetry
#%%
A=symmetry_bounds(result.coil_parts, result.target_field)
print(A.faces)