
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
import numpy as np
import warnings
np.warnings = warnings
#%%

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'x',  # definition of the target field
        'target_gradient_strength': 200,
        'coil_mesh': 'create planar mesh',
        'planar_mesh_parameter_list': [0.25,0.25,1,1,1,0,0,0,0,0,0],
        #'coil_mesh_file': 'bi_planer_rectangles_width_1000mm_distance_500mm.stl',
        'target_mesh_file': 'none',
        'b_0_direction': [0,0,1],
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.075,  # in meter
        'target_region_resolution': 2,  # MATLAB 10 is the default
        'use_only_target_mesh_verts': True,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 4,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor':0.25,
        'surface_is_cylinder_flag': False,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.01,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.01,
        'iteration_num_mesh_refinement': 1,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'skip_postprocessing': False,
        'skip_inductance_calculation': True,
        'tikhonov_reg_factor': 2,  # Tikhonov regularization factor for the SF optimization

        'output_directory': 'matrix determination',  # [Current directory]
        'project_name': 'code_edits_iii',
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
ax = plt.axes#(projection = '3d')
plt.scatter(x1,y1)
for i in range(len(x1)):
# =============================================================================
#     if i in mesh_inds:
#         ax.scatter(x1[i], y1[i], z1[i], c= 'r')
#         ax.annotate(i,(x1[i],y1[i]))
#     else:
#         ax.scatter(x1[i], y1[i], z1[i], c= 'b')
# =============================================================================
    plt.annotate(i,(x1[i],y1[i]))


ax.set_title('Coil Mesh, Coil Mesh Resolution = %d x '%(arg_dict['planar_mesh_parameter_list'][2]*arg_dict['iteration_num_mesh_refinement']) + '%d'%(arg_dict['planar_mesh_parameter_list'][3]*arg_dict['iteration_num_mesh_refinement']))
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.set_zlabel('z')
#fig1.savefig(os.path.join(mesh,'Coil_Mesh_Coil_Mesh_Resolution_=_%d'%(arg_dict['planar_mesh_parameter_list'][2]*arg_dict['iteration_num_mesh_refinement']) + '_%d'%(arg_dict['planar_mesh_parameter_list'][3]*arg_dict['iteration_num_mesh_refinement'])))
plt.show()
plt.close