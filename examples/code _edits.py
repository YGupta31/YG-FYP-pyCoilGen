
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
"""Initialize the mesh_factory package.

This module provides functions to dynamically load plugins for mesh creation.
"""

import os
import importlib

def load_mesh_factory_plugins():
    """Load all available mesh creation plugins.

    This function dynamically discovers and imports all Python files in the 
    mesh_factory directory (excluding this file), treating them as plugins.
    It returns a list of imported modules.

    Every plugin must be a module that exposes the following functions:

    - get_name()-> str       : Return the name of the mesh builder instruction.
    - get_parameters()->list : Return a list of tuples of the parameter names and default values.
    - register_args(parser)  : Called to register any required parameters with ArgParse.

    In addition, it must also provide a creator function that matches the value returned by `get_name()`, e.g.:
    - create_planar_mesh(input_args: argparse.Namespace) : Mesh or DataStructure(vertices, faces, normal)

    Returns:
        list: A list of imported plugin modules.

    """
    plugins = []

    # Load all .py files in the mesh_factory directory
    for file_name in os.listdir(os.path.dirname(__file__)):
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_name = f"pyCoilGen.mesh_factory.{file_name[:-3]}"
            module = importlib.import_module(module_name)
            plugins.append(module)

    return plugins


# System imports
from argparse import Namespace
import numpy as np
import os
# Logging
import logging

# Local imports
from pyCoilGen.mesh_factory import load_mesh_factory_plugins

from pyCoilGen.sub_functions.data_structures import DataStructure, Mesh

log = logging.getLogger(__name__)


def read_mesh(input_args)->(Mesh, Mesh, Mesh):
    """
    Read the input mesh and return the coil, target, and shielded meshes.

    Args:
        input_args (object): Input parameters for reading the mesh.

    Returns:
        coil_mesh (object): Coil mesh object, or None if no mesh was created (e.g. printing 'help').
        target_mesh (object): Target mesh object.
        shielded_mesh (object): Shielded mesh object.
    """
    # Read the input mesh
    mesh_plugins = load_mesh_factory_plugins()

    coil_mesh = get_mesh(input_args, 'coil_mesh', 'coil_mesh_file', mesh_plugins)
    if coil_mesh is None:
        return None, None, None

    # Read the target mesh surface
    target_mesh = get_mesh(input_args, 'target_mesh', 'target_mesh_file', mesh_plugins)

    # Read the shielded mesh surface
    shielded_mesh = get_mesh(input_args, 'shield_mesh', 'secondary_target_mesh_file', mesh_plugins)

    return coil_mesh, target_mesh, shielded_mesh


def get_mesh(input_args: Namespace, primary_parameter: str, legacy_parameter: str, mesh_plugins: list):
    """
    Create a mesh using the command-line parameters, with fallback.

    First try the primary/new parameter name, but also support the legacy parameter name.

    Args:
        input_args (Namespace): Input parameters for reading the mesh.
        primary_parameter (str): The name of the primary mesh creation parameter.
        legacy_parameter (str): The name of the legacy mesh creation parameter.
        mesh_plugins (list of modules): The list of modules from `load_mesh_factory_plugins`.

    Returns:
        mesh (Mesh): The created mesh, or None if no mesh was created.

    Raises:
        ValueError if the mesh builder is not found.
    """

    parameter_value = getattr(input_args, primary_parameter)

    if parameter_value == 'none':
        parameter_value = getattr(input_args, legacy_parameter)
        # Preserve legacy behaviour (version 0.x.y)
        log.debug("Using legacy method to load meshes.")
        if parameter_value == 'none':
            return None

        if parameter_value.endswith('.stl'):
            log.debug("Loading mesh from STL file.")
            # Load the stl file; read the coil mesh surface
            this_mesh = Mesh.load_from_file(input_args.geometry_source_path,  input_args.coil_mesh_file)
            log.info(" Loaded mesh from STL. Assuming representative normal is [0,0,1]!")
            this_mesh.normal_rep = np.array([0.0, 0.0, 1.0])
            return this_mesh

    # Version 0.x: Support both 'coil_mesh_file' and 'coil_mesh'. 'coil_mesh' takes priority.
    plugin_name = parameter_value.replace(' ', '_').replace('-', '_')
    print("Using plugin: ", plugin_name)
    if plugin_name == 'help':
        print('Available mesh creators are:')
        for plugin in mesh_plugins:
            name_function = getattr(plugin, 'get_name', None)
            parameters_function = getattr(plugin, 'get_parameters', None)
            if name_function:
                name = name_function()
                if parameters_function:
                    parameters = parameters_function()
                    parameter_name, default_value = parameters[0]
                    print(f"'{name}', Parameter: '{parameter_name}', Default values: {default_value}")
                    for i in range(1, len(parameters)):
                        print(f"\t\tParameter: '{parameter_name}', Default values: {default_value}")

                else:
                    print(f"'{name}', no parameters")
        return None

    found = False
    for plugin in mesh_plugins:
        mesh_creation_function = getattr(plugin, plugin_name, None)
        if mesh_creation_function:
            this_mesh = mesh_creation_function(input_args)
            found = True
            break

    if found == False:
        raise ValueError(f"No mesh creation method found for {input_args.this_mesh_file}")
    
    if this_mesh is None:
        log.warning("Mesh builder '%s' was specified but mesh is None!", plugin_name)
        return None

    if isinstance(this_mesh, Mesh):
        return this_mesh
    return create_unique_noded_mesh(this_mesh)


def create_unique_noded_mesh(non_unique_mesh):
    """
    Create a mesh with unique nodes.

    Args:
        non_unique_mesh (DataStructure): Mesh object with non-unique nodes.

    Returns:
        unique_noded_mesh (Mesh): Mesh object with unique nodes.
    """

    faces = non_unique_mesh.faces
    verts = non_unique_mesh.vertices

    mesh = Mesh(vertices=verts, faces=faces)
    # mesh.cleanup() # Changes mesh a lot.
    mesh.normal_rep = non_unique_mesh.normal
    return mesh


#%%ARGS

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'z',  # definition of the target field
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
        'levels': 10,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor':0.25,
        'surface_is_cylinder_flag': False,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.01,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.5,
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
#         #ax.annotate(i,(x1[i],y1[i]))
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



