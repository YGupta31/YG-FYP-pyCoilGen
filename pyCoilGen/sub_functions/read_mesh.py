# System imports
from argparse import Namespace
import numpy as np
import os
# Logging
import logging

# Local imports
from pyCoilGen.mesh_factory import load_mesh_factory_plugins

from .data_structures import DataStructure, Mesh

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
    ## Apply symmetry to identify symmetric indicies ##
    ####################################################
    #Span the verticies and determine which satisfy our conditions;
    
    symmetry_planes = [1,1,1] #xy, xz, yz
    # Coil Mesh defined as coil_mesh
    xm = [] #index of satisfied verticies in mesh
    ym = []
    zm = []
    #index of satisified verticies in target field
    xt = []
    yt = []
    zt = []
    #reduced mesh nodes
    # for xy symmetry
    if symmetry[0] != 0:
        for i in range(len(coil_mesh.vertices)):
            if coil_mesh.v[i][2] <=0:# for mesh
                zm = zm + [i]
        for i in range(len(target_mesh.v[2])):
            if target_mesh.v[2][i]<0.000000000001:# for target field
                zt = zt + [i]
    else:
        zm = np.arange(0, len(coil_mesh.v), 1)
        zt = np.arange(0, len(target_mesh.v[2]), 1)            
    # for xz symmetry

    if symmetry[1] != 0:
        for i in range(len(coil_mesh.v)):
            if coil_mesh.v[i][1] <=0: # for mesh
                ym = ym + [i]
        for i in range(len(target_mesh.v[1])):
            if target_mesh.v[1][i]<0.000000000001:# for target field
                yt = yt + [i]
    else:
        ym = np.arange(0, len(coil_mesh.v), 1)
        yt = np.arange(0, len(target_mesh.v[1]), 1)
        
    # for yz symmetry

    if symmetry[2] != 0:
        for i in range(len(coil_mesh.v)):
            if coil_mesh.v[i][0] <=0: # for mesh
                xm = xm + [i]
        for i in range(len(target_mesh.v)):
            if target_mesh.v[0][i]<0.000000000001:# for target field
                xt = xt + [i]
    else:
        xm = np.arange(0, len(coil_mesh.v), 1)
        xt = np.arange(0, len(target_mesh.v[0]), 1)

    # find corresponding values of satisified verticies
    mesh_inds =list(set(xm).intersection(ym, zm))
    target_inds = list(set(xt).intersection(yt, zt))


    # #boundary nodes
    # sredboundary = mesh_inds.boundary_indicies()   
    # #true boundary nodes
    # t_boundary = list(set(coil_mesh.boundary).intersection(sredboundary))
    # #symmetry plane boundary nodes
    # splane_boundary = []
    # for i in sredboundary:
    #     if i not in t_boundary:
    #         splane_boundary = splane_boundary +[i]
            

    ####################################################
    return coil_mesh, target_mesh, shielded_mesh, mesh_inds, target_inds


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
