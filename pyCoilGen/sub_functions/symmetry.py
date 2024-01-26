# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:41:26 2024

@author: yashi
"""

#%%
#Sysetem
import numpy as np
#from sympy import symbols, diff, lamdify
import trimesh
from typing import List

#logging
import logging

#local imports
from pyCoilGen.sub_functions.data_structures import DataStruture, CoilPart
from pyCoilGen.sub_functions.constants import get_level, DEBUG_NONE, DEBUG_VERBOSE
from pyCoilGen.sub_functions.data_structures import Mesh
log = logging.getLogger(__name__)

def symmetry_bounds(coil_parts: List[CoilPart],target_field):
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
        
        symmetry.planes = [1,1,1] #xy, xz, yz
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
        if symmetry[0] != 0:
            for i in range(len(coil_mesh.vertices)):
                if coil_mesh.vertices[i][2] <=0:# for mesh
                    zm = zm + [i]
            for i in range(len(target_field.coords[2])):
                if target_field.coords[2][i]<0.000000000001:# for target field
                    zt = zt + [i]
        else:
            zm = np.arange(0, len(coil_mesh.vertices), 1)
            zt = np.arange(0, len(target_field.coords[2]), 1)            
        # for xz symmetry
    
        if symmetry[1] != 0:
            for i in range(len(coil_mesh.vertices)):
                if coil_mesh.vertices[i][1] <=0: # for mesh
                    ym = ym + [i]
            for i in range(len(target_field.coords[1])):
                if target_field.coords[1][i]<0.000000000001:# for target field
                    yt = yt + [i]
        else:
            ym = np.arange(0, len(coil_mesh.vertices), 1)
            yt = np.arange(0, len(target_field.coords[1]), 1)
            
        # for yz symmetry
    
        if symmetry[2] != 0:
            for i in range(len(coil_mesh.vertices)):
                if coil_mesh.vertices[i][0] <=0: # for mesh
                    xm = xm + [i]
            for i in range(len(target_field.coords[0])):
                if target_field.coords[0][i]<0.000000000001:# for target field
                    xt = xt + [i]
        else:
            xm = np.arange(0, len(coil_mesh.vertices), 1)
            xt = np.arange(0, len(target_field.coords[0]), 1)
    
        # find corresponding values of satisified verticies
        mesh_inds =list(set(xm).intersection(ym, zm))
        target_inds = list(set(xt).intersection(yt, zt))
        mesh_verts = []
        for i in range(len(coil_mesh.vertices)):
            for j in mesh_inds:
                if i==j:
                    mesh_verts = mesh_verts+[coil_mesh.verticies[i]]
        mesh_face_inds=[]
        mesh_face=[]
        #store faces
        for i in range(len(coil_mesh.f)):
            vertices = coil_mesh.f[i]
            
            # Check if all vertices of the face are in mesh_inds
            if np.all(np.isin(vertices, mesh_inds)):
                mesh_face_inds.append(i)
                mesh_face.append(vertices)
                
    
    
        #boundary nodes
        sredboundary = mesh_inds.boundary_indicies()   
        #true boundary nodes
        t_boundary = list(set(coil_mesh.boundary).intersection(sredboundary))
        #symmetry plane boundary nodes
        splane_boundary = []
        for i in sredboundary:
            if i not in t_boundary:
                splane_boundary = splane_boundary +[i]
                
        symmetry.trimesh_obj=trimesh.Trimesh(verticies = mesh_verts, faces=mesh_face)
# =============================================================================
#         symmetry.sinds = mesh_inds
#         symmetry.sym.boundary = sredboundary
#         symmetry.sym.not_boundary=sred_not_boundary
#         symmetry.sym.t_boundary = t_boundary
#         symmetry.sym.splane_boundary = splane_boundary
# =============================================================================
        return symmetry
    
def symmetry (opt_stream_function):
    """
    Apply symmetrical boundary conditions to full mesh of coil
    
    Initialises:
        -None
    Args:
        -input_args.symmetry_plane (list): list of active symmetry planes
        -coil_parts (list): List of coil parts
        -combined mesh
        -sf_b_field
    Returns:
        -coil_parts (list): List of coil parts
        -combined mesh
        -sf_b_field
    None.
    
    """
    
    #take the points of the sf_b_field and map onto symmetric conditions
    
    #for my case do a single plane and then copy onto part coil_part[1]
    
    # apply to xy plane
    
    #apply to xz plane
    
    #apply to yz plane