# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:41:26 2024

@author: yashi
"""

#%%
#Sysetem
import numpy as np
from sympy import symbols, diff, lamdify

from typing import List

#logging
import logging

#local imports
from .data_structures import DataStruture, CoilPart
from .constants import get_level, DEBUG_NONE, DEBUG_VERBOSE
from .data_structures import Mesh
log = logging.getLogger(__name__)

def symmetry_bounds(coil_parts: List[CoilPart],target_field, input_args):
    """
    Identify the required symmetry planes and reduce the mesh of the target field and current density appropriately. 
    Identify the boundary nodes of the redcuced meshes and differentiate between true boundaries and boundariesthat lie on symmetrical planes.
    
    Intitialises:
        
        
    Args:
        -target field
        -coil_parts
        -input_args.symmetry_plane (list): list of active symmetry planes
        
    Returns:
        coil_parts (list): Updated list of coil parts

    """
    symcondition = input_args.symmetry
    if symcondition == True:
        #Span the verticies and determine which satisfy our conditions;
        
        symmetry = input_args.symmetry_parameter_list#[1,1,1] #xy, xz, yz
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
                if target_field.coords[2][i]<0.00000001:# for target field
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
                if target_field.coords[1][i]<0.00000001:# for target field
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
                if target_field.coords[0][i]<0.00000001:# for target field
                    xt = xt + [i]
        else:
            xm = np.arange(0, len(coil_mesh.vertices), 1)
            xt = np.arange(0, len(target_field.coords[0]), 1)
    
        # find corresponding values of satisified verticies
        mesh_inds =list(set(xm).intersection(ym, zm))
        target_inds = list(set(xt).intersection(yt, zt))
    
        #boundary nodes
        sredboundary = mesh_inds.boundary_indicies()   
        #true boundary nodes
        t_boundary = list(set(coil_mesh.boundary).intersection(sredboundary))
        #symmetry plane boundary nodes
        splane_boundary = []
        for i in sredboundary:
            if i not in t_boundary:
                splane_boundary = splane_boundary +[i]
        coil_part.sredboundary = sredboundary
        coil_part.t_boundary = t_boundary
        coil_part.splane_boundary = splane_boundary
        return coil_part
    
def symmetry ():
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