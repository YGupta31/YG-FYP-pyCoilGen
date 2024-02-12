import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilSolution, BasisElement, CoilPart
from .gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle

log = logging.getLogger(__name__)


def calculate_sensitivity_matrix(coil_parts: List[CoilPart], target_field, input_args) -> List[CoilPart]:
    """
    Calculate the sensitivity matrix.

    Initialises the following properties of a CoilPart:
        - sensitivity_matrix: (3, m, num vertices)

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of coil parts.
        target_field: The target field.

    Returns:
        List[CoilPart]: Updated list of coil parts with sensitivity matrix.

    """
        # Only use symmetric coordinates
    #############################
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
    #########################################################################

    for part_ind in range(len(coil_parts)-1):
        coil_part = coil_parts[part_ind]

        target_points = target_field.coords
        gauss_order = input_args.gauss_order

        # Calculate the weights and the test point for the Gauss-Legendre integration on each triangle
        u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(gauss_order)
        num_gauss_points = len(gauss_weight)
        biot_savart_coeff = 1e-7
        num_nodes = len(coil_part.basis_elements)
        num_target_points = target_points.shape[1]
        sensitivity_matrix = np.zeros((3, num_target_points, num_nodes))

        for node_ind in mesh_inds: # range(num_nodes):
            basis_element = coil_part.basis_elements[node_ind]

            dCx = np.zeros(num_target_points)
            dCy = np.zeros(num_target_points)
            dCz = np.zeros(num_target_points)

            for tri_ind in range(len(basis_element.area)):
                node_point = basis_element.triangle_points_ABC[tri_ind, :, 0]
                point_b = basis_element.triangle_points_ABC[tri_ind, :, 1]
                point_c = basis_element.triangle_points_ABC[tri_ind, :, 2]

                x1, y1, z1 = node_point
                x2, y2, z2 = point_b
                x3, y3, z3 = point_c

                vx, vy, vz = basis_element.current[tri_ind]

                for gauss_ind in range(num_gauss_points):
                    xgauss_in_uv = x1 * u_coord[gauss_ind] + x2 * v_coord[gauss_ind] + \
                        x3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    ygauss_in_uv = y1 * u_coord[gauss_ind] + y2 * v_coord[gauss_ind] + \
                        y3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    zgauss_in_uv = z1 * u_coord[gauss_ind] + z2 * v_coord[gauss_ind] + \
                        z3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])

                    distance_norm = (
                        (xgauss_in_uv - target_points[0])**2 + (ygauss_in_uv - target_points[1])**2 + (zgauss_in_uv - target_points[2])**2)**(-3/2)

                    dCx += ((-1) * vz * (target_points[1] - ygauss_in_uv) + vy * (target_points[2] - zgauss_in_uv)
                            ) * distance_norm * 2 * basis_element.area[tri_ind] * gauss_weight[gauss_ind]
                    dCy += ((-1) * vx * (target_points[2] - zgauss_in_uv) + vz * (target_points[0] - xgauss_in_uv)
                            ) * distance_norm * 2 * basis_element.area[tri_ind] * gauss_weight[gauss_ind]
                    dCz += ((-1) * vy * (target_points[0] - xgauss_in_uv) + vx * (target_points[1] - ygauss_in_uv)
                            ) * distance_norm * 2 * basis_element.area[tri_ind] * gauss_weight[gauss_ind]

            sensitivity_matrix[:, :, node_ind] = np.array([dCx, dCy, dCz]) * biot_savart_coeff

        coil_part.sensitivity_matrix = sensitivity_matrix
#### Apply symmetry across planes ######
    ## xy plane##
    sym_inds = []
    if symmetry[0] != 0:
        a = coil_parts[0].coil_mesh
        for i in range(len(a.v))
            if a.v[i][2]>0:
                for j in mesh_inds:
                    if a.v[j][2] == -1*a.v[i][2] & a.v[j][1] == a.v[i][1] & a.v[j][0] == a.v[i][0]:
                        coil_parts[0].gradient_sensitivity_matrix[:, :, i] = coil_parts[0].gradient_sensitivity_matrix[:, :, j]*symmetry[0]
                        sym _inds = sym_inds +[i]
                        
    reflect_inds = np.append(mesh_inds, sym_inds)                
    ## xz plane##
    if symmetry[1] != 0:
        a = coil_parts[0].coil_mesh
        for i in range(len(a.v))
            if a.v[i][1]>0:
                for j in reflect_inds:
                    if a.v[j][2] == a.v[i][2] & a.v[j][1] == -1*a.v[i][1] & a.v[j][0] == a.v[i][0]:
                        coil_parts[0].gradient_sensitivity_matrix[:, :, i] = coil_parts[0].gradient_sensitivity_matrix[:, :, j]*symmetry[0]
                        sym _inds = sym_inds +[i]
    ## yz plane##
    coil_parts[1].gradient_sensitivity_matrix = coil_parts[0].gradient_sensitivity_matrix
    for i in range(len(coil_parts[1].gradient_sensitivity_matrix):
        for j in range(len(coil_parts[1].gradient_sensitivity_matrix[i]):
            for k in range(len(coil_parts[1].gradient_sensitivity_matrix[i][j]):
                coil_parts[1].gradient_sensitivity_matrix[i][j][k] = coil_parts[1].gradient_sensitivity_matrix[i][j][k]*symmetry[2]
    return coil_parts
