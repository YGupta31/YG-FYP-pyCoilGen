import numpy as np

from .gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle


def calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args):
    """
    Calculate the gradient sensitivity matrix for coil parts.

    Initialises the following properties of the CoilParts:
        - gradient_sensitivity_matrix

    Depends on the following properties of the CoilParts:
        - basis_elements

    Depends on the following input_args:
        - gauss_order

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (list): List of coil parts.
        target_field: The target field.
        input: The input parameters.

    Returns:
        list: List of coil parts with the gradient sensitivity matrix calculated.
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
    target_points = target_field.coords
    gauss_order = input_args.gauss_order
    u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(gauss_order)

    # Calculate the sensitivity matrix for each coil part # only calcuate for 1 part of the coil
    for part_ind in range(len(coil_parts)-1):
        biot_savart_coeff = 10 ** (-7)
        plate_thickness = 0.001
        num_nodes = len(coil_parts[part_ind].basis_elements[mesh_inds])
        num_target_points = target_points.shape[1]  # TODO: Check 1 or 0?
        gradient_sensitivity_matrix = np.zeros((3, num_target_points, num_nodes))  # TODO: Check shape for Python

        x = target_points[0, :]
        y = target_points[1, :]
        z = target_points[2, :]

        for node_ind in mesh_inds: #range(num_nodes):
            DBzdx = np.zeros(num_target_points)
            DBzdy = np.zeros(num_target_points)
            DBzdz = np.zeros(num_target_points)

            for tri_ind in range(coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC.shape[0]):
                node_point = coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 0]
                point_b = coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 1]
                point_c = coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 2]

                x1, y1, z1 = node_point
                x2, y2, z2 = point_b
                x3, y3, z3 = point_c

                d_l_x = coil_parts[part_ind].basis_elements[node_ind].current[tri_ind, 0]
                d_l_y = coil_parts[part_ind].basis_elements[node_ind].current[tri_ind, 1]
                d_l_z = coil_parts[part_ind].basis_elements[node_ind].current[tri_ind, 2]

                for gauss_ind in range(len(gauss_weight)):
                    l_x = x1 * u_coord[gauss_ind] + x2 * v_coord[gauss_ind] + \
                        x3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    l_y = y1 * u_coord[gauss_ind] + y2 * v_coord[gauss_ind] + \
                        y3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    l_z = z1 * u_coord[gauss_ind] + z2 * v_coord[gauss_ind] + \
                        z3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])

                    E = ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (3 / 2)
                    dEdx = 3 * (x - l_x) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
                    dEdy = 3 * (y - l_y) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
                    dEdz = 3 * (z - l_z) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
                    C = 10 ** (-7)
                    phi_x = (d_l_y * (z - l_z) - d_l_z * (y - l_y))
                    phi_y = (d_l_z * (x - l_x) - d_l_x * (z - l_z))
                    phi_z = (d_l_x * (y - l_y) - d_l_y * (x - l_x))
                    theta_factor = ((E * E) ** (-1)) * (-1) * C
                    dBz_dx = theta_factor * dEdx * phi_z + (E ** (-1)) * C * (-1) * d_l_y
                    dBz_dy = theta_factor * dEdy * phi_z + (E ** (-1)) * C * d_l_x
                    dBz_dz = theta_factor * dEdz * phi_z

                    DBzdx += dBz_dx * gauss_weight[gauss_ind]
                    DBzdy += dBz_dy * gauss_weight[gauss_ind]
                    DBzdz += dBz_dz * gauss_weight[gauss_ind]

            gradient_sensitivity_matrix[:, :, node_ind] = np.vstack(
                (DBzdx, DBzdy, DBzdz)) * biot_savart_coeff * plate_thickness
        
        
        coil_parts[part_ind].gradient_sensitivity_matrix = gradient_sensitivity_matrix
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
