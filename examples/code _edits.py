
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
if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

#%%
#edit release
# Logging
import logging

# System imports
import numpy as np
from os import makedirs

# Logging
import logging


# Local imports
from pyCoilGen.sub_functions.constants import *
from pyCoilGen.sub_functions.data_structures import CoilSolution

# For visualisation
from pyCoilGen.helpers.visualisation import visualize_vertex_connections, visualize_compare_contours

# For timing
from pyCoilGen.helpers.timing import Timing

# For saving Pickle files
from pyCoilGen.helpers.persistence import save, save_preoptimised_data

# From original project
from pyCoilGen.sub_functions.read_mesh import read_mesh
from pyCoilGen.sub_functions.parse_input import parse_input, create_input
from pyCoilGen.sub_functions.split_disconnected_mesh import split_disconnected_mesh
from pyCoilGen.sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
# from pyCoilGen.sub_functions.refine_mesh import refine_mesh # Broken
from pyCoilGen.sub_functions.parameterize_mesh import parameterize_mesh
from pyCoilGen.sub_functions.define_target_field import define_target_field
# from pyCoilGen.sub_functions.temp_evaluation import temp_evaluation
from pyCoilGen.sub_functions.calculate_one_ring_by_mesh import calculate_one_ring_by_mesh
from pyCoilGen.sub_functions.calculate_basis_functions import calculate_basis_functions
from pyCoilGen.sub_functions.calculate_sensitivity_matrix import calculate_sensitivity_matrix
from pyCoilGen.sub_functions.calculate_gradient_sensitivity_matrix import calculate_gradient_sensitivity_matrix
from pyCoilGen.sub_functions.calculate_resistance_matrix import calculate_resistance_matrix
from pyCoilGen.sub_functions.stream_function_optimization import stream_function_optimization
from pyCoilGen.sub_functions.calc_potential_levels import calc_potential_levels
from pyCoilGen.sub_functions.calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts
from pyCoilGen.sub_functions.process_raw_loops import process_raw_loops
from pyCoilGen.sub_functions.find_minimal_contour_distance import find_minimal_contour_distance
from pyCoilGen.sub_functions.topological_loop_grouping import topological_loop_grouping
from pyCoilGen.sub_functions.calculate_group_centers import calculate_group_centers
from pyCoilGen.sub_functions.interconnect_within_groups import interconnect_within_groups
from pyCoilGen.sub_functions.interconnect_among_groups import interconnect_among_groups
from pyCoilGen.sub_functions.shift_return_paths import shift_return_paths
from pyCoilGen.sub_functions.generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
from pyCoilGen.sub_functions.create_sweep_along_surface import create_sweep_along_surface
from pyCoilGen.sub_functions.calculate_inductance_by_coil_layout import calculate_inductance_by_coil_layout
from pyCoilGen.sub_functions.load_preoptimized_data import load_preoptimized_data
from pyCoilGen.sub_functions.evaluate_field_errors import evaluate_field_errors
from pyCoilGen.sub_functions.calculate_gradient import calculate_gradient
from pyCoilGen.sub_functions.export_data import export_data, check_exporter_help

# Set up logging
log = logging.getLogger(__name__)


def pyCoilGen(log, input_args=None):
    # Create optimized coil finished coil layout
    # Author: Philipp Amrein, University Freiburg, Medical Center, Radiology, Medical Physics
    # 5.10.2021

    # The following external functions were used in modified form:
    # intreparc@John D'Errico (2010), @matlabcentral/fileexchange
    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan Schmidt rms@dgp.toronto.edu"
    # based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes", NS (2021).
    # Curve intersections (https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections),
    # MATLAB Central File Exchange.

    timer = Timing()
    timer.start()

    # Parse the input variables
    if type(input_args) is dict:
        try:
            if input_args['debug'] >= DEBUG_VERBOSE:
                log.debug(" - converting input dict to input type.")
        except KeyError:
            pass
        input_parser, input_args = create_input(input_args)
    elif input_args is None:
        input_parser, input_args = parse_input(input_args)
    else:
        input_args = input_args

    set_level(input_args.debug)

    project_name = f'{input_args.project_name}'
    persistence_dir = input_args.persistence_dir
    image_dir = input_args.output_directory

    # Create directories if they do not exist
    makedirs(persistence_dir, exist_ok=True)
    makedirs(image_dir, exist_ok=True)

    # Print the input variables
    # DEBUG
    if get_level() >= DEBUG_VERBOSE:
        log.debug('Parse inputs: %s', input_args)


    solution = CoilSolution()
    solution.input_args = input_args

    if check_exporter_help(input_args):
        return solution

    try:
        runpoint_tag = 'test'

        if input_args.sf_source_file == 'none':
            # Read the input mesh
            print('Load geometry:')
            coil_mesh, target_mesh, secondary_target_mesh = read_mesh(input_args)  # 01

            if coil_mesh is None:
                log.info("No coil mesh, exiting.")
                timer.stop()
                return None

            if get_level() >= DEBUG_VERBOSE:
                log.debug(" -- vertices shape: %s", coil_mesh.get_vertices().shape)  # (264,3)
                log.debug(" -- faces shape: %s", coil_mesh.get_faces().shape)  # (480,3)

            if get_level() > DEBUG_VERBOSE:
                log.debug(" coil_mesh.vertex_faces: %s", coil_mesh.trimesh_obj.vertex_faces[0:10])

            if get_level() > DEBUG_VERBOSE:
                coil_mesh.display()

            # Split the mesh and the stream function into disconnected pieces
            print('Split the mesh and the stream function into disconnected pieces.')
            timer.start()
            coil_parts = split_disconnected_mesh(coil_mesh)  # 00
            timer.stop()
            solution.coil_parts = coil_parts
            runpoint_tag = '00'

            # Upsample the mesh density by subdivision
            print('Upsample the mesh by subdivision:')
            timer.start()
            coil_parts = refine_mesh(coil_parts, input_args)  # 01
            timer.stop()
            runpoint_tag = '01'

            # Parameterize the mesh
            print('Parameterize the mesh:')
            timer.start()
            coil_parts = parameterize_mesh(coil_parts, input_args)  # 02
            timer.stop()
            runpoint_tag = '02'

            # Define the target field
            print('Define the target field:')
            timer.start()
            target_field, is_suppressed_point = define_target_field(
                coil_parts, target_mesh, secondary_target_mesh, input_args)
            timer.stop()
            solution.target_field = target_field
            solution.is_suppressed_point = is_suppressed_point
            runpoint_tag = '02b'
           
            if get_level() >= DEBUG_VERBOSE:
                log.debug(" -- target_field.b shape: %s", target_field.b.shape)  # (3, 257)
                log.debug(" -- target_field.coords shape: %s", target_field.coords.shape)  # (3, 257)
                log.debug(" -- target_field.weights shape: %s", target_field.weights.shape)  # (257,)

            # Evaluate the temp data; check whether precalculated values can be used from previous iterations
            # print('Evaluate the temp data:')
            # input_args = temp_evaluation(solution, input_args, target_field)

            # Find indices of mesh nodes for one ring basis functions
            print('Calculate mesh one ring:')
            timer.start()
            coil_parts = calculate_one_ring_by_mesh(coil_parts)  # 03
            timer.stop()
            runpoint_tag = '03'

            # Create the basis function container which represents the current density
            print('Create the basis function container which represents the current density:')
            timer.start()
            coil_parts = calculate_basis_functions(coil_parts)  # 04
            timer.stop()
            runpoint_tag = '04'

            # Calculate the sensitivity matrix Cn
            print('Calculate the sensitivity matrix:')
            timer.start()
            coil_parts = calculate_sensitivity_matrix(coil_parts, target_field, input_args)  # 05
            timer.stop()
            runpoint_tag = '05'

            # Calculate the gradient sensitivity matrix Gn
            print('Calculate the gradient sensitivity matrix:')
            timer.start()
            coil_parts = calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args)  # 06
            timer.stop()
            runpoint_tag = '06'

            # Calculate the resistance matrix Rmn
            print('Calculate the resistance matrix:')
            timer.start()
            coil_parts = calculate_resistance_matrix(coil_parts, input_args)  # 07
            timer.stop()
            runpoint_tag = '07'

            # Optimize the stream function toward target field and further constraints
            print('Optimize the stream function toward target field and secondary constraints:')
            timer.start()
            coil_parts, combined_mesh, sf_b_field = stream_function_optimization(coil_parts, target_field, input_args)
            timer.stop()
            solution.combined_mesh = combined_mesh
            solution.sf_b_field = sf_b_field
            runpoint_tag = '08'

            if input_args.sf_dest_file != 'none':
                print('Persist pre-optimised data:')
                save_preoptimised_data(solution)

        else:
            # Load the preoptimised data
            print('Load pre-optimised data:')
            timer.start()
            solution = load_preoptimized_data(input_args)
            timer.stop()
            coil_parts = solution.coil_parts
            combined_mesh = solution.combined_mesh
            target_field = solution.target_field

        # Calculate the potential levels for the discretization
        print('Calculate the potential levels for the discretization:')
        timer.start()
        coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)  # 09
        timer.stop()
        solution.primary_surface_ind = primary_surface_ind
        runpoint_tag = '09'

        # Generate the contours
        print('Generate the contours:')
        timer.start()
        coil_parts = calc_contours_by_triangular_potential_cuts(coil_parts)  # 10
        timer.stop()
        runpoint_tag = '10'

        #####################################################
        # Visualisation
        if get_level() > DEBUG_NONE:
            for part_index in range(len(coil_parts)):
                coil_part = coil_parts[part_index]
                coil_mesh = coil_part.coil_mesh

                visualize_compare_contours(coil_mesh.uv, 800, f'{image_dir}/10_{project_name}_contours_{part_index}_p.png',
                                           coil_part.contour_lines)
        #
        #####################################################

        # Process contours
        print('Process contours: Evaluate loop significance')
        timer.start()
        coil_parts = process_raw_loops(coil_parts, input_args, target_field)  # 11
        timer.stop()
        runpoint_tag = '11'

        if not input_args.skip_postprocessing:
            # Find the minimal distance between the contour lines
            print('Find the minimal distance between the contour lines:')
            timer.start()
            coil_parts = find_minimal_contour_distance(coil_parts, input_args)  # 12
            timer.stop()
            runpoint_tag = '12'

            # Group the contour loops in topological order
            print('Group the contour loops in topological order:')
            timer.start()
            coil_parts = topological_loop_grouping(coil_parts)  # 13
            timer.stop()
            runpoint_tag = '13'
            for index, coil_part in enumerate(coil_parts):
                print(f'  -- Part {index} has {len(coil_part.groups)} topological groups')

            # Calculate center locations of groups
            print('Calculate center locations of groups:')
            timer.start()
            coil_parts = calculate_group_centers(coil_parts)  # 14
            timer.stop()
            runpoint_tag = '14'

            #####################################################
            # Visualisation
            if get_level() > DEBUG_NONE:
                for part_index in range(len(coil_parts)):
                    coil_part = coil_parts[part_index]
                    coil_mesh = coil_part.coil_mesh
                    c_group_centers = coil_part.group_centers

                    visualize_compare_contours(coil_mesh.uv, 800, f'{image_dir}/14_{project_name}_contour_centres_{part_index}_p.png',
                                               coil_part.contour_lines, c_group_centers.uv)
            #
            #####################################################

            # Interconnect the single groups
            print('Interconnect the single groups:')
            timer.start()
            coil_parts = interconnect_within_groups(coil_parts, input_args)  # 15
            timer.stop()
            runpoint_tag = '15'

            # Interconnect the groups to a single wire path
            print('Interconnect the groups to a single wire path:')
            timer.start()
            coil_parts = interconnect_among_groups(coil_parts, input_args)  # 16
            timer.stop()
            runpoint_tag = '16'

            #####################################################
            # Visualisation
            if get_level() > DEBUG_NONE:
                for index1 in range(len(coil_parts)):
                    c_part = coil_parts[index1]
                    c_wire_path = c_part.wire_path

                    visualize_vertex_connections(
                        c_wire_path.uv.T, 800, f'{image_dir}/16_{project_name}_wire_path2_uv_{index1}_p.png')
            #
            #####################################################

            # Connect the groups and shift the return paths over the surface
            print('Shift the return paths over the surface:')
            timer.start()
            coil_parts = shift_return_paths(coil_parts, input_args)  # 17
            timer.stop()
            runpoint_tag = '17'

            # Create Cylindrical PCB Print
            print('Create PCB Print:')
            timer.start()
            coil_parts = generate_cylindrical_pcb_print(coil_parts, input_args)  # 18
            timer.stop()
            runpoint_tag = '18'

            # Create Sweep Along Surface
            print('Create sweep along surface:')
            timer.start()
            coil_parts = create_sweep_along_surface(coil_parts, input_args)
            timer.stop()
            runpoint_tag = '19'

        # Calculate the inductance by coil layout
        print('Calculate the inductance by coil layout:')
        # coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance
        timer.start()
        solution = calculate_inductance_by_coil_layout(solution, input_args)
        timer.stop()
        runpoint_tag = '20'

        # Evaluate the field errors
        print('Evaluate the field errors:')
        timer.start()
        coil_parts, solution_errors = evaluate_field_errors(
            coil_parts, input_args, solution.target_field, solution.sf_b_field)
        timer.stop()
        solution.solution_errors = solution_errors
        log.info("Layout error: Mean: %f, Max: %f",
                 solution_errors.field_error_vals.mean_rel_error_layout_vs_target,
                 solution_errors.field_error_vals.max_rel_error_layout_vs_target
                 )
        runpoint_tag = '21'

        # Calculate the gradient
        print('Calculate the gradient:')
        timer.start()
        coil_gradient = calculate_gradient(coil_parts, input_args, target_field)
        timer.stop()
        solution.coil_gradient = coil_gradient
        runpoint_tag = '22'

        # Export data
        print('Exporting data:')
        timer.start()
        export_data(solution)
        timer.stop()
        runpoint_tag = '23'

        # Finally, save the completed solution.
        runpoint_tag = 'final'
        print(f'Solution saved to "{save(persistence_dir, project_name, runpoint_tag, solution)}"')

        timer.stop()
    except Exception as e:
        log.error("Caught exception: %s", e)
        save(persistence_dir, project_name, f'{runpoint_tag}_exception', solution)
        raise e
    return solution


#%%
# System imports
import sys

# Logging
import logging

# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'x',  # definition of the target field
        'target_gradient_strength': 200,
        'coil_mesh': 'create bi-planar mesh',
        'planar_mesh_parameter_list': [0.25,0.25,1,1,0,0,1,0,0,0,0.2],
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
        'pot_offset_factor': 0,
        'surface_is_cylinder_flag': False,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.005,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.01,
        'iteration_num_mesh_refinement': 2,  # the number of refinements for the mesh;
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
        if result.target_field.coords[2][i]<=0:# for target field
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
        if result.target_field.coords[1][i]<=0:# for target field
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
        if result.target_field.coords[0][i]<=0:# for target field
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
fig.savefig(os.path.join(mesh,'Target_Field_Target_Resolution_=_%d'%arg_dict['target_region_resolution']))
plt.show()
plt.close()

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
for i in range(len(result.combined_mesh.vertices)):
    if i in mesh_inds:
        ax.scatter(x1[i], y1[i], z1[i], c= 'r')
    else:
        ax.scatter(x1[i], y1[i], z1[i], c= 'b')
    #plt.annotate(i, x1[i], y1[i], z1[i])
ax.set_title('Coil Mesh, Coil Mesh Resolution = %d x '%(arg_dict['planar_mesh_parameter_list'][2]*arg_dict['iteration_num_mesh_refinement']) + '%d'%(arg_dict['planar_mesh_parameter_list'][3]*arg_dict['iteration_num_mesh_refinement']))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig1.savefig(os.path.join(mesh,'Coil_Mesh_Coil_Mesh_Resolution_=_%d'%(arg_dict['planar_mesh_parameter_list'][2]*arg_dict['iteration_num_mesh_refinement']) + '_%d'%(arg_dict['planar_mesh_parameter_list'][3]*arg_dict['iteration_num_mesh_refinement'])))
plt.show()
plt.close()