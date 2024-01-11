
#%%
# System imports
import sys

# Logging
import logging

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

    arg_dict = {
        'field_shape_function': 'z',  # definition of the target field
        #'coil_mesh_file': 'bi_planer_rectangles_width_1000mm_distance_500mm.stl',
        'coil_mesh': 'create planar mesh',
        'planar_mesh_parameter_list': [0.35, 0.6, 26, 36, 0, 1, 0, np.pi/2, -0.1, 0, 0],
        #'coil_mesh': 'create bi-planar mesh',
        #'biplanar_mesh_parameter_list': [0.35,0.6,30,20,1,0,0,0,0,0,0.2],
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0,
        'target_region_radius': 0.075,  # in meter
        'target_region_resolution': 15,  # MATLAB 10 is the default
        'use_only_target_mesh_verts': False,
        'b_0_direction':[0,0,1],
        'target_gradient_strength':200,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 20,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor': 0.2,
        'smooth_factor': 5,
        'min_loop_significance': 20,
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
        'tikhonov_reg_factor': 25,  # Tikhonov regularization factor for the SF optimization

        'output_directory': 'images_i',  # [Current directory]
        'project_name': 'half_z_f',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)
#%%
    ##plot result
    from os import makedirs

    import matplotlib.pyplot as plt
    from pyCoilGen.helpers.persistence import load
    import pyCoilGen.plotting as pcg_plt

    which = 'half_z_f'
    solution = load('debug', which, 'final')
    save_dir = f'{solution.input_args.output_directory}'
    makedirs(save_dir, exist_ok=True)

    coil_solutions = [solution]

    # Plot a multi-plot summary of the solution
    pcg_plt.plot_various_error_metrics(coil_solutions, 0, f'{which}', save_dir=save_dir)

    # Plot the 2D projection of stream function contour loops.
    pcg_plt.plot_2D_contours_with_sf(coil_solutions, 0, f'{which} 2D', save_dir=save_dir)
    pcg_plt.plot_3D_contours_with_sf(coil_solutions, 0, f'{which} 3D', save_dir=save_dir)
    
    # Plot the vector fields
    coords = solution.target_field.coords
    
    # Plot the computed target field.
    plot_title=f'{which} Target Field '
    field = solution.solution_errors.combined_field_layout
    pcg_plt.plot_vector_field_xy(coords, field, plot_title=plot_title, save_dir=save_dir)
    
    # Plot the difference between the computed target field and the input target field.
    plot_title=f'{which} Target Field Error '
    field = solution.solution_errors.combined_field_layout - solution.target_field.b
    pcg_plt.plot_vector_field_xy(coords, field, plot_title=plot_title, save_dir=save_dir)