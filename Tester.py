import sys
from TestingFunctions import build_and_save_filtration, make_filtration_movie,make_persistence_diagram

from FiltrationPlotter import make_frame3D

test = 10
# test = int(sys.argv[1])


old_parameter_set = {
    "num_divisions": 50,
    "max_filtration_param": -20,
    "min_filtration_param": 0,
    "start": 0,
    "worm_length": 10000,
    "ds_rate": 50,
    "landmark_selector": "maxmin",
    "use_ne_for_maxmin": False,
    "d_speed_amplify": 1,
    "d_orientation_amplify": 1,
    "d_stretch": 1,
    "d_ray_distance_amplify": 1,
    "d_use_hamiltonian": 0,
    "d_cov" : 0,
    "simplex_cutoff": 0,
    "weak": False,
    "absolute": False,
    "use_cliques": False,
    "use_twr": False,
    "m2_d": 0,  #Set to anything but 0 to run, set 'time_order_landmarks' = TRUE (don't think i need last part anymore - CHECK)
    "straight_VB": 0, #
    "out": None,
    "program": "Perseus",
    "dimension_cutoff": 2,
    "time_order_landmarks": False,
    "connect_time_1_skeleton": False,
    "reentry_filter": False,
    "store_top_simplices": True,
    "sort_output": False,
    "graph_induced": False    # Use graph induced complex to build filtration.
}


if test == 1:
    in_data_file_name = "datasets/L63_x_m2/L63_x_m2_tau7.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate' : 60,
            'worm_length' : 10000,
            'max_filtration_param': -10,
            'd_cov': 20,

        })

    start_pt = 0   # skip first half of in data file (primitive sliding window)
    build_and_save_filtration(in_data_file_name, build_filt_params, start=start_pt) # comment out to reuse filtration

    make_filtration_movie(
        in_data_file_name,              # used to check if saved filtration is up to date, and in titlebox
        "output/L63_x_m2_tau7_movie.mp4",      		# output filename
        build_filt_params,              # passed to BuildComplex.build_filtration()

        # the following are optional plotting parameters and may be omitted
        # see documentation at line 76 of TestingFunctions.py.
        color_scheme='highlight new',
        framerate=1,
    )

    make_persistence_diagram(
        in_data_file_name,
        "output/L63_x_m2_tau7_persistence_new.png",
        build_filt_params
    )



if test == 2:
    in_data_file_name = "datasets/btc2milIC123.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate' : 20,
            'worm_length' : 5000,
            'max_filtration_param': -10,
        })

    # build_and_save_filtration(in_data_file_name, build_filt_params, start=0)

    # make_persistence_diagram(
    #     in_data_file_name,
    #     "output/persistence_diagram_test.png",
    #     build_filt_params
    # )
    #
    # make_filtration_movie(
    #     in_data_file_name,
    #     "output/3d_movie_test_start.mp4",
    #     build_filt_params,
    #     color_scheme='none',
    #     framerate=1,
    #     hide_1simplexes=True,
    #     camera_angle=[55, 135]
    # )

    make_frame3D(5, hide_1simplexes=True, alpha=1)


if test == 3:
    in_data_file_name = "datasets/L96N22F5_x1_m2tau10.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate' : 200,
            'worm_length' : 5000,
            'max_filtration_param': -20
        })

    build_and_save_filtration(in_data_file_name, build_filt_params)

    make_persistence_diagram(
        in_data_file_name,
        "persistence_diagram_test.png",
        build_filt_params
    )


if test == 4:
    in_data_file_name = "datasets/L96N22F5_x1_m2tau10.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate' : 200,
            'worm_length' : 5000,
            'max_filtration_param': -20
        })

    start_pt = .5   # skip first half of in data file (primitive sliding window)
    build_and_save_filtration(in_data_file_name, build_filt_params, start=start_pt) # comment out to reuse filtration

    make_filtration_movie(
        in_data_file_name,              # used to check if saved filtration is up to date, and in titlebox
        "test4.mp4",      # output filename
        build_filt_params,              # passed to BuildComplex.build_filtration()

        # the following are optional plotting parameters and may be omitted
        # see documentation at line 76 of TestingFunctions.py.
        color_scheme='highlight new',
        max_frames=10,
        framerate=1,
    )


if test == 5:
    in_data_file_name = "datasets/btc2milIC123.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate': 200,
            'worm_length': 5000,
            'max_filtration_param': -20
        })

    build_and_save_filtration(in_data_file_name, build_filt_params, start=0)

    make_filtration_movie(
        in_data_file_name,
        "test5.mp4",
        build_filt_params,
        color_scheme='none',
        max_frames=10,
        framerate=1,
        hide_1simplexes=True
    )
    

if test == 6:
	in_data_file_name = "datasets/Annulus1_np20r1L5dp1.txt"
	build_filt_params = old_parameter_set
	build_filt_params.update(
        {
            'ds_rate': 4,
            'worm_length': 160,
            'max_filtration_param': -8,
			'd_use_hamiltonian': -10,
			'landmark_selector': 'EST'
		})

	build_and_save_filtration(in_data_file_name, build_filt_params, start=0)

	make_filtration_movie(
		in_data_file_name,
		"test6.mp4",
		build_filt_params,
		color_scheme='none',
		max_frames= 50,
		framerate=1
	)
	
if test == 7:

	for i in xrange(3):
		print i 
		in_data_file_name = "datasets/L63_x_m2/L63_x_m2_tau%s.txt" % str(i+2)
		print '%s' % str(in_data_file_name)
		build_filt_params = old_parameter_set
		build_filt_params.update(
        	{
            	'ds_rate' : 50,
            	'worm_length' : 10000,
            	'max_filtration_param': -20
        	})

		build_and_save_filtration(in_data_file_name, build_filt_params)
		print 'Making PD %s' % str(i + 2)
		make_persistence_diagram(
        	in_data_file_name,
        	"PD_L63_x_m2_tau%s.png" % str(i+2),
        	build_filt_params
    	)


if test == 8:

	for i in xrange(3):
		print i 
		in_data_file_name = "datasets/L63_x_m2/L63_x_m2_tau%s.txt" % str(i+2)
		print '%s' % str(in_data_file_name)
		build_filt_params = old_parameter_set
		build_filt_params.update(
        	{
            	'ds_rate' : 50,
            	'worm_length' : 10000,
            	'max_filtration_param': -20
        	})

		build_and_save_filtration(in_data_file_name, build_filt_params)
		print 'Making PD %s' % str(i + 2)
		
		make_filtration_movie(
			in_data_file_name,
			"test8.mp4",
			build_filt_params,
			color_scheme='none',
			max_frames= 50,
			framerate=1
    	)

if test == 9:
    in_data_file_name = "datasets/L63_x_m2/L63_x_m2_tau10.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate': 100,
            'worm_length' : 10000,
            'd_cov': -3
        })
    build_and_save_filtration(in_data_file_name,build_filt_params)
    make_filtration_movie(
        in_data_file_name,
        "test9.mp4",
        build_filt_params,
        color_scheme='none',
        max_frames= 50,
        framerate=1
    )

if test == 10:
    in_data_file_name = "datasets/L63_x_m2/L63_x_m2_tau7.txt"
    build_filt_params = old_parameter_set
    build_filt_params.update(
        {
            'ds_rate' : 25,
            'worm_length' : 5000,
            'max_filtration_param': -10,
            'num_divisions' : 30
        })

    start_pt = 0   # skip first half of in data file (primitive sliding window)
    build_and_save_filtration(in_data_file_name, build_filt_params, start=start_pt) # comment out to reuse filtration

    make_filtration_movie(
        in_data_file_name,
        "output/L63_x_m2_tau7_movie.mp4",
        build_filt_params,
        color_scheme='highlight new',
        framerate=1,
        frame_debug=True
    )

    make_persistence_diagram(
        in_data_file_name,
        "output/L63_x_m2_tau7_persistence.png",
        build_filt_params
    )

print 'test %d complete.' % test
