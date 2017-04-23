import BuildComplex
import numpy as np

standard_parameter_set = {
    "num_divisions": 50,
    "max_filtration_param": -20,
    "min_filtration_param": 0,
    "start": 0,
    "worm_length": 0,
    "ds_rate": 50,
    "landmark_selector": "maxmin",
    "use_ne_for_maxmin": False,
    "d_speed_amplify": 1,
    "d_orientation_amplify": 1,
    "d_stretch": 1,
    "d_ray_distance_amplify": 1,
    "d_use_hamiltonian": 0,
    "d_cov":0,
    "simplex_cutoff": 0,
    "weak": False,
    "absolute": False,
    "use_cliques": False,
    "use_twr": False,
    "m2_d": 0,
    "straight_VB" : 0,
    "out": None,
    "program": "Perseus",
    "dimension_cutoff": 2,
    "time_order_landmarks": False,
    "connect_time_1_skeleton": False,
    "reentry_filter": False,
    "store_top_simplices": True,
    "sort_output": False,
    "graph_induced": False
}

def are_same(filt1,filt2):
    if len(filt1)!=len(filt2):
        print "Lists are not the same lenth"
        return False
    else:
        for i in range(0,len(filt1)):
            obj1 = filt1[i]
            obj2 = filt2[i]
            if obj1.landmark_set!=obj2.landmark_set:
                return False
            elif obj1.birth_time!=obj2.birth_time:    
                return False
    return True
filename = 'test_cases/L63_x_m2_tau35.txt'
# filename = 'datasets/btc2millC123.txt'
params = standard_parameter_set
params.update({
    'worm_length': 5000
})

filt_1 = BuildComplex.build_filtration(filename, params)
filt_1_sort = np.array(sorted(list(filt_1[0])))

filt_2 = BuildComplex.build_filtration(filename, params)
filt_2_sort = np.array(sorted(list(filt_2[0])))

print are_same(filt_1_sort,filt_2_sort)











