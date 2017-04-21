import time
import subprocess
import os
import itertools
import numpy as np
from os import system, chdir

from sys import platform

from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import BuildComplex



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

def build_filtration(in_filename, params, start=0):
    # lines = open(in_filename).readlines()


    print "building filtration..."
    filtration = BuildComplex.build_filtration(in_filename, params)
    # witness_coords = filtration[1][1]
    # landmark_coords = filtration[1][0]
    complexes = sorted(list(filtration[0]))

    return complexes


def group_by_birth_time(complex_ID_list):
    """Reformats 1D list of SimplexBirth objects into 2D array of
    landmark_set lists, where 2nd index is  birth time (? see below)"""

    # TODO: ensure that if a time t has no births, the row t is empty/skipped

    complex_ID_array = []  # list of complex_at_t lists
    complex_at_t = []  # list of simplices with same birth_time
    i = 0
    time = 0
    list_length = len(complex_ID_list)
    while i < list_length:
        birth_time = complex_ID_list[i].birth_time
        if birth_time == time:
            complex_at_t.append(complex_ID_list[i].landmark_set)
            i += 1
        else:
            complex_ID_array.append(complex_at_t)
            complex_at_t = []
            time += 1
    return complex_ID_array


def expand_to_2simplexes(filt_array):
    """for each k-simplex in filtration array, if k > 2, replaces with the
    component 2-simplexes(i.e. all length-3 subsets of landmark_ID_set) """
    for row in filt_array:
        expanded_row = []
        for landmark_ID_set in row:
            expanded_set = list(itertools.combinations(landmark_ID_set, 3)) \
                if len(landmark_ID_set) > 3 else [list(landmark_ID_set)]
            expanded_row.extend(expanded_set)
        row[:] = expanded_row


def build_perseus_in_file(filt_array):
    print 'building perseus_in.txt...'
    out_file = open('perseus/perseus_in.txt', 'a')
    out_file.truncate(0)
    out_file.write('1\n')
    for idx, row in enumerate(filt_array):
        for simplex in row:
            #   format for perseus...
            line_str = str(len(simplex) - 1) + ' ' + ' '.join(
                str(ID) for ID in simplex) + ' ' + str(idx + 1) + '\n'
            out_file.write(line_str)
    out_file.close()


def get_rank_data():
    birth_t, death_t = np.loadtxt('perseus/perseus_out_1.txt', unpack=True)

    epsilons = np.loadtxt('filtration_data/epsilons.txt')
    lim = np.max(epsilons)

    birth_e = []
    death_e = []

    for times in zip(birth_t, death_t):
        if times[1] != - 1:
            birth_e.append(epsilons[int(times[0])])
            death_e.append(epsilons[int(times[1])])


    # immortal_holes = np.asarray([[epsilons[int(birth_t[i])], lim] for i, death_time in enumerate(death_t) if death_time == -1]) # place immoral ho

    immortal_holes = []
    for i, death_time in enumerate(death_t):    # place immortal holes at [birth time, time lim]
        if death_time == -1:
            immortal_holes.append([epsilons[int(birth_t[i])], lim])
    immortal_holes = np.array(immortal_holes)

    if immortal_holes.size:
        birth_e.extend(immortal_holes[:,0])
        death_e.extend(immortal_holes[:,1])


    count = np.zeros(len(birth_t))
    for i, pt in enumerate(zip(birth_e, death_e)):
        for scanner_pt in zip(birth_e, death_e):
            if pt == scanner_pt:
                count[i] += 1



    points = np.asarray([birth_e,
                         death_e,
                         count]).T
    points = np.vstack({tuple(row) for row in points})  # toss duplicates


    edge_b = np.arange(0, 1.1 * lim, .2)
    edge_d = np.full(len(edge_b), 1.1 * lim)
    edge_count = np.zeros(len(edge_b))
    edge = np.asarray([edge_b, edge_d, edge_count]).T

    points = np.vstack((points, edge))

    x, y, z = points[:,0], points[:,1], points[:,2]
    return x, y, z, lim

def call_perseus(filt_list):
    if not isinstance(filt_list, basestring):
        filt_array = group_by_birth_time(filt_list)
        expand_to_2simplexes(filt_array)
        filt_array = np.asarray(filt_array)
        build_perseus_in_file(filt_array)


    print 'calling perseus...'
    os.chdir('perseus')

    if platform == "linux" or platform == "linux2":
        subprocess.call("./perseusLin nmfsimtop perseus_in.txt perseus_out", shell=True)

    elif platform == "darwin":  # macOS
        subprocess.call("./perseusMac nmfsimtop perseus_in.txt perseus_out", shell=True)

    else:   # Windows
        subprocess.call("perseusWin.exe nmfsimtop perseus_in.txt perseus_out", shell=True)

    os.chdir('..')


def get_volume(filt_list, i):

    def plot_surface(f):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('birth e')
        ax.set_ylabel('death_e')

        xnew = np.arange(min_lim * max_lim, max_lim, step)
        ynew = np.arange(min_lim * max_lim, max_lim, step)
        xnew, ynew = np.meshgrid(xnew, ynew)

        znew = f(xnew, ynew)
        znew[xnew > ynew] = np.nan
        ax.scatter(x, y, z, color='red', zorder=10)
        ax.plot_surface(xnew, ynew, znew, zorder=0)

        plt.show()
        plt.clf()

    def plot_contour(f, filename):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('birth e')
        ax.set_ylabel('death_e')

        xnew = np.arange(min_lim * max_lim, max_lim, step)
        ynew = np.arange(min_lim * max_lim, max_lim, step)
        xnew, ynew = np.meshgrid(xnew, ynew)

        znew = f(xnew, ynew)
        znew[xnew > ynew] = np.nan
        ax.contourf(xnew, ynew, znew)

        plt.savefig(filename)
        plt.close(fig)

    call_perseus(filt_list)

    x, y, z, max_lim = get_rank_data()
    min_lim = .1 * max_lim
    # min_lim = 0
    step = .05

    try:
        f = interpolate.Rbf(x, y, z, function='linear')
        filename = 'mrf_data/contour' + str(i) + '.png'
        # plot_surface(f)
        plot_contour(f, filename)
    except ZeroDivisionError:
        print 'zero division error!!'
        return np.nan, np.nan

    vol, err = integrate.dblquad(f, min_lim, max_lim, lambda x: x, lambda x: max_lim)
    return vol, err



def get_volumes(base_filename, i_arr, i_ref, filt_params):
    filename = base_filename + str(i_ref) + '.txt'
    filt_list = build_filtration(filename, filt_params)
    # ref_vol = get_volume('skip_build_filt')
    ref_vol, err = get_volume(filt_list, '_ref')
    print ''
    print 'ref vol:', ref_vol
    print ''

    vols = []
    for i in i_arr:
        filename = base_filename + str(i) + '.txt'
        filt_list = build_filtration(filename, filt_params)
        vol, err = get_volume(filt_list, i)
        vols.append(vol)
        print ''
        print 'i:', i, 'vol:',  (vol)
        print ''
    return ref_vol, np.array(vols)



if __name__ == '__main__':
    start = time.time()
    params = old_parameter_set
    # params = standard_parameter_set
    params.update(
        {
            'ds_rate' : 100,
            'worm_length' : 5000,
            'max_filtration_param': -10,
            'num_divisions' : 30
        })

    # i_ref = 30
    # i_arr = np.arange(30, 40, 1)
    # ref_vol, vols = get_volumes('test_cases/L63_x_m2_tau', i_arr, i_ref, params)
    #
    # distances = vols - ref_vol
    # print ''
    # print 'ref_vol:', ref_vol
    # print 'i:', i_arr
    # print 'distances:', distances
    # print
    # print 'time elapsed:', time.time() - start
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(i_arr, np.absolute(distances))
    # ax.set_xlabel('tau (samples)')
    # ax.set_ylabel('$delta$')
    # ax.set_ylim(bottom=0)
    # plt.savefig('distances.png')

    filename = 'test_cases/L63_x_m2_tau35.txt'

    filt_1 = BuildComplex.build_filtration(filename, params)
    filt_1_sort = np.array(sorted(list(filt_1[0])))


    filt_2 = BuildComplex.build_filtration(filename, params)
    filt_2_sort = np.array(sorted(list(filt_2[0])))

    print 'equal?'
    print filt_1 == filt_2
    print np.array_equal(filt_1_sort, filt_2_sort)
