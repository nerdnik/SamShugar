from __future__ import division

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

def build_and_save_filtration(in_filename, params, start=0):
	# lines = open(in_filename).readlines()


	print "building filtration..."
	filtration = BuildComplex.build_filtration(in_filename, params)
	# witness_coords = filtration[1][1]
	# landmark_coords = filtration[1][0]
	complexes = sorted(list(filtration[0]))
	np.save('filtration_data/complexes.npy', complexes)

	return complexes


def get_interval_data():
	""" formats perseus output """
	birth_t, death_t = np.loadtxt('perseus/perseus_out_1.txt', unpack=True)

	epsilons = np.loadtxt('filtration_data/epsilons.txt')
	lim = np.max(epsilons)

	birth_e = []
	death_e = []

	for times in zip(birth_t, death_t):
		if times[1] != - 1:
			birth_e.append(epsilons[int(times[0])])
			death_e.append(epsilons[int(times[1])])


	immortal_holes = []
	for i, death_time in enumerate(death_t):    # place immortal holes at [birth time, time lim]
		if death_time == -1:
			immortal_holes.append([epsilons[int(birth_t[i])], lim * .95])
	immortal_holes = np.array(immortal_holes)

	if len(immortal_holes):
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


	x, y, z = points[:,0], points[:,1], points[:,2]
	return x, y, z, lim


def get_homology(filt_list):
	""" calls perseus """

	if filt_list == 'read':
		filt_list = np.load('filtration_data/complexes.npy')

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


def get_volume(intervals):

	x, y, z, max_lim = intervals
	# min_lim = .1 * max_lim
	min_lim = 0
	step = .05

	vol, err = integrate.dblquad(f, min_lim, max_lim, lambda x: x, lambda x: max_lim)
	return vol, err


def build_rank_func(data):
	""" helper for get_rank_func()"""
	x, y, z, max_lim = data
	min_lim = 0

	div = .05 * max_lim
	x_ = np.arange(min_lim, max_lim, div)
	y_ = np.arange(min_lim, max_lim, div)
	xx, yy = np.meshgrid(x_, y_)

	pts = zip(x, y, z)
	grid_pts = zip(np.nditer(xx), np.nditer(yy))
	grid_vals = np.zeros(len(grid_pts))
	for i, grid_pt in enumerate(grid_pts):
		if grid_pt[0] <= grid_pt[1]:
			for pt in pts:
				if pt[0] <= grid_pt[0] and pt[1] >= grid_pt[1]:
					grid_vals[i] += pt[2]
		else:
			grid_vals[i] = np.nan
	grid_vals = np.reshape(grid_vals, xx.shape)

	return [xx, yy, grid_vals, max_lim]


def get_rank_func(filename, filt_params):
	filt = build_and_save_filtration(filename, filt_params)
	get_homology(filt)
	intervals = get_interval_data()
	return build_rank_func(intervals)


def PRF_dist_plots(dir, base_filename, i_ref, i_arr, filt_params, rebuild_filt=True):
	if rebuild_filt:
		# filename = '{}/{}{}'.format(dir, i_ref, base_filename)
		filename = '{}/{}{}.txt'.format(dir, base_filename, i_ref)

		ref_func = get_rank_func(filename, filt_params)

		funcs = []
		for i in i_arr:
			# filename = '{}/{}{}'.format(dir, i, base_filename)
			filename = '{}/{}{}.txt'.format(dir, base_filename, i)

			print '\n============================================='
			print '*', filename
			print '=============================================\n'

			func = get_rank_func(filename, filt_params)
			funcs.append(func)

		funcs = np.array(funcs)

		np.save('MRFs.npy', funcs)
		np.save('MRF_ref.npy', ref_func)


	funcs = np.load('MRFs.npy')
	ref_func = np.load('MRF_ref.npy')

	box_area = (ref_func[3] / len(ref_func[2])) ** 2

	diffs = np.array([np.subtract(func[2], ref_func[2]) for func in funcs])

	dists = np.array([np.abs(np.nansum(diff)) * box_area  for diff in diffs])

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111)
	ax.plot(i_arr, dists)
	ax.set_xlabel('$tau \quad (samples)$')
	ax.set_ylabel('$distance \quad ({\epsilon}^2 \; \# \; holes)$')
	ax.xaxis.set_ticks(i_arr[::2])
	ax.grid()
	ax.set_ylim(bottom=0)
	ax.set_title('reference tau: ' + str(i_ref))
	plt.savefig('distances_L63.png')



def see(filename, filt_params):

	filt = build_and_save_filtration(filename, filt_params)
	get_homology(filt)
	intervals = get_interval_data()
	max_lim = intervals[3]
	x, y, z = build_rank_func(intervals)
	z = np.log10(z + 1)

	fig = plt.figure(figsize=(12, 4), dpi = 300, tight_layout=True)
	ax = fig.add_subplot(131, projection='3d')
	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.view_init(35, 135)
	ax.plot_surface(x, y, z)

	# plt.show()

	ax = fig.add_subplot(132)
	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.set_aspect('equal')
	ax.contourf(x, y, z)

	ax = fig.add_subplot(133)
	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.set_aspect('equal')
	ax.scatter(intervals[0], intervals[1], s=intervals[2] * 5)

	plt.savefig('rank_function_log.png')

