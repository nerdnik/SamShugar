import os
from mayavi import mlab
import numpy as np
import itertools
from matplotlib import collections
import matplotlib.pyplot as pyplot

def unpack_complex_data_2D(complex_list, landmark_coords_data):

    def group_by_birth_time(complex_ID_list):
        """Reformats 1D list of SimplexBirth objects into 2D array of
        landmark_set lists, where 2nd index is  birth time (? see below)"""

        # TODO: ensure that if a time t has no births, the row t is empty/skipped

        complex_ID_array = []	# list of complex_at_t lists
        complex_at_t = []	# list of simplices with same birth_time
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

    def expand_to_2simplexes(ID_array):
        """for each k-simplex in filtration array, if k > 2, replaces with the
        component 2-simplexes(i.e. all length-3 subsets of landmark_ID_set) """
        for row in ID_array:
            expanded_row = []
            for landmark_ID_set in row:
                expanded_set = list(itertools.combinations(landmark_ID_set, 3))\
                    if len(landmark_ID_set) > 3 else [landmark_ID_set]
                expanded_row.append(expanded_set)
            row[:] = expanded_row

    def IDs_to_coords(ID_array):
        """Replaces each landmark_ID with corresponding coordinates"""
        for row in ID_array:
            for parent_simplex in row:
                new_parent_simplex = []
                for child in parent_simplex:
                     new_parent_simplex.append(list(child))
                for child in new_parent_simplex:
                    new_child = []
                    for landmark_ID in child:
                        landmark_coords = landmark_coords_data[landmark_ID]
                        new_child.append(landmark_coords)
                    child[:] = new_child
                parent_simplex[:] = new_parent_simplex

    def flatten_rows(ID_array):
        for row in ID_array:
            new_row = []
            for parent in row:
                for child in parent:
                    new_row.append(child)
            row[:] = new_row

    ID_array = group_by_birth_time(complex_list)
    expand_to_2simplexes(ID_array)
    IDs_to_coords(ID_array)
    flatten_rows(ID_array)
    coords_array = ID_array
    return coords_array

def unpack_complex_data_3D(complex_list):

    def group_by_birth_time(complex_ID_list):
        """Reformats 1D list of SimplexBirth objects into 2D array of
        landmark_set lists, where 2nd index is  birth time (? see below)"""

        # TODO: ensure that if a time t has no births, the row t is empty/skipped

        complex_ID_array = []	# list of complex_at_t lists
        complex_at_t = []	# list of simplices with same birth_time
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

    def expand_to_2simplexes(ID_array):
        """accepts a k-simplex and, if k > 2, returns the component 2-simplexes
        (i.e. all length-3 subsets of landmark_ID_set), else returns input"""
        for row in ID_array:
            expanded_row = []
            for landmark_ID_set in row:
                expanded_set = itertools.combinations(landmark_ID_set, 3)\
                    if len(landmark_ID_set) > 3 else [tuple(landmark_ID_set)]
                expanded_row.extend(expanded_set)
            row[:] = expanded_row

    def separate_by_k (array):
        lines = []
        triangles = []
        for row in array:
            lines_row = []
            triangles_row = []
            for simplex in row:
                if len(simplex) == 2:
                    lines_row.append(simplex)
                else: #if len(simplex) == 3:
                    triangles_row.append(simplex)
            triangles.append(triangles_row)
            lines.append(lines_row)
        return [lines, triangles]

    ID_array = group_by_birth_time(complex_list)
    expand_to_2simplexes(ID_array)
    complexes = separate_by_k(ID_array)
    return complexes

def load_data():
    witness_data = np.load('filtration_data/witness_coords.npy')
    landmark_data = np.load('filtration_data/landmark_coords.npy')
    complex_data = np.load('filtration_data/complexes.npy')
    ambient_dim = len(witness_data[1])
    return ambient_dim, [witness_data, landmark_data, complex_data]

def remove_old_frames():
    os.chdir('filtration_movie_frames')
    for f in os.listdir("."):
        if f.endswith(".png"):
            os.remove(f)
    os.chdir('..')

def add_title(subplot, title_block_info, i):
    in_file_name = title_block_info[0]
    out_file_name = title_block_info[1]
    parameter_set = title_block_info[2]
    color_scheme = title_block_info[3]
    camera_angle = title_block_info[4]
    alpha = title_block_info[5]
    dpi = title_block_info[6]
    max_frames = title_block_info[7]
    hide_1simplexes = title_block_info[8]

    # subplot.axis('tight')
    subplot.axis('off')
    # subplot.xaxis.set_ticks([])
    # subplot.yaxis.set_ticks([])
    subplot.set_xlim([0,1])
    subplot.set_ylim([0,1])

    title_table = subplot.table(
        cellText = [[in_file_name.split('/')[-1]],   # remove leading "datasets/"
                    [out_file_name],
                    ["color scheme:" + str(color_scheme) + ", alpha: " + str(alpha)]],
        bbox=[0, .9, 1, .1],    # x0, y0, width, height
        fontsize=8,
    )

    time_box = subplot.table(
        cellText= [['time', '%d' % i]],
        bbox=[.5, .825, .5, .05])    # x0, y0, width, height
    # time_box.auto_set_font_size(False)
    # time_box.set_fontsize(6)

    param_data = np.array([[key, parameter_set[key]] for key in parameter_set.keys()])
    param_table = subplot.table(
        cellText=param_data,
        colLabels=('PARAM', 'VALUE'),
        colColours=(['c', 'c']),
        colWidths=[1.5, .5],
        bbox=[0, 0, 1, .8], # x0, y0, width, height
    )
    param_table.auto_set_font_size(False)
    param_table.set_fontsize(6)


def get_simplex_color(scheme, birth_time, current_birth_time, max_birth_time):
    """helper for plot_complex()"""
    if scheme == 'none':
        color = (.4, .6, .8)
    elif scheme == 'highlight new':
        color = (1, 0, 0) if birth_time == current_birth_time - 1 else (0, 0, 1)
    elif scheme == 'birth_time gradient':
        prog = birth_time / float(max_birth_time)
        color = (0, prog, 1 - prog)
    else:
        print 'error:', scheme, 'is not a valid color scheme'
    return color

def add_filtration_plot_2D(subplot, birth_time, data, color_scheme, alpha):
    def plot_witnesses(plot, attractor_data):
        attractor_data = np.array(attractor_data)
        x = attractor_data[:, 0]
        y = attractor_data[:, 1]
        plot.scatter(x, y, color='black', s=1, zorder=1)

    def plot_landmarks(plot, landmark_data):
        landmark_data = np.array(landmark_data)
        x = landmark_data[:, 0]
        y = landmark_data[:, 1]
        plot.scatter(x, y, color='green', zorder=2)

    def plot_complex(plot, complex_data, current_birth_time):
        """plots all simplices with birth time =< current_birth_time"""
        max_birth_time = len(complex_data) - 1
        birth_time = 0
        while birth_time < current_birth_time:
            color = get_simplex_color(color_scheme, birth_time, current_birth_time, max_birth_time)
            simplexes_coords = complex_data[birth_time]
            simplexes = collections.PolyCollection(simplexes_coords, edgecolors='black', facecolors=color)
            plot.add_collection(simplexes)
            birth_time += 1



    subplot.set_aspect('equal')
    # subplot.axis('off')
    subplot.xaxis.set_ticks([])
    subplot.yaxis.set_ticks([])

    witness_data = data[0]
    landmark_data = data[1]
    complex_data = data[2]

    plot_witnesses(subplot, witness_data)
    plot_landmarks(subplot, landmark_data)
    plot_complex(subplot, complex_data, birth_time)

def add_filtration_plot_3D(subplot, birth_time, filt_data, color_scheme, alpha, camera_angle, hide_1simplexes):

    def plot_witnesses(witness_data):
        x = witness_data[:, 0]
        y = witness_data[:, 1]
        z = witness_data[:, 2]
        mlab.points3d(x, y, z, mode='point', color=(0, 0, 0))

    def plot_landmarks(landmark_data):
        x = landmark_data[:, 0]
        y = landmark_data[:, 1]
        z = landmark_data[:, 2]
        mlab.points3d(x, y, z, scale_factor=.02, color=(0, .7, 0))

    def plot_complex(complex_data, current_birth_time, landmark_data):  # how to specify color per simplex??
        """plots plots all simplices with birth time =< birth_time"""
        max_birth_time = len(complex_data) - 1
        birth_time = 0
        while birth_time < current_birth_time:
            # color = get_simplex_color_3D(color_scheme, birth_time, current_birth_time, max_birth_time, landmark_data)
            # color = ((<float> for id in triangle_ID) for triangle_ID in triangle_IDs)
            # then, in triangular_mesh(...., scalers=color)

            color = get_simplex_color(color_scheme, birth_time, current_birth_time, max_birth_time)

            triangle_IDs =  complex_data[1][birth_time]
            x = landmark_data[:, 0]
            y = landmark_data[:, 1]
            z = landmark_data[:, 2]
            mlab.triangular_mesh(x, y, z, triangle_IDs, color=color, opacity=alpha, representation='surface')
            mlab.triangular_mesh(x, y, z, triangle_IDs, color=(0, 0, 0), representation='wireframe')

            if hide_1simplexes == False:
                lines = complex_data[0][birth_time]
                for simplex_IDs in lines:
                    ID_coords = np.array([landmark_data[simplex_IDs[0]], landmark_data[simplex_IDs[1]]])
                    x = ID_coords[:, 0]
                    y = ID_coords[:, 1]
                    z = ID_coords[:, 2]
                    mlab.plot3d(x, y, z, tube_radius=None, color=(0,0,0))
                    # mlab.pipeline.line_source(x, y, z, figure=fig)

            birth_time += 1

    if os.name == 'nt':
        mlab.options.offscreen = True
    mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    mlab.view(azimuth=camera_angle[0], elevation=camera_angle[1],focalpoint='auto', distance='auto')

    plot_witnesses(filt_data[0])
    plot_landmarks(filt_data[1])
    plot_complex(filt_data[2], birth_time, filt_data[1])

    filt_plot = mlab.screenshot(antialiased=True)

    subplot.imshow(filt_plot, aspect='auto')
    subplot.xaxis.set_ticks([])
    subplot.yaxis.set_ticks([])
    mlab.clf()

def make_movie_frames(title_block_info, color_scheme, alpha, dpi, camera_angle, max_frames, hide_1simplexes):

    remove_old_frames()

    print 'loading filtration arrays...'
    ambient_dim, filt_data = load_data()
    if ambient_dim == 2:
        filt_data[2] = unpack_complex_data_2D(filt_data[2], filt_data[1])
    elif ambient_dim == 3:
        filt_data[2] = unpack_complex_data_3D(filt_data[2])
    print 'ambient dimension: %d' % ambient_dim

    print 'plotting filtration...'
    if not max_frames: max_frames = len(filt_data[2])
    for i in xrange(max_frames):
        print 'frame %d of %d' % (i + 1, max_frames)
        fig = pyplot.figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
        # print fig.properties()

        title_block = pyplot.subplot2grid((3, 4), (0, 0), rowspan=3)
        filt_plot = pyplot.subplot2grid((3, 4), (0, 1), rowspan=3, colspan=3)

        add_title(title_block, title_block_info, i)
        if ambient_dim == 2:
            add_filtration_plot_2D(filt_plot, i, filt_data, color_scheme, alpha)
        elif ambient_dim == 3:
            add_filtration_plot_3D(filt_plot, i, filt_data, color_scheme, alpha, camera_angle, hide_1simplexes)

        img_name = 'filtration_movie_frames/image%03d.png' % i
        pyplot.savefig(img_name, dpi=dpi)
        pyplot.close(fig)


if __name__ == '__main__':
    pass