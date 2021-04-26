import os
import numpy as np
import h5py


def read_stars(path, directory, update=False):
    '''
    Reads in the unique star IDs that have .abu files in the target directory
    at the given path and saves this list to a file with the name of
    {directory}_stars.txt to save load time on later calls, unless the file
    does not exist or update is set to True.
    '''

    filename = f'{directory}_stars.txt'

    if not os.path.exists(filename) or update:
        with open(filename, 'w') as star_file:
            stars = []
            # For each file in the target directory read the name of the file
            # and locate the star ID used by BACCHUS (usually ELEM-STARID.abu)
            for paths, directories, files in os.walk(f'{path}/{directory}'):
                for file in files:
                    file = file.split(u'-')
                    file = file[1]
                    file = file.rstrip('.abu')
                    file = file.split('_')
                    star = file[0] + '_' + file[1]
                    stars += [star]
            stars = np.array(stars)
            stars = np.unique(stars)
            print('# STAR_ID', file=star_file)
            for star in stars:
                print(star, file=star_file)
    else:
        stars = np.genfromtxt(filename, names=True,
                              dtype=None, encoding=None)['STAR_ID']

    return stars


def get_element_list(path):
    '''
    Reads the BACCHUS line list files and returns a dictionary whose keys
    are the elements and the values are lists of the lines that BACCHUS used.
    '''
    elem_list = []
    lines_list = []
    line_list_file = open(f'{path}/elements.wln', 'r')
    mp_line_list_file = open(f'{path}/elements_MP.wln', 'r')

    # Each line in the elements.wlm files should a single element
    # in the form
    # "Atomic_Number Element_Abbreviation space_delim_list_of_lines"
    # loop through each element and add it to elem_list and a list
    # of lines used for that element to lines_list

    # The elements.wln and elements_MP.wlm should have the same number of
    # elements so we can zip together their lines
    for line, line_mp in zip(line_list_file, mp_line_list_file):
        # Split up each line from the files
        sep_line = line.split()
        sep_line_mp = line_mp.split()

        # Make sure that we have zipped together lines from that correspond
        # to the same element
        assert (sep_line[1] == sep_line_mp[1]
                ), 'Must add the same element to the list at the same time'

        # Add the element to the elem_list
        elem_list += [sep_line[1]]

        # Convert the line list wavelengths to floats and add them to a list
        temp_line_list = []
        for wave in sep_line[2:]:
            temp_line_list += [float(wave)]
        for wave_mp in sep_line_mp[2:]:
            if float(wave_mp) not in temp_line_list:
                temp_line_list += [float(wave_mp)]

        # Sort the list (this is probably unnecessary because unique should
        # sort as well) and get the unique line list wavelengths to avoid
        # duplicates between the normal and metal-poor line lists. Round
        # the wavelengths as they appear in the .abu files and save this
        # list of lines to lines_list
        temp_line_list.sort()
        temp_line_list = list(
            np.round(np.unique(np.array(temp_line_list)), 1))
        lines_list += [temp_line_list]

    # Make a dictionary of each element as keys with the list of its lines
    # as items
    elem_line_dict = {elem_list[i]: lines_list[i]
                      for i in range(len(elem_list))}

    return elem_line_dict
