import os
import numpy as np
import h5py


def read_stars(path, directory, update_starlist=False):
    '''
    Reads in the unique star IDs that have .abu files in the target directory
    at the given path and saves this list to a file with the name of
    {directory}_stars.txt to save load time on later calls, unless the file
    does not exist or update is set to True.
    '''

    filename = f'{directory}_stars.txt'

    if not os.path.exists(filename) or update_starlist:
        with open(filename, 'w') as star_file:
            stars = []
            # For each file in the target directory read the name of the file
            # and locate the star ID used by BACCHUS (usually ELEM-STARID.abu)
            for paths, directories, files in os.walk(f'{path}/{directory}'):
                for file in files:
                    if file == '.DS_Store':
                        pass
                    else:
                        file = file.split(u'-')
                        file = file[1]
                        file = file.rstrip('.abu')
                        file = file.split('_')
                        star = f'{file[0]}_{file[1]}'
                        # star = file[0]
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


def compile_measurements(path, directory, filename, stars=None, overwrite=False,
                         update_group=False, update_starlist=False):
    '''
    Reads the .abu files at {path}/{directory} and compiles the line-by-line
    abundance measurements and flags in an hdf5 file with {filename}.hdf5.
    Creates an hdf5 group with the name {directory} and within that group
    each element will be its own dataset named according to the abbreviated
    element name.
    '''

    # Currently assuming that the elements.wln and elements_MP.wln are in
    # the same path as the target directory
    elem_line_dict = get_element_list(path)
    if stars is None:
        stars = read_stars(path, directory, update_starlist=update_starlist)

    # Create a new hdf5 file if it doesn't exist or overwrite is true, otherwise
    # append to it
    if not os.path.exists(filename) or overwrite:
        comp_file = h5py.File(f'{filename}.hdf5', 'w')
    else:
        comp_file = h5py.File(f'{filename}.hdf5', 'a')

    # Check to see if the target directory already has a group in the hdf5 file
    # If it does either delete it if update is set to true or raise an error for
    # the user that the file already exists
    if directory in comp_file.keys():
        if update_group:
            del comp_file[directory]
        else:
            raise FileExistsError(
                f'The group {directory} already exists.  Set overwrite=True to overwrite the whole file, or update=True to overwrite this group.')

    # Create a new group for the target directory
    new_group = comp_file.create_group(directory)

    # We need are using a fixed format, so if a star is missing a line for an element
    # we need to fill it with a null row which is setup here for ease later
    nan_row = [np.nan, np.nan, np.nan, 9, np.nan, 9,
               np.nan, 9, np.nan, 9, np.nan, np.nan,
               np.nan, np.nan, np.nan, np.nan]

    for element, lines in elem_line_dict.items():
        element_comp_list = []
        fail_dir = 'failure_log'
        if not os.path.exists(fail_dir):
            os.makedirs(fail_dir)
        fail_path = f'./{fail_dir}/'
        failed_filename = f'{fail_path}{element}_{directory}_bacchus_failed.txt'
        failed_log = open(failed_filename, 'w')
        print('# STAR_ID', file=failed_log)
        for i, star in enumerate(stars):
            star_elem_file = f'{path}/{directory}/{element}-{star}.abu'
            row = []
            row += [star]
            # if os.path.exists(star_elem_file):
            try:
                data = np.genfromtxt(star_elem_file, names=True,
                                     dtype=None, encoding=None, skip_header=1)
                col_id = 1
                for line in lines:
                    if np.any(data['lambda'] == line):
                        # print(np.array(data[data['lambda'] == line][0]))
                        row += [*data[data['lambda'] == line][0]]
                    else:
                        row += nan_row
            except:
                print(star, file=failed_log)
                row += nan_row * len(lines)
            # else:
            #    row += [np.nan] * 16 * len(lines)

            element_comp_list += [tuple(row)]

        header = ['lambda', 'eqw_obs', 'syn', 'flag_syn', 'eqw',  'flag_eqw', 'int',
                  'flag_int', 'chi2', 'flag_chi2', 'chi2_val', 'SNR', 'rvcor',
                  'limit_syn', 'limit_eqw', 'limit_int']

        dtypes = [np.float_, np.float, np.float, np.int8, np.float, np.int8,
                  np.float_, np.int8, np.float_, np.int8, np.float_, np.float_,
                  np.float_, np.float_, np.float_, np.float_]

        new_header = []
        new_header += ['STAR_ID']
        new_dtypes = []
        new_dtypes += ['S18']

        for i in range(len(lines)):
            for label, dtype in zip(header, dtypes):
                new_header += [element + '_' + str(i + 1) + '_' + label]
                new_dtypes += [dtype]

        input_dtype = [(value) for value in zip(new_header, new_dtypes)]
        data_array = np.array(element_comp_list, dtype=input_dtype)
        #data_array[:, 0] = data_array[:, 0].astype('S')
        data = new_group.create_dataset(element, data=data_array)
