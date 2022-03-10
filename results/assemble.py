import os
import numpy as np
import h5py
from bacchus_tools.solar_abundances import asplund_2005
from bacchus_tools.read_element_linelist import get_element_list


def read_stars(path, directory, update_starlist=False):
    '''
    Reads in the unique star IDs that have .abu files in the target directory
    at the given path and saves this list to a file with the name of
    {directory}_stars.txt to save load time on later calls, unless the file
    does not exist or update is set to True.
    '''

    filename = f'{directory}_stars.txt'
    elem_filename = f'{directory}_elements.txt'

    if not os.path.exists(filename) or not os.path.exists(elem_filename) or update_starlist:
        stars = []
        elements = []
        # For each file in the target directory read the name of the file
        # and locate the star ID used by BACCHUS (usually ELEM-STARID.abu)
        for paths, directories, files in os.walk(f'{path}/{directory}'):
            for file in files:
                if '.abu' in file:
                    file = file.split(u'-', 1)
                    element = file[0]
                    elements += [element]
                    # file = file[1]
                    # file = file.rstrip('.abu')
                    # file = file.split('_')
                    # star = f'{file[0]}_{file[1]}'
                    # # star = file[0]
                    # star = star.rstrip('tmp')
                    # stars += [star]
                elif '.par' in file:
                    file = file.rstrip('.par')
                    star = file
                    stars += [star]
                else:
                    pass
            # for directory in directories:
            #     star = directory
            #     stars += [star]
        stars = np.array(stars)
        stars = np.unique(stars)
        elements = np.array(elements)
        elements = np.unique(elements)

        with open(filename, 'w') as star_file:
            print('# STAR_ID', file=star_file)
            for star in stars:
                print(star, file=star_file)
        with open(elem_filename, 'w') as elem_file:
            print('# elements', file=elem_file)
            for element in elements:
                print(element, file=elem_file)
    else:
        stars = np.genfromtxt(filename, names=True,
                              dtype=None, encoding=None)['STAR_ID']
        elements = np.genfromtxt(elem_filename, names=True, dtype=None,
                                 encoding=None)['elements']

    return stars, elements


def compile_measurements(path, directory, filename, elem_line_dict=None,
                         stars=None, elements=None, overwrite=False,
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
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

    if stars is None or elements is None:
        stars, elements = read_stars(path, directory,
                                     update_starlist=update_starlist)

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

    data_array = extract_parameters(path, directory, stars)
    data = new_group.create_dataset('param', data=data_array)

    for element in elements:
        data_array = extract_element(path, directory, stars, element,
                                     elem_line_dict[element])
        data = new_group.create_dataset(element, data=data_array)


def read_flag_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = np.array([[file_line.split()[0], float(
                file_line.split()[1]), int(file_line.split()[2])] for file_line in file if len(file_line.split()) >= 3])
    else:
        data = np.array([[None, None, None]])

    return data


def flag_file_value(data, element, line):
    if np.any(data[:, 1].astype(np.float_) == line) and data[data[:, 1].astype(np.float_) == line][0, 0] == element:
        flag_value = int(data[data[:, 1].astype(np.float_) == line][0, 2])
    else:
        flag_value = 9

    return flag_value


def extract_element(path, directory, stars, element, lines):

    # We need are using a fixed format, so if a star is missing a line for an element
    # we need to fill it with a null row which is setup here for ease later
    nan_row = [np.nan, np.nan, np.nan, 9, np.nan, 9,
               np.nan, 9, np.nan, 9, np.nan, 9, np.nan, np.nan,
               np.nan, np.nan, np.nan, np.nan, 9, 9]

    element_comp_list = []
    fail_dir = 'failure_log'
    if not os.path.exists(fail_dir):
        os.makedirs(fail_dir)
    fail_path = f'./{fail_dir}/'
    failed_filename = f'{fail_path}{element}_{directory}_bacchus_failed.txt'
    failed_log = open(failed_filename, 'w')
    print('# STAR_ID', file=failed_log)
    for i, star in enumerate(stars):
        star_elem_file = f'{path}/{directory}/{star}/{element}-{star}.abu'
        star_blend_file = f'{path}/{directory}/{star}/blend.flags'
        star_cont_file = f'{path}/{directory}/{star}/cont.flags'

        blend_data = read_flag_file(star_blend_file)
        cont_data = read_flag_file(star_cont_file)

        row = []
        row += [star]
        # if os.path.exists(star_elem_file):
        if os.path.exists(star_elem_file):
            data = np.genfromtxt(star_elem_file, names=True,
                                 dtype=None, encoding=None, skip_header=1)
            col_id = 1
            for line in lines:
                if np.any(data['lambda'] == line):
                    # print(np.array(data[data['lambda'] == line][0]))
                    row += [*data[data['lambda'] == line][0]]

                    row += [flag_file_value(blend_data, element, line)]
                    row += [flag_file_value(cont_data, element, line)]

                else:
                    row += nan_row

        else:
            print(star, file=failed_log)
            row += nan_row * len(lines)
        # else:
        #    row += [np.nan] * 16 * len(lines)

        element_comp_list += [tuple(row)]

    header = ['lambda', 'eqw_obs', 'syn', 'flag_syn', 'eqw',  'flag_eqw', 'int',
              'flag_int', 'chi2', 'flag_chi2', 'wln', 'flag_wln', 'chi2_val',
              'SNR', 'rvcor', 'limit_syn', 'limit_eqw', 'limit_int', 'flag_blend',
              'flag_cont']

    dtypes = [np.float_, np.float, np.float, np.int8, np.float, np.int8,
              np.float_, np.int8, np.float_, np.int8,  np.float_, np.int8,
              np.float_, np.float_, np.float_, np.float_, np.float_, np.float_,
              np.int8, np.int8]

    new_header = []
    new_header += ['STAR_ID']
    new_dtypes = []
    new_dtypes += ['S24']

    for i in range(len(lines)):
        for label, dtype in zip(header, dtypes):
            new_header += [element + '_' + str(i + 1) + '_' + label]
            new_dtypes += [dtype]

    input_dtype = [(value) for value in zip(new_header, new_dtypes)]
    data_array = np.array(element_comp_list, dtype=input_dtype)

    return data_array


def extract_parameters(path, directory, stars):
    param_comp_list = []

    # Need to extract all of the parameters (below) and then compile them
    # into an array like in extract elements
    # starid, teff, logg, metallicity, microturbulence, alpha_model, c_model
    # convolution

    def format_settings(string):
        return float(string.split('= ')[-1].strip().strip("'"))

    for i, star in enumerate(stars):
        row = []
        row += [star]
        star_par_file = f'{path}/{directory}/{star}/{star}.par'
        star_cno_file = f'{path}/{directory}/{star}/updatedCNO'

        if os.path.exists(star_par_file):
            with open(star_par_file, 'r') as par_file:
                par_lines = par_file.readlines()
                model_lines = [
                    line for line in par_lines if line.find('set MODEL') != -1]
                metallic_lines = [
                    line for line in par_lines if line.find('set METALLIC') != -1]
                turbvel_lines = [
                    line for line in par_lines if line.find('set TURBVEL') != -1]
                alpha_lines = [
                    line for line in par_lines if line.find('set alpha') != -1]
                c_lines = [
                    line for line in par_lines if line.find('set C') != -1]
                convol_lines = [
                    line for line in par_lines if line.find('set convol') != -1]
                snr_lines = [
                    line for line in par_lines if line.find('set SNR') != -1]

                if model_lines == []:
                    row += [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan]
                else:
                    teff = float(model_lines[-1].split()[-1].split('g')[0])
                    logg = float(model_lines[-1].split()
                                 [-1].split('g')[1].split('m')[0].split('z')[0])
                    metallic = format_settings(metallic_lines[-1])
                    turbvel = format_settings(turbvel_lines[-1])
                    alpha = format_settings(alpha_lines[0])
                    cfe = format_settings(
                        c_lines[0]) - asplund_2005()['C'] - metallic
                    convol = format_settings(convol_lines[-1])
                    snr = format_settings(snr_lines[-1])
                    row += [teff, logg, metallic,
                            turbvel, alpha, cfe, convol, snr]

        else:
            row += [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan]

        if os.path.exists(star_cno_file):
            with open(star_cno_file, 'r') as cno_file:
                cno_line = cno_file.readlines()[0]
                cno_items = cno_line.strip().split()
                cno_row_addition = []
                for cno_elem in ['C', 'N', 'O']:
                    if f'updated{cno_elem}=' in cno_items:
                        cno_row_addition += [
                            int(cno_items[cno_items.index(f'updated{cno_elem}=') + 1])]
                    else:
                        cno_row_addition += [9]
                row += cno_row_addition
        else:
            row += [9, 9, 9]

        param_comp_list += [tuple(row)]

    header = ['STAR_ID', 'teff', 'logg', 'fe_h', 'vmicro', 'alpha_fe',  'c_fe',
              'convol', 'snr', 'c_iter', 'n_iter', 'o_iter']

    dtypes = ['S24', np.float_, np.float_, np.float_, np.float_, np.float_,
              np.float_, np.float_, np.float_, np.int8, np.int8, np.int8]

    input_dtype = [(value) for value in zip(header, dtypes)]
    data_array = np.array(param_comp_list, dtype=input_dtype)

    return data_array
