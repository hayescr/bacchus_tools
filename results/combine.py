import numpy as np
from bacchus_tools.results import assemble
from bacchus_tools.solar_abundances import asplund_2005
from bacchus_tools.read_element_linelist import get_element_list

# use_line = [0, 0, 0, 0, 1, 1]
#
# use_method = {
#     'syn': [0, 0, 0, 0, 1, 1],
#     'eqw': [0, 0, 0, 0, 1, 1],
#     'int': [0, 0, 0, 0, 1, 1],
#     'chi2': [0, 0, 0, 0, 1, 1]
# }
#
# method_flags = {
#     'syn': [[1], [1], [1], [1], [1], [1], [1, 3], [1], [1]],
#     'eqw': [[1], [1], [1], [1], [1], [1], [1, 3], [1], [1]],
#     'int': [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
#     'chi2': [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
# }


def calculate_abundances(table, group, elem, path='.', solar_abu=None,
                         elem_line_dict=None, best_lines=None, use_line=None,
                         use_method=None, method_flags=None, bc_flags=None,
                         limit_setting=None, zero_points=None, convol_limit=None,
                         updatedo_flags=None):

    # Load solar abundances if they aren't supplied
    if solar_abu is None:
        solar_abu = asplund_2005()

    # Load the BACCHUS element line lists if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

    # Set up the best lines settings if they aren't supplied
    if best_lines is None:
        best_lines = {elem: None}

    if limit_setting is None or elem not in limit_setting:
        limit_setting = {
            elem: {'source': 'bacchus_limit', 'default_limit': 'limit_int'}}

    if zero_points is None or elem not in zero_points:
        zero_points = {elem: None}

    if convol_limit is None:
        def convol_limit(param_table):
            return -9999

    if updatedo_flags is None:
        updatedo_flags = np.full(len(table[f'{group}/param']), 2)

    # Check the provided line, method, and flag settings
    use_line, use_method, method_flags, bc_flags = check_settings(elem, elem_line_dict,
                                                                  use_line, use_method,
                                                                  method_flags, bc_flags)

    elem_vals, errors, elem_counts, elem_limits = combine_measurements(table, group, elem,
                                                                       elem_line_dict,
                                                                       best_lines, use_line,
                                                                       use_method,
                                                                       method_flags, bc_flags,
                                                                       limit_setting,
                                                                       zero_points,
                                                                       convol_limit,
                                                                       updatedo_flags)

    if elem not in solar_abu:
        return elem_vals, errors, elem_counts, elem_limits
    else:
        return elem_vals - solar_abu[elem], errors, elem_counts, elem_limits - solar_abu[elem]


def check_settings(elem, elem_line_dict, use_line, use_method, method_flags, bc_flags):
    '''
    Check the line use, method use and flag settings to confirm that they
    have the same length as the number of lines.  Will set missing methods on
    so that they are used, and missing flags will be set to only use BACCHUS
    method flag=1. If integers are supplied for use_line or a given method
    instead of lists, will apply that single value to all lines.
    '''

    # Set up the line, method and flag settings if they aren't supplied
    method_names = ['syn', 'eqw', 'int', 'chi2', 'wln']

    bc_names = ['blend', 'cont']

    # Use all lines if no specific lines are specified.  Otherwise use the
    # specified lines (make sure there are enough) or apply a single value for
    # all lines
    if use_line is None:
        use_line = [1] * len(elem_line_dict[elem])
    elif isinstance(use_line, int):
        use_line = [use_line] * len(elem_line_dict[elem])
    else:
        assert len(use_line) == len(
            elem_line_dict[elem]), f'use_line should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'

    if use_method is None:
        use_method = {}
    if method_flags is None:
        method_flags = {}
    if bc_flags is None:
        bc_flags = {}

    # Check to make sure that settings are provided for which methods to use
    # and what flags to accept.  If a single value is provided for a given
    # method use that setting for all lines.  Flags must either be not set
    # (defaulting to using flag=1 only) or must be provided for each line
    # separately
    for method in method_names:
        if method not in use_method:
            use_method[method] = use_line
        elif isinstance(use_method[method], int):
            use_method[method] = [use_method[method]] * \
                len(elem_line_dict[elem])
        else:
            assert len(use_method[method]) == len(
                elem_line_dict[elem]), f'the number of settings for use_method["{method}"] should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'
        if method not in method_flags:
            method_flags[method] = [[1]] * len(elem_line_dict[elem])
        else:
            assert len(method_flags[method]) == len(
                elem_line_dict[elem]), f'the number of settings for method_flags["{method}"] should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'

    for bc in bc_names:
        if bc not in bc_flags:
            bc_flags[bc] = [[0, 1, 9]] * len(elem_line_dict[elem])
        else:
            assert len(bc_flags[bc]) == len(
                elem_line_dict[elem]), f'the number of settings for bc_flags["{bc}"] should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'

    return use_line, use_method, method_flags, bc_flags


def combine_measurements(table, group, elem, elem_line_dict, best_lines,
                         use_line, use_method, method_flags, bc_flags,
                         limit_setting, zero_points, convol_limit, updatedo_flags):
    '''
    A function to average line-by-line abundances from BACCHUS (converted to a
    compiled hdf5 table) with an arbitrary choice of lines, methods and flags.
    Returns [X/H] and the [X/H] error for the input element by measuring the
    dispersion between the abundances from each line and method and taking the
    standard error of the mean (assuming n samples = # lines).
    '''

    # -------------------- Calculate average abundances ---------------------- #

    elem_table = table[f'{group}/{elem}']
    param_table = table[f'{group}/param']

    elem_vals = np.zeros(len(elem_table))
    elem_counts = np.zeros(len(elem_table))
    elem_limit_counts = np.zeros(len(elem_table))
    elem_limits = np.full(len(elem_table), np.nan)

    # Keep track of which stars have the best lines measured
    best_line_not_meas = np.zeros(len(elem_table), dtype=bool)

    for i in range(len(elem_line_dict[elem])):
        # skip lines that are not being used
        if use_line[i] == 0:
            continue
        else:
            line_abu = np.zeros(len(elem_table))
            line_count = np.zeros(len(elem_table))
            flags = np.ones(len(elem_table))

            # Loop through each method if it is used and recored which stars
            # have met the input flag conditions
            for method, setting in use_method.items():
                # if setting[i] == 1:
                flags = np.logical_and(
                    flags,
                    np.in1d(elem_table[f'{elem}_{i+1}_flag_{method}'],
                            method_flags[method][i])
                )

            for bc, settings in bc_flags.items():
                flags = np.logical_and(
                    flags,
                    np.in1d(elem_table[f'{elem}_{i+1}_flag_{bc}'],
                            settings[i])
                )

            # Loop through each method and add its measurement
            for method, setting in use_method.items():
                if setting[i] == 1:
                    column = f'{elem}_{i + 1}_{method}'
                    # if elem == 'C12C13':
                    #     line_abu[np.logical_and(
                    #         flags, elem_table[column] > 0.)] += np.log10(elem_table[column][np.logical_and(flags, elem_table[column] > 0.)])
                    #     line_count += np.logical_and(
                    #         flags, elem_table[column] > 0.)
                    # else:
                    #     line_abu[flags] += elem_table[column][flags]
                    #     line_count += flags
                    line_abu[flags] += elem_table[column][flags]
                    line_count += flags

            if limit_setting[elem]['source'] == 'bacchus_limit':
                climit = f"{elem}_{i+1}_{limit_setting[elem]['default_limit']}"
                line_limit = elem_table[climit]
            elif limit_setting[elem]['source'] == 'line_by_line':
                line_settings = limit_setting[elem][elem_line_dict[elem][i]]
                climit = f"{elem}_{i+1}_{line_settings['default_limit']}"
                calc_limit = line_settings['function'](
                    param_table, **line_settings['func_param'])
                # line_limit = np.fmax(elem_table[climit], calc_limit)
                line_limit = calc_limit

            if zero_points[elem] is None:
                line_zero_point = 0.
            elif zero_points[elem]['source'] == 'line_by_line':
                line_zero_point = zero_points[elem]['values'][elem_line_dict[elem][i]]

            line_limit[np.logical_not(flags)] = np.nan
            elem_limits = np.fmin(elem_limits, line_limit - line_zero_point)

            if elem == 'C12C13':
                line_measured = np.logical_and(line_abu / line_count < line_limit,
                                               line_count != 0)

                limit_measured = np.logical_and(
                    line_abu / line_count >= line_limit, line_count != 0)

            else:
                line_measured = np.logical_and(line_abu / line_count > line_limit,
                                               line_count != 0)

                limit_measured = np.logical_and(
                    line_abu / line_count <= line_limit, line_count != 0)

            # Average the method abundances and add one to the count if this
            # line is measured
            elem_vals[line_measured] += (line_abu[line_measured]
                                         / line_count[line_measured]) - line_zero_point
            elem_counts += line_measured

            elem_limit_counts += limit_measured

            # If there are any beest lines, check that they are measured
            if best_lines[elem] is None:
                best_line_not_meas = np.zeros(len(elem_table), dtype=bool)
            elif elem_line_dict[elem][i] in best_lines[elem]:
                best_line_not_meas = np.logical_or(best_line_not_meas,
                                                   np.logical_not(line_measured))

    # Average all of the lines and fill bad measurements with nans
    elem_vals[elem_counts != 0] = elem_vals[elem_counts != 0] / \
        elem_counts[elem_counts != 0]
    elem_vals[elem_counts == 0] = np.nan
    elem_vals[best_line_not_meas] = np.nan
    elem_counts[best_line_not_meas] = 0
    elem_limits[np.logical_or(np.logical_not(
        np.isnan(elem_vals)), elem_limit_counts == 0)] = np.nan
    if elem == 'C12C13':
        elem_counts[elem_vals < 0.] = 0
        elem_vals[elem_vals < 0.] = np.nan

    # ----------------------- Calculate uncertainties ------------------------ #

    errors = np.zeros(len(elem_table))
    elem_error_counts = np.zeros(len(elem_table))

    for i in range(len(elem_line_dict[elem])):
        # skip lines that are not being used
        if use_line[i] == 0:
            continue
        else:
            line_abu = np.zeros(len(elem_table))
            line_count = np.zeros(len(elem_table))
            line_error = np.zeros(len(elem_table))
            line_error_count = np.zeros(len(elem_table))
            flags = np.ones(len(elem_table))

            # Loop through each method if it is used and recored which stars
            # have met the input flag conditions
            for method, setting in use_method.items():
                # if setting[i] == 1:
                flags = np.logical_and(
                    flags,
                    np.in1d(elem_table[f'{elem}_{i+1}_flag_{method}'],
                            method_flags[method][i])
                )

            for bc, settings in bc_flags.items():
                flags = np.logical_and(
                    flags,
                    np.in1d(elem_table[f'{elem}_{i+1}_flag_{bc}'],
                            settings[i])
                )

            if zero_points[elem] is None:
                line_zero_point = 0.
            elif zero_points[elem]['source'] == 'line_by_line':
                line_zero_point = zero_points[elem]['values'][elem_line_dict[elem][i]]

            # Loop through each method and add its measurement
            for method, setting in use_method.items():
                if setting[i] == 1:
                    column = f'{elem}_{i + 1}_{method}'
                    line_abu[flags] += elem_table[column][flags]
                    line_count += flags
                # if setting[i] == 1:
                if method in ['chi2', 'wln']:
                    column = f'{elem}_{i + 1}_{method}'
                    line_error_count += flags
                    line_error[flags] += (elem_table[column]
                                          [flags] - elem_vals[flags] - line_zero_point)**2.

            if limit_setting[elem]['source'] == 'bacchus_limit':
                climit = f"{elem}_{i+1}_{limit_setting[elem]['default_limit']}"
                line_limit = elem_table[climit]
            elif limit_setting[elem]['source'] == 'line_by_line':
                line_settings = limit_setting[elem][elem_line_dict[elem][i]]
                climit = f"{elem}_{i+1}_{line_settings['default_limit']}"
                calc_limit = line_settings['function'](
                    param_table, **line_settings['func_param'])
                # line_limit = np.fmax(elem_table[climit], calc_limit)
                line_limit = calc_limit

            if elem == 'C12C13':
                line_measured = np.logical_and(line_abu / line_count < line_limit,
                                               line_count != 0)
            else:
                line_measured = np.logical_and(line_abu / line_count > line_limit,
                                               line_count != 0)

            errors[line_measured] += line_error[line_measured]
            elem_error_counts[line_measured] += line_error_count[line_measured]

    elem_measured = np.logical_not(np.isnan(elem_vals))
    # Take the standard deviation of different line and method abundance
    # measurements and divide by the number of lines used to get a standard
    # error of the mean where the sample size is n lines not n measurements
    # because the different methods aren't independent samples

    old_errors = np.array(errors)

    errors[elem_measured] = (np.sqrt(errors[elem_measured] / (elem_error_counts[
        elem_measured] - 1))) / np.sqrt(elem_counts[elem_measured])

    errors[np.isnan(elem_vals)] = np.nan

    elem_vals[param_table['convol'] <= convol_limit(param_table)] = np.nan
    errors[param_table['convol'] <= convol_limit(param_table)] = np.nan
    elem_counts[param_table['convol'] <= convol_limit(param_table)] = 0
    elem_limits[param_table['convol'] <= convol_limit(param_table)] = np.nan

    elem_vals[updatedo_flags == 1] = np.nan
    errors[updatedo_flags == 1] = np.nan
    elem_counts[updatedo_flags == 1] = 0
    elem_limits[updatedo_flags == 1] = np.nan

    if elem == 'C12C13':
        errors = np.minimum(errors, 50.)

    return elem_vals, errors, elem_counts, elem_limits


def package_lines(table, group, elem, path='.', elem_line_dict=None):

    # Make an n stars x 4 x nlines array, loop through each method and each
    # line and copy the array to storage[:,j,i] and then do the same for the
    # flags and return these arrays (hope to save them as 4xnlines elements in
    # a fits column)

    # Load solar abundances if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

    elem_table = table[f'{group}/{elem}']

    methods = ['syn', 'eqw', 'int', 'chi2', 'wln']
    bc_methods = ['blend', 'cont']

    abu_array = np.zeros((len(elem_table), len(elem_line_dict[elem]), 5))
    flag_array = np.zeros((len(elem_table), len(elem_line_dict[elem]), 5))
    bc_flag_array = np.zeros((len(elem_table), len(elem_line_dict[elem]), 2))

    for j, method in enumerate(methods):
        for i in range(len(elem_line_dict[elem])):
            abu_column = f'{elem}_{i + 1}_{method}'
            flag_column = f'{elem}_{i+1}_flag_{method}'
            abu_array[:, i, j] = elem_table[abu_column]
            flag_array[:, i, j] = elem_table[flag_column]

    for j, method in enumerate(bc_methods):
        for i in range(len(elem_line_dict[elem])):
            bc_flag_column = f'{elem}_{i+1}_flag_{method}'
            bc_flag_array[:, i, j] = elem_table[bc_flag_column]

    return abu_array, flag_array, bc_flag_array
