import numpy as np
import h5py
from bacchus_tools.results import assemble
from bacchus_tools.solar_abundances import asplund_2005

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
                         use_method=None, method_flags=None):

    # Load solar abundances if they aren't supplied
    if solar_abu is None:
        solar_abu = asplund_2005()

    # Load solar abundances if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = assemble.get_element_list(path)

    # Set up the best lines settings if they aren't supplied
    if best_lines is None:
        best_lines = {elem: None}

    # Set up the line, method and flag settings if they aren't supplied
    # method_names = ['syn', 'eqw', 'int', 'chi2']
    # if use_line is None:
    #     use_line = [1] * len(elem_line_dict[elem])
    # else:
    #     assert len(use_line) == len(
    #         elem_line_dict[elem]), f'use_line should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'
    # if use_method is None:
    #     use_method = {}
    # if method_flags is None:
    #     method_flags = {}
    #
    # for method in method_names:
    #     if method not in use_method:
    #         use_method[method] = use_line
    #     else:
    #         assert len(use_method[method]) == len(
    #             elem_line_dict[elem]), f'the number of settings for use_method["{method}"] should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'
    #     if method not in method_flags:
    #         method_flags[method] = [[1]] * len(elem_line_dict[elem])
    #     else:
    #         assert len(method_flags[method]) == len(
    #             elem_line_dict[elem]), f'the number of settings for method_flags["{method}"] should be the same length as the number of lines ({len(elem_line_dict[elem])}) for this element ({elem})'

    # Check the provided line, method, and flag settings
    use_line, use_method, method_flags = check_settings(elem, elem_line_dict,
                                                        use_line, use_method,
                                                        method_flags)

    elem_vals, errors, elem_counts = combine_measurements(table, group, elem,
                                                          elem_line_dict,
                                                          best_lines, use_line,
                                                          use_method,
                                                          method_flags)

    return elem_vals - solar_abu[elem], errors, elem_counts


def check_settings(elem, elem_line_dict, use_line, use_method, method_flags):
    '''
    Check the line use, method use and flag settings to confirm that they
    have the same length as the number of lines.  Will set missing methods on
    so that they are used, and missing flags will be set to only use BACCHUS
    method flag=1. If integers are supplied for use_line or a given method
    instead of lists, will apply that single value to all lines.
    '''

    # Set up the line, method and flag settings if they aren't supplied
    method_names = ['syn', 'eqw', 'int', 'chi2']

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

    return use_line, use_method, method_flags


def combine_measurements(table, group, elem, elem_line_dict, best_lines,
                         use_line, use_method, method_flags):
    '''
    A function to average line-by-line abundances from BACCHUS (converted to a
    compiled hdf5 table) with an arbitrary choice of lines, methods and flags.
    Returns [X/H] and the [X/H] error for the input element by measuring the
    dispersion between the abundances from each line and method and taking the
    standard error of the mean (assuming n samples = # lines).
    '''

    # -------------------- Calculate average abundances ---------------------- #

    elem_table = table[f'{group}/{elem}']

    elem_vals = np.zeros(len(elem_table))
    elem_counts = np.zeros(len(elem_table))

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
                if setting[i] == 1:
                    flags = np.logical_and(
                        flags,
                        np.in1d(elem_table[f'{elem}_{i+1}_flag_{method}'],
                                method_flags[method][i])
                    )

            # Loop through each method and add its measurement
            for method, setting in use_method.items():
                if setting[i] == 1:
                    column = f'{elem}_{i + 1}_{method}'
                    line_abu[flags] += elem_table[column][flags]
                    line_count += flags

            climit = f'{elem}_{i+1}_limit_int'
            line_measured = np.logical_and(line_abu > elem_table[climit],
                                           line_count != 0)

            # Average the method abundances and add one to the count if this
            # line is measured
            elem_vals[line_measured] += line_abu[line_measured] / \
                line_count[line_measured]
            elem_counts += line_measured

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
            flags = np.ones(len(elem_table))

            # Loop through each method if it is used and recored which stars
            # have met the input flag conditions
            for method, setting in use_method.items():
                if setting[i] == 1:
                    flags = np.logical_and(
                        flags,
                        np.in1d(elem_table[f'{elem}_{i+1}_flag_{method}'],
                                method_flags[method][i])
                    )

            # Loop through each method and add its measurement
            for method, setting in use_method.items():
                if setting[i] == 1:
                    column = f'{elem}_{i + 1}_{method}'
                    line_abu[flags] += elem_table[column][flags]
                    line_count += flags
                    line_error[flags] += (elem_table[column]
                                          [flags] - elem_vals[flags])**2.

            climit = f'{elem}_{i+1}_limit_int'
            line_measured = np.logical_and(line_abu > elem_table[climit],
                                           line_count != 0)

            errors[line_measured] += line_error[line_measured]
            elem_error_counts[line_measured] += line_count[line_measured]

    elem_measured = np.logical_not(np.isnan(elem_vals))
    # Take the standard deviation of different line and method abundance
    # measurements and divide by the number of lines used to get a standard
    # error of the mean where the sample size is n lines not n measurements
    # because the different methods aren't independent samples
    errors[elem_measured] = (np.sqrt(errors[elem_measured] / (elem_error_counts[
        elem_measured] - 1))) / np.sqrt(elem_counts[elem_measured])

    errors[np.isnan(elem_vals)] = np.nan

    return elem_vals, errors, elem_counts
