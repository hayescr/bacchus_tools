import numpy as np


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
    for line in line_list_file:

        # Split up each line from the files
        sep_line = line.split()

        if len(sep_line) < 3:
            continue

        # Don't include the "element" if its a molecular isotope or continuum
        # Unless it's C12C13 which we are interested in
        if float(sep_line[0]) > 90 and sep_line[1] != 'C12C13':
            continue
        else:
            # Add the element to the elem_list
            elem_list += [sep_line[1]]

            # Convert the line list wavelengths to floats and add them to a list
            temp_line_list = []
            for wave in sep_line[2:]:
                temp_line_list += [float(wave)]

            for line_mp in mp_line_list_file:
                sep_line_mp = line_mp.split()
                if len(sep_line_mp) < 3:
                    continue
                if sep_line_mp[1] != sep_line[1]:
                    continue

                else:
                    # Make sure that we have zipped together lines from that
                    # correspond to the same element
                    assert (sep_line[1] == sep_line_mp[1]
                            ), 'Must add the same element to the list at the same time'

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
                np.round(np.unique(np.array(temp_line_list)), 3))
            lines_list += [temp_line_list]

    # Make a dictionary of each element as keys with the list of its lines
    # as items
    elem_line_dict = {elem_list[i]: lines_list[i]
                      for i in range(len(elem_list))}

    return elem_line_dict
