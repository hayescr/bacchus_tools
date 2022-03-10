import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table
from bacchus_tools.results import combine
from bacchus_tools.solar_abundances import asplund_2005
from bacchus_tools.read_element_linelist import get_element_list


def write_fits(tablenames, groups, elements, elements_filepath='.',
               output_filename=None, best_lines_dict=None, use_line_dict=None,
               use_method_dict=None, method_flags_dict=None, bc_flags_dict=None,
               limit_setting=None, zero_points=None, convol_limit=None,
               updatedo_flags_dict=None, overwrite=False, use_apogee_param=False,
               apogee_table=None, repeat_errors_dict=None, spec_proc=None):

    # Load solar abundances
    solar_abu = asplund_2005()

    # Load the BACCHUS element line lists
    elem_line_dict = get_element_list(elements_filepath)

    # Configure the setting dictionaries if they aren't provided
    if use_line_dict is None:
        use_line_dict = {element: None for element in elements}

    if use_method_dict is None:
        use_method_dict = {element: None for element in elements}

    if method_flags_dict is None:
        method_flags_dict = {element: None for element in elements}

    if bc_flags_dict is None:
        bc_flags_dict = {element: None for element in elements}

    if updatedo_flags_dict is None:
        updatedo_flags_dict = {element: None for element in elements}

    if type(tablenames) == str:
        tablenames = [tablenames]

    if type(groups) == str:
        groups = [groups]

    if type(apogee_table) != list:
        apogee_table_list = [apogee_table]
    else:
        apogee_table_list = apogee_table

    if type(updatedo_flags_dict) != list:
        updatedo_flags_dict_list = [updatedo_flags_dict]
    else:
        updatedo_flags_dict_list = updatedo_flags_dict

    data_hdus = []

    for hdf5_tablename, group, apogee_table, updatedo_flags_dict in zip(tablenames, groups, apogee_table_list, updatedo_flags_dict_list):
        # Open the hdf5 table of assembled BACCHUS outputs
        hdf5_tablename = hdf5_tablename.replace('.hdf5', '')
        hdf5_table = h5py.File(f'{hdf5_tablename}.hdf5', 'r')

        # If no output filename is supplied name it after the input table and group
        if output_filename is None:
            output_filename = f'{hdf5_tablename}_{group}.fits'

        # Set up a list to store each of the fits columns
        columns_list = []

        # -------------------------- Paramater Columns --------------------------- #

        if use_apogee_param:
            apogee_param_names = ['APOGEE_ID', 'FIELD', 'TELESCOPE', 'EXTRATARG',
                                  'SNR', 'TEFF', 'LOGG', 'M_H', 'ALPHA_M', 'FE_H']

            apogee_param_formats = ['A19', 'A20',
                                    'A6', 'J', 'E', 'E', 'E', 'E', 'E', 'E']

            apogee_param_col_names = [
                'APOGEE_ID', 'FIELD', 'TELESCOPE', 'EXTRATARG', 'SNR', 'TEFF', 'LOGG', 'M_H', 'ALPHA_M', 'FE_H']

            for name, fits_format, col_name in zip(apogee_param_names, apogee_param_formats, apogee_param_col_names):
                if col_name == 'ALPHA_M':
                    alpha_m = apogee_table['PARAM'][:, 6]
                    alpha_m[alpha_m <
                            - 0.72] = hdf5_table[f'{group}/param']['alpha_fe'][alpha_m < -0.72]
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=alpha_m)]
                elif col_name == 'M_H':
                    m_h = apogee_table['PARAM'][:, 3]
                    m_h[m_h < -2.49] = 0.0
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=m_h)]
                else:
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=apogee_table[col_name])]

            # bacchus_param_names = ['STAR_ID', 'TEFF_B', 'LOGG_B', 'FE_H_B', 'VMICRO',
            #                        'ALPHA_FE_MODEL', 'C_FE_MODEL', 'CONVOL', 'SNR_B', 'UPDATE_C',
            #                        'UPDATE_N', 'UPDATE_O']
            #
            # bacchus_param_formats = ['A19', 'E', 'E', 'E', 'E',
            #                          'E',  'E', 'E', 'E', 'I', 'I', 'I']
            #
            # bacchus_param_col_names = ['STAR_ID', 'teff', 'logg', 'fe_h', 'vmicro', 'alpha_fe',
            #                            'c_fe', 'convol', 'snr', 'c_iter', 'n_iter', 'o_iter']

            bacchus_param_names = ['VMICRO', 'CONVOL', 'UPDATE_C',
                                   'UPDATE_N', 'UPDATE_O']

            bacchus_param_formats = ['E', 'E', 'I', 'I', 'I']

            bacchus_param_col_names = ['vmicro',
                                       'convol', 'c_iter', 'n_iter', 'o_iter']

            for name, fits_format, col_name in zip(bacchus_param_names, bacchus_param_formats, bacchus_param_col_names):
                if col_name == 'convol':
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=-hdf5_table[f'{group}/param'][col_name])]
                elif col_name in ['c_iter', 'n_iter', 'o_iter']:
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=hdf5_table[f'{group}/param'][col_name])]
                else:
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=hdf5_table[f'{group}/param'][col_name])]

        else:
            param_names = ['STAR_ID', 'TEFF', 'LOGG', 'FE_H', 'VMICRO',
                           'ALPHA_FE_MODEL', 'C_FE_MODEL', 'CONVOL', 'SNR', 'UPDATE_C',
                           'UPDATE_N', 'UPDATE_O']

            param_formats = ['A24', 'E', 'E', 'E', 'E',
                             'E',  'E', 'E', 'E', 'I', 'I', 'I']

            param_col_names = ['STAR_ID', 'teff', 'logg', 'fe_h', 'vmicro', 'alpha_fe',
                               'c_fe', 'convol', 'snr', 'c_iter', 'n_iter', 'o_iter']

            for name, fits_format, col_name in zip(param_names, param_formats, param_col_names):
                if col_name == 'convol':
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=-hdf5_table[f'{group}/param'][col_name])]
                elif col_name in ['c_iter', 'n_iter', 'o_iter']:
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=hdf5_table[f'{group}/param'][col_name])]
                else:
                    columns_list += [fits.Column(name=name, format=fits_format,
                                                 array=hdf5_table[f'{group}/param'][col_name])]

        # ---------------------------- Element Columns --------------------------- #

        if use_apogee_param:
            metallicity = apogee_table['FE_H']
            teff = apogee_table['TEFF']
            feh_err = apogee_table['FE_H_ERR']
        else:
            metallicity = hdf5_table[f'{group}/param']['fe_h']
            teff = hdf5_table[f'{group}/param']['teff']
            feh_err = np.zeros(len(hdf5_table[f'{group}/param']['fe_h']))

        for element in elements:
            x_h, x_h_err, x_n_lines, x_h_lim = combine.calculate_abundances(hdf5_table,
                                                                            group, element,
                                                                            solar_abu=solar_abu,
                                                                            elem_line_dict=elem_line_dict,
                                                                            best_lines=best_lines_dict,
                                                                            use_line=use_line_dict[element],
                                                                            use_method=use_method_dict[element],
                                                                            method_flags=method_flags_dict[element],
                                                                            bc_flags=bc_flags_dict[element],
                                                                            limit_setting=limit_setting,
                                                                            zero_points=zero_points,
                                                                            convol_limit=convol_limit,
                                                                            updatedo_flags=updatedo_flags_dict[element])

            if spec_proc is not None:
                if spec_proc[element] is not None:
                    x_h, x_h_err, x_n_lines, x_h_lim = spec_proc[element](x_h,
                                                                          x_h_err, x_n_lines, x_h_lim,
                                                                          hdf5_table=hdf5_table,
                                                                          group=group,
                                                                          element=element,
                                                                          solar_abu=solar_abu,
                                                                          elem_line_dict=elem_line_dict,
                                                                          best_lines_dict=best_lines_dict,
                                                                          use_line_dict=use_line_dict,
                                                                          use_method_dict=use_method_dict,
                                                                          method_flags_dict=method_flags_dict,
                                                                          bc_flags_dict=bc_flags_dict,
                                                                          limit_setting=limit_setting,
                                                                          zero_points=zero_points,
                                                                          convol_limit=convol_limit,
                                                                          updatedo_flags_dict=updatedo_flags_dict
                                                                          )

            x_line_abu, x_line_flags, x_bc_flags = combine.package_lines(
                hdf5_table, group, element, elem_line_dict=elem_line_dict)

            if repeat_errors_dict is not None:
                if element in repeat_errors_dict:
                    # repeat_err_input = np.vstack(
                    #     (teff, metallicity)).transpose()
                    repeat_err_input = np.vstack(
                        (teff, x_h)).transpose()
                    x_h_repeat_err = repeat_errors_dict[element]['function'](
                        repeat_err_input, *repeat_errors_dict[element]['params'])
                else:
                    x_h_repeat_err = np.zeros(len(metallicity))
            else:
                x_h_repeat_err = np.zeros(len(metallicity))

            # x_h_lim = np.zeros(len(x_h))

            array_format = f'{int(5*len(elem_line_dict[element]))}E'
            array_dim = f'(5, {len(elem_line_dict[element])})'
            array_flag_format = f'{int(5*len(elem_line_dict[element]))}I'
            array_flag_dim = f'(5, {len(elem_line_dict[element])})'
            array_bc_format = f'{int(2*len(elem_line_dict[element]))}I'
            array_bc_dim = f'(2, {len(elem_line_dict[element])})'

            if element not in solar_abu:
                columns_list += [
                    fits.Column(name=f'{element.upper()}', format='E',
                                array=x_h),
                    fits.Column(name=f'{element.upper()}_ERR_MEAS',
                                format='E', array=x_h_err),
                    fits.Column(name=f'{element.upper()}_ERR_EMP',
                                format='E', array=np.minimum(x_h_repeat_err, 50.)),
                    fits.Column(name=f'{element.upper()}_LIM',
                                format='E', array=x_h_lim),
                    fits.Column(name=f'{element.upper()}_N_LINES',
                                format='I', array=x_n_lines.astype(int)),
                    fits.Column(name=f'{element.upper()}_METHOD_ABUND',
                                format=array_format, dim=array_dim,
                                array=x_line_abu),
                    fits.Column(name=f'{element.upper()}_METHOD_FLAGS',
                                format=array_flag_format, dim=array_flag_dim,
                                array=x_line_flags),
                    fits.Column(name=f'{element.upper()}_SPECTRA_FLAGS',
                                format=array_bc_format, dim=array_bc_dim,
                                array=x_bc_flags)
                ]

            else:
                columns_list += [
                    fits.Column(name=f'{element.upper()}_FE', format='E',
                                array=x_h - metallicity),
                    fits.Column(name=f'{element.upper()}_FE_ERR_MEAS',
                                format='E', array=np.hypot(x_h_err, feh_err)),
                    fits.Column(name=f'{element.upper()}_FE_ERR_EMP',
                                format='E', array=np.hypot(x_h_repeat_err, feh_err)),
                    fits.Column(name=f'{element.upper()}_FE_LIM',
                                format='E', array=x_h_lim - metallicity),
                    fits.Column(name=f'{element.upper()}_N_LINES',
                                format='I', array=x_n_lines.astype(int)),
                    fits.Column(name=f'{element.upper()}_METHOD_ABUND',
                                format=array_format, dim=array_dim,
                                array=x_line_abu),
                    fits.Column(name=f'{element.upper()}_METHOD_FLAGS',
                                format=array_flag_format, dim=array_flag_dim,
                                array=x_line_flags),
                    fits.Column(name=f'{element.upper()}_SPECTRA_FLAGS',
                                format=array_bc_format, dim=array_bc_dim,
                                array=x_bc_flags)
                ]
        cols = fits.ColDefs(columns_list)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        hdu = fits.BinTableHDU.from_columns(tbhdu.columns)

        data_hdus += [hdu]

    dtype = [('METHOD_NAMES', 'S5', 5), ('SPECTRA_FLAG_NAMES', 'S5', 2)]
    dtype += [(f'{element.upper()}_LINE_LAMBDA', np.float_,
               len(elem_line_dict[element])) for element in elements]

    method_array = np.empty(1, dtype=dtype)

    # method_array = np.zeros((1, 5)).astype(str)
    # spectra_flag_array = np.zeros((1, 2)).astype(str)

    # for i, method in enumerate(['syn', 'eqw', 'int', 'chi2', 'wln']):
    method_array['METHOD_NAMES'] = np.array(
        ['syn', 'eqw', 'int', 'chi2', 'wln'])

    # for i, flag in enumerate(['blend, cont']):
    method_array['SPECTRA_FLAG_NAMES'] = np.array(['blend', 'cont'])

    for element in elements:
        method_array[f'{element.upper()}_LINE_LAMBDA'] = np.array(
            elem_line_dict[element])

    # desc_columns_list = []
    #
    # desc_columns_list += [fits.Column(name='METHOD_NAMES', format='30A',
    #                                   array=method_array['METHOD'])]
    # desc_columns_list += [fits.Column(name='SPECTRA_FLAG_NAMES',
    #                                   format='20A', array=method_array['FLAG'])]
    #
    # for element in elements:
    #     column_format = f'{len(elem_line_dict[element])}E'
    #
    #     line_array = np.empty(
    #         1, dtype=[('WAVE', np.float_, len(elem_line_dict[element]))])
    #
    #     # line_array = np.zeros((1, len(elem_line_dict[element])))
    #
    #     # for i, wave in enumerate(elem_line_dict[element]):
    #     line_array['WAVE'] = np.array(elem_line_dict[element])
    #     desc_columns_list += [fits.Column(
    #         name=f'{element}_LINE_LAMBDA', format=column_format, array=line_array['WAVE'])]

    hdu0 = fits.PrimaryHDU()

    if len(data_hdus) == 1:
        hdu = data_hdus[0]
    else:
        for i in range(len(data_hdus) - 1):
            hdu1_unique_id = np.core.defchararray.add(np.char.strip(data_hdus[i].data['TELESCOPE']), np.core.defchararray.add(
                np.char.strip(data_hdus[i].data['FIELD']), np.char.strip(data_hdus[i].data['APOGEE_ID'])))
            for j in range(len(data_hdus) - 1 - i):
                hdu2_unique_id = np.core.defchararray.add(np.char.strip(data_hdus[i + j + 1].data['TELESCOPE']), np.core.defchararray.add(
                    np.char.strip(data_hdus[i + j + 1].data['FIELD']), np.char.strip(data_hdus[i + j + 1].data['APOGEE_ID'])))
                newdata = data_hdus[i + j + 1].data
                newdata = newdata[np.logical_not(
                    np.in1d(hdu2_unique_id, hdu1_unique_id))]
                data_hdus[i + j + 1] = fits.BinTableHDU(data=newdata)
        nrows_list = []
        for i in range(len(data_hdus)):
            nrows_list += [data_hdus[i].data.shape[0]]
        hdu = fits.BinTableHDU.from_columns(
            data_hdus[0].columns, nrows=sum(nrows_list))
        for colname in data_hdus[0].columns.names:
            for i in range(len(data_hdus) - 1):
                hdu.data[colname][sum(
                    nrows_list[0:i + 1]):sum(nrows_list[0:i + 2])] = data_hdus[i + 1].data[colname]

        hdu.data = hdu.data[np.argsort(hdu.data['APOGEE_ID'])]

    # desc_cols = fits.ColDefs(desc_columns_list)
    #
    # desc_tbhdu = fits.BinTableHDU.from_columns(desc_cols)
    #
    # desc_hdu = fits.BinTableHDU.from_columns(desc_tbhdu.columns)

    method_array = Table(method_array)

    desc_hdu = fits.table_to_hdu(method_array)

    new_hdul = fits.HDUList([hdu0, hdu, desc_hdu])

    new_hdul.writeto(output_filename, overwrite=overwrite)
