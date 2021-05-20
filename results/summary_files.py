import numpy as np
import h5py
from astropy.io import fits
from bacchus_tools.results import combine
from bacchus_tools.solar_abundances import asplund_2005
from bacchus_tools.read_element_linelist import get_element_list


def write_fits(hdf5_tablename, group, elements, elements_filepath='.',
               output_filename=None, best_lines_dict=None, use_line_dict=None,
               use_method_dict=None, method_flags_dict=None, overwrite=False):

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

    # Open the hdf5 table of assembled BACCHUS outputs
    hdf5_tablename = hdf5_tablename.replace('.hdf5', '')
    hdf5_table = h5py.File(f'{hdf5_tablename}.hdf5', 'r')

    # If no output filename is supplied name it after the input table and group
    if output_filename is None:
        output_filename = f'{hdf5_tablename}_{group}.fits'

    # Set up a list to store each of the fits columns
    columns_list = []

    # -------------------------- Paramater Columns --------------------------- #
    param_names = ['STAR_ID', 'TEFF', 'LOGG', 'FE_H', 'VMICRO',
                   'ALPHA_FE_MODEL', 'C_FE_MODEL', 'CONVOL', 'SNR']

    param_formats = ['A18', 'E', 'E', 'E', 'E', 'E',  'E', 'E', 'E']

    param_col_names = ['STAR_ID', 'teff', 'logg', 'fe_h', 'vmicro', 'alpha_fe',
                       'c_fe', 'convol', 'snr']

    for name, fits_format, col_name in zip(param_names, param_formats, param_col_names):
        if col_name == 'convol':
            columns_list += [fits.Column(name=name, format=fits_format,
                                         array=-hdf5_table[f'{group}/param'][col_name])]
        else:
            columns_list += [fits.Column(name=name, format=fits_format,
                                         array=hdf5_table[f'{group}/param'][col_name])]

    # ---------------------------- Element Columns --------------------------- #

    metallicity = hdf5_table[f'{group}/param']['fe_h']

    for element in elements:
        x_h, x_h_err, x_n_lines = combine.calculate_abundances(hdf5_table,
                                                               group, element,
                                                               solar_abu=solar_abu,
                                                               elem_line_dict=elem_line_dict,
                                                               best_lines=best_lines_dict,
                                                               use_line=use_line_dict[element],
                                                               use_method=use_method_dict[element],
                                                               method_flags=method_flags_dict[element])

        x_line_abu, x_line_flags = combine.package_lines(
            hdf5_table, group, element, elem_line_dict=elem_line_dict)

        x_lim = np.zeros(len(x_h))

        array_format = f'{int(4*len(elem_line_dict[element]))}E'
        array_dim = f'(4, {len(elem_line_dict[element])})'

        if element not in solar_abu:
            columns_list += [
                fits.Column(name=f'{element.upper()}', format='E',
                            array=x_h),
                fits.Column(name=f'{element.upper()}_ERR',
                            format='E', array=x_h_err),
                fits.Column(name=f'{element.upper()}_LIM',
                            format='E', array=x_lim),
                fits.Column(name=f'{element.upper()}_N_LINES',
                            format='I', array=x_n_lines.astype(int)),
                fits.Column(name=f'{element.upper()}_ABU_EPS',
                            format=array_format, dim=array_dim,
                            array=x_line_abu),
                fits.Column(name=f'{element.upper()}_ABU_FLAGS',
                            format=array_format, dim=array_dim,
                            array=x_line_flags)
            ]

        else:
            columns_list += [
                fits.Column(name=f'{element.upper()}_FE', format='E',
                            array=x_h - metallicity),
                fits.Column(name=f'{element.upper()}_FE_ERR',
                            format='E', array=x_h_err),
                fits.Column(name=f'{element.upper()}_FE_LIM',
                            format='E', array=x_lim),
                fits.Column(name=f'{element.upper()}_N_LINES',
                            format='I', array=x_n_lines.astype(int)),
                fits.Column(name=f'{element.upper()}_ABU_EPS',
                            format=array_format, dim=array_dim,
                            array=x_line_abu),
                fits.Column(name=f'{element.upper()}_ABU_FLAGS',
                            format=array_format, dim=array_dim,
                            array=x_line_flags)
            ]

    cols = fits.ColDefs(columns_list)

    tbhdu = fits.BinTableHDU.from_columns(cols)

    hdu = fits.BinTableHDU.from_columns(tbhdu.columns)

    hdu.writeto(output_filename, overwrite=overwrite)
