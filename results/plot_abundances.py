import os
import h5py
import numpy as np
import h_plot as h
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
from bacchus_tools.solar_abundances import asplund_2005
from bacchus_tools.read_element_linelist import get_element_list


def make_lbl_plots(hdf5_tablename, group, elements, elements_filepath='.',
                   plot_contents=['xfe', 'xteff', 'hrd']):

    # Load solar abundances
    solar_abu = asplund_2005()

    # Load the BACCHUS element line lists
    elem_line_dict = get_element_list(elements_filepath)

    # Open the hdf5 table of assembled BACCHUS outputs
    hdf5_tablename = hdf5_tablename.replace('.hdf5', '')
    hdf5_table = h5py.File(f'{hdf5_tablename}.hdf5', 'r')

    for element in elements:
        for plot_content in plot_contents:
            line_by_line_plots(hdf5_table, group, element, plot_content,
                               solar_abu=solar_abu,
                               elem_line_dict=elem_line_dict)


def line_by_line_plots(table, group, elem, plot_content='xfe',
                       solar_abu=None, elem_line_dict=None, path='.'):

    # Load solar abundances if they aren't supplied
    if solar_abu is None:
        solar_abu = asplund_2005()

    # Load the BACCHUS element line lists if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

    methods = ['syn', 'eqw', 'int', 'chi2']
    flags = [0, 1, 2, 3]

    nlines = len(elem_line_dict[elem])

    if nlines == 1:
        figsize = (4, 3)
        nrow = 1
        ncol = 1
    elif nlines <= 10:
        figsize = (8, ceil(nlines / 2) * 2)
        nrow = ceil(nlines / 2)
        ncol = 2
    else:
        figsize = (ceil(nlines / 5) * 4, 10)
        ncol = ceil(nlines / 5)
        nrow = ceil(nlines / ncol)

    for method in methods:
        plot_directory = f'{group}_line_by_line/{elem}/{method}'
        os.makedirs(plot_directory, exist_ok=True)

        for flag in flags:
            fig = plt.figure(figsize=figsize)

            for i in range(len(elem_line_dict[elem])):

                plot_filename = f'{elem}_{method}_flag{flag}_{plot_content}.png'

                column = f'{elem}_{str(i + 1)}_{method}'
                cflag = f'{elem}_{str(i + 1)}_flag_{method}'

                flag_selection = table[f'{group}/{elem}'][cflag] == flag

                if plot_content == 'hrd':
                    xlim = [3000, 6000]
                    ylim = [-1, 6]
                    xinvert = True
                    yinvert = True
                    xlabel = 'Teff'
                    ylabel = 'logg'
                    xs = table[f'{group}/param']['teff'][flag_selection]
                    ys = table[f'{group}/param']['logg'][flag_selection]
                elif plot_content == 'xteff':
                    xlim = [3000, 6000]
                    ylim = [-1, 1]
                    xinvert = True
                    yinvert = False
                    xlabel = 'Teff'
                    ylabel = f'[{elem}/Fe]'
                    xs = table[f'{group}/param']['teff'][flag_selection]
                    ys = table[f'{group}/{elem}'][column][flag_selection] - \
                        solar_abu[elem] - \
                        table[f'{group}/param']['fe_h'][flag_selection]
                else:
                    xlim = [-2.5, 1]
                    ylim = [-1, 1]
                    xinvert = False
                    yinvert = False
                    xlabel = '[Fe/H]'
                    ylabel = f'[{elem}/Fe]'
                    xs = table[f'{group}/param']['fe_h'][flag_selection]
                    ys = table[f'{group}/{elem}'][column][flag_selection] - \
                        solar_abu[elem] - \
                        table[f'{group}/param']['fe_h'][flag_selection]

                ind1 = i // ncol
                ind2 = i % ncol
                ax = plt.subplot2grid((nrow, ncol), (ind1, ind2))

                if len(ys) != 0:
                    cset1 = h.density2d_to_points(xs, ys, ax=ax,
                                                  bins=150, masklim=5, xlim=xlim,
                                                  ylim=ylim, cmap='viridis',
                                                  norm=colors.LogNorm(),
                                                  interpolation='nearest',
                                                  label=str(
                                                      elem_line_dict[elem][i]),
                                                  zorder=1)
                else:
                    cset1 = ax.plot(-9999., -9999., color='k', marker='.',
                                    ms=2, ls='none', zorder=1,
                                    rasterized=True,
                                    label=str(elem_line_dict[elem][i]))

                ax.tick_params(axis='both', direction='in', labelsize=8)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if ind1 == 4 or i >= nlines - ncol:
                    ax.set_xlabel(xlabel, fontsize=11)
                if ind2 == 0:
                    ax.set_ylabel(ylabel, fontsize=11)

                if xinvert:
                    ax.invert_xaxis()
                if yinvert:
                    ax.invert_yaxis()

                leg = ax.legend(loc=0, numpoints=1, scatterpoints=1,
                                framealpha=0.9, prop={
                                    'size': 8}, markerscale=2.)
                leg.get_frame().set_linewidth(0.0)

            # fig.subplots_adjust(hspace=0.3, wspace=0.3)
            plt.tight_layout()
            plt.savefig(f'{plot_directory}/{plot_filename}', dpi=150)
            plt.close(fig)
