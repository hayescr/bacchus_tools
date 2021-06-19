import os
import h5py
import numpy as np
import h_plot as h
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
from bacchus_tools.results.combine import calculate_abundances
from bacchus_tools.solar_abundances import asplund_2005
from bacchus_tools.read_element_linelist import get_element_list


def make_lbl_plots(hdf5_tablename, group, elements, elements_filepath='.',
                   plot_contents=['xfe', 'xteff', 'xlogg']):

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
        line_by_line_plots(hdf5_table, group, element, 'xfe',
                           solar_abu=solar_abu,
                           elem_line_dict=elem_line_dict,
                           plotcolor=True, plotcolor_param='teff',
                           filename_addition='_teffcolor',
                           color_order='dsc', color_lim=[3500, 5000])
        line_by_line_plots(hdf5_table, group, element, 'xfe',
                           solar_abu=solar_abu,
                           elem_line_dict=elem_line_dict,
                           plotcolor=True, plotcolor_param='logg',
                           filename_addition='_loggcolor',
                           color_order='dsc', color_lim=[0, 4])
        line_by_line_plots(hdf5_table, group, element, 'xfe',
                           solar_abu=solar_abu,
                           elem_line_dict=elem_line_dict,
                           plotcolor=True, plotcolor_param='alpha_fe',
                           filename_addition='_alphacolor', color_lim=[-0.1, 0.4])


def make_combined_lbl_plots(hdf5_tablename, group, elements,
                            elements_filepath='.',
                            plot_contents=['xfe', 'xteff', 'xlogg', 'hrd'],
                            use_method_dict=None, method_flags_dict=None,
                            bc_flags_dict=None,
                            limit_setting=None, zero_points=None):

    # Load solar abundances
    solar_abu = asplund_2005()

    # Load the BACCHUS element line lists
    elem_line_dict = get_element_list(elements_filepath)

    if use_method_dict is None:
        use_method_dict = {element: None for element in elements}

    if method_flags_dict is None:
        method_flags_dict = {element: None for element in elements}

    if bc_flags_dict is None:
        bc_flags_dict = {element: None for element in elements}

    # Open the hdf5 table of assembled BACCHUS outputs
    hdf5_tablename = hdf5_tablename.replace('.hdf5', '')
    hdf5_table = h5py.File(f'{hdf5_tablename}.hdf5', 'r')

    for element in elements:
        for plot_content in plot_contents:
            line_by_line_combined_plots(hdf5_table, group, element, plot_content,
                                        solar_abu=solar_abu,
                                        elem_line_dict=elem_line_dict,
                                        use_method=use_method_dict[element],
                                        method_flags=method_flags_dict[element],
                                        bc_flags=bc_flags_dict[element],
                                        limit_setting=limit_setting,
                                        zero_points=zero_points)

        line_by_line_combined_plots(hdf5_table, group, element, 'xfe',
                                    solar_abu=solar_abu,
                                    elem_line_dict=elem_line_dict,
                                    use_method=use_method_dict[element],
                                    method_flags=method_flags_dict[element],
                                    bc_flags=bc_flags_dict[element],
                                    plotcolor=True, plotcolor_param='teff',
                                    filename_addition='_teffcolor',
                                    color_order='dsc', color_lim=[3500, 5000],
                                    limit_setting=limit_setting,
                                    zero_points=zero_points)
        line_by_line_combined_plots(hdf5_table, group, element, 'xfe',
                                    solar_abu=solar_abu,
                                    elem_line_dict=elem_line_dict,
                                    use_method=use_method_dict[element],
                                    method_flags=method_flags_dict[element],
                                    bc_flags=bc_flags_dict[element],
                                    plotcolor=True, plotcolor_param='logg',
                                    filename_addition='_loggcolor',
                                    color_order='dsc', color_lim=[0, 4],
                                    limit_setting=limit_setting,
                                    zero_points=zero_points)

        line_by_line_combined_plots(hdf5_table, group, element, 'xfe',
                                    solar_abu=solar_abu,
                                    elem_line_dict=elem_line_dict,
                                    use_method=use_method_dict[element],
                                    method_flags=method_flags_dict[element],
                                    bc_flags=bc_flags_dict[element],
                                    zero_points=zero_points,
                                    plotcolor=True, plotcolor_param='alpha_fe',
                                    filename_addition='_alphacolor', color_lim=[-0.1, 0.4],
                                    limit_setting=limit_setting)


def make_combined_element_plots(hdf5_tablename, group, elements,
                                elements_filepath='.',
                                plot_contents=['xfe', 'xteff', 'xlogg', 'hrd'],
                                best_lines_dict=None, use_line_dict=None,
                                use_method_dict=None, method_flags_dict=None,
                                bc_flags_dict=None,
                                limit_setting=None, zero_points=None,
                                convol_limit=None, updatedo_flags_dict=None):

    # Load solar abundances
    solar_abu = asplund_2005()

    # Load the BACCHUS element line lists
    elem_line_dict = get_element_list(elements_filepath)

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

    # Open the hdf5 table of assembled BACCHUS outputs
    hdf5_tablename = hdf5_tablename.replace('.hdf5', '')
    hdf5_table = h5py.File(f'{hdf5_tablename}.hdf5', 'r')

    for element in elements:
        for plot_content in plot_contents:
            line_combination_plots(hdf5_table, group, element, plot_content,
                                   solar_abu=solar_abu,
                                   elem_line_dict=elem_line_dict,
                                   best_lines_dict=best_lines_dict,
                                   use_line=use_line_dict[element],
                                   use_method=use_method_dict[element],
                                   method_flags=method_flags_dict[element],
                                   bc_flags=bc_flags_dict[element],
                                   limit_setting=limit_setting,
                                   zero_points=zero_points,
                                   convol_limit=convol_limit,
                                   updatedo_flags=updatedo_flags_dict[element])

        line_combination_plots(hdf5_table, group, element, 'xfe',
                               solar_abu=solar_abu,
                               elem_line_dict=elem_line_dict,
                               best_lines_dict=best_lines_dict,
                               use_line=use_line_dict[element],
                               use_method=use_method_dict[element],
                               method_flags=method_flags_dict[element],
                               bc_flags=bc_flags_dict[element],
                               plotcolor=True, plotcolor_param='teff',
                               filename_addition='_teffcolor',
                               color_order='dsc', color_lim=[3500, 5000],
                               limit_setting=limit_setting,
                               zero_points=zero_points,
                               convol_limit=convol_limit,
                               updatedo_flags=updatedo_flags_dict[element])

        line_combination_plots(hdf5_table, group, element, 'xfe',
                               solar_abu=solar_abu,
                               elem_line_dict=elem_line_dict,
                               best_lines_dict=best_lines_dict,
                               use_line=use_line_dict[element],
                               use_method=use_method_dict[element],
                               method_flags=method_flags_dict[element],
                               bc_flags=bc_flags_dict[element],
                               plotcolor=True, plotcolor_param='logg',
                               filename_addition='_loggcolor',
                               color_order='dsc', color_lim=[0, 4],
                               limit_setting=limit_setting,
                               zero_points=zero_points,
                               convol_limit=convol_limit,
                               updatedo_flags=updatedo_flags_dict[element])

        line_combination_plots(hdf5_table, group, element, 'xfe',
                               solar_abu=solar_abu,
                               elem_line_dict=elem_line_dict,
                               best_lines_dict=best_lines_dict,
                               use_line=use_line_dict[element],
                               use_method=use_method_dict[element],
                               method_flags=method_flags_dict[element],
                               bc_flags=bc_flags_dict[element],
                               plotcolor=True, plotcolor_param='alpha_fe',
                               filename_addition='_alphacolor', color_lim=[-0.1, 0.4],
                               limit_setting=limit_setting,
                               zero_points=zero_points,
                               convol_limit=convol_limit,
                               updatedo_flags=updatedo_flags_dict[element])

        line_combination_plots(hdf5_table, group, element, 'xfe',
                               solar_abu=solar_abu,
                               elem_line_dict=elem_line_dict,
                               best_lines_dict=best_lines_dict,
                               use_line=use_line_dict[element],
                               use_method=use_method_dict[element],
                               method_flags=method_flags_dict[element],
                               bc_flags=bc_flags_dict[element],
                               plotcolor=True, plotcolor_param='convol',
                               filename_addition='_convolcolor', color_lim=[-20, -10],
                               limit_setting=limit_setting,
                               zero_points=zero_points,
                               convol_limit=convol_limit,
                               updatedo_flags=updatedo_flags_dict[element])


def line_by_line_plots(table, group, elem, plot_content='xfe',
                       solar_abu=None, elem_line_dict=None, path='.',
                       plotcolor=False, plotcolor_param='teff',
                       filename_addition='', color_order='asc',
                       color_lim=[None, None]):

    # Load solar abundances if they aren't supplied
    if solar_abu is None:
        solar_abu = asplund_2005()

    # Load the BACCHUS element line lists if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

    methods = ['syn', 'eqw', 'int', 'chi2', 'wln']
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

                plot_filename = f'{elem}_{method}_flag{flag}_{plot_content}{filename_addition}.png'

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
                    if elem not in solar_abu:
                        ylim = [0, 40]
                        ylabel = elem
                        ys = table[f'{group}/{elem}'][column][flag_selection]
                    else:
                        ys = table[f'{group}/{elem}'][column][flag_selection] - \
                            solar_abu[elem] - \
                            table[f'{group}/param']['fe_h'][flag_selection]
                elif plot_content == 'xlogg':
                    xlim = [-1, 5]
                    ylim = [-1, 1]
                    xinvert = True
                    yinvert = False
                    xlabel = 'Logg'
                    ylabel = f'[{elem}/Fe]'
                    xs = table[f'{group}/param']['logg'][flag_selection]
                    if elem not in solar_abu:
                        ylim = [0, 40]
                        ylabel = elem
                        ys = table[f'{group}/{elem}'][column][flag_selection]
                    else:
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
                    if elem not in solar_abu:
                        ylim = [0, 50]
                        ylabel = elem
                        ys = table[f'{group}/{elem}'][column][flag_selection]
                    else:
                        ys = table[f'{group}/{elem}'][column][flag_selection] - \
                            solar_abu[elem] - \
                            table[f'{group}/param']['fe_h'][flag_selection]

                ind1 = i // ncol
                ind2 = i % ncol
                ax = plt.subplot2grid((nrow, ncol), (ind1, ind2))

                if len(ys) != 0:
                    if plotcolor:
                        if color_order == 'dsc':
                            select = np.argsort(-table[f'{group}/param']
                                                [plotcolor_param][flag_selection])
                        else:
                            select = np.argsort(table[f'{group}/param']
                                                [plotcolor_param][flag_selection])
                        cs = table[f'{group}/param'][plotcolor_param][flag_selection]
                        cset1 = ax.scatter(xs[select], ys[select], c=cs[select],
                                           marker='.', s=5, cmap='inferno_r',
                                           label=str(elem_line_dict[elem][i]),
                                           vmin=color_lim[0], vmax=color_lim[1])
                    else:
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


def line_by_line_combined_plots(table, group, elem, plot_content='xfe',
                                solar_abu=None, elem_line_dict=None,
                                use_method=None, method_flags=None,
                                bc_flags=None,
                                limit_setting=None, zero_points=None,
                                path='.', plotcolor=False, plotcolor_param='teff',
                                filename_addition='', color_order='asc',
                                color_lim=[None, None]):

    # Load solar abundances if they aren't supplied
    if solar_abu is None:
        solar_abu = asplund_2005()

    # Load the BACCHUS element line lists if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

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

    plot_directory = f'{group}_line_by_line_comb/{elem}'
    os.makedirs(plot_directory, exist_ok=True)

    if use_method is not None:
        input_use_method = dict(use_method)
    else:
        input_use_method = None

    if method_flags is not None:
        input_method_flags = dict(method_flags)
    else:
        input_method_flags = None

    if bc_flags is not None:
        input_bc_flags = dict(bc_flags)
    else:
        input_bc_flags = None

    fig = plt.figure(figsize=figsize)

    for i in range(len(elem_line_dict[elem])):
        if use_method is not None:
            use_method = dict(input_use_method)
        else:
            use_method = None
        if method_flags is not None:
            method_flags = dict(input_method_flags)
        else:
            method_flags = None

        if bc_flags is not None:
            bc_flags = dict(input_bc_flags)
        else:
            bc_flags = None

        plot_filename = f'{elem}_comb_{plot_content}{filename_addition}.png'

        use_line = [0] * nlines
        use_line[i] = 1

        elem_abu, errors, elem_counts, elem_limits = calculate_abundances(table, group, elem,
                                                                          path='.',
                                                                          solar_abu=solar_abu,
                                                                          elem_line_dict=elem_line_dict,
                                                                          use_line=use_line,
                                                                          use_method=use_method,
                                                                          method_flags=method_flags,
                                                                          bc_flags=bc_flags,
                                                                          limit_setting=limit_setting,
                                                                          zero_points=zero_points)

        if plot_content == 'hrd':
            xlim = [3000, 6000]
            ylim = [-1, 6]
            xinvert = True
            yinvert = True
            xlabel = 'Teff'
            ylabel = 'logg'
            xs = table[f'{group}/param']['teff'][np.logical_not(
                np.isnan(elem_abu))]
            ys = table[f'{group}/param']['logg'][np.logical_not(
                np.isnan(elem_abu))]
        elif plot_content == 'xteff':
            xlim = [3000, 6000]
            ylim = [-1, 1]
            xinvert = True
            yinvert = False
            xlabel = 'Teff'
            ylabel = f'[{elem}/Fe]'
            xs = table[f'{group}/param']['teff']
            if elem not in solar_abu:
                ylim = [0, 40]
                ylabel = elem
                ys = elem_abu
            else:
                ys = elem_abu - table[f'{group}/param']['fe_h']
        elif plot_content == 'xlogg':
            xlim = [-1, 5]
            ylim = [-1, 1]
            xinvert = True
            yinvert = False
            xlabel = 'Logg'
            ylabel = f'[{elem}/Fe]'
            xs = table[f'{group}/param']['logg']
            if elem not in solar_abu:
                ylim = [0, 40]
                ylabel = elem
                ys = elem_abu
            else:
                ys = elem_abu - table[f'{group}/param']['fe_h']
        else:
            xlim = [-2.5, 1]
            ylim = [-1, 1]
            xinvert = False
            yinvert = False
            xlabel = '[Fe/H]'
            ylabel = f'[{elem}/Fe]'
            xs = table[f'{group}/param']['fe_h']
            if elem not in solar_abu:
                ylim = [0, 40]
                ylabel = elem
                ys = elem_abu
            else:
                ys = elem_abu - table[f'{group}/param']['fe_h']

        ind1 = i // ncol
        ind2 = i % ncol
        ax = plt.subplot2grid((nrow, ncol), (ind1, ind2))

        if len(ys) != 0:
            if plotcolor:
                if color_order == 'dsc':
                    select = np.argsort(-table[f'{group}/param']
                                        [plotcolor_param])
                else:
                    select = np.argsort(table[f'{group}/param']
                                        [plotcolor_param])
                cs = table[f'{group}/param'][plotcolor_param]
                cset1 = ax.scatter(xs[select], ys[select], c=cs[select],
                                   marker='.', s=5, cmap='inferno_r',
                                   label=str(elem_line_dict[elem][i]),
                                   vmin=color_lim[0], vmax=color_lim[1])

            else:
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


def line_combination_plots(table, group, elem, plot_content='xfe',
                           solar_abu=None, elem_line_dict=None, best_lines_dict=None,
                           use_line=None, use_method=None, method_flags=None,
                           bc_flags=None,
                           limit_setting=None, zero_points=None,
                           convol_limit=None, updatedo_flags=None, path='.',
                           plotcolor=False, plotcolor_param='teff',
                           filename_addition='', color_order='asc',
                           color_lim=[None, None]):

    # Load solar abundances if they aren't supplied
    if solar_abu is None:
        solar_abu = asplund_2005()

    # Load the BACCHUS element line lists if they aren't supplied
    if elem_line_dict is None:
        elem_line_dict = get_element_list(path)

    flags = [0, 1, 2, 3]

    figsize = (4, 3)
    nrow = 1
    ncol = 1

    plot_directory = f'{group}_combined_abundances/{elem}'
    os.makedirs(plot_directory, exist_ok=True)

    fig = plt.figure(figsize=figsize)

    plot_filename = f'{elem}_{plot_content}{filename_addition}.png'

    elem_abu, errors, elem_counts, elem_limits = calculate_abundances(table, group, elem,
                                                                      path='.',
                                                                      solar_abu=solar_abu,
                                                                      elem_line_dict=elem_line_dict,
                                                                      best_lines=best_lines_dict,
                                                                      use_line=use_line,
                                                                      use_method=use_method,
                                                                      method_flags=method_flags,
                                                                      bc_flags=bc_flags,
                                                                      limit_setting=limit_setting,
                                                                      zero_points=zero_points,
                                                                      convol_limit=convol_limit,
                                                                      updatedo_flags=updatedo_flags)

    if plot_content == 'hrd':
        xlim = [3000, 6000]
        ylim = [-1, 6]
        xinvert = True
        yinvert = True
        xlabel = 'Teff'
        ylabel = 'logg'
        xs = table[f'{group}/param']['teff'][np.logical_not(
            np.isnan(elem_abu))]
        ys = table[f'{group}/param']['logg'][np.logical_not(
            np.isnan(elem_abu))]
    elif plot_content == 'xteff':
        xlim = [3000, 6000]
        ylim = [-1, 1]
        xinvert = True
        yinvert = False
        xlabel = 'Teff'
        ylabel = f'[{elem}/Fe]'
        xs = table[f'{group}/param']['teff']
        if elem not in solar_abu:
            ylim = [0, 40]
            ylabel = elem
            ys = elem_abu
        else:
            ys = elem_abu - table[f'{group}/param']['fe_h']
    elif plot_content == 'xlogg':
        xlim = [-1, 5]
        ylim = [-1, 1]
        xinvert = True
        yinvert = False
        xlabel = 'Teff'
        ylabel = f'[{elem}/Fe]'
        xs = table[f'{group}/param']['logg']
        if elem not in solar_abu:
            ylim = [0, 40]
            ylabel = elem
            ys = elem_abu
        else:
            ys = elem_abu - table[f'{group}/param']['fe_h']
    else:
        xlim = [-2.5, 1]
        ylim = [-1, 1]
        xinvert = False
        yinvert = False
        xlabel = '[Fe/H]'
        ylabel = f'[{elem}/Fe]'
        xs = table[f'{group}/param']['fe_h']
        if elem not in solar_abu:
            ylim = [0, 40]
            ylabel = elem
            ys = elem_abu
        else:
            ys = elem_abu - table[f'{group}/param']['fe_h']

    ind1 = 0
    ind2 = 0
    ax = plt.subplot2grid((nrow, ncol), (ind1, ind2))

    if len(ys) != 0:
        if plotcolor:
            if color_order == 'dsc':
                select = np.argsort(-table[f'{group}/param']
                                    [plotcolor_param])
            else:
                select = np.argsort(table[f'{group}/param']
                                    [plotcolor_param])
            cs = table[f'{group}/param'][plotcolor_param]
            cset1 = ax.scatter(xs[select], ys[select], c=cs[select],
                               marker='.', s=5, cmap='inferno_r',
                               vmin=color_lim[0], vmax=color_lim[1])

        else:
            cset1 = h.density2d_to_points(xs, ys, ax=ax,
                                          bins=150, masklim=5, xlim=xlim,
                                          ylim=ylim, cmap='viridis',
                                          norm=colors.LogNorm(),
                                          interpolation='nearest',
                                          zorder=1)
    else:
        cset1 = ax.plot(-9999., -9999., color='k', marker='.',
                        ms=2, ls='none', zorder=1,
                        rasterized=True,)

    ax.tick_params(axis='both', direction='in', labelsize=8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if xinvert:
        ax.invert_xaxis()
    if yinvert:
        ax.invert_yaxis()

    # leg = ax.legend(loc=0, numpoints=1, scatterpoints=1,
    #                 framealpha=0.9, prop={
    #                     'size': 8}, markerscale=2.)
    # leg.get_frame().set_linewidth(0.0)

    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_directory}/{plot_filename}', dpi=150)
    plt.close(fig)
