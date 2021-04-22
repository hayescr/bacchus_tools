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
