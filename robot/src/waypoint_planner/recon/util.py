import io
import itertools
import os

import PIL
import numpy as np
import utm


def bytes2im(arrs):
    if len(arrs.shape) == 1:
        return np.array([bytes2im(arr_i) for arr_i in arrs])
    elif len(arrs.shape) == 0:
        return np.array(PIL.Image.open(io.BytesIO(arrs)))
    else:
        raise ValueError


def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return sorted(list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders])))


def latlong_to_utm(latlong):
    """
    :param latlong: latlong or list of latlongs
    :return: utm (easting, northing)
    """
    latlong = np.array(latlong)
    if len(latlong.shape) > 1:
        return np.array([latlong_to_utm(p) for p in latlong])

    easting, northing, _, _ = utm.from_latlon(*latlong)
    return np.array([easting, northing])