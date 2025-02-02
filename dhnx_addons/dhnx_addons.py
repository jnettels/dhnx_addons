r"""Collection of generalized functions for ALKIS and OpenStreetMap data.

Installation and usage
----------------------

See the readme.md in the repository for general information.


Special dependencies
--------------------

See or environment.yaml in the repository.


Useful resources
----------------

- Download test reference year weather data from DWD
    - https://kunden.dwd.de/obt/
- Download Open GeoData (e.g. LoD1 and LoD2 3D-models)
    - https://opengeodata.lgln.niedersachsen.de/#lod2
    - https://www.lvermgeo.sachsen-anhalt.de/de/gdp-open-data.html
- Download digital terrain/elevation model files
    - https://earthexplorer.usgs.gov/
    - https://gdz.bkg.bund.de/index.php/default/open-data.html

Common errors & solutions
-------------------------

Error:

    .. code::

        Cannot mix incompatible Qt library (5.15.8) with this library (5.15.6)

Solution:

    Qt is a library for graphical user interfaces.
    This error can occur when you have separate conda environments with
    different versions of the package "PyQt5". It should be solved by
    making sure your new environments use the same version as your base
    environment.
    In the example above, we can run the command

    ``conda list qt``

    to find the actual package name that causes the issue and then e.g.

    ``conda install qt-main==5.15.6``

Error:

    .. code::

        OSError: exception: access violation reading 0xFFFFFFFFFFFFFFFF

Solution:

    This could occur with pygeos-0.12.0 (comes with tobler-0.9.0). It
    should not happen with up-to-date packages.

Error:

    .. code::

        from rasterio._version import gdal_version, get_geos_version, get_proj_version
        ImportError: DLL load failed while importing _version:
            Die angegebene Prozedur wurde nicht gefunden.

Solution:

    Place ``import osgeo`` before rasterio is imported by fiona
    or contextily.
    See https://gis.stackexchange.com/a/450445/135438

Error:

    .. code::

        AttributeError: partially initialized module 'fiona' has no
        attribute '_loading' (most likely due to a circular import)

Solution:

    Place ``import fiona`` before ``import dhnx_addons`` in your script.

Error:

    .. code::

        WARNING: Cannot find header.dxf (GDAL_DATA is not defined)

Solution:

    This is not an error, just a warning. But it can be fixed by setting
    the 'GDAL_DATA' environment variable:

    .. code::

        os.environ['GDAL_DATA'] = os.path.join(
            os.path.dirname(sys.executable), 'Library/share/gdal')

Error:

    .. code::

        Windows fatal exception: stack overflow

        Main thread:
        Current thread 0x0000a5bc (most recent call first):
          File "C:\Users\**\anaconda3\envs\work\lib\pickle.py", line 531 in get
          File "C:\Users\**\anaconda3\envs\work\lib\pickle.py", line 547 in save
          ...

Solution:

    I implemented the cache from joblib.Memory, because it can save a lot
    of time when running with the exact same settings multiple times.
    However, sometimes it seems to cause this error. In that case it should
    help to clear the cache by putting one of the following lines in front
    of the line where ``lpagg_run()`` or ``dhnx_run()`` are called.

    - ``dhnx_addons.lpagg_run.clear()  # Clear cached results``
    - ``dhnx_addons.dhnx_run.clear()  # Clear cached results``

Error:

    .. code::

        ModuleNotFoundError: No module named '_gdal'

Solution:

    This error appeared during the import of osgeo or fiona. It happend
    after creating a python environment with a gdal version different
    from the one installed in the base environment. That version of gdal
    was used, because it's gdal.dll was found in the system path.
    Removing that entry from the system path and setting an env
    variable recommended by gdal solved the issue.

    .. code::

        # Remove path that contains the conflicting gdal.dll
        PATH = os.environ['PATH'].split(os.pathsep)
        path_anaconda_bin = os.path.join(os.path.expanduser('~'),
                                         'anaconda3', 'Library', 'bin')
        while path_anaconda_bin in PATH:
            PATH.remove(path_anaconda_bin)
        os.environ['PATH'] = os.pathsep.join(PATH)

        # Let gdal search for the required dll
        os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'

"""

import os
import sys
import subprocess
import re
import io
import json
import logging
import warnings
from joblib import Memory
from packaging.version import parse
import numpy as np
import shapely
if parse(shapely.__version__) >= parse("2.0"):
    # There is a weird dll import error that occurs either if osgeo is
    # imported or not, and it seems to be related to the shapely version
    import osgeo  # import before geopandas fixes issue with rasterio, fiona

import pandas as pd
import geopandas as gpd
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from inspect import signature
import yaml

try:
    from . import cbc_installer  # local import
except ImportError:
    import cbc_installer  # local import for running dhnx_addons.py

logger = logging.getLogger(__name__)  # Create a logger for this module

# Define a memory to be used as a decorator, which enables caching for
# the most expensive functions like lpagg_run and dhnx_run
memory = Memory(location='cache', verbose=0)

try:
    import fiona
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'fiona' can be installed with "
                   "'conda install fiona'")
try:
    import libpysal
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'libpysal' can be installed with "
                   "'conda install libpysal'")
try:
    import tobler
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'tobler' can be installed with "
                   "'conda install tobler>=0.8.0 -c conda-forge'")
try:
    import dhnx
    if parse(dhnx.__version__) < parse("0.0.3"):
        raise ImportError(f"Installed dhnx version ({dhnx.__version__}"
                          ") is lower than the tested version (0.0.3)")
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'dhnx' can be installed with "
                   "pip install dhnx==0.0.3")

try:
    import contextily
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'contextily' can be installed with "
                   "'conda install contextily -c conda-forge'")
try:
    import osmnx as ox
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'osmnx' can be installed with "
                   "'conda install osmnx -c conda-forge'")

try:
    import demandlib
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'demandlib' can be installed from "
                   "'https://github.com/jnettels/demandlib'")

try:
    import pandapipes
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'pandapipes' can be installed with "
                   "'pip install pandapipes'")


def main():
    """Run an example main method."""
    setup()
    logger.info("Welcome to the 'DHNx Addons' example main method")
    # workflow_example_openstreetmap(show_plot=True)
    workflow_example_openstreetmap(show_plot=False)


def setup(log_level='INFO'):
    """Set up the logger and other settings."""
    logger.setLevel(level=log_level.upper())  # Logger for this module
    # logging.getLogger('osmnx').setLevel(level='ERROR')
    # logging.getLogger('dhnx').setLevel(level='ERROR')
    logging.basicConfig(
        format='%(asctime)s %(module)-12s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S')

    # Loading the pandapipes library modifies global pandas options
    # for printing dataframes. Set them back to the default values
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    pd.set_option('display.max_rows', 60)  # default: 60
    pd.set_option('display.max_columns', 0)  # default: 0

    # Make a specific UserWarning by joblib.Memory silent.
    # It warns about the caching action taking too long, which is fine.
    warnings.filterwarnings(action='ignore', category=UserWarning,
                            module='Memory')

    # Silencing specific FutureWarning by message content
    warnings.filterwarnings(
        action="ignore", module='oemof.solph',
        message=("For backward compatibility, the option investment "
                 "overwrites the option nominal_value. Both options cannot "
                 "be set at the same time."), category=FutureWarning)
    warnings.filterwarnings(
        action="ignore", module='oemof.network',
        message=("Usage of oemof.network.Component is deprecated. "
                 "Use oemof.network.Node instead."), category=FutureWarning)


def workflow_example_openstreetmap(
        gdf_area=None,
        gdf_prod=None,
        gdf_streets=None,
        show_plot=True,
        crs_default='EPSG:25832',
        ):
    """Run an example workflow from OpenStreetMap data."""
    if gdf_area is None:
        gdf_area = load_example_area()

    # Make sure GeoDataFrames are in a projected coordinate system
    for gdf in [gdf_area, gdf_prod, gdf_streets]:
        if gdf is not None:
            gdf.to_crs(crs_default, inplace=True)

    gdf_houses = download_buildings_from_osm(
        gdf_area,
        dropna_tresh=0.01,
        building_keys=True,  # if True, use all the keys
        # building_keys=[
        #     'apartments',
        #     'commercial',
        #     'detached',
        #     'house',
        #     'industrial',
        #     'residential',
        #     'retail',
        #     'semidetached_house',
        #     'yes',  # This key defines all "other" buildings
        # ],
        rename_dict={'building': 'building_osm'},
        show_plot=show_plot)

    gdf_houses = workflow_default(gdf_houses, show_plot=show_plot)
    if show_plot:
        plot_heated(gdf_houses)
    # breakpoint()

    if gdf_prod is None:
        # Skip this step if generators / producers are given
        gdf_houses, gdf_prod = assign_random_producer_building(gdf_houses)

    # Only the buildings designated as 'heated' should remain
    gdf_houses.drop(gdf_houses[gdf_houses['heated'] == False].index,
                    inplace=True)

    # Now we have alternatives for the thermal power input for dhnx
    # a) Use the power estimated from fixed full load hours
    # b.1) Run lpagg, then use the maximum thermal power for each building
    # b.2) Run lpagg, then use the time series of power for each building

    # Run "load profile aggregator" to get maximum thermal load for buildings
    # lpagg_run.clear()  # Clear cached results
    gdf_houses, df_load_ts_slice = lpagg_run(
        gdf_houses,
        sigma=3,
        E_th_col='e_th_total_kWh',
        result_folder='./result_lpagg',
        print_file='load.dat',
        # intervall='15 minutes',
        show_plot=show_plot,
        save_plot_filetypes=['svg'],
        print_columns=['HOUR', 'E_th_RH_HH', 'E_th_TWE_HH', 'E_el_HH',
                       'E_th_RH_GHD', 'E_th_TWE_GHD', 'E_el_GHD'],
        house_type_replacements={
            'SFH': 'EFH',
            'MFH': 'MFH',
            'business': 'GHD/G1G',
            'other-heated-non-residential': 'GHD/G1G',
        },
        use_demandlib="auto",
        # use_demandlib=False,
        # print_houses_xlsx=True,  # Print individual profiles for each house
        # print_GLF_stats=True,  # Print time shift distribution
        # log_level='debug',
        # unique_profile_workflow=False,
        )
    save_path = './result_dhnx'

    save_geojson(gdf_houses, 'consumers_polygon', path=save_path,
                 save_excel=True)
    save_geojson(gdf_prod, 'producers_polygon', path=save_path)

    if gdf_streets is None:
        if not gdf_area.intersects(gdf_prod, align=False).any():
            # If producer is outside of (building) area, use their envelope
            # as search area for street data download
            gdf_area = gpd.GeoSeries(
                pd.concat([gdf_area, gdf_prod]).union_all().envelope,
                crs=gdf_area.crs)

        gdf_streets = download_streets_from_osm(
            gdf_area, dropna_tresh=0.01, show_plot=show_plot)
        save_geojson(gdf_streets, 'streets_input', path=save_path,
                     type_errors='coerce')

    """
    # In cases where there are multiple producers, it is possible to simulate
    # multiple time steps in order to optimize the network for all of them.
    # Set the nominal heating capacity of each producer. This example
    # would only work if gdf_prod already has two producer entries.
    gdf_prod['heat.source.nominal_value'] = [2500, 2500]  # kW
    # Define e.g. two time steps for each producer. In the first time step
    # only the first producer is active and vice versa. This should result
    # in a grid that can be heated from both producers.
    gdf_prod['heat.source.max'] = [[1, 0.001], [0.001, 1]]
    # A producer can also be forced to be connected by specifying minimum power
    gdf_prod['heat.source.min'] = [[0, 0], [0, 1]]
    # Create identical demand time steps for each time step defined above
    df_load_ts_slice = pd.DataFrame(columns=gdf_houses.index, index=[0, 1])
    df_load_ts_slice.loc[0] = gdf_houses['P_heat_max'].values
    df_load_ts_slice.loc[1] = gdf_houses['P_heat_max'].values
    # Then pass df_load_ts_slice to dhnx_run().

    # The approach above is for using P_heat_max of each building.
    # If the buildings' time series should be used, the following can help
    n_ts = len(df_load_ts_slice)  # Store number of time steps
    # For two producers, append 1 copy of the time series to itself
    df_load_ts_slice = pd.concat([df_load_ts_slice, df_load_ts_slice])

    # Create a producer DataFrame with the desired settings
    gdf_prod['heat.source.nominal_value'] = [5000, 400]
    gdf_prod['heat.source.max'] = [
        [1]*n_ts + [1]*n_ts,  # Condition 1: source 1 | source 2
        [0.001]*n_ts + [1]*n_ts  # Condition 2: source 1 | source 2
        ]
    gdf_prod['heat.source.min'] = [
        [0]*n_ts + [0]*n_ts,
        [0]*n_ts + [1]*n_ts
        ]

    """

    # dhnx_run.clear()  # Clear cached results
    network, gdf_pipes, df_pipes, df_DN = dhnx_run(
        gdf_streets, gdf_prod, gdf_houses,
        save_path=save_path,
        show_plot=show_plot,
        # path_invest_data='invest_data',
        # path_pipe_data="input/Pipe_data.csv",
        # df_load_ts_slice=None,
        df_load_ts_slice=df_load_ts_slice,
        col_p_th=None,
        # col_p_th='p_th_guess_kW',
        # col_p_th='P_heat_max',
        # simultaneity=0.8,
        # reset_index=False,
        method='boundary',
        solver=None,
        solver_cmdline_options={  # gurobi
            # 'MIPGapAbs': 1e-5,  # (absolute gap) default: 1e-10 (gurobi)
            # 'MIPGap': 0.03,  # (0.2 = 20% gap) default: 0 (gurobi)
            'ratioGap': 0.01,  # (0.2 = 20% gap) default: 0 (cbc)
            'seconds': 60 * 10 * 1,  # (seconds of maximum runtime) (cbc)
            # 'TimeLimit': 60 * 1,  # (seconds of maximum runtime)
            'TimeLimit': 60 * 10 * 1,  # (seconds of maximum runtime) (gurobi)
            # 'TimeLimit': 60 * 60 * 1,  # (seconds of maximum runtime) (gurobi)
            # 'TimeLimit': 60 * 60 * 3,  # (seconds of maximum runtime)
        },
        )

    # Run a pandapipes simulation with the results of DHNx. Choose e.g.
    # the time of overall maximum thermal power demand in df_load_ts_slice
    # or the maximum thermal power multiplied with a simultaneity factor
    # as an input for the thermal power of each consumer
    p_pipes, p_forks, p_consumers, p_producers = pandapipes_run(
        network, gdf_pipes, df_DN, show_plot=show_plot,
        # P_th_kW=df_load_ts_slice.loc[df_load_ts_slice.sum(axis=1).idxmax()],
        P_th_kW=network.components['consumers']['P_heat_max']*0.5,
        )

    # Download the elevation data for the current area
    download_elevation_data(gdf_houses, show_plot=show_plot)


def workflow_default(buildings, show_plot=True):
    """Run many of the functions with default values in correct order.

    This is meant as an example on how to use these functions.
    """
    buildings = fill_residential_osm_building_types(
        buildings, discard_types=['yes'],
        notna_columns=['addr:street', 'addr:housenumber'])

    buildings = building_type_from_osm(buildings)

    buildings = identify_heated_buildings(
        buildings,
        notna_columns=['addr:street', 'addr:housenumber'])

    buildings = assign_random_construction_classification(
        buildings,
        col_refurbished_state=None,
        refurbished_weights=None,
        year_mu=1950, year_sigma=18,
        )
    buildings = assign_construction_classification_from_arge(
        buildings,
        aliases_MFH=['business', 'other-heated-non-residential'],
        )
    buildings = calculate_building_areas(
        buildings,
        aliases_business=['other-heated-non-residential'],
        aliases_unknown=['non-heated'],
        )
    buildings = set_heat_demand_from_source_arge(
        buildings,
        aliases_MFH=['business', 'other-heated-non-residential'],
        )
    buildings = set_heat_demand_for_new_buildings(buildings)
    buildings = apply_climate_correction_factor(buildings)

    # Choose one of these functions for estimating hot water demand
    buildings = set_domestic_hot_water_from_DIN18599(buildings)
    # buildings = set_domestic_hot_water_from_values(buildings)

    buildings = separate_heating_and_DHW(buildings)
    buildings = guess_thermal_power_from_full_load_hours(buildings)
    buildings = set_n_persons_and_flats(buildings)
    log_statistics(buildings, show_plot=show_plot)

    gdf_hex = create_hexgrid(
        buildings, clip=False,
        extensive_variables=['e_th_total_kWh'],
        show_plot=show_plot,
        resolution=None, buffer_distance=100,
        )

    if show_plot:
        plot_hexgrid(gdf_hex, 'e_th_total_kWh', buildings,
                     title='Wärmebedarf [MWh]', scale=1/1000,
                     plot_basemap=True,
                     )

    return buildings


def load_example_area(crs='epsg:4647'):
    """Load example area defined by a list of lat/lon coordinates."""
    bbox = [(9.1008896, 54.1954005),
            (9.1048374, 54.1961024),
            (9.1090996, 54.1906397),
            (9.1027474, 54.1895923),
            ]
    polygon = shapely.geometry.Polygon(bbox)
    gdf_polygon = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')

    if crs:
        gdf_polygon.to_crs(crs=crs, inplace=True)

    return gdf_polygon


# Section "Input/Output"

def load_xml_geodata(file, layer=None, driver='GML', crs="EPSG:25833",
                     **kwargs):
    """Load (3D) geographical data from XML files (citygml).

    This should not be necessary, since gpd.read_file(file) should do the
    same. But somehow this approach works, using fiona directly, while
    geopandas fails for some xml files.
    """
    with fiona.open(file, 'r', driver=driver, layer=layer, **kwargs) as src:
        features = [feature for feature in src]
    gdf = gpd.GeoDataFrame.from_features([feature for feature in features],
                                         crs=crs)
    return gdf


def save_gis_generic(gdf, file, ext, driver, path='.', crs=None,
                     type_errors='coerce', save_excel=False, **kwargs):
    """Save a GeoDataFrame to a given GIS file type at the given path.

    This is 'just' a wrapper around geopandas.to_file(), but includes
    logic to give a chance to close an opened target file instead of failing
    with PermissionError and to handle type errors, by conversion to string.

    crs "EPSG:4647" is recommended for good length calculation results.

    Parameters
    ----------
    gdf : GeoDataFrame
        Data to save.
    file : str
        File name to save to.
    ext : str
        File extension
    driver : str
        An OGR format driver accepted by geopandas.to_file()
    path : str, optional
        File path to save to. The default is '.'.
    crs : pyproj.CRS, optional
        Coordinate reference system to use for the saved file.
        The default is None. In this case the crs of gdf is not changed.
    type_errors : str, optional
        The GeoPandas to_file() method can sometimes cause TypeErrors.
        If ‘raise’, then invalid parsing will raise an exception.
        If ‘coerce’, then invalid columns will be converted to string.
        The default is '‘coerce’'.
    save_excel : bool, optional
        If True, also save an Excel file with the same name. The default
        is False.
    **kwargs :
        Other keyword arguments are passed on to geopandas.to_file().

    Returns
    -------
    None.

    """
    file = file.removesuffix(ext)  # Drop extension from the file name

    filepath = os.path.join(path, file+ext)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    gdf = gdf.copy()  # Make a copy to keep the input object unchanged

    if crs is None:
        if gdf.crs is None:
            crs = "EPSG:4647"  # 'backup' crs
            logger.error(f"Geometry has no CRS. Assigning default {crs}, "
                         "but that may be incorrect.")
        else:
            crs = gdf.crs
    try:
        logger.info('Saving... %s', filepath)
        if gdf.crs is not None and gdf.crs != crs:
            gdf.to_crs(crs=crs, inplace=True)

        gdf.to_file(filepath, driver=driver, **kwargs)
    except PermissionError:
        try:
            input("Please close QGIS to allow saving the file '{}.geojon'. "
                  "Then hit Enter, or CTRL+C to abort.\n".format(file))
            save_gis_generic(
                gdf, file, ext=ext, driver=driver, path=path, crs=crs,
                type_errors=type_errors, save_excel=save_excel, **kwargs)
        except KeyboardInterrupt:
            logger.info('Saving %s.geojson skipped!', file)
    except ValueError as e:
        # This may be an error like:
        # ValueError: Invalid field type <class 'numpy.int32'>
        # ValueError("Invalid field type <class 'list'>")
        # Test which column causes the error
        for col in gdf.columns:
            if col in gdf.select_dtypes(gdf.geometry.name).columns:
                continue  # Do not mess with the geometry column
            try:
                # Try to trigger the exception again, but only for this column
                json.dumps(gdf[col].tolist())  # Convert the column to JSON
                # <class 'list'> causes errors in gdf.to_file(), but
                # not in json.dumps. So if a list is present, we have
                # to manually raise an exception
                if list in gdf[col].map(type).unique():
                    raise e
            except (ValueError, TypeError) as e2:
                if type_errors == 'raise':
                    raise TypeError(
                        f"Error in column '{col}' when saving '{file}': "
                        f"'{e2}'. You may try the option "
                        "type_errors='coerce' to attempt fixing this "
                        "by forcing conversion to string") from e
                else:
                    logger.warning("Converting column '%s' to string to "
                                   "avoid error: %s", col, e2)
                    gdf[col] = gdf[col].astype(str)

        if type_errors == 'coerce':
            # All errors should have been fixed now, so try saving again
            # Pass type_errors='raise' to avoid getting caught in a loop
            save_gis_generic(
                gdf, file, ext=ext, driver=driver, path=path, crs=crs,
                type_errors='raise', save_excel=save_excel, **kwargs)
    else:  # Execute when there is no error
        if save_excel:
            _save_excel(gdf, os.path.join(path, file+'.xlsx'))


def save_geojson(gdf, file, path='.', crs=None, type_errors='coerce',
                 save_excel=False, **kwargs):
    """Save a GeoDataFrame to geojson file at the given path.

    For large datasets, consider using save_geopackage() or save_sql()
    instead, which yield better performance in e.g. QGIS.
    """
    save_gis_generic(gdf, file, ext='.geojson', driver='GeoJSON',
                     path=path, crs=crs, type_errors=type_errors,
                     save_excel=save_excel, **kwargs)


def save_geopackage(gdf, file, path='.', crs=None, type_errors='coerce',
                    save_excel=False, **kwargs):
    """Save a GeoDataFrame to a GeoPackage database file at the given path.

    GeoPackage has great performance in QGIS, even for large data sets.
    """
    save_gis_generic(gdf, file, ext='.gpkg', driver='GPKG',
                     path=path, crs=crs, type_errors=type_errors,
                     save_excel=save_excel, **kwargs)


def save_sql(gdf, file, path='.', mode='w', layer=None):
    """Save a GeoDataFrame to a SQLite database file at the given path.

    Note for QGIS: Reading the result as a regular Vector Layer should work
    fine, while reading this as a Spalite Database failes with error
    "Failure getting table metadata".
    The function save_spatialite() was an attempt to fix that, but does not
    work yet.

    Notes:
        - Loading SQL in QGIS yield much better performance than e.g. geojson
        - SQLite files can be updated inplace, without needing to close QGIS
        - Removes any unicode characters that cause encoding errors
        - Saving with driver "SQLite" forces lowercase on all column names
        - Saving would fail in case of duplicate column names, so they are
          renamed instead (e.g. 'column' to 'column1')
    """
    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, file+'.sqlite')
    logger.info('Saving... %s', filepath)

    gdf.columns = gdf.columns.str.lower()
    if gdf.columns.duplicated().any():
        fix_duplicate_column_names(gdf)

    try:
        gdf.to_file(filepath, driver="SQLite", mode=mode, layer=layer,
                    overwrite=True)
    except UnicodeEncodeError as e:
        # Message example: codec can't encode character '\u0308' in position
        logger.error(e)
        bad_char = re.findall((r"character '(.+)' in position"), str(e))[0]

        logger.warning("Removing bad character '%s'", bad_char)
        gdf.replace(bad_char, '', regex=True, inplace=True)

        # for col in gdf.columns:
        #     if gdf[col].dtype == 'object':
        #         if gdf[col].str.contains(bad_char).any():
        #             logger.warning(
        #                 "Replacing bad character '%s' in column '%s'",
        #                 bad_char, col)
        #             gdf[col] = gdf[col].str.replace(bad_char, '', regex=True)

        save_sql(gdf, file, path)  # Run function recursively


def save_spatialite(gdf_in, file, path='.', mode='w', layer=None):
    """Save a GeoDataFrame to a SpatiaLite database file at the given path.

    Not working correctly yet!

    https://www.giacomodebidda.com/posts/export-a-geodataframe-to-spatialite/
    https://gis.stackexchange.com/questions/141818/insert-geopandas-geodataframe-into-spatialite-database
    """
    import os
    import sqlite3

    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, file+'.sqlite')
    logger.info('Saving... %s', filepath)

    # breakpoint()
    DB_PATH = os.path.join(os.getcwd(), 'your-database6.db')
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    gdf = gdf_in.copy()
    # Note: 'unique_idx' is just the name of any column that exists in the
    # DataFrame and has unique values.
    gdf.index.set_names("unique_idx", inplace=True)
    gdf = gdf.reset_index()

    # breakpoint()
    if len(gdf.geometry.type.unique()) == 1:
        geom_type = gdf.geometry.type.unique()[0]
    else:
        raise ValueError("Multiple geometry types not supported")

    epsg = gdf.crs.to_epsg()
    if gdf.geometry.has_z.all():
        n_dims = 3
    elif not gdf.geometry.has_z.any():
        n_dims = 2
    else:
        raise ValueError("Gemetries with mixed dimensions (2 and 3) not "
                         "supported")

    # Drop all geospatial data
    # df = gdf.drop(['geometry'], axis='columns')

    # gdfb = gdf[['geometry', 'unique_idx']].copy()  # Convert geometry to binary
    # gdfb['geometry'] = gdf['geometry'].to_wkb()
    gdf['geometry'] = gdf['geometry'].to_wkb()
    # tuples = gdfb.to_records(index=False).tolist()

    # Create the table and populate it with non-geospatial datatypes
    with sqlite3.connect(DB_PATH) as conn:
        # df.to_sql('your_table_name', conn, if_exists='replace', index=False)
        gdf.to_sql('your_table_name', conn, if_exists='replace', index=False)

    with sqlite3.connect(DB_PATH) as conn:
        conn.enable_load_extension(True)
        conn.load_extension("mod_spatialite")
        conn.execute("SELECT InitSpatialMetaData(1);")
        conn.execute(
            """
            SELECT AddGeometryColumn('your_table_name', 'wkb_geometry', {epsg}, '{geom_type}', {n_dims});
            """.format(epsg=epsg, geom_type=geom_type, n_dims=n_dims)
        )

    # import shapely.wkb as swkb
    # records = [
    #     {'gml_id': gdf.gml_id.iloc[i],
    #      'wkb': swkb.dumps(gdf.geometry.iloc[i])}
    #     for i in range(gdf.shape[0])
    # ]
    # tuples = tuple((d['wkb'], d['gml_id']) for d in records)
    # tuples2 = gdfb.to_numpy()

    # gpd.GeoSeries.from_wkb(records['wkb'])
    # gpd.GeoSeries.from_wkb(gdfb['geometry'])

    with sqlite3.connect(DB_PATH) as conn:
        conn.enable_load_extension(True)
        conn.load_extension("mod_spatialite")
        conn.execute(
            """UPDATE your_table_name SET wkb_geometry=GeomFromWKB(geometry, {epsg});""".format(epsg=epsg))
        # conn.executemany(
        #     """
        #     UPDATE your_table_name
        #     SET wkb_geometry=GeomFromWKB(?, {epsg})
        #     WHERE your_table_name.unique_idx = ?
        #     """.format(epsg=epsg), (tuples)
        # )
    breakpoint()

    with sqlite3.connect(DB_PATH) as conn:
        conn.enable_load_extension(True)
        conn.load_extension("mod_spatialite")
        cur = conn.execute(
            """
            SELECT wkb_geometry FROM your_table_name
            """
        )
        results = cur.fetchall()

    print(results)

def fix_duplicate_column_names(df):
    """Append integer 1, 2, ... to duplicate column names of DataFrame df."""
    logger.warning("Fixing duplicate column names")
    s = df.columns.to_series().groupby(df.columns)
    df.columns = np.where(s.transform('size') > 1,
                          df.columns + s.cumcount().add(1).astype(str),
                          df.columns)


def _save_excel(df, path, **kwargs):
    """Save (Geo-)DataFrame as Excel (alias function)."""
    save_excel(df, path, **kwargs)


def save_excel(df, path, **kwargs):
    """Save (Geo-)DataFrame as Excel file (without 'geometry' column).

    Can also be automatically called by save_geojson().

    Just a wrapper around 'df.to_excel(path, **kwargs)' that creates the
    directory and asks to close the file in case of permission error.
    """
    try:
        df_save = df.drop(columns=[df.geometry.name])
    except AttributeError:
        df_save = df.copy()

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    try:
        logger.info('Saving... %s', path)
        df_save.to_excel(path, **kwargs)
    except PermissionError:
        input("Please close the file to allow saving! Then hit Enter.")
        save_excel(df_save, path, **kwargs)


def merge_with_ogr(paths, merge_target, crs="EPSG:25832", layer=None):
    """Merge all files in paths into one output file with ogr2ogr.

    Sometimes loading geodata files with pandas or even fiona fails for
    various reasons. It may instead help to convert files into another format
    with ogr2ogr directly.

    ogr2ogr comes with the installation of gdal (``conda install gdal``)

    https://gdal.org/programs/ogr2ogr.html

    Useful formats:
        - "ESRI Shapefile"
        - "GPKG"

    Known issue:
        - The maximum length of each field is defined by the first file
          loaded. If longer values in the same field appear in later
          files, a warning is displayed
    """
    # This function requires access to the program ogr2ogr.exe
    # If python gdal is installed, it should be available in the following
    # path, so that is added to the system PATH.
    ogr2ogr_exe = os.path.join(os.path.dirname(sys.executable),
                               'Library/bin/ogr2ogr.exe')
    if os.path.exists(ogr2ogr_exe):
        os.environ['PATH'] += os.pathsep + os.path.dirname(ogr2ogr_exe)

    filename, ext = os.path.splitext(os.path.basename(merge_target))
    if ext == '.gpkg':
        format_ = 'GPKG'
    elif ext == '.shp':
        format_ = 'ESRI Shapefile'
    else:
        raise ValueError(f"Undefined extension '{ext}'")

    if layer is None:
        layer = filename

    if not os.path.exists(os.path.dirname(merge_target)):
        os.makedirs(os.path.dirname(merge_target))

    if os.path.exists(merge_target):
        # The file has to be removed before recreating it
        os.remove(merge_target)

    for src_file in paths:
        logger.debug(src_file)
        # Create one merged gpkg-file for all gml files
        output = subprocess.check_output(
            ['ogr2ogr', '-f', format_, merge_target, src_file,
             '-dim', '3', '-s_SRS', crs, '-t_SRS', crs,
             # '-oo', 'REMOVE_UNUSED_FIELDS=YES',
             # '-oo', 'REMOVE_UNUSED_LAYERS=YES',
             '-append', '-update',
             '-nln', layer,  # new layer name, instead of one layer per file
             # "building", "consistsofbuildingpart",
             ],
            stderr=subprocess.STDOUT,
            universal_newlines=True
            )
        if len(output) > 0:
            logging.warning(output)


# Section "Buildings and heat demand"
def identify_heated_buildings(
        gdf, col_heated='heated',
        area_threshold=40, notna_columns=None,
        col_building_type='building_type',
        types_heated=['SFH', 'MFH', 'business',
                      'other-heated-non-residential'],
        ):
    """Identify heated buildings depending on certain criteria.

    All provided criteria must be true (logical AND) to assign heated status.

    area_threshold (float, optional):
        Minimum required area in m³ to be counted as heated building.

    notna_columns (list, optional):
        List of names of columns that must not have
        missing (na) values. Common example: Entries for street name and house
        number are required for a building to be considered heated.
        For OpenStreetMap data, use e.g. ['addr:street', 'addr:housenumber'].
    """
    logger.info('Identify heated buildings')

    if area_threshold is not None:
        mask1 = gdf.area >= area_threshold
    else:
        mask1 = True

    if notna_columns is not None:
        mask2 = gdf[notna_columns].notna().all(axis='columns')
    else:
        mask2 = True

    if col_building_type in gdf.columns:
        mask3 = gdf[col_building_type].isin(types_heated)
    else:
        mask3 = True

    gdf.loc[mask1 & mask2 & mask3, col_heated] = True
    gdf.loc[~(mask1 & mask2 & mask3), col_heated] = False

    return gdf


def assign_alkis_functions_to_osm_building_keys(
        buildings,
        col_building_osm='building_osm',
        col_alkis_function='funktion',
        overwrite=['yes'],
        show_plot=False,
        ):
    """Synchronize openstreetmap "building" and ALKIS "FUNKTION" tags.

    overwrite: If True, overwrite all values. If list, only overwrite nan and
    values in the list. E.g: ['yes'] ('yes' is the most generic tag for a
    building)

    https://wiki.openstreetmap.org/wiki/Key:building
    """
    logger.info('Assign alkis functions to osm building keys')

    alkis_to_osm_building = dict({
        'Allgemein bildende Schule': 'school',
        'Arkade': 'retail',
        'Auskragende/zurückspringende Geschosse': 'roof',
        'Aussichtsturm': 'tower',
        # 'BAUWERK_AX_BauwerkImGewaesserbereich': '',
        'BAUWERK_AX_VorratsbehaelterSpeicherbauwerk': 'storage_tank',
        'Berufsbildende Schule': 'college',
        'Betriebsgebäude für Flugverkehr': 'hangar',
        'Betriebsgebäude für Schienenverkehr': 'train_station',
        'Betriebsgebäude für Schiffsverkehr': 'hangar',
        'Betriebsgebäude für Straßenverkehr': 'hangar',
        'Durchfahrt an überbauter Verkehrsstraße': 'bridge',
        'Durchfahrt im Gebäude': 'bridge',
        'Feuerwehr': 'fire_station',
        'Finanzamt': 'government',
        'Forschungsinstitut': 'university',
        'Forsthaus': 'barn',
        'Friedhofsgebäude': 'cemetery',
        'Funkmast': 'mast',
        'Garage': 'garage',
        'Gebäude für Erholungszwecke': 'civic',
        'Gebäude für Gesundheitswesen': 'civic',
        'Gebäude für Gewerbe und Industrie': 'industrial',
        'Gebäude für Gewerbe und Industrie mit Wohnen': 'industrial',
        'Gebäude für Handel und Dienstleistung mit Wohnen': 'retail',
        'Gebäude für Handel und Dienstleistungen': 'retail',
        'Gebäude für Wirtschaft oder Gewerbe': 'commercial',
        'Gebäude für soziale Zwecke': 'civic',  # KITA, Sporthalle
        'Gebäude für öffentliche Zwecke': 'public',
        'Gebäude für öffentliche Zwecke mit Wohnen': 'public',
        'Gebäude im Stadion': 'stadium',
        'Gebäude zur Entsorgung': 'public',
        'Gebäude zur Versorgung': 'public',
        'Gemischt genutztes Gebäude mit Wohnen': 'apartments',
        'Gericht': 'public',
        'Gotteshaus': 'church',
        'Hallenbad': 'civic',
        'Hochschulgebäude (Fachhochschule, Universität)': 'university',
        'Jugendherberge': 'hotel',
        'Justizvollzugsanstalt': 'civic',
        'Kapelle': 'chapel',
        'Kirche': 'church',
        'Krankenhaus': 'hospital',
        'Kreisverwaltung': 'government',
        'Land- und forstwirtschaftliches Betriebsgebäude': 'barn',
        'Land- und forstwirtschaftliches Wohngebäude': 'barn',
        'Messehalle': 'public',
        'Museum': 'museum',
        'Parkdeck': 'parking',
        'Parkhaus': 'parking',
        'Polizei': 'public',
        'Rathaus': 'government',
        'Schloss': 'castle',
        'Schornstein im Gebäude': 'chimney',
        'Schornstein, Schlot, Esse': 'chimney',
        'Schutzhütte': 'hut',
        'Sendeturm, Funkturm, Fernmeldeturm': 'tower',
        'Silo': 'silo',
        'Sport-, Turnhalle': 'sports_hall',
        'Tank': 'storage_tank',
        'Tankstelle': 'service',
        'Theater, Oper': 'civic',
        'Trauerhalle': 'civic',
        'Treibhaus, Gewächshaus': 'greenhouse',
        'Turm im Gebäude': 'tower',
        'Veranstaltungsgebäude': 'public',
        'Verwaltungsgebäude': 'government',
        'Waschstraße, Waschanlage, Waschhalle': 'service',
        'Wasserbehälter': 'storage_tank',
        'Wasserturm': 'water_tower',
        'Windmühle': 'tower',
        'Wohngebäude': 'residential',
        'Wohngebäude mit Gemeinbedarf': 'apartments',
        'Wohngebäude mit Gewerbe und Industrie': 'apartments',
        'Wohngebäude mit Handel und Dienstleistungen': 'apartments',
        'Zuschauertribüne, überdacht': 'grandstand',
        'Überdachung': 'roof'
        })

    for alkis_function, building in alkis_to_osm_building.items():
        mask1 = buildings[col_alkis_function] == alkis_function
        mask2 = buildings[col_building_osm].isna()
        buildings.loc[mask1 & mask2, col_building_osm] = building

    if overwrite is True:
        buildings[col_building_osm] = \
            buildings[[col_alkis_function]].replace(
                {col_alkis_function: alkis_to_osm_building})
    else:
        # Mask the rows that should be written (nan and 'yes'), but only
        # if those rows have information found in the keyword dict
        mask = (buildings[col_building_osm].isna()
                | (buildings[col_building_osm].isin(overwrite)
                   &
                   buildings[col_alkis_function].isin(alkis_to_osm_building
                                                      .keys())))
        # Paste those values from "FUNKTION" column to "building_osm"
        buildings[col_building_osm] = buildings[col_building_osm].where(
            ~mask, buildings[col_alkis_function])
        # Replace the values in "building_osm"
        buildings.replace({col_building_osm: alkis_to_osm_building},
                          inplace=True)

    if show_plot:
        buildings.plot(column=col_building_osm, figsize=(20, 20), legend=True)
    return buildings


def fill_residential_osm_building_types(
        gdf,
        col_building_osm='building_osm',
        col_heated=None,
        area_threshold=200,  # m²
        discard_types=None,
        notna_columns=None,
        ):
    """Fill undefined buildings with residential osm building keys.

    Often enough, the building types in OpenStreetMap are not properly
    defined and the generic ``building=yes`` is given.
    We cannot work with that, but rather than delete all those buildings,
    we can make a guess and assume that most of these are residential
    buildings.

    Buildings with col_heated == True (if given) and larger than
    area_threshold (in m²) are labelled 'apartments' (a OSM definition
    for multi-family-houses), while smaller buildings are labelled
    'house' (single-family).

    https://wiki.openstreetmap.org/wiki/Key:building

    Alternative functions:
        - fill_residential_osm_building_types()
        - fill_random_osm_building_types()

    Args:
        gdf (GeoDataFrame): A GeoDataFrame of the buildings

        col_building_osm (str): Column name storing the building type

        col_heated (str): Column name used for an additional filter. Only
        heated buildings (column value == True) are assigned a building type.

        area_threshold (float): Area in m² that should separate houses
        and appartments

        discard_types (list): List of keys in the column 'col_building_osm'
        that will be discarded and replaced by the new types. Default
        is None, but recommendation is ['yes']. This generic building tag
        may make further processing problamatic otherwise.

        notna_columns (list, optional):
        List of names of columns that must not have
        missing (na) values. Common example: Entries for street name and house
        number are required for a building to be considered heated.
        For OpenStreetMap data, use e.g. ['addr:street', 'addr:housenumber'].

    """
    logger.info('Fill residential OpenStreetMap building types')

    if col_building_osm not in gdf.columns:
        gdf[col_building_osm] = np.nan

    if discard_types is not None:
        mask_na = gdf[col_building_osm].replace(discard_types, np.nan).isna()
    else:
        mask_na = gdf[col_building_osm].isna()

    mask_area = gdf.area >= area_threshold

    if notna_columns is not None:
        mask_notna = gdf[notna_columns].notna().all(axis='columns')
    else:
        mask_notna = pd.Series([True]*len(gdf))

    if col_heated is not None:
        mask_heated = gdf[col_heated] == True
    else:
        mask_heated = pd.Series([True]*len(gdf))

    gdf.loc[mask_na & mask_area & mask_heated & mask_notna,
            col_building_osm] = 'apartments'
    gdf.loc[mask_na & ~mask_area & mask_heated & mask_notna,
            col_building_osm] = 'house'

    return gdf


def fill_random_osm_building_types(
        buildings,
        col_building_osm='building_osm',
        col_heated='heated',
        osm_building_types=['house', 'residential', 'detached',
                            'semidetached_house', 'apartments'],
        ):
    """Fill undefined buildings with osm building keys.

    https://wiki.openstreetmap.org/wiki/Key:building

    Use carefully. Assigning types at random is only useful for tests.
    However, this is useful if you want to use
    building_type_from_osm() as a next step.

    Alternative functions:
        - fill_residential_osm_building_types()
        - fill_random_osm_building_types()

    Args:
        buildings (gdf): A GeoDataFrame of the buildings

        col_building_osm (str): Column name storing the building type

        col_heated (str): Column name used for an additional filter. Only
        heated buildings (column value == True) are assigned a building type.

        osm_building_types (list): List of types to choose from for the
        random assignment. E.g. 'house', 'residential', 'detached',
        'semidetached_house', 'apartments'.

    """
    logger.info('Fill random osm building types')

    if col_building_osm not in buildings.columns:
        buildings[col_building_osm] = np.nan

    # Get all buildings where col_heated is True and building is not yet
    # defined, i.e. None
    buildings.sort_values(by=[col_heated, col_building_osm],
                          ascending=False,
                          na_position='first',
                          ignore_index=True,
                          inplace=True)
    mask1 = buildings[col_heated] == True
    mask2 = buildings[col_building_osm].isna()
    n_candidates = (mask1 & mask2).value_counts().get(True, default=0)
    logger.info("Assigned %s random building types to column %s",
                n_candidates, col_building_osm)

    rng = np.random.default_rng(42)
    buildings.loc[0:n_candidates-1, col_building_osm] = rng.choice(
        osm_building_types, size=n_candidates,
        # p=[0.6, 0.3, 0.1],
        )
    return buildings


def building_type_from_osm(
        gdf,
        col_building_osm='building_osm',
        col_building_type='building_type',
        assign_key_yes='unknown',
        warn_undefined=True,
        ):
    """Assign OpenStreetMap's "building" key values to general types.

    This function translates specific "building" key values from OpenStreetMap
    to broader building types, based on their potential heat demand. This
    categorization is designed as preparation for functions like:
        - assign_construction_classification_from_arge()
        - calculate_building_areas()
        - set_heat_demand_from_source_arge()

    Parameters:
        gdf : GeoDataFrame
            The input geospatial data.
        col_building_osm : str, optional
            Column name for OSM building types. Default is 'building_osm'.
        col_building_type : str, optional
            Column name for general building types. Default is 'building_type'.
        assign_key_yes : str, optional
            Defines the type to which the 'yes' key is assigned.
            Default is 'unknown'. This may need further attention in your
            following workflow. It is strongly advised to make a conscious
            decision here about how to handle those buildings. E.g. mark them
            as 'non-heated' and discard them in a following step.

    Returns:
        GeoDataFrame
            The updated GeoDataFrame with general building types.

    Reference:
    https://wiki.openstreetmap.org/wiki/Key:building
    """
    logger.info('Assign building type from OpenStreetMap')

    translate_dict = {
        'SFH': ['house', 'residential', 'detached', 'semidetached_house',
                'terrace', 'farm', 'bungalow', 'villa',
                ],
        'MFH': ['apartments', 'dormitory'],
        'business': ['commercial', 'office', 'retail', 'supermarket', 'shop',],
        'other-heated-non-residential':
            ['school', 'college', 'university', 'fire_station', 'government',
             'civic', 'industrial', 'public', 'hotel', 'hospital', 'museum',
             'sports_hall', 'kindergarten', 'warehouse',
             'sports_centre', 'mosque', 'religious', 'brewery', 'cathedral',
             'presbytery', 'hostel', 'restaurant',
             ],
        'non-heated':
            ['roof', 'tower', 'hangar', 'train_station', 'bridge', 'barn',
             'cemetery', 'mast', 'garage', 'garages', 'stadium', 'church',
             'chapel', 'parking', 'castle', 'hut', 'silo', 'storage_tank',
             'greenhouse', 'water_tower', 'grandstand', 'roof', 'stadium',
             'shed', 'service', 'carport', 'farm_auxiliary', 'construction',
             'toilets', 'bunker', 'disused', 'ruins', 'allotment_house',
             'no', 'kiosk', 'hall', 'electricity', 'chimney', 'container',
             'substation', 'lighthouse', 'ship', 'shelter', 'roof_terrace',
             'stable',
             ],
        'unknown':
            [],
        }

    # The user needs to decide how to handle the building tag 'yes'
    # It is appended to whichever type they choose.
    if assign_key_yes in translate_dict.keys():
        translate_dict[assign_key_yes].append('yes')
    else:
        translate_dict[assign_key_yes] = ['yes']

    for b_type, b_list in translate_dict.items():
        gdf.loc[gdf[col_building_osm].isin(b_list), col_building_type] = b_type

    if warn_undefined:
        undefined = gdf.loc[gdf[col_building_type].isna(),
                            col_building_osm].value_counts()
        if not undefined.empty:
            logger.warning("During assignment of general building types from "
                           "OpenStreetMap types, the following tags where "
                           "found to be undefined. Consider updating "
                           "building_type_from_osm() with appropriate "
                           "assignments:\n%s", undefined)

    return gdf


def assign_random_construction_classification(
        gdf,
        col_construction_year='construction_year',
        col_refurbished_state='refurbished_state',
        refurbished_weights=[0.6, 0.3, 0.1],
        year_mu=None,
        year_sigma=None,
        ):
    """Assign random construction years and/or refurbished states to buildings.

    Instead of assigning both years and refurbished states with this function,
    you can also only assign years and then assign refurbishments based on
    propabilities from the literature with
    assign_construction_classification_from_arge() instead.

    Args:
        col_refurbished_state (str): If not None, this colun name is used
        to store the generated random states of refurbishment.

        refurbished_weights (list): List of three floats, defining the
        probabilities associated with the states 'not refurbished',
        'slightly refurbished' and 'mostly refurbished'. If None, a uniform
        distribution is used (which is probably not desired).

        col_construction_year(str): If not None, this column name is used
        to store the generated random year values.

        year_mu (int): If mean mu and sigma of year are given, a normal
        distribution with those properties is used to assign the random years.
        If not, a regular random distribution between 1900 and 2008 is used.

        year_sigma (int): (See year_mu)


    Columns 'refurbished_state' and 'construction_year' are set.
    Afterwards, set_heat_demand_from_source_arge() can be used.
    """
    logger.info('Assign random construction classification')

    # These names must match those used in set_heat_demand_from_source_arge()
    # and assign_construction_classification_from_arge()
    refurbished_states = [
        'not refurbished', 'slightly refurbished', 'mostly refurbished']

    rng = np.random.default_rng(42)
    if col_construction_year is not None:
        if col_construction_year not in gdf.columns:
            gdf[col_construction_year] = np.nan
        if year_mu is None or year_sigma is None:
            gdf.fillna({col_construction_year:
                        pd.Series(rng.integers(low=1900, high=2008,
                                               size=len(gdf)),
                                  index=gdf.index)},
                       inplace=True)
        else:
            gdf.fillna({col_construction_year:
                        pd.Series(rng.normal(year_mu, year_sigma,
                                             size=len(gdf)),
                                  index=gdf.index)},
                       inplace=True)
            gdf[col_construction_year] = \
                gdf[col_construction_year].astype('int')

    if col_refurbished_state is not None:
        if col_refurbished_state not in gdf.columns:
            gdf[col_refurbished_state] = np.nan

        gdf.fillna({col_refurbished_state:
                    pd.Series(rng.choice(refurbished_states, size=len(gdf),
                                         p=refurbished_weights),
                              index=gdf.index)},
                   inplace=True)

    return gdf


def assign_construction_classification_from_arge(
        gdf,
        col_building_type='building_type',
        col_refurbished_state='refurbished_state',
        col_construction_year='construction_year',
        fillna_value='not refurbished',
        aliases_SFH=None,
        aliases_MFH=None,
        ):
    """Assign refurbishment states based on probabilities defined in ARGE.

    The source ARGE defines probabilities of three refurbishement states,
    based on building type and construction year.
    For each building, draw a random refurbishement state based on its
    assigned probabilities. Existing refurbishment states are not overwritten.

    The original source defines single- and multi family homes (SFH, MFH).
    By setting the arguments 'aliases_SFH' and 'aliases_MFH' to lists of
    building type names, those will be treated as SFH or MFH, respectively.

    'construction_year':
        - Various age classes until 2008

    'building_type':
        - 'SFH'
        - 'MFH'

    'refurbished_state':
        - 'not refurbished'
        - 'slightly refurbished'
        - 'mostly refurbished'

    Args:
        fillna_value (str): If not None, use this string to fill all
        missing values, e.g. 'not refurbished'.
    """
    df = load_src_data_arge_refurbishmend_probabilities()
    df = process_src_data_arge(
        df, col_construction_year, [col_building_type, col_refurbished_state],
        'probabilities',
        aliases_SFH=aliases_SFH, aliases_MFH=aliases_MFH)
    # Reformat to the probabilities for each type of building and year
    df = df.unstack(col_refurbished_state)
    choices = df.columns  # 'not refurbished', 'slightly refurbished', ...

    # Define the DataFrame of probability distributions for each
    # building by matching type and year with the source data
    df_probabilities = (pd.merge(
        gdf[[col_building_type, col_construction_year]],
        df.reset_index(),
        how='left',
        on=[col_building_type, col_construction_year])
        .set_index(gdf.index)  # Keep index so we can drop NaN
        )
    # Keep only the choice columns and drop NaN
    df_probabilities = df_probabilities[choices].dropna()

    # Generate a DataFrame of selections. In each row, a 1 denotes
    # which choice was selected.
    rng = np.random.default_rng(42)
    df_selections = pd.DataFrame(
        index=df_probabilities.index,  # Keep the index intact!
        data=rng.multinomial(n=1, pvals=df_probabilities),
        columns=df_probabilities.columns)

    # Finally, reduce the DataFrame to one column with the selected choice
    # and assign it to the original DataFrame. Since the index was kept,
    # everything should align, despite dropping NaN.
    if col_refurbished_state not in gdf.columns:
        gdf[col_refurbished_state] = np.nan

    gdf.fillna({col_refurbished_state: df_selections.idxmax(axis=1)},
               inplace=True)

    # Test the results
    # for building_type in ['SFH', 'MFH']:
    #     dfx = gdf[gdf[col_building_type] == building_type]
    #     for year in sorted(dfx[col_construction_year].unique()):
    #         dfy = dfx[dfx[col_construction_year] == year]
    #         hist = dfy[col_refurbished_state].value_counts()/len(dfy)
    #         print(building_type, year)
    #         print(hist)

    if fillna_value is not None:
        gdf.fillna({col_refurbished_state: fillna_value}, inplace=True)

    return gdf


def set_heat_demand_from_source_arge(
        df_in,
        col_heated='heated',
        col_spec_total='e_th_spec_total',
        col_total='e_th_total_kWh',
        col_building_type='building_type',
        col_refurbished_state='refurbished_state',
        col_construction_year='construction_year',
        fillna_value=None,
        warnings='ignore',
        decimals=2,
        eta=0.85,
        aliases_SFH=None,
        aliases_MFH=None,
        ):
    """Get estimated annual heat demand for heating and domestic hot water.

    Applies only to residential buildings constructed until 2008.
    Distinguishes:
        'construction_year':
            - Various age classes until 2008
        'building_type':
            - 'SFH'
            - 'MFH'
        'refurbished_state':
            - 'not refurbished'
            - 'slightly refurbished'
            - 'mostly refurbished'


    Data is assumed to be valid for TRY-region 4 (Potsdam). Data
    can be corrected to actual building location with
    apply_climate_correction_factor()

    Alternative functions:
        - set_heat_demand_from_source_arge()
        - set_heat_demand_from_source_B()

    Parameters
    ----------
    df_in : DataFrame
        A table including the columns 'building_type', 'refurbished_state',
        'construction_year'.
    result_col : str, optional
        Name of the column added to the given DataFrame.
        The default is 'E_th_spec'.
    fillna_value : float, optional


    Returns
    -------
    DataFrame
        Input DataFrame with the added column containing the heat demand in
        unit kWh/(m² * a) (annual energy per Area A_N defined in German EnEV)

    Source:
    Walberg, Dietmar (Hg.) (2011): Wohnungsbau in Deutschland - 2011 -
    Modernisierung oder Bestandsersatz. Studie zum Zustand und der
    Zukunftsfähigkeit des deutschen „Kleinen Wohnungsbaus“.
    Unter Mitarbeit von Astrid Holz, Timo Gniechwitz und Thorsten Schulze.
    Arbeitsgemeinschaft für Zeitgemäßes Bauen e.V. Kiel
    (Bauforschungsbericht, 59).
    Online verfügbar unter
    https://www.bfw-bund.de/wp-content/uploads/2015/10/ARGE-Wohnungsbau-Deutschland-2011-Textband-1.pdf

    The original source tables on pp. 48 & 53 describe the energy demand as
    final energy demand ("Endenergie"). These values need converted to (net)
    heat demand with a generic efficiency for e.g. a boiler.
    In this implementation, a default of 85% efficiency is used.

    Definitions for the states of refurbishment (in German):

        a) Nicht modernisiert: Seit der Erbauung gab es keine
        wesentlichen Modernisierungen, d.h. maximal eine
        Maßnahme an der Gebäudehülle und/oder der
        Anlagentechnik im Standard nach WSchV 1977/1984
        bzw. maximal eine Maßnahme an der Gebäudehülle im
        Flächenumfang von 50% des Bauteils oder der
        Anlagentechnik im Standard nach WSchV 1995.

        b) Gering modernisiert: An wesentlichen Bauteilen oder
        Komponenten wurden teilweise Modernisierungen
        durchgeführt, d.h. maximal zwei Maßnahmen an der
        Gebäudehülle und/oder der Anlagentechnik im Standard
        nach WSchV 1977/1984 bzw. maximal eine Maßnahme an
        der Gebäudehülle und/oder der Anlagentechnik im
        Standard nach WSchV 1995.

        c) Mittel/größtenteils modernisiert: An wesentlichen
        Bauteilen oder Komponenten wurden größtenteils
        Modernisierungen durchgeführt, d.h. mehr als zwei
        Maßnahmen an der Gebäudehülle und/oder der
        Anlagentechnik im Standard nach WSchV 1977/1984
        bzw. mehr als eine Maßnahme an der Gebäudehülle
        und/oder der Anlagentechnik im Standard nach
        WSchV 1995.

    """
    logger.info('Set heat demand from construction classification')

    if not pd.api.types.is_integer_dtype(df_in[col_construction_year]):
        raise ValueError("Years in column '{}' must be integers, not {}"
                         .format(col_construction_year,
                                 df_in[col_construction_year].dtype))

    df = load_src_data_arge_heat_demand()
    df = process_src_data_arge(
        df, col_construction_year, [col_building_type, col_refurbished_state],
        col_spec_total, aliases_SFH=aliases_SFH, aliases_MFH=aliases_MFH)
    # Convert from final energy to net energy demand:
    df = df * eta
    # Merging assigns the correct heat demand to each row in the DataFrame
    df_out = (pd.merge(df_in, df, how='left',
                       on=[col_building_type, col_refurbished_state,
                           col_construction_year])
              .set_index(df_in.index))

    if fillna_value is not None:
        df_out.fillna({col_spec_total: fillna_value}, inplace=True)

    elif df_out[col_spec_total].isna().any() and warnings == 'raise':
        n = df_out[col_spec_total].isna().value_counts().get(True, default=0)
        logger.warning("%s buildings did not receive a heat demand, because "
                       "they did not match the criteria of the data source. "
                       "You may use the argument 'fillna_value' to fill in a "
                       "specific heat demand, or ignore this warning.", n)

    if col_heated is not None:
        # Set heat demand of non-heated buildings to nan
        df_out.loc[df_out[col_heated] == False, col_spec_total] = np.nan

    df_out[col_total] = (df_out[col_spec_total] * df_out['a_N']
                         ).round(decimals)

    return df_out


def process_src_data_arge(
        df, idx_name, col_names, val_name, year_min=1800,
        year_max=2050, aliases_SFH=None, aliases_MFH=None):
    """Process data from source ARGE."""
    df = df.rename_axis(index=idx_name, columns=col_names)

    try:
        # Convert 2nd column index level to categorical (for sorting)
        df_idx = df.columns.to_frame(index=False)
        df_idx[col_names[1]] = pd.Categorical(
            df.columns.get_level_values(1),
            categories=df.columns.unique(level=1),
            ordered=True)
        df.columns = pd.MultiIndex.from_frame(df_idx)
    except IndexError:
        pass

    # Apply the aliases by making copies of the correct columns
    if aliases_SFH is not None:
        for alias in aliases_SFH:
            df = pd.concat([df, df[['SFH']].rename(columns={'SFH': alias})],
                           axis='columns')
    if aliases_MFH is not None:
        for alias in aliases_MFH:
            df = pd.concat([df, df[['MFH']].rename(columns={'MFH': alias})],
                           axis='columns')

    # Make year index continuous for easier merging
    df = df.reindex(index=list(range(year_min, year_max)), method='bfill')
    # Convert the DataFrame to a named series, to prepare for merging
    df = df.unstack().rename(val_name)
    return df


def load_src_data_arge_heat_demand():
    """Load head demand data from source ARGE.

    Unit: kWh/(m² * a) (annual energy per Area A_N defined in German EnEV)

    Walberg, Dietmar (2011): Wohnungsbau in Deutschland, pp. 48 & 53
    Year is inclusive: 1948 means 1918-1948, 1957 means 1949 - 1957
    """
    df = pd.DataFrame(
        index=[1917, 1948, 1957, 1968, 1978, 1987, 1993, 2001, 2008],
        data={
         ('SFH', 'not refurbished'):
         [226.6, 237.5, 235.2, 231.9, 213.5, 168.9, 148.5, 116, 91.8],
         ('SFH', 'slightly refurbished'):
         [197.1, 208.8, 209.8, 203.1, 187.6, 152.9, 135.7, 105.8, 84.9],
         ('SFH', 'mostly refurbished'):
         [167.4, 175.3, 175.4, 165.3, 154.7, 127.1, 111.1, None, None],
         ('MFH', 'not refurbished'):
         [189.4, 194.4, 193, 182.6, 171.2, 140.8, 126.3, 116.3, 96.3],
         ('MFH', 'slightly refurbished'):
         [163, 165.5, 163.1, 159.4, 148.5, 125.5, 115.7, 107.5, 90.1],
         ('MFH', 'mostly refurbished'):
         [139.8, 142.7, 141.3, 137.4, 131, 111.5, 103.6, None, None],
         },
        )
    return df


def load_src_data_arge_refurbishmend_probabilities():
    """Load occurence of refurbishment data from source ARGE.

    Walberg, Dietmar (2011): Wohnungsbau in Deutschland, pp. 48 & 53
    Year is inclusive: 1948 means 1918-1948, 1957 means 1949 - 1957
    """
    df = pd.DataFrame(
        index=[1917, 1948, 1957, 1968, 1978, 1987, 1993, 2001, 2008],
        data={
         ('SFH', 'not refurbished'):
         [0.03, 0.02, 0.03, 0.05, 0.11, 0.29, 0.75, 0.85, 0.95],
         ('SFH', 'slightly refurbished'):
         [0.64, 0.67, 0.73, 0.74, 0.74, 0.64, 0.20, 0.15, 0.05],
         ('SFH', 'mostly refurbished'):
         [0.33, 0.31, 0.24, 0.21, 0.15, 0.07, 0.05, 0, 0],
         ('MFH', 'not refurbished'):
         [0.02, 0.02, 0.03, 0.04, 0.10, 0.36, 0.72, 0.88, 0.97],
         ('MFH', 'slightly refurbished'):
         [0.61, 0.67, 0.64, 0.69, 0.74, 0.54, 0.21, 0.12, 0.03],
         ('MFH', 'mostly refurbished'):
         [0.37, 0.31, 0.33, 0.27, 0.16, 0.10, 0.07, 0, 0],
         },
        )
    return df


def set_heat_demand_for_new_buildings(
        df,
        col_heated='heated',
        col_spec_total='e_th_spec_total',
        col_total='e_th_total_kWh',
        col_building_type='building_type',
        col_construction_year='construction_year',
        fillna_value=None,
        decimals=2,
        ):
    """Set the total heat demand for new (modern) buildings.

    This function is designed for use after set_heat_demand_from_source_arge()
    which does not include buildings constructed after 2008
    """
    df_data = pd.DataFrame(
        index=[2009, 2030],
        data={
         'SFH': [70],
         'MFH': [65],
         'business': [40],
         'other-heated': [60],
         'other-heated-non-residential': [40],
         },
        )
    df_data = process_src_data_arge(
        df_data, col_construction_year, col_building_type,
        col_spec_total, year_min=2009)

    # Merging assings the correct heat demand to each row in the DataFrame
    df_tmp = df[[col_heated, col_building_type, col_construction_year, 'a_N']]
    df_tmp = (pd.merge(df_tmp, df_data, how='left',
                       on=[col_building_type, col_construction_year])
              .set_index(df.index))

    if col_heated is not None:
        # Set heat demand of non-heated buildings to zero
        df_tmp.loc[df_tmp[col_heated] == False, col_spec_total] = np.nan

    df_tmp[col_total] = (df_tmp[col_spec_total]*df_tmp['a_N']).round(decimals)

    # Modify df in place using non-NA values from df_tmp
    df.update(df_tmp)
    return df


def load_src_degree_days(col_try='try_code', col_degreedays='G20/15'):
    """Load the degree days ("Gradtagszahlen") for the 15 TRY-regions.

    The degree days "G20/15" in unit [K*d] for each of the 15 German regions,
    calculated from the test reference years (TRY) by Deutscher Wetterdienst
    (DWD), Version 2011.

    The degree days "G20/15" were calculated with fixed indoor reference
    of 20°C and heating temperature limit of 15°C according to
    VDI 3807-1, 2013.

    If the degree days of a given TRY region are divided by those found
    at Potsdam (TRY 4), this yields the "Klimakorrekturfaktor".
    """
    df = pd.DataFrame(data=[
        [1, 3401.5],  # Bremerhaven
        [2, 3604.1],  # Rostock-Warnemünde
        [3, 3678.3],  # Hamburg-Fuhlsbüttel
        [4, 3666.8],  # Potsdam
        [5, 3360.3],  # Essen
        [6, 4358.9],  # Bad Marienberg
        [7, 3733.7],  # Kassel
        [8, 4724.7],  # Braunlage
        [9, 3974.6],  # Chemnitz
        [10, 4440.0],  # Hof (Hof-Hohensaas)
        [11, 5855.8],  # Fichtelberg
        [12, 3131.9],  # Mannheim
        [13, 4034.4],  # Mühldorf-am-Inn
        [14, 4335.1],  # Stötten
        [15, 4463.9],  # Garmisch-Partenkirchen
        ],
        columns=[col_try, col_degreedays])
    df.set_index(col_try, inplace=True)
    return df


def set_heat_demand_from_source_B(
        df_in,
        col_spec_heat='E_th_spec_heat',
        col_heat='E_th_heat_kWh',
        col_construction_year='construction_year',
        ):
    """Get estimated annual heat demand for heating from source "B".

    Source "B":
    IBS Ingenieurbüro für Haustechnik Schreiner
    http://energieberatung.ibs-hlk.de/eb_begr.htm

    Alternative functions:
        - set_heat_demand_from_source_arge()
        - set_heat_demand_from_source_B()

    Using this data source yields much (up to factor 2, depending on the
    constrution year) higher energy demands than source "A".

    """
    df = pd.DataFrame(
            index=pd.Index(
                data=[1800, 1977, 1984, 1995, 2002],
                name='construction_year'),
            data={
             col_spec_heat:
             [(280+360)/2, (200+260)/2, (140+180)/2, (100+120)/2, (70+80)/2],
             })
    # Make year index continus for easier merging
    df = df.reindex(index=list(range(1800, 2050)), method='ffill')

    # Merging assings the correct heat demand to each row in the DataFrame
    df_out = (pd.merge(df_in, df, how='left', on=[col_construction_year])
              .set_index(df_in.index))

    df_out[col_heat] = df_out[col_spec_heat] * df_out['a_N']

    return df_out


def apply_climate_correction_factor(
        gdf, col_energy=['e_th_spec_total', 'e_th_total_kWh']):
    """Apply a climate correction factor to specific columns in a GeoDataFrame.

    To adjust for regional weather effects, correct the given columns
    with a climate correction factor derived from the degree days in
    the 15 TRY-regions of Germany (DWD 2011), depending on the geographic
    location of the objects in the GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame containing the energy consumption data.
    col_energy : list of str, optional
        The names of the columns in the GeoDataFrame that hold the energy
        consumption data (default is ['e_th_spec_total', 'e_th_total_kWh']).

    Returns
    -------
    GeoDataFrame
        A modified GeoDataFrame where the energy consumption columns have
        been adjusted for regional variations in energy consumption.

    """
    climate_factor = 'climate_factor'
    col_try = 'try_code'
    # Find the TRY region for each object in the GeoDataFrame
    gdf = find_TRY_regions(gdf, col_try=col_try, show_plot=False)
    # Load degree days values and calculate factor with reference Potsdam
    df_Kd = load_src_degree_days(col_degreedays=climate_factor)
    df_corr = df_Kd / df_Kd.loc[4]  # TRY-region 4 (Potsdam)

    # Create a temporary DataFrame that holds the factor for each row
    gdf_temp = pd.merge(left=gdf[[col_try]], right=df_corr,
                        how='left', on=col_try
                        ).set_index(gdf.index)
    # Multiply the given energy columns with the climate factor
    gdf[col_energy] = gdf[col_energy].mul(gdf_temp[climate_factor], axis=0)

    return gdf


def set_domestic_hot_water_from_DIN18599(
        df,
        col_heated='heated',
        col_DHW='e_th_DHW_kWh',
        col_spec_DHW='e_th_spec_DHW',
        col_building_osm='building_osm',
        decimals=2):
    """Set domestic hot water energy demand from DIN 18599.

    Residential:
    qw b = max[16,5 - (A_NGF,WE,m · 0,05); 8,5] kWh/(m2 ∙ a)

    DIN V 18599-10:2016, page 17

    Non-residential:
    Demand for hot water for non-residential buildings
    DIN V 18599-10:2016, page 30

    Categories assigned to OpenStreetMap building type keys.

    Alternative functions:
        - set_domestic_hot_water_from_DIN18599()
        - set_domestic_hot_water_from_values()

    """
    # DIN V 18599-10:2016, page 17
    mask = df[col_building_osm].isin(
        ['SFH', 'MFH', 'house', 'residential', 'detached',
         'semidetached_house', 'apartments'])
    df.loc[mask, col_spec_DHW] = ((16.5 - df['a_NRF'] * 0.05)
                                  .clip(lower=8.5)
                                  .round(decimals))

    # DIN V 18599-10:2016, page 30
    df_non_residential = pd.DataFrame(
        columns=[col_building_osm, col_spec_DHW],
        data=[  # Wh/(m² * d)
            ['civic', 30],  # Bürogebäude
            ['college', 130],  # Schule
            ['commercial', 10],  # Einzelhandel, Kaufhaus
            ['government', 30],  # Bürogebäude
            ['hospital', 400],  # Krankenhaus
            ['hotel', 350],  # Hotel mittel
            ['industrial', 90],  # Werkstatt, Industriebetrieb
            ['public', 0],
            ['retail', 10],  # Einzelhandel, Kaufhaus
            ['school', 130],  # Schule
            ['university', 130],  # Schule

            # Not in # DIN V 18599-10:2016
            ['warehouse', 0],  # Lagerhaus / Lagerhalle
            ],
        )
    # Convert Wh/(m² * d) to kWh/m²  (reference area: NGF)
    df_non_residential[col_spec_DHW] = df_non_residential[col_spec_DHW].mul(
        365/1000)

    # Assign the each building type in df its specific heat demand
    df_tmp = (df[[col_building_osm]]
              .merge(df_non_residential, on=col_building_osm, how='left')
              .set_index(df.index)  # preserve index for correct update
              )

    try:
        geometry_col = df.geometry.name  # Store name of geometry column
        crs = df.crs
    except AttributeError:
        geometry_col = None
        crs = None

    df.update(df_tmp, overwrite=False)  # Update function breaks GeoDataFrame

    if geometry_col is not None:
        df.set_geometry(geometry_col, inplace=True)  # Restore geometry column
        df.set_crs(crs, inplace=True)

    if col_heated is not None:
        # Set heat demand of non-heated buildings to zero
        df.loc[df[col_heated] == False, col_spec_DHW] = 0

    df[col_DHW] = (df[col_spec_DHW] * df['a_NRF']).round(decimals)

    return df


def set_domestic_hot_water_from_values(
        df, col_DHW='e_th_DHW_kWh', col_spec_DHW='e_th_spec_DHW',
        col_building_type='building_type', A_ref='a_NRF',
        E_th_spec_DHW_dict=dict({'SFH': 11, 'MFH': 15, 'business': 9,
                                 'other-heated-non-residential': 8})
        ):
    """Set a fixed specific domestic hot water heat per building type.

    This can be used as an alternative to the function
    set_domestic_hot_water_from_DIN18599(), to define custom DHW values.
    The example values of 11 kWh/m² for SFH and 15 kWh/m² for MFH are taken
    from the (outdated) DIN V 18599-10:2011, referring to the area NRF.

    Applies only to the 'building_type' options provided in the input
    dictionary E_th_spec_DHW_dict.
    Unit of values is kWh/(m² * a). Make sure the argument A_ref matches
    the reference area (A_NRF, a_N, etc.) of the given values.

    Alternative functions:
        - set_domestic_hot_water_from_DIN18599()
        - set_domestic_hot_water_from_values()
    """
    logger.info("Set fixed DHW energy demand")
    df_spec = pd.Series(E_th_spec_DHW_dict, name=col_spec_DHW)
    df = df.join(df_spec, on=col_building_type, how='left')
    df[col_DHW] = df[col_spec_DHW] * df[A_ref]
    return df


def separate_heating_and_DHW(
        df, col_total='e_th_total_kWh', col_heat='e_th_heat_kWh',
        col_DHW='e_th_DHW_kWh'):
    """Subtract domestic hot water from total heat to calculate space heat.

    Allows NaN values in either column and treats them as zero.
    """
    logger.info("Separate heating and DHW")
    df[col_heat] = df[col_total].sub(df[col_DHW], fill_value=0).clip(lower=0)
    df[col_total] = df[col_heat].add(df[col_DHW], fill_value=0)

    return df


def calculate_avg_level_height(
        gdf, col_height, col_levels=['building:levels', 'roof:levels'],
        col_level_height=None, height_min=2, kind='single_mean',
        group_col1=None, group_col2=None, show_plot=False, decimals=4):
    """Calculate level height from building height and number of levels.

    If col_level_height is given, store the height for each building in
    that column.

    height_min: Entries in col_height lower than this will be removed,
    because they are considered implausible

    kind (str):
        Given enough input data, can provide average height for groups
        of buildings. Then those grouped averages can be used to fill the
        missing number of levels.
        Choose kind='grouped_means' for this approach. Then also
        group_col1='osm_building' and group_col2='baujahr' need to be defined

    """
    if not isinstance(col_levels, list):
        col_levels = [col_levels]

    # Remove implausible building heights
    gdf.loc[gdf[col_height] < height_min, col_height] = np.nan

    _col_levels = [col for col in col_levels if col in gdf.columns]
    if len(_col_levels) > 0:
        levels = gdf[_col_levels].sum(axis='columns', min_count=1)
        levels.replace(0, np.nan, inplace=True)

        df_level_height = (gdf[col_height]
                           .div(levels)
                           .replace(np.inf, np.nan)
                           )
        if col_level_height:
            gdf[col_level_height] = df_level_height.round(decimals)
    else:
        df_level_height = pd.Series()

    if kind == 'single_mean':
        level_height = df_level_height.mean()

    elif kind == 'grouped_means':
        # There are two approaches here (fit1 and fit2):
        #    1) Group by building type and year first, then create a fit
        #       through that data. This does not take into account the count
        #       of each class, which could be used as a weight. I.e. the mean
        #       of a  class with 1000 entries should be more reliable that
        #       one with only 100 entries. However, this allows to simply
        #       drop classes below a certain count that is not deemed
        #       representative
        #    2) Do not group, but instead fit through all individual buildings
        #       with each building type. This will automatically included the
        #       weight described above. However, for building types with few
        #       buildings outliers can drastically influence the result
        #
        # I currently favor 1), because judging from the plots it creats more
        # 'stable' results. I have tested fitting degrees of 2 but return to 1
        # (linear fit) because others are too unreliable.
        group_cols = [group_col1, group_col2]
        # gdf[col_height].describe()
        # gdf[col_height].value_counts().sort_index().head(30)

        test = gdf.groupby(by=group_cols)[col_level_height].describe()
        # test.loc[test['count'] <= 10, 'mean'] = np.nan
        for building in test.index.unique(group_col1):
            if test.loc[building, 'mean'].count() == 0:
                test.at[building, 'mean'] = \
                    test.loc[building, 'mean'].fillna(test['mean'].mean())

            elif test.loc[building, 'mean'].count() == 1:
                test.at[building, 'mean'] = \
                    test.loc[building, 'mean'].fillna(
                        test.loc[building, 'mean'].mean())

            if len(test.loc[building, 'mean']) == 1:
                continue

            df_fit1 = test.loc[building, 'mean'].dropna()
            df_fit2 = gdf.loc[gdf[group_col1] == building,
                              [group_col2, col_level_height]
                              ].set_index(group_col2)
            df_fit2 = df_fit2[col_level_height].sort_index()
            if df_fit2.count() == 0:
                df_fit2.fillna(test['mean'].mean(), inplace=True)
            df_fit2.dropna(inplace=True)

            # print(building)
            # print(test.loc[building, ['mean', 'count']])

            p1 = np.polyfit(df_fit1.index.astype(int), df_fit1, 1)
            p2 = np.polyfit(df_fit2.index.astype(int), df_fit2, 1)
            f1 = np.poly1d(p1)
            f2 = np.poly1d(p2)
            test.loc[building, 'fit1'] = f1(test.loc[building, 'mean'].index)
            test.loc[building, 'fit2'] = f2(test.loc[building, 'mean'].index)
            test['fit1'] = test['fit1'].clip(lower=height_min)
            test['fit2'] = test['fit2'].clip(lower=height_min)

            if show_plot:
                fig, ax = plt.subplots()
                test.loc[building].reset_index().plot.scatter(
                    ax=ax, x=group_col2, y='mean', label=building,
                    s='count', c='count', xlabel=group_col2)
                test.loc[building, 'fit1'].plot(ax=ax, label='fit1',
                                                c='tab:red')
                test.loc[building, 'fit2'].plot(ax=ax, label='fit2',
                                                c='tab:orange')
                ax.set_ylim(bottom=0,
                            # top=top
                            )
                plt.legend()
                plt.show()

        # Create another column 'mix' where the mean is replaced with the
        # fit function if the count is too low.
        test['mix'] = test['mean']
        test.loc[(test['count'] < 100) & test['fit1'].notna(),
                 'mix'] = test['fit1']

        # We can now choose the mean per group or one of the fit functions
        # as the input for all buildings with undefined level height
        # method_select = 'mean'
        # method_select = 'fit1'
        # method_select = 'fit2'
        method_select = 'mix'

        if test.loc[test['count'] > 0, method_select].isna().any():
            raise ValueError("Some building groups do not have a mean level "
                             "height assigned. Check the calculation.")

        # After all this, test[method_select] contains the mean level height
        # for each building type and year class. Now the NaN values in
        # gdf[col_level_height] need to be filled correctly
        test.rename(columns={method_select: col_level_height}, inplace=True)
        df_tmp = pd.merge(gdf[group_cols],
                          test[col_level_height], how='left',
                          on=group_cols).set_index(gdf.index)
        # Some buildings will still be NaN, e.g. those with building type None
        # Or those that only occur a single time (where no polyfit is possible)
        # Fill them with the global mean
        df_tmp.fillna({col_level_height: df_level_height.mean()}, inplace=True)

        # Fill the missing values in the input df with the group means
        gdf.fillna({col_level_height: df_tmp[col_level_height]}, inplace=True)
        gdf[col_level_height] = gdf[col_level_height].round(decimals)

        # Set the return value to None as a signal that instead the
        # column col_level_height was filled
        level_height = None

    return level_height


def calculate_levels_from_height(
        gdf, col_height, col_levels='building:levels', level_height=None,
        col_level_height=None, upper=None, lower=1, decimals=1):
    """Calculate the number of levels per building from the height.

    Given the height of a building in column 'col_height' and the
    height per level 'level_height', store the number of levels in
    column 'col_levels'. Existing values are not overwritten.

    Instead of a global height per level ``level_height``, a column name
    ``col_level_height`` with the value for each building can be given.

    level_height (float): The average height of each level in a building.
    E.g. 3.5m.

    limit (int): Set a maximum limit for the number of levels
    decimals (int): Number of decimal places to round the result to

    """
    if level_height is None and col_level_height is None:
        raise ValueError("One of level_height and col_level_height "
                         "must be defined")
    if level_height is not None and col_level_height is not None:
        raise ValueError("Only one of level_height and col_level_height "
                         "must be defined")
    if col_level_height is not None:
        level_height = gdf[col_level_height]

    gdf.loc[gdf[col_levels].isna(), col_levels] = (
        gdf[col_height]
        .div(level_height)  # Can be single value or a whole column
        .clip(lower=lower, upper=upper)
        .round(decimals))
    # print(gdf[col_levels].describe())
    return gdf


def calculate_building_areas(
        gdf, col_levels=['building:levels', 'roof:levels'], levels_default=1.5,
        area_list=['a_N', 'a_WFL', 'a_NRF'], decimals=2,
        col_building_type='building_type', **kwargs):
    """Calculacte the desired areas for each building in the GeoDataFrame.

    In OpenStreetMap, the total number of levels are defined by three keys:
        - building:levels
        - roof:levels
        - building:levels:underground (ignored here)

    https://wiki.openstreetmap.org/wiki/Key:building:levels

    Use the ground area of the building polygon and the number of levels
    to calculate the BGF of the building. Then use convert_building_area()
    to convert to the other desired areas.

    If col_levels is a list of column names, use their sum to calculate
    the total number of levels. Otherwise, col_levels is assumed to be
    a single column name that has the total number of floors. If it does
    not exist, the default number of levels will be stored in that column.

    TODO: Maybe count roof:levels only with a factor like 0.5
    """
    if not isinstance(col_levels, list):
        col_levels = [col_levels]

    _col_levels = [col for col in col_levels if col in gdf.columns]
    if len(_col_levels) > 0:
        try:
            make_columns_numeric(gdf, _col_levels, errors='raise')
        except ValueError as e:
            logger.warning(f"Parsing the column(s) {_col_levels} caused "
                           f"error (which is ignored): {e}")
            make_columns_numeric(gdf, _col_levels, errors='coerce')
        gdf.fillna({col_levels[0]: levels_default}, inplace=True)
        levels = gdf[_col_levels].sum(axis='columns', min_count=1)

    else:  # None of col_levels are in gdf.columns
        logger.warning("Column '%s' (number of levels) not in DataFrame. "
                       "Setting number of levels to %s for each building",
                       col_levels[0], levels_default)
        gdf[col_levels[0]] = levels_default
        levels = gdf[col_levels[0]]

    # Calculate Bruttogrundfläche from polygon area and number of floors
    # Use a minimum number of 1 levels for each building
    gdf['a_BGF'] = (gdf.area * levels.clip(lower=1)).round(decimals)

    for A in area_list:
        gdf = convert_building_area(gdf, A_input='a_BGF', A_output=A, **kwargs)

    return gdf


def calculate_heat_demand(
        buildings, col_heated='heated'):
    """Calculate total heat demands from specific demand and area.

    This function is deprecated.

    Needs a colum 'col_floors' with number of floors to multiply with
    ground area for total area. If column is missing, a default number of
    floors for all buildings is used instead.
    """
    logger.info('Calculate heat demand')

    buildings.loc[buildings[col_heated] == True, 'E_th_heat_kWh'] = (
        buildings['E_th_spec_heat'].mul(buildings['a_N']))
    buildings.loc[buildings[col_heated] == True, 'e_th_DHW_kWh'] = (
        buildings['E_th_spec_DHW'].mul(buildings['a_N']))

    buildings.fillna({'E_th_heat_kWh': 0,
                      'e_th_DHW_kWh': 0},
                      inplace=True)

    buildings['E_th_total_kWh'] = buildings[
        ['E_th_heat_kWh', 'e_th_DHW_kWh']].sum('columns')

    return buildings


def guess_thermal_power_from_full_load_hours(
        gdf, flh=2000, decimals=2,
        col_total='e_th_total_kWh',
        col_flh='vbh_th_guess',
        col_p_th='p_th_guess_kW'):
    """Guess the thermal power of the buildings from given full load hours.

    Rule of thumb for full load hours
        - 1200 to 1800 h: Efficient
        - 800 to 1200 h: Avarage
        - below 800 h: Inefficient
    """
    gdf.loc[gdf[col_total] > 0, col_flh] = flh
    gdf[col_p_th] = (gdf[col_total] / gdf[col_flh]).round(decimals)
    return gdf


def set_n_persons_and_flats(
        gdf, area_per_flat=80, persons_per_flat=2.5,
        col_building_type='building_type',
        col_N_pers='N_pers', col_N_flats='N_flats'):
    """Set number of persons and flats per building.

    Do not overwrite existing values in the columns col_N_pers and col_N_flats.
    """
    for col in [col_N_flats, col_N_pers]:
        if col not in gdf.columns:
            gdf[col] = np.nan

    mask_s = gdf[col_building_type] == 'SFH'
    mask_m = gdf[col_building_type] == 'MFH'
    mask_na = gdf[col_N_flats].isna()
    gdf.loc[mask_s & mask_na, col_N_flats] = 1
    gdf.loc[mask_m & mask_na, col_N_flats] = (gdf.loc[mask_m, 'a_WFL']
                                              / area_per_flat).round(0)

    mask_na = gdf[col_N_pers].isna()
    gdf.loc[mask_na, col_N_pers] = gdf[col_N_flats] * persons_per_flat

    return gdf


def make_columns_numeric(df, columns=None, downcast='integer',
                         errors='ignore'):
    """Make as many of the columns as possible numeric with pd.to_numeric."""
    if columns is None:
        columns = df.columns
    df[columns] = df[columns].apply(pd.to_numeric, downcast=downcast,
                                    errors=errors)
    # print(df.dtypes)
    return df


def log_statistics(
        buildings,
        col_heated='heated',
        col_total='e_th_total_kWh',
        col_refurbished_state='refurbished_state',
        col_construction_year='construction_year',
        col_p_th='p_th_guess_kW',
        show_plot=True):
    """Print and plot some statistical data."""
    mask = (buildings[col_heated] == True)
    n_buildings = mask.value_counts().get(True, default=0)
    E_th_total = buildings.loc[mask, col_total].sum()
    A_BGF = buildings.loc[mask, 'a_BGF'].sum().round(2)
    A_N = buildings.loc[mask, 'a_N'].sum().round(2)

    logger.info('Total heat demand (space heating and domestic hot water) '
                'for %s heated buildings: %s GWh (%s MWh/building)',
                n_buildings, round(E_th_total/1000000, 2),
                round(E_th_total / (1000 * n_buildings), 2),
                )

    logger.info('Sum of area in heated buildings: %s m² BGF, %s m² A_N',
                A_BGF, A_N)
    # breakpoint()
    logger.info('Mean of specific heat demand building area: '
                '%s kWh/m² BGF, %s kWh/m² A_N',
                round(E_th_total / A_BGF, 2),
                round(E_th_total / A_N, 2),
                )

    if col_p_th is not None:
        P_th_W = buildings.loc[mask, col_p_th].sum() * 1000
        logger.info('Heating power relative to building area: '
                    '%s W/m² BGF, %s W/m² A_N',
                    round(P_th_W / A_BGF, 2),
                    round(P_th_W / A_N, 2),
                    )

    if show_plot:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8.27, 11.69))
        fig.suptitle("Häufigkeitsverteilung von Baualtersklasse "
                     "und Sanierungsgrad")
        for i, col in enumerate([col_construction_year,
                                 col_refurbished_state]):
            buildings.loc[mask, col].hist(ax=axs[i],
                                          bins=50,
                                          rwidth=0.9)
            axs[i].set_ylabel('Gebäudeanzahl')
            plt.grid(False)
        fig.tight_layout()
        plt.show()


def convert_building_area(
        buildings, A_input='a_BGF', A_output='a_N',
        decimals=2, col_building_type='building_type',
        aliases_SFH=[], aliases_MFH=[], aliases_business=[],
        aliases_unknown=[]):
    """Convert various area types of buildings.

    Use the existing column A_input to calculate and add the column A_output.

    Grundlage der nachstehenden Flächenumrechnungsfaktoren sind die
    Grundflächen und sonstigen Planungskennwerte von über 300 dokumentierten
    Wohn- und Nichtwohngebäuden. Sie wurden vom Baukosteninformationszentrum
    Deutscher Architektenkammern (BKI), Stuttgart, im Band "BKI Baukosten
    2017 Neubau Teil 1 - Statistische Kostenkennwerte für Gebäude"
    zusammengestellt.

    Die Nutzfläche nach EnEV (AN) wird dabei vereinfacht nach dem Ansatz
    der DIN V 18599 wie folgt berechnet.
    A_N = Wohnfläche · 1,35	(gilt für EFH mit beheiztem Keller)
    A_N = Wohnfläche · 1,20	(gilt für EFH ohne Keller und MFH)

    Returns
    -------
    None.

    """
    data = """building_type,,SFH,SFH,MFH,MFH,MFH,business
subtype,,w cellar,w/o cellar,<=6 WE,7-19 WE,>=20 WE,business
A,A_full,,,,,,
a_BGF,Bruttogrundfläche,       1.00000,1.00000,1.00000,1.00000,1.00000,1.00000
a_KGF,Konstruktionsgrundfläche,0.20825,0.21540,0.16667,0.16183,0.16858,0.15220
a_NRF,Nettoraumfläche,         0.79175,0.78393,0.83333,0.83748,0.83142,0.84780
a_NUF,Nutzungsfläche,          0.65488,0.67522,0.66401,0.69156,0.67705,0.64767
a_TF,Technikfläche,            0.02947,0.02363,0.02125,0.01660,0.01219,0.03433
a_VF,Verkehrsfläche,           0.10740,0.08575,0.14807,0.12932,0.14218,0.16580
a_WFL,Wohnfläche,              0.53242,0.62520,0.53549,0.54028,0.52081,
a_BRI,Bruttorauminhalt,        3.06483,3.18028,2.86853,2.88382,3.01286,3.63990
a_N,Nutzfläche nach EnEV,      0.71877,0.75024,0.64259,0.64834,0.62497,0.89437
"""
    # Read the string table into DataFrame, then drop the description index
    df_ratio = pd.read_csv(io.StringIO(data), index_col=[0, 1], header=[0, 1])
    df_ratio = df_ratio.droplevel('A_full', axis='index')

    # Include an "unknown" category, using the mean for each area type
    df_ratio[('unknown', 'unknown')] = df_ratio.mean(axis='columns')

    # Create a temporary DataFrame from the input
    gdf_tmp = buildings[[col_building_type]].copy()
    gdf_tmp.fillna({col_building_type: 'unknown'}, inplace=True)
    # Set the subtype for each building. Treat the aliases as if they
    # belonged to the original categories
    for alias in ['SFH'] + aliases_SFH:
        gdf_tmp.loc[gdf_tmp[col_building_type] == alias, 'subtype'
                    ] = 'w/o cellar'
        gdf_tmp.replace({col_building_type: {alias: 'SFH'}}, inplace=True)
    for alias in ['MFH'] + aliases_MFH:
        gdf_tmp.loc[gdf_tmp[col_building_type] == 'MFH', 'subtype'
                    ] = '7-19 WE'
        gdf_tmp.replace({col_building_type: {alias: 'MFH'}}, inplace=True)
    for alias in ['business'] + aliases_business:
        gdf_tmp.loc[gdf_tmp[col_building_type] == alias, 'subtype'
                    ] = 'business'
        gdf_tmp.replace({col_building_type: {alias: 'business'}}, inplace=True)
    for alias in ['unknown'] + aliases_unknown:
        gdf_tmp.loc[gdf_tmp[col_building_type] == alias, 'subtype'
                    ] = 'unknown'
        gdf_tmp.replace({col_building_type: {alias: 'unknown'}}, inplace=True)

    # Merging these DataFrames produces columns with the correct ratio for
    # each building
    gdf_tmp = (pd.merge(gdf_tmp, df_ratio.T, how='left',
                        on=[col_building_type, 'subtype'],)
               .set_index(gdf_tmp.index))

    # All ratios are in relation to A_BGF
    buildings[A_output] = (buildings[A_input]
                           .div(gdf_tmp[A_input])  # Yields A_BGF
                           .mul(gdf_tmp[A_output])  # Yields A_output
                           .round(decimals)
                           )

    # Special case: Businesses have no living area, which needs to be set to 0
    if A_output == 'a_WFL':
        mask = buildings[col_building_type].isin(['business']
                                                 + aliases_business)
        buildings.loc[mask, 'a_WFL'] = buildings.loc[mask, 'a_WFL'].fillna(0)

    if buildings[A_output].isna().any():
        logger.warning("Not all buildings received an area")
        breakpoint()

    return buildings


def apply_adoption_rate(gdf, adoption_rate=0.5,
                        col_candidates='DISTRICT_HEATING'):
    """Apply an adoption rate to a GeoDataFrame of buildings.

    We may not want to supply all given buildings with heat, to simulate
    a low adoption rate among the building owners.

    Args:
        gdf (GeoDataFrame): A building GeoDataFrame

        adoption_rate (float): The ratio of buildings that will be chosen
        among the valid candidates

        col_candidates (str): Name of the column that marks candidates
        for the adoption rate (by being True in this column).

    Returns:
        gdf (GeoDataFrame): A GeoDataFrame with 'col_candidates' modified
        or added if it did not exist

    """
    logger.info("Apply %s %% adoption rate to buildings", adoption_rate*100)
    if col_candidates not in gdf.columns:
        gdf[col_candidates] = True

    # To prepare for this, all available candiates are sorted to the top
    n_candidates = gdf[col_candidates].value_counts().get(True, default=0)
    if n_candidates == 0:
        raise ValueError(f"No valid candidates in column {col_candidates}")

    gdf.sort_values(by=col_candidates, ascending=False, inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    # ... then a list of random choices (indices) is generated and applied
    # (equals a fraction of total buildings to connect to DHN):
    ids_DH = np.random.choice(n_candidates,
                              size=int(adoption_rate*n_candidates),
                              replace=False)
    # Now set all candidates to False, before setting the chosen ones to True
    gdf[col_candidates] = False
    gdf.loc[ids_DH, col_candidates] = True
    return gdf


# Section "Geographic tools"

def check_crs(gdf, crs="EPSG:4647"):
    """Convert CRS to EPSG:4647 - ETRS89 / UTM zone 32N (zE-N).

    This is the (only?) Coordinate Reference System that gives the correct
    results for distance calculations.
    """
    if gdf.crs != crs:
        gdf.to_crs(crs=crs, inplace=True)
        logging.info('CRS of GeoDataFrame converted to {0}'.format(crs))

    return gdf


def merge_xyz_dtm(file_list, output_file=None, show_plot=False):
    """Merge given 'digital terrain/elevation/surface model' files into one.

    The DTM/DEM/DSM files must be in the xyz format.
    """
    df = pd.concat(
        [pd.read_csv(file, delim_whitespace=True) for file in file_list]
        )

    # Sorting is essential to make a valid xyz file
    df.sort_values(['y', 'x'], inplace=True)
    df.reset_index(inplace=True)

    if output_file is not None:
        df.to_csv(output_file, sep=' ', index=False)

    if show_plot:
        df_plot = (df.set_index(['x', 'y'])
                   .unstack('x')
                   .droplevel(0, axis='columns')
                   .sort_index(ascending=False)
                   )
        plt.matshow(df_plot)

    return df


def join_area_weighted(gdf_target, gdf_source, columns=None):
    """Join columns from gdf_source with gdf_target where both intersect.

    This is just a wrapper around tobler.area_weighted.area_join()
    https://pysal.org/tobler/generated/tobler.area_weighted.area_join.html

    This is somewhat similar to GeoDataFrame.sjoin(), but that one will
    create duplicate geometries, if there are multiple matches.
    Sometimes that may be desired.

    Args:
        gdf_target (gdf): Target GeoDataFrame

        gdf_source (gdf): Source GeoDataFrame

        columns (list): columns in parcels dataframe for variables to be
        joined. If None, use all available columns. May result in
        duplicate columns errors.

    Returns:
        buildings (gdf): GeoDataFrame with joined variables as new columns

    """
    logger.info('Spatial weighted area join of two GeoDataFrames...')

    if columns is None:
        geom = gdf_source.geometry.name
        columns = gdf_source.drop(columns=[geom]).columns

    gdf_target = tobler.area_weighted.area_join(
        gdf_source.to_crs(gdf_target.crs), gdf_target, columns)
    return gdf_target


def join_area_interpolate(gdf_target, gdf_source,
                          intensive_variables=None,
                          extensive_variables=None,
                          categorical_variables=None,
                          n_jobs=1,
                          ):
    """Join columns from gdf_source with gdf_target via area interpolation.

    This is just a wrapper around tobler.area_weighted.area_interpolate()
    https://pysal.org/tobler/generated/tobler.area_weighted.area_interpolate.html

    It can be used to e.g. sum up the heat demands of all buildings in a
    district and store the sum as an attribute in the district GeoDataFrame.
    Tobler finds the buildings from gdf_source that lie within the district
    polygons of gdf_target.

    In addition to tobler, the index and columns from gdf_target are included.

    Args:
        gdf_target (gdf): Target GeoDataFrame

        gdf_source (gdf): Source GeoDataFrame

        intensive_variables : list, optional
            Columns in DataFrame for intensive variables. An average is calculated.
        extensive_variables : TYPE, optional
            Columns in DataFrame for extensive variables. These will be summed up.

    Returns:
        gdf_target (gdf): GeoDataFrame with joined variables as new columns

    """
    logger.info('Spatial weighted area interpolation of two GeoDataFrames...')

    gdf_target2 = tobler.area_weighted.area_interpolate(
        source_df=gdf_source.to_crs(gdf_target.crs),
        target_df=gdf_target,
        intensive_variables=intensive_variables,
        extensive_variables=extensive_variables,
        categorical_variables=categorical_variables,
        allocate_total=True,
        n_jobs=n_jobs,
        )
    # Keep index name
    gdf_target2.index.set_names(gdf_target.index.name, inplace=True)
    # Keep columns from original gdf_target
    gdf_target = gdf_target2.join(gdf_target.drop(
        columns=[gdf_target.geometry.name]))

    return gdf_target


def combine_alkis_and_osm_buildings(
        gdf_alkis,
        gdf_osm,
        columns=None,
        address_col=None):
    """Add the columns from gdf_osm to gdf_alkis where buildings intersect.

    This is useful for copying data from osm to an alkis dataset.

    columns (list) :
        List of columns to transfer from gdf_osm to gdf_alkis. E.g.:
        ['addr:street', 'addr:housenumber', 'building:levels', 'roof:levels']
    """
    gdf_alkis = join_area_weighted(gdf_alkis, gdf_osm, columns)

    if address_col is not None:
        if ('addr:street' in gdf_alkis.columns
           and 'addr:housenumber' in gdf_alkis.columns):
            gdf_alkis[address_col] = \
                gdf_alkis['addr:street'] + ' ' + gdf_alkis['addr:housenumber']

    return gdf_alkis


def combine_buildings_and_parcels(
        buildings, parcels, columns=None):
    """Store information from parcels in buildings within those parcels.

    Join values in columns from parcels based on the largest intersection with
    buildings. In case of a tie, pick the first one.

    Most useful to transfer e.g. the parcel identification text from the
    parcels to the buildings within those parcels.

    This is just a wrapper around tobler.area_weighted.area_join()
    https://pysal.org/tobler/generated/tobler.area_weighted.area_join.html

    This is somewhat similar to GeoDataFrame.sjoin(), but that one will
    create duplicate geometries, if there are multiple matches.
    Sometimes that may be desired.

    Args:
        buildings (gdf): Target GeoDataFrame

        parcels (gdf): Source GeoDataFrame

        columns (list): columns in parcels dataframe for variables to be
        joined. If None, use all available columns. May result in
        duplicate columns errors.

    Returns:
        buildings (gdf): GeoDataFrame with joined variables as new columns

    """
    buildings = join_area_weighted(buildings, parcels, columns=columns)
    return buildings


def make_geographic_selection(
        buildings, gdf_selection, col_candidates=None, show_plot=False,
        drop=False, warn_empty=True):
    """Make a geographic selection of buildings.

    In column 'col_candidates' of GeoDataFrame 'buildings', only those within
    GeoDataFrame 'gdf_selection' and previously True will remain True,
    all other buildings will be set to False.
    """
    logger.info('Make geographic selection')
    if col_candidates is not None:
        if col_candidates not in buildings.columns:
            buildings[col_candidates] = True

    # Length of GeoDataFrame needs to be reduced to 1
    if len(gdf_selection) > 1:
        gdf_selection = gpd.GeoDataFrame(
            geometry=[gdf_selection.union_all()],
            crs=gdf_selection.crs)

    # "Within" only works if both gdf share the same coordinate reference
    gdf_selection.to_crs(crs=buildings.crs, inplace=True)
    mask1 = buildings.within(gdf_selection.geometry.iloc[0])
    n_dropped = mask1.value_counts().get(False, default=0)
    n_remain = mask1.value_counts().get(True, default=0)
    logger.info('Buildings discarded by area selection: %s', n_dropped)
    if n_remain == 0 and warn_empty:
        logger.warning('No buildings left after area selection!')

    if drop:  # Actually remove all buildings outside of selection area
        buildings = buildings.loc[mask1].copy()

    if col_candidates is not None:
        # Keep previous selection of candidates intact
        mask2 = buildings[col_candidates].isin([True])
        # Mark the buildings available for district heating
        buildings[col_candidates] = False
        buildings.loc[mask1 & mask2, col_candidates] = True

    if show_plot:
        _, ax = plt.subplots(figsize=(20, 10), dpi=300)
        gdf_selection.plot(ax=ax, color='green')
        buildings.plot(ax=ax, column=col_candidates)
        plt.show()
    return buildings


def dissolve_sort(df):
    """Define custom aggregation function with special sorting."""
    df = df.sort_values(by=['heated', 'area'], ascending=False)
    return df.iloc[0]


def merge_all_in_parcel(
        buildings, parcel_text='LAGEBEZTXT', col_heated='heated',
        address_empty=""):
    """Merge all the buildings within a parcel.

    This does not check if the buildings touch. Should not be used, because
    seperate buildings are more likely to be garden sheds, garages, etc.

    Use merge_touching_buildings_in_parcels() instead.
    """
    # Often, only the front building in a parcel has an assigned address,
    # but buildings in the back could be counted into the heated area
    # as well. Assign them the LAGEBEZTXT of the parcel they are in
    mask1 = buildings[parcel_text] == address_empty
    mask2 = buildings['FUNKTION'] == 'Wohngebäude'
    buildings.loc[mask1 & mask2, parcel_text] = (
        buildings.loc[mask1 & mask2, 'LAGEBEZTXT_parcel'])

    # After assigning the same addresses (LAGEBEZTXT) to multiple buildings,
    # now merge these: https://geopandas.org/aggregation_with_dissolve.html
    buildings['area'] = buildings.area  # store area for use in aggfunc_sort()
    buildings_merged = buildings.dissolve(by=parcel_text,
                                          sort=False,  # better performance?
                                          aggfunc=dissolve_sort,
                                          as_index=False)
    # My custom aggfunc dissolve_sort() works, but makes all columns dtype
    # object. This causes problems later on, so we have to reset the dtypes
    buildings_merged[col_heated] = buildings_merged[col_heated].astype('bool')
    buildings_merged.drop('area', axis='columns', inplace=True)

    # However, this merges all objects without address into one, which we
    # need to get rid off
    drop = buildings_merged[buildings_merged[parcel_text] == address_empty]
    buildings_merged.drop(index=drop.index.tolist(), inplace=True)

    # Discard all buildings that do not have an address
    mask = buildings[parcel_text] == address_empty
    buildings.loc[mask, col_heated] = False
    # Append these unused buildings to the merged ones
    buildings = pd.concat([buildings_merged, buildings[mask]])

    return buildings


def merge_all_touching_buildings(buildings, sort_columns=['heated']):
    """Merge all touching polygons.

    If you only want to merge buildings within each parcel, use
    merge_touching_buildings_in_parcels() instead.
    """
    buildings['area'] = buildings.area  # store area for use in sort_values()

    # When merging multiple polygons, we want to keep the information of the
    # polygon with the largest area, which is also heated.
    # To achieve that, we sort the buildings now, and use the aggfunc 'first'
    # when calling dissolve()
    buildings = buildings.sort_values(by=sort_columns + ['area'],
                                      ascending=False)

    weights = libpysal.weights.Queen.from_dataframe(buildings,
                                                    silence_warnings=True,
                                                    )
    buildings_merged = buildings.dissolve(by=weights.component_labels,
                                          sort=False,  # better performance?
                                          aggfunc='first',
                                          # aggfunc=dissolve_sort,
                                          as_index=False)

    buildings_merged.drop('area', axis='columns', inplace=True)
    return buildings_merged


def merge_touching_buildings_in_parcels_slow(buildings, col_heated='heated'):
    """Merge all buildings that touch, but only if they share a parcel.

    Very slow for large number of buildings, due to iteration.
    merge_touching_buildings_in_parcels() yields the same result, much faster.
    """
    logger.info('Merging...')
    buildings['area'] = buildings.area  # store area for use in aggfunc_sort()
    buildings_merged = gpd.GeoDataFrame(geometry=[], crs=buildings.crs)

    for name, parcel_buildings in buildings.groupby(by='LAGEBEZTXT_parcel'):
        # create spatial weights matrix
        weights = libpysal.weights.Queen.from_dataframe(parcel_buildings,
                                                        silence_warnings=True,
                                                        )
        buildings_new = parcel_buildings.dissolve(by=weights.component_labels,
                                                  sort=False,
                                                  aggfunc=dissolve_sort,
                                                  as_index=False)
        buildings_merged = pd.concat([buildings_merged, buildings_new])

    # My custom aggfunc dissolve_sort() works, but makes all columns dtype
    # object. This causes problems later on, so we have to reset the dtypes
    buildings_merged[col_heated] = buildings_merged[col_heated].astype('bool')
    buildings_merged.drop('area', axis='columns', inplace=True)
    logger.info('Merging done.')

    return buildings_merged


def merge_touching_buildings_in_parcels(
        buildings, parcel_text='LAGEBEZTXT', sort_columns=[],
        parcel_text_alt=None, address_empty="", mask_skip=None,
        predicate='intersects', silence_warnings=True,
        **kwargs_contiguity):
    """Merge all buildings that touch, but only if they share a parcel.

    All buildings must have a column 'parcel_text' defined. They share a
    parcel if the value in this column is identical. To assign such a column,
    you may need to run 'combine_buildings_and_parcels()' first.

    Args:
        buildings (gdf): A GeoDataFrame of the buildings

        parcel_text (str): Column name in 'buildings' of e.g. unique parcel
        numbers that define if buildings share a parcel

        sort_columns (list): List of column names that are supposed to affect
        the sorting of the merged buildings, in addition to the building area.
        Only the information of the first building (after sorting) with the
        same parcel_text are kept.

        parcel_text_alt (str): Alternative column name to use for the parcel
        text definition, where the column 'parcel_text' is equal to the
        argument 'address_empty'

        address_empty (str): String to determine when to use 'parcel_text_alt'

        mask_skip (index):
            An index mask as e.g. returned by ``buildings['HEATED'] == False``.
            Where this mask is True, the objects are not merged with their
            neighbours. This allows to skip specific buildings.

        predicate (string):
            The predicate to use for determination of neighbors.
            Options: 'intersects', 'within', 'contains', 'overlaps',
            'crosses', 'touches'. Default is ‘intersects’.
            (See libpysal.weights.fuzzy_contiguity())

        silence_warnings (boolean):
            Silence libpysal warnings. Default is True.

        kwargs_contiguity:
            Keyword arguments passed to libpysal.weights.fuzzy_contiguity()


    Returns:
        buildings (gdf): GeoDataFrame with merged buildings

    Employs the libpysal library to make computation very fast.
    """
    logger.info('Merge touching buildings in parcels')

    parcel_text_tmp = 'LAGEBEZTXT_tmp'  # Column is added and later deleted

    # Create a new temporary column for the parcel text. If 'parcel_text' is
    # empty, use the text from the surrounding parcel, which should have been
    # stored in the separate column 'parcel_text_alt'.
    # This allows to match building parts that belong together
    buildings[parcel_text_tmp] = buildings[parcel_text]
    if parcel_text_alt is not None:
        mask = buildings[parcel_text_tmp] == address_empty
        buildings.loc[mask, parcel_text_tmp] = (
            buildings.loc[mask, parcel_text_alt])

    if mask_skip is not None:
        # The rows defined by this index mask are not to be merged
        # Giving each a unique string ensures they are never merged
        buildings.loc[mask_skip, parcel_text_tmp] = (
            'unique_' + (buildings.loc[mask_skip].reset_index()
                         .index.astype(str)))

    # Store area for use in aggfunc_sort()
    buildings['area'] = buildings.area

    # When merging multiple polygons, we want to keep the information of the
    # polygon with the largest area, which is also heated.
    # To achieve that, we sort the buildings now, and use the aggfunc 'first'
    # when calling dissolve()
    buildings = (
        buildings
        .sort_values(by=[parcel_text]+sort_columns+['area'], ascending=False)
        .reset_index(drop=True))

    # Use libpysal magic
    # 1: Create "weights" from the buildings with the same parcel text
    regimes = buildings[parcel_text_tmp]
    w_block = libpysal.weights.block_weights(
        regimes, silence_warnings=silence_warnings)
    # 2a: Create "weights" from buildings that touch each other
    # w_contiguity = libpysal.weights.Queen.from_dataframe(
    #     buildings, use_index=False, silence_warnings=silence_warnings)
    # 2b: Create "weights" from buildings that intersect
    w_contiguity = libpysal.weights.fuzzy_contiguity(
        buildings, predicate=predicate, silence_warnings=silence_warnings,
        **kwargs_contiguity)
    # 3: Get the intersetions of all buildings that touch in the same parcel
    weights = libpysal.weights.w_intersection(
        w_contiguity, w_block, silence_warnings=silence_warnings)

    # Merge those selected buildings
    buildings_merged = buildings.dissolve(by=weights.component_labels,
                                          sort=False,  # better performance?
                                          aggfunc='first',
                                          as_index=True)

    # Drop unused columns:
    buildings_merged.drop(columns=[parcel_text_tmp, 'area'], inplace=True)
    return buildings_merged


def merge_with_test(df1, df2, on, find_closest_matches=False):
    """Merge df2 into df1 with pd.merge() while testing for missing matches.

    Optionally find the closest matches for the missing matches.

    .. Note ::

        Returns a tuple ``(df1, df_missing)``!

    If inputs are GeoDataFrames, the geometry of df1 is kept, while geometry
    of df2 is dropped.

    Parameters
    ----------
    df1 : DataFrame
        The DataFrame that will be returned, with new colums from df2.
        Represents ``left`` in
        ``pd.merge(left=df1, right=df2, on=on, how='left')``.
    df2 : DataFrame
        The DataFrame from which columns will be added to df1 where rows match
        in the column ``on``. Represents ``right`` in
        ``pd.merge(left=df1, right=df2, on=on, how='left')``.
    on : str
        Column name to join on, e.g. 'address'. Must be a single column.
    find_closest_matches : bool, optional
        If True, the closest matches are searched for those rows that were not
        merged. This can be quite slow, so it is only recommended
        for debugging. The potential matches are never actually merged to df1,
        only included in df_missing. The default is False.

    Returns
    -------
    tuple :
        tuple (df1, df_missing) of the merged DataFrame and a DataFrame
        containing information about the missing matches.

    """
    if find_closest_matches:
        try:
            from fuzzywuzzy import process as fuzzywuzzy_process
        except ImportError as e:
            raise ImportError("Finding closest matches requires 'fuzzywuzzy'. "
                              "Please install it with pip or conda.") from e

    df2_duplicates = df2[on].duplicated().value_counts().get(True, 0)
    if df2_duplicates > 0:
        logger.warning("There are %s duplicates in column '%s' of df2. This "
                       "will probably cause a failure of the merge",
                       df2_duplicates, on)

    # Find out which elements of df2 (the new data) cannot be matched to df1
    df_test = pd.merge(left=df1, right=df2, on=on, how='outer', indicator=True)
    len_right_only = df_test['_merge'].value_counts().get('right_only', 0)
    if len_right_only > 0:
        logger.info("%s rows in df2 could not be matched to df1",
                    len_right_only)

    # If df2 is a GeoDataFrame, we want to be able to plot it on a map.
    # Keeping track of the geometry column is a little convoluted:
    cols_keep = [on]
    if isinstance(df2, gpd.GeoDataFrame):
        if df2.geometry.name in df_test.columns:
            cols_keep.append(df_test.geometry.name)
        else:  # Assume df1 and df2 were renamed to geometry_x and geometry_y
            cols_keep.append(df2.geometry.name + '_y')

    # Create a (Geo)DataFrame with the rows that could not be merged
    df_missing = df_test.loc[df_test['_merge'] == 'right_only', cols_keep]
    try:  # Try to restore geometry
        df_missing.set_geometry(df2.geometry.name + '_y', inplace=True)
    except (ValueError, AttributeError):
        pass

    if not df_missing.empty:
        if not find_closest_matches:
            logger.info("Missing matches are not tested for close matches")
        else:
            logger.info("Finding closest matches...")
            # Isolate the missing rows in df2 and find the closest match
            # in column 'on' of df1 using 'fuzzywuzzy'

            def search_match(row):
                """For the given row, find the closest match in df1."""
                highest = fuzzywuzzy_process.extractOne(row[on], df1[on])
                return pd.Series([highest[0], highest[1]])  # match and score

            df_missing[['match', 'score']] = df_missing.apply(search_match,
                                                              axis='columns')

    df = pd.merge(left=df1,
                  right=df2.drop(columns="geometry", errors='ignore'),
                  on=on, how='left').set_index(df1.index)
    return df, df_missing


def snap_line_network(gdf, tolerance=0.5, tmp_length="tmp_length_calc"):
    """Snap each line in gdf to the vertices of the remaining lines.

    This is useful if the input network is not connected everywhere.
    Shorter lines are edited first (by sorting by length).

    Parameters
    ----------
    gdf : GeoDataFrame
        The network of lines, e.g. streets
    tolerance : float, optional
        Distance tolerance for shapely.snap. The default is 0.5.
        Unit should be [m] with the default crs.
    tmp_length : str, optional
        Name of a temporary column. The default is "tmp_length_calc".

    Returns
    -------
    gdf : GeoDataFrame
        Network with snapped lines.

    """

    gdf[tmp_length] = gdf.length
    gdf.sort_values(tmp_length, inplace=True)
    for idx in gdf.index:
        geom = gdf.loc[idx, gdf.geometry.name]
        other_geoms = gdf[gdf.geometry != geom]
        geom_snapped = shapely.ops.snap(geom, other_geoms.union_all(),
                                        tolerance=tolerance)
        gdf.loc[idx, gdf.geometry.name] = geom_snapped
    gdf.sort_index(inplace=True)
    gdf.drop(columns=[tmp_length], inplace=True)
    return gdf


def create_hexgrid(gdf_buildings, gdf_area=None, resolution=None, clip=False,
                   buffer_distance=200,
                   intensive_variables=None,
                   extensive_variables=None,
                   categorical_variables=None,
                   n_jobs=1,
                   save_path=None,
                   show_plot=True,
                   ):
    """Create hexplot of the sum of col_heat from gdf_buildings in gfd_parcels.

    Parameters
    ----------
    gdf_buildings : TYPE
        DESCRIPTION.
    gdf_area : TYPE, optional
        Area that the hexgrid is constructed from. If None, the convex hull
        of gdf_buildings plus buffer of 'buffer_distance' is used.
    resolution : TYPE, optional
        DESCRIPTION. H3 hex resolution. If none, a resolution is estimated.
    clip : TYPE, optional
        DESCRIPTION. The default is False.
    intensive_variables : list, optional
        Columns in DataFrame for intensive variables. An average is calculated.
    extensive_variables : TYPE, optional
        Columns in DataFrame for extensive variables. These will be summed up.

    scheme: str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used under the hood.
        Supported are all schemes provided by mapclassify
        e.g. ‘BoxPlot’, ‘EqualInterval’, ‘FisherJenks’, ‘FisherJenksSampled’,
        ‘HeadTailBreaks’, ‘JenksCaspall’, ‘JenksCaspallForced’,
        ‘JenksCaspallSampled’, ‘MaxP’, ‘MaximumBreaks’, ‘NaturalBreaks’,
        ‘Quantiles’, ‘Percentiles’, ‘StdMean’, ‘UserDefined’.

    Returns
    -------
    gdf_hex_interp : TYPE
        DESCRIPTION.

    See the following for reference:
    https://pysal.org/tobler/generated/tobler.area_weighted.area_interpolate.html
    https://pysal.org/tobler/notebooks/census_to_hexgrid.html

    """
    logger.info("Create hexgrid")
    if gdf_area is None:
        gdf_area = gpd.GeoDataFrame(
            geometry=[gdf_buildings
                      .union_all()
                      .convex_hull
                      .buffer(buffer_distance)],
            crs=gdf_buildings.crs)
        if show_plot:
            plot_geometries([gdf_area, gdf_buildings],
                            title="'Buffered' area used for hexgrid")

    if resolution is None:
        resolution = fit_hexgrid_resolution(gdf_area.area.sum())

    gdf_hex = tobler.util.h3fy(gdf_area,
                               resolution=resolution,
                               clip=clip)

    if gdf_hex.empty:
        logger.error("Hexgrid is not available. A higher resolution than {} "
                     "might help. Skipping hexgrid plot".
                     format(resolution))
        return gdf_hex

    gdf_hex = tobler.area_weighted.area_interpolate(
        source_df=gdf_buildings,
        target_df=gdf_hex,
        intensive_variables=intensive_variables,
        extensive_variables=extensive_variables,
        categorical_variables=categorical_variables,
        allocate_total=True,
        n_jobs=n_jobs,
        )

    logger.debug("Relative deviation of sums in input and hexgrid:\n%s",
                 (gdf_buildings[extensive_variables].sum()
                  - gdf_hex[extensive_variables].sum()
                  ) / gdf_buildings[extensive_variables].sum()
                 )

    if save_path is not None:
        save_geojson(gdf_hex, file='buildings_hex', path=save_path)

    return gdf_hex


def plot_hexgrid(
        gdf_hex,
        plot_col,
        gdf_buildings=None,
        scale=1,
        title=None,
        figsize=(20, 10),
        scheme='fisherjenkssampled',
        k=7,
        show_plot=True,
        plot_basemap=False,
        ):
    """Plot the result from create_hexgrid()."""
    if not show_plot or gdf_hex.empty:
        return

    fig, ax = plt.subplots(figsize=figsize)

    gdf_hex_plot = gdf_hex.copy()
    gdf_hex_plot[plot_col] *= scale  # e.g. from kWh to MWh with 1/1000

    def plot_recursive(k):
        """Decrease the number of classes k until the plot is valid."""
        try:
            gdf_hex_plot.plot(
                column=plot_col,
                ax=ax,
                scheme=scheme,
                k=k,  # Number of classes
                alpha=0.5,
                legend=True,
                legend_kwds=dict(title=title),
                )
        except ValueError as e:
            k -= 1
            if k > 0:
                plot_recursive(k)
            else:
                raise e
    try:
        plot_recursive(k)
    except ValueError as e:
        raise ValueError("Column '{}' is missing enough valid values "
                         "for plot".format(plot_col)) from e

    if plot_basemap:
        add_basemap(ax, crs=gdf_hex.crs)
    if gdf_buildings is not None:
        gdf_buildings.plot(ax=ax)
    ax.axis('off')
    plt.show()



def fit_hexgrid_resolution(area):
    """Get an estimated appropriate H3 resolution for the given area."""
    # Define a logarithmic function to be fit to the data.
    def func(x, a, b):
        return a * np.log(x) + b

    # Define some useful combinations of area and resolution
    df_res = pd.Series({
        448283.43868: 11,
        2.125123e+07: 9,
        3.260295e+08: 8,
    })

    # Fit the function to the data
    coeff = np.polyfit(np.log(df_res.index.to_numpy()), df_res.values, 1)
    # Get an estimated appropriate resolution for the given area
    resolution = func(area, coeff[0], coeff[1])
    resolution = round(max(min(resolution, 15), 0))
    return resolution


def add_basemap(ax, crs, provider='OSM'):
    """Add a contextily basemap in given crs to plot ax."""
    if provider == 'Toner':
        source = contextily.providers.Stamen.TonerLite
    elif provider == 'OSM':
        source = contextily.providers.OpenStreetMap.Mapnik
    else:
        source = contextily.providers.OpenStreetMap.Mapnik

    contextily.add_basemap(ax=ax, source=source, crs=crs)
    return ax


def sort_from_north_to_south(gdf, col_id=None, set_index=False):
    """Sort the objects in gdf from north to south.

    Insert an increasing index into column 'col_id', if given.
    """
    gdf['centroid_y'] = gdf.centroid.y
    gdf['centroid_x'] = gdf.centroid.x
    gdf.sort_values(by=['centroid_y', 'centroid_x'],
                    ascending=False, inplace=True)
    gdf.drop(columns=['centroid_x', 'centroid_y'], inplace=True)

    if col_id is not None:
        gdf[col_id] = range(0, len(gdf))
        gdf[col_id] = gdf[col_id].astype(str)

        if set_index:
            gdf.set_index(col_id, inplace=True)

    return gdf


def check_duplicate_geometries(gdf, drop=True, keep='first', show_plot=False):
    """Test the input GeoDataFrame for duplicate geometries and plot them."""
    geom_col = gdf.geometry.name
    idx = gdf.duplicated(subset=geom_col)
    if idx.any():
        if show_plot:
            fig, ax = plt.subplots(dpi=400)
            gdf.loc[~idx].plot(ax=ax, color='green')
            gdf.loc[idx].plot(ax=ax, color='red')
            plt.title("Red are duplicate geometries. Please fix!")
            plt.show()

        if drop:
            logger.info("Dropping %s duplicate geometries",
                        idx.value_counts()[True])
            gdf = gdf.drop_duplicates(subset="geometry", keep=keep)
    return gdf


def clean_3d_geometry(gdf):
    """Attempt to make 3D geometry valid (experimental)."""
    from shapely.validation import explain_validity, make_valid

    logger.debug("Number of objects in GeoDataFrame: %s", len(gdf))
    # Drop objects with empty geometry
    gdf.dropna(subset='geometry', inplace=True)
    gdf = gdf[~gdf.is_empty]

    # There are lots of 'objects' with tiny area, and often these are not
    # valid geometries. So we just delete them.
    # Careful, these may be vertical objects (walls)??
    gdf = gdf[gdf.area > 0.00001]
    # gdf['area'] = gdf.area

    # There is a function make_valid() that repairs invalid geometries.
    # But the repaired ones still often cause problems, so we just discard them
    # gdf['validity'] = gdf.geometry.apply(explain_validity)
    # print(gdf['validity'].value_counts())
    # gdf = gdf[gdf['validity'] == 'Valid Geometry']

    # make_valid() allows processing of 2d data, but kind of destroys 3d data
    gdf.geometry = gdf.geometry.apply(make_valid)

    logger.debug("Number of objects in GeoDataFrame after cleaning: %s",
                 len(gdf))
    return gdf


def add_z_height_to_polygons(gdf, z=0):
    """Set (or overwrite) the z (height) component of each polygon.

    This creates a "2.5D" (Multi-) Polygon Z. It is still a flat object at a
    constant height.

    z (float / list / str) :
        A single float value, a list of values (must be the same length)
        as gdf, or the name of a column in gdf.
    """
    if isinstance(z, list):
        z_values = z
    elif isinstance(z, str):
        z_values = gdf[z]
    else:
        z_values = [z] * len(gdf)

    def add_z(geom, z_new):
        def _add_z(x, y, z_old=None):
            return x, y, [z_new for _ in x]
        return shapely.ops.transform(_add_z, geom)

    gdf[gdf.geometry.name] = [add_z(geometry, z)
                              for geometry, z in zip(gdf.geometry, z_values)]

    return gdf


def add_height_from_3d_geometries(gdf, gdf_3d, columns):
    """Add height information from 3D-Model to 2D building GeoDataFrame.

    # Links for obtaining buildings with 3D data
    https://opengeodata.lgln.niedersachsen.de/#lod1

    TODO Read the actual 3D polygon information, instead of depending
    on a data column with the height value

    TODO: Deal with data where height is stored in multiple columns
    # Combine the values from two columns into one:
    gdf_lod1['height'] = gdf_lod1['measuredheight']
    gdf_lod1.loc[gdf_lod1['height'].isna(), 'height'] = (
        gdf_lod1.loc[gdf_lod1['height'].isna(), 'buildingpart_measuredheight'])

    """
    return combine_buildings_and_parcels(gdf, gdf_3d, columns)


def choose_random_thermal_load(gdf_buildings, low=10, high=50):
    """Choose a random maximum thermal power for each building.

    Set the column 'P_heat_max' to a random value between 10 and 50 kW
    for all houses. This column is required for dhnx.
    """
    np.random.seed(42)
    gdf_buildings['P_heat_max'] = \
        np.random.randint(low, high, size=len(gdf_buildings))
    return gdf_buildings


# Section "OpenStreetMap downloads"
def download_area_by_name(places, crs='EPSG:25832', show_plot=True, **kwargs):
    """Download the polygon of an area from OpenStreetMap by its name.

    Parameters
    ----------
    places : list
        A list of strings, defining the places to search for.
        Example: ``places=['Braunschweig, Germany']``

    crs : str, optional
        A coordinate reference system string to convert the data into.
        Default is 'EPSG:25832'

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the polygon(s) of the found location(s).

    """
    gdf = ox.geocode_to_gdf(places, **kwargs)
    if crs is not None:
        gdf.to_crs(crs, inplace=True)

    if show_plot:
        plot_geometries(gdf, plt_kwargs=dict(alpha=0.5), plot_basemap=True,
                        title=f"Detected area '{', '.join(places)}'")

    return gdf


def download_zip_code_layer(places, crs='EPSG:25832', **kwargs):
    """Download polygon of zip codes by name of given place."""
    # Define custom filter for OSM geometries
    kwargs.setdefault('tags', {'boundary': 'postal_code'})
    gdf = gdf = download_features_from_place(places, crs=crs, **kwargs)
    return gdf


def download_admin_level(places, level=9, crs='EPSG:25832', **kwargs):
    """Download polygon of administration level by name of given place."""
    # Define custom filter for OSM geometries
    kwargs.setdefault('tags', {'admin_level': str(level)})
    gdf = download_features_from_place(places, crs=crs, **kwargs)
    return gdf


def download_features_from_place(places, crs='EPSG:25832', **kwargs):
    """Download features from a plce from OpenStreetMap with osmnx."""
    # Use OSMnx to download the data as a GeoDataFrame
    gdf = ox.features_from_place(places, **kwargs)
    gdf = gdf.xs('relation', level='element_type')
    gdf.drop(columns=['nodes', 'ways'], errors='ignore', inplace=True)
    if crs is not None:
        gdf.to_crs(crs, inplace=True)

    return gdf


def download_country_borders_from_osm(
        places=["Braunschweig, Germany"], show_plot=False):
    """Download borders of country, city etc. by the name(s) of the place(s).

    https://pygis.io/docs/d_access_osm.html
    https://max-coding.medium.com/getting-administrative-boundaries-from-open-street-map-osm-using-pyosmium-9f108c34f86

    Returns
    -------
    None.

    # Unused experiments:

    tags = {"boundary": 'administrative'}
    gdf = ox.features_from_polygon(area.geometry[0], tags=tags)

    plot_geometries([gdf], plot_basemap=True)
    plot_geometries([area], plot_basemap=True)

    gdf.dropna(subset=['admin_level'])
    gdf_al2 = gdf[gdf['admin_level'] == '2']
    gdf_al3 = gdf[gdf['admin_level'] == '3']
    gdf_al2 = gdf_al2[gdf_al2['addr:country'] == 'DE']
    gdf_DE = gdf[gdf['addr:country'] == 'DE']
    gdf_al2.plot()
    gdf_al3.plot()
    gdf['admin_level'].unique()
    gdf['place'].unique()
    gdf['addr:country'].unique()
    gdf.index.unique('element_type')

    """
    gdf = download_area_by_name(places=places, show_plot=show_plot)
    return gdf


def download_buildings_from_osm(
        gdf_polygon,
        building_keys=True,
        crs=None,
        show_plot=False,
        dropna_tresh=None,
        rename_dict=None,  # e.g. {'building': 'building_osm'},
        prefix=None,
        ):
    """Download building data from OpenStreetMap.

    gdf_polygon must define the region from which to download data. If
    it contains more than one geometry, the convex hull of the unary union
    is used to create the download region.

    Select the building types you want to import
    See: https://wiki.openstreetmap.org/wiki/Key:building

    Args:
        crs (str): A target coordinate reference system, e.g. "EPSG:3857".
        If "None", use crs of gdf_polygon.

        building_keys (bool, list): Either ``True`` or a list of buildings
        tags used in OpenStreetMap to filter which building types to select.
        Example: building_keys=['apartments', 'commercial', 'detached',
        'house', 'industrial', 'residential', 'retail', 'semidetached_house',
        'yes']

        dropna_tresh (int): The input 'thresh' for df.dropna(). If not None,
        dropna() is called, allowing to remove all columns that have only
        a few entries. This can reduce the number of (mostly empty) columns
        returned dramatically. If dropna_tresh < 1.0, it is multiplied with
        the length of the DataFrame. I.e. dropna_tresh = 0.02 means
        all columns with less than 2% entries are discarded.

    """
    building_tags = dict({'building': building_keys})

    if isinstance(gdf_polygon, str):  # If this is a file path, load the file
        gdf_polygon = gpd.read_file(gdf_polygon)

    if not isinstance(gdf_polygon, gpd.GeoDataFrame):
        logger.warning("The input geometry is not a GeoDataFrame.")

    if crs is None:  # Store original crs of input
        crs = gdf_polygon.crs

    if len(gdf_polygon) > 1:
        gdf_polygon = gpd.GeoDataFrame(
            geometry=[gdf_polygon
                      .union_all()
                      .convex_hull],
            crs=gdf_polygon.crs)

    polygon = gdf_polygon.to_crs(epsg=4326).geometry[0]  # for osmnx
    try:  # osmnx>=1.5.0
        gdf = ox.features_from_polygon(polygon, tags=building_tags)
    except AttributeError:  # osmnx<1.5.0
        gdf = ox.geometries_from_polygon(polygon, tags=building_tags)

    # Convert to target crs
    gdf.to_crs(crs=crs, inplace=True)
    if show_plot:
        plot_geometries(gdf, title='Downloaded buildings',
                        plot_basemap=True)

    # Make sure that only polygon geometries are used
    # gdf = gdf[gdf['geometry'].apply(
    #     lambda x: isinstance(x, geometry.Polygon)
    # )].copy()

    if dropna_tresh is not None:
        if dropna_tresh < 1:  # Interpret as a relative threshold
            dropna_tresh = int(len(gdf) * dropna_tresh)
        gdf.dropna(axis='columns', thresh=dropna_tresh, inplace=True)

    gdf.reset_index(inplace=True)
    for element_col in ['element_type', 'element']:
        try:  # 'node' elements are unwanted point geometries
            gdf = gdf.loc[gdf[element_col] != 'node'].copy()
        except KeyError:
            pass
    # Remove nodes column (that somehow makes trouble for exporting .geojson)
    gdf.drop(columns=['nodes'], inplace=True, errors='ignore')

    if rename_dict:
        gdf.rename(columns=rename_dict, inplace=True)

    if prefix is not None:
        geom_col = gdf.geometry.name
        gdf = gdf.add_prefix(prefix)  # Rename all columns with prefix
        gdf.set_geometry(prefix+geom_col, inplace=True)  # Restore geometry

    return gdf


def download_streets_from_osm(
        gdf_polygon,
        highway_keys=[
            'residential',
            'service',
            'unclassified',
            'primary',
            'secondary',
            'tertiary',
            'living_street',
            'footway',
            'steps',
            'pedestrian',
            # 'path',
            # 'track'
            # 'yes',
            ],
        crs="EPSG:3857",
        show_plot=False,
        dropna_tresh=None,
        simplify=True,
        retain_all=False,  # retain only largest connected component for dhnx
        truncate_by_edge=True,  # Otherwise roads might be cut off
        ):
    """Download street network data from OpenStreetMap.

    Select the street types you want to consider as district heating routes

    For documentation on the key system for types of streets, see
    https://wiki.openstreetmap.org/wiki/Key:highway

    Args:
        gdf_polygon (GeoDataFrame): An object defining the selection area

        highway_keys (list): List of osm highway tags or empty list for
        all highway objects (but can include points)

        dropna_tresh (int): The input 'thresh' for df.dropna(). If not None,
        dropna() is called, allowing to remove all columns that have only
        a few entries. This can reduce the number of (mostly empty) columns
        returned dramatically. If dropna_tresh < 1.0, it is multiplied with
        the length of the DataFrame. I.e. dropna_tresh = 0.02 means
        all columns with less than 2% entries are discarded.

        simplify (bool): if True, simplify graph topology by removing all
        nodes that are not intersections or dead-ends

        retain_all (bool): if True, return the entire graph even if it
        is not connected. otherwise, retain only the largest weakly
        connected component.

        truncate_by_edge (bool): if True, retain nodes outside boundary
        polygon if at least one of node’s neighbors is within the polygon

    """
    logger.info("Download street data from OpenStreetMap")

    if len(gdf_polygon) > 1:
        gdf_polygon = gpd.GeoDataFrame(
            geometry=[gdf_polygon
                      .union_all()
                      .convex_hull],
            crs=gdf_polygon.crs)
    polygon = gdf_polygon.to_crs(epsg=4326).geometry[0]  # for osmnx

    # Download the street network data from OpenStreetMap
    streets = dict({'highway': highway_keys})
    # Option 1) Create GeoDataFrame object from the polygon
    # try:  # osmnx>=1.5.0
    #     gdf_lines_streets = ox.features_from_polygon(polygon, tags=streets)
    # except AttributeError:  # osmnx<1.5.0
    #     gdf_lines_streets = ox.geometries_from_polygon(polygon, tags=streets)

    # Option 2) Download as a graph and convert to GeoDataFrame
    # Gives more options to use osmnx functions for simplifying the network
    if len(highway_keys) > 0:
        key, values = list(streets.items())[0]
        custom_filter = f'[\"{key}\"~\"{"|".join(values)}\"]'
        network_type = None
    else:
        network_type = 'all'
        custom_filter = None

    graph = ox.graph_from_polygon(
        polygon,
        simplify=simplify,
        retain_all=retain_all,  # retain only largest connected component
        truncate_by_edge=truncate_by_edge,  # Otherwise roads might be cut off
        network_type=network_type,
        custom_filter=custom_filter,
        )
    # graph = ox.truncate.largest_component(graph, strongly=True)
    graph = ox.convert.to_undirected(graph)  # drop "duplicates"
    gdf_lines_streets = ox.graph_to_gdfs(graph, nodes=False)

    if show_plot:
        # ox.plot_graph(graph)
        plot_geometries([gdf_polygon, gdf_lines_streets],
                        title='Downloaded streets')

    # Remove nodes column (that make somehow trouble for exporting .geojson)
    gdf_lines_streets.drop(columns=['nodes'], inplace=True, errors='ignore')
    # Filter out e.g. 'polygon' types that would cause issues with dhnx
    accepted_types = ['LineString', 'MultiLineString']
    gdf_lines_streets = gdf_lines_streets.loc[
        gdf_lines_streets['geometry'].type.isin(accepted_types)].copy()

    if crs is not None:
        # Convert to target crs
        gdf_lines_streets.to_crs(crs=crs, inplace=True)

    if dropna_tresh is not None:
        if dropna_tresh < 1:  # Interpret as a relative threshold
            dropna_tresh = int(len(gdf_lines_streets) * dropna_tresh)
        gdf_lines_streets.dropna(axis='columns', thresh=dropna_tresh,
                                 inplace=True)

    return gdf_lines_streets


# Section 'Plotting functions'

def plot_geometries(
        gdf_list,
        title='',
        crs_default="EPSG:4647",
        plot_basemap=False,
        plt_kwargs=None,
        set_axis_off=False,
        show_plot=True,
        save_path=None,
        **fig_kwargs,
        ):
    """Plot the given list of geometry objects.

    Geometries can be GeoDataFrames, GeoSeries and shapely geometries.
    They are converted to a common crs. If taking the crs of the first
    entry in gdf_list fails, 'crs_default' is used instead.

    plt_args : list
        List of dictionaries for each gdf in gdf_list, with arguments to
        hand over to each gdf.plot() call.
    """
    fig_kwargs.setdefault('figsize', (20, 10))
    fig, ax = plt.subplots(**fig_kwargs)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if isinstance(gdf_list, gpd.GeoDataFrame):
        gdf_list = [gdf_list]  # Allow single GeoDataFrame as list
    if isinstance(plt_kwargs, dict):
        plt_kwargs = [plt_kwargs]  # Allow single dict as list

    try:
        crs_use = gdf_list[0].crs
    except Exception:
        crs_use = crs_default

    if plt_kwargs is None:  # Need to create a list of empty dicts
        plt_kwargs = [dict() for x in range(len(gdf_list))]

    handles = []

    for gdf, color, args in zip(gdf_list, colors[:len(gdf_list)], plt_kwargs):
        if not (isinstance(gdf, gpd.GeoDataFrame)
                or isinstance(gdf, gpd.GeoSeries)):
            # Assume that this is a shapely geometry that can be converted
            gdf = gpd.GeoDataFrame(geometry=[gdf], crs=crs_default)

        if 'column' not in args.keys():
            # gdf.plot() accepts only one of 'color' and 'column'
            args.setdefault('color', color)
        gdf.to_crs(crs=crs_use).plot(ax=ax, **args)

        # Unless the "columns" argument is used with gdf.plot(), by default
        # no legend is created. Create an artificial legend:
        if 'label' in args.keys():
            allowed_args = (list(signature(Patch).parameters.keys())
                            + list(signature(Patch.set).parameters.keys()))
            for key in [x for x in args.keys() if x not in allowed_args]:
                args.pop(key, None)
            if 'linewidth' in args.keys():
                args.setdefault('color', 'black')
                handles.append(Line2D([0], [0], **args))
            else:
                handles.append(Patch(**args))

    if plot_basemap:
        add_basemap(ax, crs=crs_use)

    if set_axis_off:
        ax.set_axis_off()

    if len(handles) > 0:
        plt.legend(handles=handles)
    plt.title(title)
    if save_path is not None:
        custom_plot_save(save_path)
    if show_plot:
        plt.show()


def custom_plot_save(filename, folder='', dpi=750,
                     transparent=False, extensions=['.png', '.svg']):
    """Save plot figures to different file formats."""
    filepath = os.path.join(folder, filename)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    for ext in extensions:
        plt.savefig(filepath+ext, dpi=dpi, bbox_inches='tight',
                    transparent=transparent)


def plot_heated(gdf, col_heated='heated', **fig_kwargs):
    """Plot the buildings where col_heated is True."""
    logger.info("Plot map of column '%s'", col_heated)
    fig_kwargs.setdefault('figsize', (10, 10))
    fig_kwargs.setdefault('dpi', 100)

    _, ax = plt.subplots(**fig_kwargs)
    gdf.plot(ax=ax, column=col_heated, legend=True,
             legend_kwds={'title': col_heated},
             )
    plt.show()


# Section "DHNX Processing"
# Comfort functions for using the DHNX library

def assign_random_producer_building(gdf_houses, seed=42):
    """Choose one random building as a 'generator' for district heating."""
    rng = np.random.default_rng(seed=seed)
    id_producer = rng.integers(len(gdf_houses))
    gdf_prod = gdf_houses.iloc[[id_producer]].copy()
    gdf_houses.drop(index=gdf_houses.index[id_producer], inplace=True)
    return gdf_houses, gdf_prod


def clean_previous_street_results(gdf):
    """Clean a lines GeoDataFrame that was a previous optimization result.

    It sometimes can be desirable to re-use the distribution lines
    result from a previous optimization run, e.g. to keep placement of pipe
    location consistent and cut optimization time, while still adapting to
    changes in e.g. thermal power per building or pipe properties.

    In such a case, the generator line(s) and house lines must be removed
    from the network, and old columns are removed for savety.
    """
    if 'type' in gdf.columns:
        mask = gdf['type'].isin(['DL'])
        gdf = gdf.loc[mask, [gdf.geometry.name]]

    return gdf


# @memory.cache  # Deactivated, as it sometimes causes stack overflow exception
def dhnx_run(gdf_lines_streets, gdf_poly_gen, gdf_poly_houses,
             save_path='./out', show_plot=True,
             path_invest_data=None,  # 'invest_data',
             path_pipe_data=None,  # "input/Pipe_data.csv",
             pipe_data_sheet_name=0,
             df_load_ts_slice=None,
             col_p_th='P_heat_max',
             bidirectional_pipes=False,
             simultaneity=1,
             reset_index=True,
             n_conn=1,
             n_conn_prod=1,
             method='midpoint',
             welding=True,
             solver=None,
             solve_kw={'tee': True},  # print solver output
             solver_cmdline_options=None,
             ):
    """Run the dhnx (district heating networks) process.

    This function uses a 'cache': The results from a run are stored in
    a local folder and reused for as long as the input stays the same.
    This can save a lot of time for repeated runs.
    To enforce clearing the cache and running the function again, run
    the following line of code before calling this function:

    ``dhnx_run.clear()``

    Parameters
    ----------
    gdf_lines_streets : GeoDataFrame
        A line network (typically streets) used as potential paths for
        district heating grid.
    gdf_poly_gen : GeoDataFrame
        Polygon layer of one or multiple generator buildings which supply
        the heat for the district heating grid.
    gdf_poly_houses : GeoDataFrame
        Polygon layer of the houses/buildings that should be connected
        to the district heating grid.
    save_path : str, optional
        Path where result files are saved. The default is './out'.
    show_plot : boolean, optional
        If True, show plots generated during runtime. The default is True.
    path_invest_data : str, optional
        Path to folder with special dhnx input (consumer and producer files).
        If None, all default settings are used. The default is None.
    path_pipe_data : str, optional
        Path to a .csv or .xlsx file with input data for district heating
        pipes (U-value, cost, etc. per norm diameter DN). If None, a
        default dataset for steel pipes is used. The default is None.
    pipe_data_sheet_name : str, optional
        Name of the sheet to use if path_pipe_data points to an .xlsx file
        with input data for district heating pipes. Reads the first sheet
        by default. The default is 0.
    df_load_ts_slice : DataFrame, optional
        A DataFrame with timeseries of the thermal load
        in kW for buildings. The column names (identifier for each building)
        need to match the index of gdf_poly_houses, to be able to match.
        If this is defined, ``col_p_th`` must be set to ``None``.
        The default is None.
    col_p_th : str, optional
        A column name of ``gdf_poly_houses`` containing the thermal power
        of each building. The default is 'P_heat_max'.
    bidirectional_pipes : boolean, optional
        If False, try to prevent bidirectional pipes in DHNx.
        The default is False.
    simultaneity : float, optional
        An overall simultaneity factor used by DHNx. It simply reduces
        the thermal power defined by col_p_th, without taking any actual
        simultaneity effects into account. Use with care. The default is 1.
    reset_index : boolean, optional
        DHNx requires resetting the index of the input GeoDataFrames.
        Skipping this is currently not unsupported. The default is True.
    n_conn : int, optional
        Number of connection lines created from each consumer/producer to
        the nearest line segments in the street network. This allows the
        placement of the connection lines to be part of the optimization
        process. The default is 1.
    n_conn_prod : int, optional
        Number of connection lines created from each producer to
        the nearest line segments in the street network. This allows the
        placement of the connection lines to be part of the optimization
        process. The default is 1.
    method : string, optional
        Method for creating the point if polygons are given for the consumers
        and producers. Method 'midpoint' uses the centroid of each building
        polygon. Method 'boundary' moves the point to the boundary (wall) of
        the building, along the line constructed from centroid to the street.
    welding : bool, optional
        Weld continuous line segments together and cut loose ends. This
        can improve the performance of the optimization, as it decreases
        the total number of line elements. Default is True.
    solver : str, optional
        Name of the solver used by dhnx, e.g. 'cbc' or 'gurobi'.
        If None, try to find an installed solver or install cbc.
        Default is None.
    solve_kw : dict, optional
        Special keywords used for the solver. {'tee': True}, prints the solver
        output to the console, while {'tee': False} hides it.
    solver_cmdline_options : dict, optional
        Command line options used when calling the solver. Some useful
        examples are 'seconds', 'allowableGap' and 'ratioGap' for cbc and
        'TimeLimit', 'MIPGapAbs' and 'MIPGap' for gurobi. Please refer to
        the documentation of each solver for more information.

    Raises
    ------
    ValueError
        DESCRIPTION.
    OSError
        DESCRIPTION.

    Returns
    -------
    gdf_pipes : TYPE
        DESCRIPTION.
    df_pipes : TYPE
        DESCRIPTION.

    """
    if df_load_ts_slice is not None and col_p_th is not None:
        raise ValueError("Only one of 'col_p_th' and 'df_load_ts_slice' "
                         "can be defined. (To use a single thermal power "
                         "value or a time series of thermal powers for "
                         "each building.)")

    elif df_load_ts_slice is None and col_p_th is not None:
        # Specific column name for thermal power is required for DHNX
        if gdf_poly_houses[col_p_th].isna().any():
            raise ValueError("Each building connected to the district "
                             "heating grid needs to have an associated "
                             f"thermal power in column '{col_p_th}'")

        gdf_poly_houses['P_heat_max'] = gdf_poly_houses[col_p_th]

    if ((df_load_ts_slice is None) and
       (col_p_th not in gdf_poly_houses.columns)):
        raise ValueError("The thermal load of each house in gdf_poly_houses "
                         f"needs to be given via column '{col_p_th}' or via "
                         "a separate timeseries df_load_ts_slice")

    if df_load_ts_slice is not None and simultaneity != 1:
        logger.warning("By defining 'df_load_ts_slice', a time series is "
                       "used for the thermal power of each building, which "
                       "is a method of introducing simultaneity effects. "
                       "Defining an additional simultaneity factor might "
                       "lead to undesired results (i.e. too small pipes).")

    # process the geometry
    tn_input = dhnx.gistools.connect_points.process_geometry(
        lines=gdf_lines_streets,
        producers=gdf_poly_gen.copy(),
        consumers=gdf_poly_houses.copy(),
        method=method,
        reset_index=reset_index,
        n_conn=n_conn,
        n_conn_prod=n_conn_prod,
        welding=welding,
    )

    if show_plot:
        # plot output after processing the geometry
        _, ax = plt.subplots(figsize=(20, 10), dpi=300)
        tn_input['consumers'].plot(ax=ax, color='green')
        tn_input['producers'].plot(ax=ax, color='red')
        tn_input['pipes'].plot(ax=ax, color='blue')
        tn_input['forks'].plot(ax=ax, color='grey')
        plt.title('Geometry after pre-processing')
        plt.show()

    # Optionally export the geodataframes and load it into qgis
    # for checking the results of the geometry processing
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for filename, gdf in tn_input.items():
            try:
                save_geojson(gdf, file=filename, path=save_path,
                             type_errors='coerce')
            except Exception as e:
                print(gdf)
                breakpoint()
                logger.exception(e)

    # Part III: Initialise the ThermalNetwork and perform the Optimisation

    # initialize a ThermalNetwork
    network = dhnx.network.ThermalNetwork()

    # add the pipes, forks, consumer, and producers to the ThermalNetwork
    for k, v in tn_input.items():
        network.components[k] = v

    if df_load_ts_slice is not None:
        if len(gdf_poly_houses) != len(df_load_ts_slice.columns):
            raise ValueError("Buildings GeoDataFrame and time series must "
                             "describe the same number of buildings")
        if reset_index:  # Replace given ts columns with range index
            # Assume consumers and their time series have the same order
            df_load_ts_slice.columns = tn_input['consumers'].index
        # Enforce string column names, because dhnx expects them
        df_load_ts_slice.columns = df_load_ts_slice.columns.astype(str)
        # Add the time series slice of the thermal load to the Network
        # (required for using settings["heat_demand"]="series") later on
        network.sequences['consumers']['heat_flow'] = df_load_ts_slice

    # check if ThermalNetwork is consistent
    network.is_consistent()

    # Create or overwrite the input file "invest_data/network/pipes.csv"
    # Create pipe data table from manufacturer data
    df_DN, constants_costs, constants_loss = (
        calc_lineralized_pipe_input(
            path_pipe_data=path_pipe_data,
            pipe_data_sheet_name=pipe_data_sheet_name,
            T_FF=80, T_RF=50, T_ground=10, dP_max=150,
            show_plot=show_plot))
    # Export the linearized form required by DHNx
    df_invest_opt_pipes = export_lineralized_pipe_input(
        df_DN, constants_costs, constants_loss, path_invest_data)
    if save_path is not None:
        save_excel(df_DN, os.path.join(save_path, 'DN_table.xlsx'))

    # load the specification of the oemof-solph components
    # For 'consumers' and 'producers', load default options
    invest_opt = get_default_dhnx_invest_options()
    # Add the pipes investment options
    invest_opt['network'] = dict(pipes=df_invest_opt_pipes)
    # If any investment options are present as a file structure, use those
    if path_invest_data is not None and os.path.exists(path_invest_data):
        invest_opt.update(
            dhnx.input_output.load_invest_options(path_invest_data))

    # Define which solver to use. This function checks if 'gurobi' or 'cbc'
    # are installed and installs 'cbc' if necessary
    if solver is None:
        solver = get_installed_solver()

    # Optionally, define some settings for the solver. Especially increasing
    # the solution tolerance with 'ratioGap' or setting a maximum runtime
    # in 'seconds' helps if large networks take too long to solve
    if solver_cmdline_options is None:
        solver_cmdline_options = {}

    if solver == 'gurobi':
        # Try to remove common 'cbc' arguments from dict that would cause
        # an error with 'gurobi'
        solver_cmdline_options.pop('allowableGap', None)
        solver_cmdline_options.pop('ratioGap', None)
        solver_cmdline_options.pop('seconds', None)

    settings = dict(
        solver=solver,
        solve_kw=solve_kw,
        # solver_cmdline_options={  # cbc
        #     # 'allowableGap': 1e-5,  # (absolute gap) default: 1e-10
        #     'ratioGap': 0.01,  # (0.2 = 20% gap) default: 0
        #     # 'seconds': 60 * 1 * 1,  # (maximum runtime) default: 1e+100
        #     'seconds': 60 * 60 * 8,  # (maximum runtime) default: 1e+100
        # },
        # solver_cmdline_options={  # gurobi
            # 'MIPGapAbs': 1e-5,  # (absolute gap) default: 1e-10
            # 'MIPGap': 0.03,  # (0.2 = 20% gap) default: 0
            # 'TimeLimit': 60 * 1,  # (seconds of maximum runtime)
            # 'TimeLimit': 60 * 10 * 1,  # (seconds of maximum runtime)
            # 'TimeLimit': 60 * 60 * 6,  # (seconds of maximum runtime)
        # },
        solver_cmdline_options=solver_cmdline_options,
        bidirectional_pipes=bidirectional_pipes,
        simultaneity=simultaneity,
        )
    if df_load_ts_slice is not None:
        settings["heat_demand"] = "series"
        settings["num_ts"] = len(df_load_ts_slice)
        settings["frequence"] = 'h'  # default value

    # perform the investment optimisation
    try:
        network.optimize_investment(invest_options=invest_opt, **settings)
    except ValueError as e:
        logger.exception(e)
        logger.error("The district heating grid network optimization "
                     "failed with an error. Examples for possible reasons:\n"
                     "- The time limit for the solver was too short to reach "
                     "any useful solution.\n"
                     "- Largest pipe cannot carry the required capacity. "
                     "Adding larger DNs to the list of available pipes "
                     "might solve the error.\n"
                     "- The given producers, if limited in their capacity, "
                     "cannot provide the required capacity.")
        breakpoint()
        # save_geopackage(network.components['forks'], 'debug_forks')
        # save_geopackage(network.components['pipes'], 'debug_pipes')
        # save_geopackage(gdf_lines_streets, 'debug_streets')

    # Part IV: Check the results #############

    # get results
    results_edges = network.results.optimization['components']['pipes']
    # print(results_edges[['from_node', 'to_node', 'hp_type', 'capacity',
    #                      'direction', 'costs', 'losses']])

    # print(results_edges[['costs']].sum())
    # print('Objective value: ', network.results.optimization['oemof_meta']['objective'])
    logger.info('Total costs: {}'.format(results_edges[['costs']].sum()))
    logger.info('Objective value: {}'.format(
        network.results.optimization['oemof_meta']['objective']))
    # The costs of the objective value and the investment costs of the DHS
    # pipelines are the same, since no additional costs (e.g. for energy
    # sources) are considered in this example.

    # add the investment results to the geoDataFrame
    gdf_pipes = network.components['pipes']
    cols_drop = [c for c in results_edges.columns if c in gdf_pipes]
    gdf_pipes = gdf_pipes.drop(columns=cols_drop)  # Drop duplicate columns
    gdf_pipes = gdf_pipes.join(results_edges, rsuffix='_results')

    gdf_pipes = apply_DN(gdf_pipes, df_DN)  # Apply DN from capacity
    gdf_pipes = get_total_costs_and_losses(gdf_pipes, df_DN)

    if show_plot:
        plot_geometries(
            [gdf_poly_houses, gdf_poly_gen, gdf_pipes[gdf_pipes['DN'] > 0]],
            plt_kwargs=[dict(label='Consumer'),
                        dict(label='Producer'),
                        dict(column='DN', linewidth=2, legend=True,
                             label='Pipelines', legend_kwds={'label': 'DN'})],
            plot_basemap=True,
            title='Invested pipelines',
            set_axis_off=True,
            dpi=300)

    # Export results
    gdf_pipes = gdf_pipes[gdf_pipes['DN'] > 0]  # Keep only DN>0 in output
    if save_path is not None:
        save_geojson(gdf_pipes, file='pipes_result', path=save_path)
        save_excel(gdf_pipes, os.path.join(save_path, 'pipes_result.xlsx'))

    # Save grouped info about pipes
    df_pipes = gdf_pipes.copy()
    df_pipes = df_pipes[df_pipes['DN'] > 0]
    df_pipes.replace({'type': {'DL': 'Verteilleitung',
                               'GL': 'Erzeugerleitung',
                               'HL': 'Hausanschlussleitung'}},
                     inplace=True)

    cols_mean = ['Inner diameter [m]', 'Roughness [mm]',
                 'U-value [W/mK]', 'Costs [€/m]', 'T_forward [°C]',
                 'T_return [°C]', 'T_ground [°C]', 'T_mean [°C]',
                 'Max delta p [Pa/m]', 'v_max [m/s]', 'Mass flow [kg/s]',
                 'P_max [kW]', 'P_loss [kW/m]']
    cols_sum = ['length', 'capacity', 'Cost_lin [€]',
                'P_loss_lin [kW]', 'Cost [€]', 'P_loss [kW]',
                'E_loss [MWh]']
    agg_dict = dict()
    for col in df_pipes.columns:
        if col in cols_mean:
            agg_dict[col] = 'mean'
        elif col in cols_sum:
            agg_dict[col] = 'sum'

    df_pipes_agg = (df_pipes.set_index('type')
                    .groupby(['type', 'DN'])
                    .agg(agg_dict))

    df_pipes_stat = pd.Series(dtype='float')
    df_pipes_stat['Sum length consumer [m]'] = \
        df_pipes_agg.loc['Hausanschlussleitung', 'length'].sum()
    df_pipes_stat['Sum length producer [m]'] = \
        df_pipes_agg.loc['Erzeugerleitung', 'length'].sum()
    df_pipes_stat['Sum length distribution [m]'] = \
        df_pipes_agg.loc['Verteilleitung', 'length'].sum()
    df_pipes_stat['Sum length [m]'] = df_pipes_agg['length'].sum()

    df_pipes_stat['Sum E_loss [MWh]'] = df_pipes_agg['E_loss [MWh]'].sum()

    df_pipes_stat['Sum thermal power consumers [kW]'] = (
        df_pipes_agg.loc['Hausanschlussleitung', 'capacity'].sum()
        - df_pipes_agg.loc['Hausanschlussleitung', 'P_loss_lin [kW]'].sum())

    df_pipes_stat['ratio pipe capacity producer / consumer [-]'] = (
        df_pipes_agg.loc['Erzeugerleitung', 'capacity'].sum()
        / df_pipes_agg.loc['Hausanschlussleitung', 'capacity'].sum())

    df_pipes_stat['DN_mean [-]'] = (
        (df_pipes_agg['length']
         * df_pipes_agg.index.get_level_values('DN')).sum()
        / df_pipes_agg['length'].sum())

    df_pipes_stat['U_mean [W/mK]'] = (
        (df_pipes_agg['length'] * df_pipes_agg['U-value [W/mK]']).sum()
        / df_pipes_agg['length'].sum())

    if save_path is not None:
        save_excel(df_pipes_agg, os.path.join(save_path, 'WN_pipes.xlsx'))
        save_excel(df_pipes_stat, os.path.join(save_path, 'WN_pipes_stat.xlsx'))

    return_dict = {'network': network,
                   'gdf_pipes': gdf_pipes,
                   'df_pipes': df_pipes_agg,
                   'df_DN': df_DN,
                   }

    class DHNx_Return():
        """Create an object for storing the return values."""
        def __init__(self, network, gdf_pipes, df_pipes, df_DN):
            self.network = network
            self.gdf_pipes = gdf_pipes
            self.df_pipes = df_pipes
            self.df_DN = df_DN

        def __str__(self):
            return str(self.__class__) + ' containing:\n' + '\n'.join(
                ('{}:\n{}'
                 .format(item, self.__dict__[item]) for item in self.__dict__))

    # Package the return values in a dedicated object. Using a dict would
    # be more straightforward, but caused issues when trying to pickle it.
    # dhnx_return = DHNx_Return(network, gdf_pipes, df_pipes, df_DN)
    # Not in use, because this breaks the @memory.cache function.

    # return dhnx_return
    return network, gdf_pipes, df_pipes, df_DN


def get_installed_solver(auto_install_cbc=True):
    """Check if the solvers 'gurobi' or 'cbc' are installed.

    Install 'cbc' if necessary.
    """
    installer = cbc_installer.CBCInstaller()

    if installer.is_solver_installed('gurobi'):
        return 'gurobi'
    else:
        if installer.is_solver_installed():
            return 'cbc'
        elif auto_install_cbc:
            installer.install()
            return 'cbc'
        else:
            raise ValueError("Neither gurobi nor cbc could be found")


# Section "DHNX Postprocessing"
# Use these functions on results from DHNX

def calc_lineralized_pipe_input(
        path_pipe_data="input/Pipe_data.csv",
        pipe_data_sheet_name=0,
        T_FF=80, T_RF=50, T_ground=10, dP_max=150,
        show_plot=False):
    # Besides the geometries, we need the techno-economic data for the
    # investment optimisation of the DHS piping network. Therefore, we load
    # the pipes data table. This is the information you need from your
    # manufacturer / from your project.
    if path_pipe_data is not None and os.path.exists(path_pipe_data):
        if os.path.splitext(path_pipe_data)[1] == '.xlsx':
            df = pd.read_excel(path_pipe_data, sheet_name=pipe_data_sheet_name)
        else:
            df = pd.read_csv(path_pipe_data, sep=",")
    else:
        logger.info("Provide a file '%s' with pipe properties per DN for "
                    "a custom lineralized optimization input. File not found, "
                    "using default pipe properties instead.", path_pipe_data)
        df = get_default_df_DN(T_FF=T_FF, T_RF=T_RF, T_ground=T_ground)

    # This is an example of input data. The Roughness refers to the roughness of
    # the inner surface and depends on the material (steel, plastic). The U-value
    # and the costs refer to the costs of the whole pipeline trench, so including
    # forward and return pipelines. The design process of DHNx is based on
    # a maximum pressure drop per meter as design criteria:
    # You could also define the maximum pressure drop individually for each DN
    # number.
    if "Max delta p [Pa/m]" not in df.columns:
        df["Max delta p [Pa/m]"] = dP_max

    # As further assumptions, you need to estimate the operation temperatures
    # of the district heating network in the design case:
    if "T_forward [°C]" not in df.columns:
        df["T_forward [°C]"] = T_FF
    if "T_return [°C]" not in df.columns:
        df["T_return [°C]"] = T_RF
    if "T_ground [°C]" not in df.columns:
        df["T_ground [°C]"] = T_ground
    df['T_mean [°C]'] = df[['T_forward [°C]', 'T_return [°C]']].mean(
        axis='columns')

    # The user has the option to assign a factor to each U-value and cost
    # value, to mark whether the meter unit represents trench length (single
    # line length) or total pipe length (double line length)
    if 'f_length_loss' in df.columns:
        df['U-value [W/mK]'] = df['U-value [W/mK]'] * df['f_length_loss']
        df = df.drop(columns=['f_length_loss'])  # Drop the used factor
    if 'f_length_cost' in df.columns:
        df['Costs [€/m]'] = df['Costs [€/m]'] * df['f_length_cost']
        df = df.drop(columns=['f_length_cost'])  # Drop the used factor

    # Now the maximum power capacity can be calculated
    df = calc_pipes_p_max(df)

    # The last step is the linearisation of the cost and loss parameter for
    # the DHNx optimisation (which is based on the MILP optimisation package
    # oemof-solph)

    # It is possible to use different accuracies: you could linearize the cost
    # and loss values with 1 segment, or many segment, or you can also perform
    # an optimisation with discrete DN numbers (which is of course
    # computationally more expensive).
    # See also the DHNx example "discrete_DN_numbers"

    # Here follows a linear approximation with 1 segment
    constants_costs = np.polyfit(df['P_max [kW]'], df['Costs [€/m]'], 1)
    constants_loss = np.polyfit(df['P_max [kW]'], df['P_loss [kW/m]'], 1)

    if show_plot:
        # Plot the economic assumptions:
        for constants, col_y in zip(
                [constants_costs, constants_loss],
                ["Costs [€/m]", 'P_loss [kW/m]']
                ):
            x_min = df['P_max [kW]'].min()
            x_max = df['P_max [kW]'].max()
            y_min = constants[0] * x_min + constants[1]
            y_max = constants[0] * x_max + constants[1]

            _, ax = plt.subplots()
            x = df['P_max [kW]']
            y = df[col_y]
            ax.plot(x, y, lw=0, marker="o", label="DN numbers",)
            ax.plot(
                [x_min, x_max], [y_min, y_max],
                ls=":", color='r', marker="x"
            )
            ax.set_xlabel("Transport capacity [kW]")
            ax.set_ylabel(col_y)
            plt.title(
                "Linear approximation of {} in district heating \n"
                "pipelines based on maximum pressure drop "
                "of {:.0f} Pa/m".format(col_y, df["Max delta p [Pa/m]"][0])
            )
            plt.legend()
            plt.ylim(0, None)
            plt.grid(ls=":")
            plt.show()

    return df, constants_costs, constants_loss


def export_lineralized_pipe_input(df, constants_costs, constants_loss,
                                  path_invest_data='invest_data'):
    """Create and export the pipes investment options to the expected location.

    See DHNx documentation for details about these settings.
    """
    df_invest_opt_pipes = pd.DataFrame({
        "label_3": "pipe-generic",
        "active": 1,
        "nonconvex": 1,
        "l_factor": constants_loss[0],
        "l_factor_fix": constants_loss[1],
        "cap_max": df['P_max [kW]'].max(),
        "cap_min": min(1, df['P_max [kW]'].min()),  # 1 kW or smallest DN
        "capex_pipes": constants_costs[0],
        "fix_costs": constants_costs[1],
        }, index=[0],
    )

    if path_invest_data is not None:
        # Export the optimisation parameter of the dhs pipelines to the
        # investment data and replace the default csv file.
        # This file will be read by DHNx and is expected at a specific location
        filepath = os.path.join(path_invest_data, "network/pipes.csv")
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        df_invest_opt_pipes.to_csv(filepath, index=False)

    return df_invest_opt_pipes


def get_default_dhnx_invest_options():
    """Generate a dictionary with the default investments options for DHNx.

    dhnx.input_output.load_invest_options()
    """
    consumers_bus = """
label_2,active,excess,shortage,shortage costs,excess costs
heat,1,0,0,999999,9999
"""
    consumers_demand = """
label_2,active,nominal_value
heat,1,1
"""
    producers_bus = """
,label_2,active,excess,shortage,shortage costs,excess costs
1,heat,1,0,0,9999,9999
"""
    producers_source = """
label_2,active
heat,1
"""

    invest_options = dict(
        consumers=dict(
            bus=pd.read_csv(io.StringIO(consumers_bus)),
            demand=pd.read_csv(io.StringIO(consumers_demand)),
            ),
        producers=dict(
            bus=pd.read_csv(io.StringIO(producers_bus)),
            source=pd.read_csv(io.StringIO(producers_source)),
            )
        )
    return invest_options


def get_default_df_DN(T_FF=80, T_RF=50, T_ground=10, save_path=None):
    """If no pipes data is given, calculate default values here.

    Use derive_dhnx_pipe_invest_options() to update the hardcoded
    coefficients used for U-value and costs from manufacturer data.
    """
    df_DN = pd.DataFrame(
        {'DN': [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300,
                350, 400, 450, 500, 600, 700, 800, 900, 1000]})

    df_DN['Inner diameter [m]'] = df_DN['DN']/1000
    df_DN['Max delta p [Pa/m]'] = 150
    df_DN['Roughness [mm]'] = 0.075  # 0.075 for steel, 0.01 for PEX
    df_DN['T_forward [°C]'] = T_FF  # °C forward flow
    df_DN['T_return [°C]'] = T_RF  # °C return flow
    df_DN['T_mean [°C]'] = (
        (df_DN['T_forward [°C]'] + df_DN['T_return [°C]']) / 2)
    df_DN["T_ground [°C]"] = T_ground

    # Generated by derive_dhnx_pipe_invest_options() with data from the
    # Logstor calculater http://calc.logstor.com/de/energitab/
    # Systemtyp: Pair (not Twin-Pipes, they are only available for DN<100)
    # Rohrsytem: Stahl Konti
    # Serie: 2 (Isolation standard for KMR-Systems)
    constants_costs = [1.23375521e-02, 2.33183815e+00, 1.76722278e+02]
    constants_loss = [6.56173877e-09, -8.03009888e-06,  3.38556713e-03,
                      1.53014666e-01]

    df_DN['Costs [€/m]'] = np.polyval(constants_costs, df_DN['DN'])
    df_DN['U-value [W/mK]'] = np.polyval(constants_loss, df_DN['DN'])

    return df_DN


def derive_dhnx_pipe_invest_options(filename=None):
    """Derive default U-value and cost values from manufacturer data.

    A user should specify their own verified manufacturer data in
    calc_lineralized_pipe_input(). In cases where that is not available,
    some plausible default values should be used. get_default_df_DN()
    provides such a default pipe data table.

    derive_dhnx_pipe_invest_options() can be used to update the hardcoded
    coefficients used in get_default_df_DN()
    """
    df = pd.read_excel(filename,
                       # sheet_name=None,
                       )

    constants_costs = np.polyfit(df['DN'], df['Costs [€/m]'], 2)
    constants_loss = np.polyfit(df['DN'], df['U-value [W/mK]'], 3)

    DN_range = np.linspace(df['DN'].min(), df['DN'].max(), 500)
    costs_fit = np.polyval(constants_costs, DN_range)
    loss_fit = np.polyval(constants_loss, DN_range)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Plot Costs
    axes[0].scatter(df['DN'], df['Costs [€/m]'], color='blue', label='Data')
    axes[0].plot(DN_range, costs_fit, color='red', label='Polynomial Fit')
    axes[0].set_title('Costs vs DN')
    axes[0].set_xlabel('DN')
    axes[0].set_ylabel('Costs [€/m]')
    axes[0].legend()
    axes[0].grid(True)

    # Plot U-value
    axes[1].scatter(df['DN'], df['U-value [W/mK]'], color='blue', label='Data')
    axes[1].plot(DN_range, loss_fit, color='red', label='Polynomial Fit')
    axes[1].set_title('U-value vs DN')
    axes[1].set_xlabel('DN')
    axes[1].set_ylabel('U-value [W/mK]')
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()



def apply_DN(gdf_pipes, df_DN):
    """Apply norm diameter of pipes in gdf_pipes from capacity in df_DN."""
    # Now apply the norm diameter to the pipes dataframe
    gdf_pipes['DN'] = 0  # default for (near) zero capacities

    for idx in gdf_pipes.index:
        capacity = gdf_pipes.loc[idx, 'capacity']

        if capacity > 0.01:
            if capacity > df_DN["P_max [kW]"].max():
                index = df_DN.sort_values(by=["P_max [kW]"],
                                          ascending=False).index[0]
                logger.error('Maximum heat demand exceeds capacity of biggest '
                             'pipe! The biggest pipe type is selected.')
            else:
                index = df_DN[df_DN["P_max [kW]"] >= capacity].sort_values(
                    by=["P_max [kW]"]).index[0]

            gdf_pipes.loc[idx, 'DN'] = df_DN.loc[index, "DN"]

        elif capacity == 0:
            continue

    # Also add other columns from df_DN to gdf_pipes
    cols_select = ['DN', 'Inner diameter [m]', 'Roughness [mm]',
                   'U-value [W/mK]', 'Costs [€/m]', 'T_forward [°C]',
                   'T_return [°C]', 'T_ground [°C]', 'T_mean [°C]',
                   'Max delta p [Pa/m]', 'v_max [m/s]', 'Mass flow [kg/s]',
                   'P_max [kW]', 'P_loss [kW/m]']
    cols_select = [c for c in cols_select if c in df_DN.columns]
    gdf_pipes = gdf_pipes.join(
        df_DN[cols_select].set_index('DN'), on='DN', how='left')

    return gdf_pipes


def get_total_costs_and_losses(
        gdf_pipes, df_DN, f_length_cost=1, f_length_loss=1):
    """Calculate total pipe costs and heat losses from specific values.

    If the columns ``'Costs [€/m]'`` and/or ``'P_loss [kW/m]'`` are present in
    ``df_DN``, their respective total values are calculated for each pipe
    segment in ``gdf_pipes``.

    Please carefully consider the factors ``f_length_cost`` and
    ``f_length_loss``. The length used for calculation is the single length
    of either forward or return pipes ('Trassenmeter') multiplied with this
    factor.
    This is necessary, because the given cost and loss data may describe
    a single actual pipe, of which two are needed for forward and return.
    In this case, the factor needs to be set to ``2``. If the input
    data describes a double pipe and/or relates to 'Trassenmeter', the factor
    ``1`` is appropriate.

    With the default factor 1 for both ``'P_loss [kW/m]'`` and
    ``'Costs [€/m]'``, it is assumed that they already describe the
    losses and costs of forward and return pipe, i.e. are related to
    the trench length.

    If the factors are already present as columns in ``df_DN``, those
    values are used instead.
    """
    # Store the assumed/selected length factor
    if 'f_length_loss' not in df_DN.columns:
        df_DN['f_length_loss'] = f_length_loss
    if 'f_length_cost' not in df_DN.columns:
        df_DN['f_length_cost'] = f_length_cost

    # Rename the values resulting from the linearization
    gdf_pipes.rename(columns={'costs': 'Cost_lin [€]',
                              'losses': 'P_loss_lin [kW]'},
                     inplace=True)
    # Create a temporary df (with more columns than we need)
    _gdf_pipes = gdf_pipes.join(df_DN.set_index('DN'), on='DN', how='left',
                                rsuffix='_r')

    # Calculate the total costs and losses from specifics values and length
    try:
        gdf_pipes['Cost [€]'] = (_gdf_pipes['Costs [€/m]']
                                 * gdf_pipes.length
                                 * _gdf_pipes['f_length_cost'])
    except KeyError as e:
        logger.warning("Cost not calculated due to missing data: %s", e)
    try:
        gdf_pipes['P_loss [kW]'] = (_gdf_pipes['P_loss [kW/m]']
                                    * gdf_pipes.length
                                    * _gdf_pipes['f_length_loss'])
        # Estimate annual thermal losses with constant conditions
        gdf_pipes['E_loss [MWh]'] = gdf_pipes['P_loss [kW]'] * 8760/1000
    except KeyError as e:
        logger.warning("Losses not calculated due to missing data: %s", e)
    return gdf_pipes


def calc_pipes_p_max(df):
    """Calculate maximum capacity of district heating pipes."""
    # Calculate the maximum flow velocity
    df['v_max [m/s]'] = df.apply(
        lambda row: dhnx.optimization.precalc_hydraulic.v_max_bisection(
            d_i=row['Inner diameter [m]'],
            T_average=row['T_mean [°C]'],
            k=row['Roughness [mm]'],
            p_max=row['Max delta p [Pa/m]']), axis=1)

    # Calculate the maximum mass flow per pipe
    df['Mass flow [kg/s]'] = df.apply(
        lambda row: dhnx.optimization.precalc_hydraulic.calc_mass_flow(
            v=row['v_max [m/s]'], di=row['Inner diameter [m]'],
            T_av=row['T_mean [°C]'],
            ), axis=1)

    # Calculate the maximum thermal transport capacity of each DN type
    df['P_max [kW]'] = df.apply(
        lambda row: dhnx.optimization.precalc_hydraulic.calc_power(
            T_vl=row['T_forward [°C]'],
            T_rl=row['T_return [°C]'],
            mf=row['Mass flow [kg/s]']
            ) * 0.001,  # Unit conversion from W to kW
        axis=1)

    try:
        df['P_loss [kW/m]'] = df.apply(
            lambda row: dhnx.optimization.precalc_hydraulic.calc_pipe_loss(
                temp_average=row["T_mean [°C]"],
                u_value=row["U-value [W/mK]"],
                temp_ground=row["T_ground [°C]"],
            ) * 0.001,  # Unit conversion from W to kW
            axis=1,
        )
    except KeyError as e:
        logger.debug("Skip calculation of pipe losses due to missing input "
                     "data: %s", str(e))
    return df


def pandapipes_run(network, gdf_pipes, df_DN=None, show_plot=False,
                   save_path='result_pandapipes',
                   P_th_kW=None,
                   f_length_loss=2,
                   pressure_net=12,  # [bar] (Pressure at the heat supply)
                   pressure_pn=20,  # [bar] The nominal pressure (used as initial value)
                   elevation_col=None,
                   direction='forward',
                   **kwargs,
                   ):
    r"""Run pandapipes simulation with result network from dhnx.

    While DHNx uses a thermal transmittance (U-value) in unit W/mK for
    heat loss calcultion, pandapipes requires a heat transfer coefficient
    input 'alpha_w_per_m2k'

    U-value as given by pipe manufacturers typically describes the W of
    thermal loss per m pipe length and K temperature difference between mean
    operational temperature and external (ground) temperature:

    :math:`\frac{P_{loss}}{l} = (T_{op} - T_{ext}) \cdot U`

    As per documentation of pandapipes the losses are calculated as:

    :math:`P_{loss} = \alpha\cdot l \cdot\pi\cdot d \cdot (T_{op} - T_{ext})`

    It is also mentioned that :math:`d` describes the inner diameter.
    https://pandapipes.readthedocs.io/en/latest/components/pipe/pipe_component.html

    Therefore the area referenced by :math:`\alpha` is determined by the
    diameter :math:`d` and :math:`\alpha` can be calculated by:

    :math:`\alpha = \frac{U}{d\cdot \pi}`

    f_length_loss: It is assumed that the U-value in df_DN describes the total
    losses of forward and return flow, because this is required for DHNx
    to include the total losses in its calculation. In contrast, for the
    pandapipes calculation, we only want to evaluate pressure losses and
    temperature drop along the forward flow. Therefore the U-value in
    df_DN is divided by f_length_loss, i.e. halfed by default.

    """
    import math
    import pandapipes as pp
    from CoolProp.CoolProp import PropsSI

    # Define the pandapipes parameters
    p = pressure_net * 100000  # pressure in [Pa]

    if df_DN is None:  # Use some default values
        dT = 30  # [K]
        feed_temp = 348  # 75 °C (Feed-in temperature at the heat supply)
        ext_temp = 283  # 10 °C (temperature of the ground)
    else:
        # Assume that the following values are the same for all DN types
        dT = (df_DN['T_forward [°C]'] - df_DN['T_return [°C]']).values[0]
        feed_temp = df_DN['T_forward [°C]'].values[0] + 273.15  # K
        ext_temp = df_DN['T_ground [°C]'].values[0] + 273.15  # K

    # Calculate heat transfer coefficient for pandapipes (see docstring above)
    df_DN["alpha [W/m2K]"] = df_DN['U-value [W/mK]'].div(
        df_DN['Inner diameter [m]'] * math.pi * f_length_loss)

    # Get required physical properties of water
    cp = PropsSI('C', 'T', feed_temp, 'P', p, 'IF97::Water') * 0.001  # [kJ/(kg K)]
    d = PropsSI('D', 'T', feed_temp, 'P', p, 'IF97::Water')  # [kg/m³]

    # Prepare the component tables of the DHNx network
    forks = network.components['forks'].copy()
    consumers = network.components['consumers'].copy()
    producers = network.components['producers'].copy()
    pipes = gdf_pipes.copy()

    # elevation_col can be used to indicate which column contains elevation
    # data (height in meters). If it does not exist, assign 0m to all data.
    if elevation_col is None:
        elevation_col = 'height_m'
        forks[elevation_col] = 0
        consumers[elevation_col] = 0
        producers[elevation_col] = 0
    else:  # Perform test for missing values
        for df_test in [forks, consumers, producers]:
            if df_test[elevation_col].isna().any():
                logger.error("Height information from column '%s' has "
                             "missing values", elevation_col)

    # prepare the consumers dataframe
    # calculate massflow for each consumer and producer
    # Do not use the maximum power of each consumer, but a power that
    # considers the simultaneity. This must be provided by the user, either
    # as a name of a column in 'consumers', or as a series (or list, array)
    # with an entry for each consumer
    if P_th_kW is None:
        P_th_kW = 'P_heat_max'
        logger.warning("The pandapipes simulation requires a thermal power "
                       f"for each consumer. Per default, the column {P_th_kW} "
                       "in the consumers table is used. But this only makes "
                       "sense if the network was designed with a "
                       "simultaneity factor = 1, which is rarely the case. "
                       "Instead, please provide the name of a column that "
                       "contains the thermal power including the "
                       "simultaneity, or a list of those values with the "
                       "argument 'P_th_kW'.")
    if isinstance(P_th_kW, str):
        if P_th_kW in consumers.columns:
            P_th_kW = consumers[P_th_kW]
        else:
            raise ValueError(
                "Pandapipes requires a thermal power for each consumer. "
                f"The provided string {P_th_kW} is not found in the columns "
                "of the consumer table")
    elif isinstance(P_th_kW, pd.Series) or isinstance(P_th_kW, pd.DataFrame):
        # The dhnx consumer index is always of dtype string. Catch case where
        # the index of P_th_kW might be int or float and can be converted
        if (P_th_kW.index != consumers.index).any():
            P_th_kW.index = P_th_kW.index.astype(str)
            if (P_th_kW.index != consumers.index).any():
                raise ValueError(
                    "Cannot match the indices of the DHNx consumers and "
                    "the given thermal power for each consumer.")

    consumers['massflow'] = P_th_kW / (cp * dT)  # [kg/s]

    # prepare the pipes dataframe

    # delete pipes with capacity of 0
    pipes = pipes.drop(pipes[pipes['capacity'] == 0].index)
    # reset the index to later on merge the pandapipes results, that
    # do not know an 'id' or 'name' anymore
    idx_name = pipes.index.name
    if idx_name is None:
        idx_name = 'index'
    pipes = pipes.reset_index()

    # Add data of technical data sheet with the DN numbers to the pipes table
    cols_select = ["Inner diameter [m]", "Roughness [mm]",
                   "U-value [W/mK]", "alpha [W/m2K]"]
    cols_select = [col for col in cols_select if col not in pipes.columns]
    if len(cols_select) > 0:
        cols_select.append("DN")
        pipes = pipes.join(df_DN[cols_select].set_index('DN'), on='DN')

    # Create the pandapipes model

    # Now, we create the pandapipes network (pp_net).
    # Note that we only model the forward pipeline system in this example and
    # focus on the pressure losses due to the pipes (no pressure losses e.g.
    # due to expansion bend and so on).
    # However, if we assume the same pressure drop for the return pipes and
    # add a constant value for the substation, we can get a first idea of the
    # hydraulic feasibility of the drafted piping system, and we can check,
    # if the temperature at the consumers is sufficiently high.
    pp_net = pp.create_empty_network(fluid="water")

    for index, fork in forks.iterrows():
        pp.create_junction(
            pp_net, pn_bar=pressure_pn, tfluid_k=feed_temp,
            name=fork['id_full'],
            height_m=fork[elevation_col],
        )

    for index, consumer in consumers.iterrows():
        pp.create_junction(
            pp_net, pn_bar=pressure_pn, tfluid_k=feed_temp,
            name=consumer['id_full'],
            height_m=consumer[elevation_col],
        )

    for index, producer in producers.iterrows():
        pp.create_junction(
            pp_net, pn_bar=pressure_pn, tfluid_k=feed_temp,
            name=producer['id_full'],
            height_m=producer[elevation_col],
        )

    if direction == 'forward':
        # create sink for consumers
        for index, consumer in consumers.iterrows():
            pp.create_sink(
                pp_net,
                junction=pp_net.junction.index[
                    pp_net.junction['name'] == consumer['id_full']][0],
                mdot_kg_per_s=consumer['massflow'],
                name=consumer['id_full']
            )

        # create source for producers
        for index, producer in producers.iterrows():
            pp.create_source(
                pp_net,
                junction=pp_net.junction.index[
                    pp_net.junction['name'] == producer['id_full']][0],
                mdot_kg_per_s=consumers['massflow'].sum(),
                name=producer['id_full']
            )

        # EXTERNAL GRID as slip (Schlupf)
        for index, producer in producers.iterrows():
            pp.create_ext_grid(
                pp_net,
                junction=pp_net.junction.index[
                    pp_net.junction['name'] == producer['id_full']][0],
                p_bar=pressure_net,
                t_k=feed_temp,
                name=producer['id_full'],
            )

    elif direction == 'return':
        # create source for consumers
        for index, consumer in consumers.iterrows():
            pp.create_source(
                pp_net,
                junction=pp_net.junction.index[
                    pp_net.junction['name'] == consumer['id_full']][0],
                mdot_kg_per_s=consumer['massflow'],
                name=consumer['id_full']
            )

        # create sink for producers
        for index, producer in producers.iterrows():
            pp.create_sink(
                pp_net,
                junction=pp_net.junction.index[
                    pp_net.junction['name'] == producer['id_full']][0],
                mdot_kg_per_s=consumers['massflow'].sum(),
                name=producer['id_full']
            )

        # EXTERNAL GRID as slip (Schlupf)
        for index, consumer in consumers.iterrows():
            pp.create_ext_grid(
                pp_net,
                junction=pp_net.junction.index[
                    pp_net.junction['name'] == consumer['id_full']][0],
                p_bar=pressure_net,
                t_k=feed_temp,
                name=consumer['id_full'],
            )

    # create pipes
    # TODO translate to create_pipes_from_parameters()
    for index, pipe in pipes.iterrows():
        pp.create_pipe_from_parameters(
            pp_net,
            from_junction=pp_net.junction.index[
                pp_net.junction['name'] == pipe['from_node']][0],
            to_junction=pp_net.junction.index[
                pp_net.junction['name'] == pipe['to_node']][0],
            length_km=pipe['length'] / 1000,  # convert to km
            diameter_m=pipe["Inner diameter [m]"],
            k_mm=pipe["Roughness [mm]"],
            alpha_w_per_m2k=pipe["alpha [W/m2K]"],
            text_k=ext_temp,
            name=pipe[idx_name],
        )

    # Set default options for pandapipes simulation
    kwargs.setdefault('iter', 10)
    kwargs.setdefault('tol_p', 1e-4)
    kwargs.setdefault('tol_v', 1e-4)
    kwargs.setdefault('stop_condition', "tol")
    kwargs.setdefault('friction_model', "colebrook")
    kwargs.setdefault('nonlinear_method', "automatic")
    kwargs.setdefault('mode', "sequential")
    kwargs.setdefault('transient', False)

    # Execute the pandapipes simulation
    pp.pipeflow(pp_net, **kwargs)

    # print(pp_net.res_junction.head(n=8))
    # print(pp_net.res_pipe.head(n=8))

    # Export results to Excel
    if save_path is not None:
        filepath = os.path.join(save_path, 'pandapipes_result.xlsx')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with pd.ExcelWriter(filepath) as writer:
            pipes.to_excel(
                writer, sheet_name='pipes',
                columns=[idx_name, 'type', 'from_node', 'to_node', 'length',
                         'capacity', 'Cost [€]', 'P_loss [kW]',
                         "Inner diameter [m]", "Roughness [mm]",
                         'U-value [W/mK]', "alpha [W/m2K]", 'DN']
            )
            pp_net.res_pipe.to_excel(writer, sheet_name='pandapipes_pipes')
            pp_net.res_junction.to_excel(writer,
                                         sheet_name='pandapipes_junctions')

    # Merge results of pipes to GeoDataFrame
    pipes = pd.merge(
        pipes, pp_net.res_pipe, left_index=True, right_index=True,
        how='left'
        )
    pipes.set_index(idx_name, inplace=True)  # restore the original index

    # Convert Kelvin to degrees Celsius temperature columns
    pipes['t_from_°C'] = pipes['t_from_k'] - 273.15
    pipes['t_to_°C'] = pipes['t_to_k'] - 273.15
    pp_net.res_junction['t_°C'] = pp_net.res_junction['t_k'] - 273.15

    junctions = pd.merge(
        pp_net.res_junction, pp_net.junction, left_index=True,
        right_index=True, how='left'
        ).set_index(pp_net.res_junction.index)
    forks = pd.merge(
        forks, junctions, left_on='id_full', right_on='name',
        how='left'
        ).set_index(forks.index)
    consumers = pd.merge(
        consumers, junctions, left_on='id_full', right_on='name',
        how='left', suffixes=('', '_pp')
        ).set_index(consumers.index)
    producers = pd.merge(
        producers, junctions, left_on='id_full', right_on='name',
        how='left', suffixes=('', '_pp')
        ).set_index(producers.index)

    if save_path is not None:
        # export the GeoDataFrames with the simulation results to .geojson
        save_geojson(pipes, 'pandapipes_pipes', path=save_path)
        save_geojson(forks, 'pandapipes_forks', path=save_path)
        save_geojson(consumers, 'pandapipes_consumers', path=save_path)
        save_geojson(producers, 'pandapipes_producers', path=save_path)

    # Determine the low pressure and low temperature points ("Schlechtpunkt")
    forks_p_min = forks.loc[[forks['p_bar'].idxmin()]]
    forks_t_min = forks.loc[[forks['t_°C'].idxmin()]]

    # Plot the results of pandapipes simulation
    if show_plot:
        # Plot pressure of pipes' ending nodes
        plot_geometries(
            [consumers,
             producers,
             pipes,
             forks_p_min],
            plt_kwargs=[dict(label='Consumer', color='green'),
                        dict(label='Producer',
                             color=matplotlib.colormaps['cividis'](1.0)),
                        dict(column='p_to_bar', linewidth=2, legend=True,
                             label='Pipelines',
                             cmap='cividis',
                             legend_kwds={'label': 'Pressure [bar]'}),
                        dict(label='Minimum pressure',
                             color=matplotlib.colormaps['cividis'](0.0)),
                        ],
            # plot_basemap=True,
            title='Pressure distribution',
            set_axis_off=True,
            dpi=300,
            # save_path=os.path.join(save_path, 'plots', 'Pressure'),
            )

        # Plot temperature of pipes' ending nodes
        plot_geometries(
            [consumers,
             producers,
             pipes,
             forks_t_min],
            plt_kwargs=[dict(label='Consumer', color='black'),
                        dict(label='Producer',
                             color=matplotlib.colormaps['Wistia'](1.0)),
                        dict(column='t_to_°C', linewidth=2, legend=True,
                             label='Pipelines',
                             cmap='Wistia',
                             legend_kwds={'label': 'Temperature [°C]'}),
                        dict(label='Minimum temperature',
                             color=matplotlib.colormaps['Wistia'](0.0)),
                        ],
            # plot_basemap=True,
            title='Temperature distribution',
            set_axis_off=True,
            dpi=300)

        # Plot volume flow rate per pipe segment
        pipes['vdot_m3_per_s_abs'] = pipes['vdot_m3_per_s'].abs()
        pipes['vdot_m3_per_h_abs'] = pipes['vdot_m3_per_s_abs']*3600
        plot_geometries(
            [consumers,
             producers,
             pipes],
            plt_kwargs=[dict(label='Consumer', color='black'),
                        dict(label='Producer',
                             color=matplotlib.colormaps['Wistia'](1.0)),
                        dict(column='vdot_m3_per_h_abs', linewidth=2,
                             legend=True, label='Pipelines', cmap='Wistia',
                             legend_kwds={'label': 'Flow rate [m³/h]'})
                        ],
            # plot_basemap=True,
            title='Flow rate distribution',
            set_axis_off=True,
            dpi=300)

        # Plot volume flow velocity per pipe segment
        pipes['v_mean_m_per_s_abs'] = pipes['v_mean_m_per_s'].abs()
        plot_geometries(
            [consumers,
             producers,
             pipes],
            plt_kwargs=[dict(label='Consumer', color='black'),
                        dict(label='Producer',
                             color=matplotlib.colormaps['Wistia'](1.0)),
                        dict(column='v_mean_m_per_s_abs', linewidth=2,
                             legend=True, label='Pipelines', cmap='Wistia',
                             legend_kwds={'label': 'Velocity [m/s]'})
                        ],
            # plot_basemap=True,
            title='Velocity distribution',
            set_axis_off=True,
            dpi=300)

        # Find and plot pressure along shorest path from start to point with
        # lowest pressure
        gdf_shortest = find_shortest_path(pipes, producers, forks_p_min)

        plot_geometries(
                [pipes, producers, forks_p_min, gdf_shortest],
                plt_kwargs=[
                    dict(label='Pipelines', linewidth=0.5, color='red'),
                    dict(label='Producer',
                         color=matplotlib.colormaps['viridis'](1.0)),
                    dict(label='Minimum pressure',
                         color=matplotlib.colormaps['viridis'](0.0)),
                    dict(column='p_to_bar', legend=True, cmap='viridis',
                         legend_kwds=dict(label='Pressure [bar]')),],
                title='Pressure distribution to point of minimum pressure',
                set_axis_off=True,
                )

    return pipes, forks, consumers, producers


def find_shortest_path(gdf, point_start, point_end, distance_col='distance',
                       show_plot=False):
    """Find shortest path from point_start to point_end in line network gdf."""
    import networkx as nx
    import momepy

    gdf['length'] = gdf.length
    G = momepy.gdf_to_nx(gdf, approach="primal")
    # nx.draw(G, {n: [n[0], n[1]] for n in list(G.nodes)}, node_size=1)

    # Extract coordinates from point_start and point_end GeoDataFrames
    start = list(point_start.geometry.centroid.iloc[0].coords)[0]
    end = list(point_end.geometry.centroid.iloc[0].coords)[0]

    # Compute the shortest path
    shortest_path_nodes = nx.shortest_path(G, source=start, target=end,
                                           weight='length')

    G_s = nx.subgraph(G, shortest_path_nodes)
    # nx.draw(G_s, {n: [n[0], n[1]] for n in list(G_s.nodes)}, node_size=1)

    # Compute the distance from the start to each node in the shortest path
    cumulative_distances = {node: nx.shortest_path_length(
        G, source=start, target=node, weight='length')
        for node in shortest_path_nodes}

    nx.set_node_attributes(G_s, values=cumulative_distances, name=distance_col)

    nodes_gdf, edges_gdf = momepy.nx_to_gdf(G_s)
    gdf_s = pd.merge(edges_gdf, nodes_gdf[['nodeID', distance_col]],
                     left_on='node_start', right_on='nodeID', how='left')
    gdf_s = gdf_s.sort_values(distance_col).reset_index()

    if show_plot:
        plot_geometries(
            [gdf, point_start, point_end, gdf_s],
            plt_kwargs=[
                dict(label='Network', color='red'),
                dict(label='Start'),
                dict(label='End'),
                dict(column=distance_col, legend=True,
                     legend_kwds=dict(label='Cumulative distance [m]')),],
            )

    # Test if the total length of the shortest path is all within gdf
    # equal_distances = np.isclose(
    #     gdf[gdf.within(gdf_s.geometry.union_all())].length.sum(),
    #     gdf_s['distance'].max(),
    #     atol=1)

    # if not equal_distances:
    #     logger.warning("There might be a problem with the shortest path")

    return gdf_s


# Section "lpagg" (load profile aggregator)
def find_TRY_regions(gdf, show_plot=False, buffer=2000, col_try='try_code'):
    """Find DWD typical reference year (TRY) region of objects in gdf."""
    import lpagg.misc

    # Find the TRY-region for each building
    gdf_TRY = lpagg.misc.get_TRY_polygons_GeoDataFrame(col_try)

    gdf = tobler.area_weighted.area_join(
        gdf_TRY.to_crs(gdf.crs), gdf, [col_try])

    if len(gdf[col_try].unique()) > 1:
        logger.info("Buildings are located in multiple TRY-regions.")

    if gdf[col_try].isna().any():
        logger.warning("Some buildings are not located in any TRY-region.")
        plot_geometries(
            gdf.fillna({col_try: 'TRY Missing'}),
            plt_kwargs=[
                dict(column=col_try, legend=True),
                ],
            title='TRY Regions',
            # plot_basemap=True,
            set_axis_off=True)

    if show_plot:
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        gdf_TRY.to_crs(gdf.crs).plot(ax=ax, column=col_try, legend=True)
        gdf.buffer(buffer).plot(ax=ax, color='red')
        plt.title('Building location and TRY-Regions')

    return gdf


def find_country_code(gdf):
    """Get the codes of country and province as input for holidays module.

    This service requires an internet connection.
    See https://en.wikipedia.org/wiki/ISO_3166-2:DE for reference of
    country codes.

    Parameters
    ----------
    gdf : GeoDataFrame
        The centroid of the convex hull of all geometries in the input
        GeoDataFrame is used as the location for which to receive the
        codes of country and province.

    Returns
    -------
    country_dict : dict
        A dictionary containing ISO 3166-2 codes of country and province.

    """
    try:
        import geopy
        geolocator = geopy.geocoders.Nominatim(user_agent='dhnx_addons')
        point = gdf.to_crs(4326).union_all().convex_hull.centroid
        location = geolocator.reverse((point.y, point.x)).raw
        country_code = location['address']['ISO3166-2-lvl4']
        country_dict = {'country': country_code[0:2],
                        'province': country_code[3:5]}
    except Exception as e:
        logger.error("Finding the country code failed, using Germany instead.")
        logger.error(e)
        country_dict = {'country': 'DE', 'province': None}

    return country_dict


def lpagg_prepare_cfg(gdf, sigma=0, show_plot=False,
                      col_building_type='building_type',
                      col_N_pers='N_pers',
                      col_N_flats='N_flats',
                      col_try='try_code',
                      col_heat='e_th_heat_kWh',
                      col_DHW='e_th_DHW_kWh',
                      weather_file=None,
                      house_type_replacements={
                          'SFH': 'EFH',
                          'MFH': 'MFH',
                      },
                      **cfg_kwargs,
                      ):
    """Prepare a table of building data for use in lpagg.

    The index of gdf will define the name of the buildings.

    lpagg uses the nomenclature from VDI 4655. Therefore the following
    columns need to be created or renamed:

    'Q_Heiz_a': 0,  # Set VDI4655 profile to zero, use external profile
    'Q_Kalt_a': None,  # Cooling is not used
    'Q_TWW_a': Q_TWW_a,  # use VDI4655 for hot water
    'W_a': W_a,
    'house_type': house_type,
    'N_Pers': N_Pers,
    'N_WE': N_WE,
    'copies': 1,  # Each building is only used 1 time
    'sigma': sigma,
    'TRY': TRY,
    """
    import lpagg
    if col_try not in gdf.columns:
        gdf = find_TRY_regions(gdf, show_plot=show_plot, col_try=col_try)
    TRY_list = gdf[col_try].unique()
    TRY = TRY_list[0]
    country_dict = find_country_code(gdf)
    df_lpagg = gdf.copy()

    rename_dict = {
        col_heat: 'Q_Heiz_a',  # space heating
        col_DHW: 'Q_TWW_a',  # domestic hot water
        'E_el_kWh': 'W_a',  # electricity
        col_building_type: 'house_type',
        col_N_pers: 'N_Pers',
        col_N_flats: 'N_WE',
        col_try: 'TRY',
        }
    df_lpagg.rename(columns=rename_dict, inplace=True)
    df_lpagg.replace({'house_type': house_type_replacements}, inplace=True)

    for col in ['N_Pers', 'N_WE', 'W_a']:
        if col not in df_lpagg.columns:
            # Number of persons and number of flats is required for VDI4655
            # Lpagg will fill in default values if we do not do it here.
            # Try set_n_persons_and_flats() to fill with default values
            df_lpagg[col] = float('nan')

    df_lpagg = df_lpagg[list(rename_dict.values())]

    # Set some more columns:
    df_lpagg['Q_Kalt_a'] = None  # Cooling is not used
    df_lpagg['copies'] = 1  # Each building is only considered once (no copies)
    df_lpagg['sigma'] = sigma  # Standard deviation for simultaneity shift

    houses_dict = df_lpagg.to_dict(orient='index')

    if weather_file is None:
        # Use a regular DWD TRY-2010 weather file
        weather_file = os.path.join(
            os.path.dirname(lpagg.__file__),
            "resources_weather",
            "TRY2010_{:02d}_Jahr.dat".format(int(TRY)))
        logger.info("Using default DWD weather file for TRY region %s", TRY)

    settings = cfg_kwargs
    # Settings for weather data
    settings.setdefault('weather_file', weather_file)
    settings.setdefault('weather_data_type', 'DWD')
    # Overwrite the default folder where print_file is saved
    settings.setdefault('print_file', 'load_profile.dat')
    settings.setdefault('result_folder', './LPagg_Result')
    # Overwrite the default column names
    settings.setdefault('rename_columns', {'Q_Heiz_TT': 'E_th_RH',
                                           'Q_Kalt_TT': 'E_th_KL',
                                           'Q_TWW_TT': 'E_th_TWE',
                                           'Q_loss': 'E_th_loss',
                                           'W_TT': 'E_el',
                                           })
    # Which columns do you want to include in the printed file?
    settings.setdefault('print_columns',
                        ['HOUR', 'E_th_RH_HH', 'E_th_TWE_HH', 'E_el_HH',
                         'E_th_RH_GHD', 'E_th_TWE_GHD', 'E_el_GHD'])
    # Do you want to print the index (i.e. time stamp for each row)?
    # Set this to 'False' for TRNSYS (It cannot handle the datetime index)
    settings.setdefault('print_index', False)
    # Display a plot of the energy demand types on screen when finished
    settings.setdefault('show_plot', show_plot)
    # Save plots if defined. Available: ['png', 'svg', 'pdf']
    settings.setdefault('save_plot_filetypes', ['svg'])
    # Time step used for interpolation, e.g. '15 minutes' or '1 hours'
    settings.setdefault('intervall', '1 hours')
    # Start and end date & time for interpolation
    settings.setdefault('start', pd.Timestamp('2018-01-01 00:00:00'))
    settings.setdefault('end', pd.Timestamp('2019-01-01 00:00:00'))
    # Account for daylight saving time (DST)
    settings.setdefault('apply_DST', True)
    # The VDI 4655 default heat limit is 15°C (definition of summer days).
    # For modern building types, this can be set to a lower value e.g. 12°C
    settings.setdefault('Tamb_heat_limit', 15)
    # In addition to the Tamb_heat_limit, you may overwrite the VDI4655 and
    # set heat demand on all summer days to zero:
    settings.setdefault('zero_summer_heat_demand', False)
    # Holidays are treated as sundays. You can select a country and
    # (optionally) a province: https://pypi.org/project/holidays/
    # BW,BY,BYP,BE,BB,HB,HH,HE,MV,NI,NW,RP,SL,SN,ST,SH,TH
    settings.setdefault('holidays', country_dict)
    # Use daily mean as constant value for domestic hot water
    settings.setdefault('flatten_daily_TWE', False)
    # Activate the debugging flag to display more info in the terminal
    settings.setdefault('log_level', 'info')
    # Print *_houses.dat / *_houses.xlsx file.
    # Be careful, can create large file sizes that take long to save!
    settings.setdefault('print_houses_dat', False)
    settings.setdefault('print_houses_xlsx', False)
    # Calculate peak power
    settings.setdefault('calc_P_max', True)  # needed for dhnx_addons
    # Print peak thermal power to separate file
    settings.setdefault('print_P_max', False)
    settings.setdefault('print_GLF_stats', True)
    # Language used for certain plots ('de', or 'en')
    settings.setdefault('language', 'en')

    cfg = dict(settings=settings,
               houses=houses_dict)

    return df_lpagg, cfg


def cache_validation_cb(metadata):
    """Only retrieve cached results if use_cache is True.

    Use in a function decorator:
    @memory.cache(cache_validation_callback=cache_validation_cb)

    Is taken from the documentation, but not available in my version of joblib.
    """
    return metadata['input_args'].get('use_cache', True)


# @memory.cache  # Useful if it works, but can cause issues
def lpagg_run(gdf, sigma=0, E_th_col='E_th_total_kWh', show_plot=True,
              power_slice_buffer=0, **cfg_kwargs):
    """Replace the __main__.py script from the regular LPagg program.

    This function uses a 'cache': The results from a run are stored in
    a local folder and reused for as long as the input stays the same.
    This can save a lot of time for repeated runs.
    To enforce clearing the cache and running the function again, run
    the following line of code before calling this function:

    ``lpagg_run.clear()``

    TODO: Eliminate argument E_th_col and get total heat from lpagg
    settings instead

    """
    import lpagg.agg
    import lpagg.misc
    lpagg.misc.setup()

    if cfg_kwargs.get('use_demandlib', False) == 'auto':
        try:  # if import is successfull, use demandlib
            from demandlib import vdi
            cfg_kwargs['use_demandlib'] = True
        except ImportError:  # otherwise, do not use demandlib
            cfg_kwargs['use_demandlib'] = False

    df_lpagg, cfg = lpagg_prepare_cfg(
        gdf, sigma=sigma, show_plot=show_plot, **cfg_kwargs)

    #  Import the cfg from the YAML config_file
    cfg = lpagg.agg.perform_configuration(cfg=cfg, ignore_errors=False)

    # Store the current configuration
    if not os.path.exists(cfg['print_folder']):
        os.makedirs(cfg['print_folder'])
    with open(os.path.join(cfg['print_folder'], 'lpagg_cfg.yaml'), 'w') as f:
        yaml.dump(cfg['settings'], f, default_flow_style=False)

    # Now let the aggregator do its job
    agg_dict = lpagg.agg.aggregator_run(cfg)

    # Plot and print the results
    lpagg.agg.plot_and_print(agg_dict['weather_data'], cfg)

    # Postprocessing
    df_load_ts_slice = lpagg_get_max_power_slice(
        agg_dict['load_curve_houses'],
        buffer=power_slice_buffer,
        show_plot=show_plot)
    gdf = lpagg_merge_houses_and_load(gdf, agg_dict['P_max_houses'],
                                      E_th_col=E_th_col)

    return gdf, df_load_ts_slice


def lpagg_get_max_power_slice(df_load_ts, buffer=0, show_plot=True):
    """Get the time slice with the maximum thermal power from lpagg.

    This time slice can be used as input for dhnx.

    Parameters:
        buffer (float): Hours of "buffer" time to create around each peak time
    """
    from pandas.tseries.frequencies import to_offset

    freq = to_offset(pd.infer_freq(df_load_ts.index)) / pd.Timedelta('1 hours')

    # Take the sum of all the thermal energies (so drop electrical)
    df_load_ts = (df_load_ts
                  .stack(["house"], future_stack=True)
                  .drop(columns=['W_TT'], level='energy', errors='ignore')
                  .sum(axis='columns')
                  .div(freq)  # convert kWh to kW
                  .unstack(["house"])
                  .rename_axis(columns=None)  # remove label 'house'
                  )

    # Choose time slice with maximum power
    # Select a time "buffer" around the maximum moment for each building
    df_idxmax = pd.DataFrame()
    df_idxmax['time'] = pd.to_datetime(df_load_ts.idxmax())
    df_idxmax['start'] = df_idxmax['time'] - pd.Timedelta(hours=buffer)
    df_idxmax['end'] = df_idxmax['time'] + pd.Timedelta(hours=buffer)

    # Combine all individual slices and drop the duplicate time steps
    slice_list = []
    for idx in df_idxmax.itertuples():
        slice_list.append(df_load_ts.loc[str(idx.start):str(idx.end)])
    df_load_ts_slice = (pd.concat(slice_list)
                        .drop_duplicates()
                        .sort_index()
                        )

    if show_plot:
        fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
        df_load_ts.sum(axis='columns').plot(ax=ax)
        df_load_ts.plot(ax=ax, legend=False)
        ax.set_ylabel("Thermal power [kW]")
        plt.title('Sum and individual thermal loads')
        plt.show()

        fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
        df_load_ts_slice.sum(axis='columns').plot(ax=ax, label='sum')
        df_load_ts_slice.plot(ax=ax, legend=False)
        ax.set_ylabel("Thermal power [kW]")
        plt.title('Sum and individual thermal loads (time selection)')
        plt.show()

    # df_load_ts.max().sum()
    # df_load_ts_slice.max().sum()

    return df_load_ts_slice


def lpagg_merge_houses_and_load(
        df_houses, df_load, E_th_col='E_th_total_kWh'):
    """Merge lpagg peak load result back with houses (Geo)DataFrame.

    Assumes that the index defines the house name in both DataFrames.
    Also calculates full use hours.
    """
    df_houses = df_houses.merge(df_load, left_index=True, right_index=True)

    # Set a column P_heat_max for using a single value in DHNx
    df_houses['P_heat_max'] = df_houses['P_th']  # for DHNX

    df_houses['Vbh_th'] = (df_houses[E_th_col] / df_houses['P_th'])

    logger.debug('Thermal energy demand: %s MWh',
                 df_houses[E_th_col].sum()/1000)

    return df_houses


def download_elevation_data(
        gdf, crs=None, ext='geotiff', path='./cache/wms_dgm200_inspire',
        show_plot=False):
    """Download elevation data for a region defined by the bounds of gdf.

    Returns:
        gdf_elevation (geopandas GeoDataFrame): A GeoDataFrame containing the
        elevation data as raster.
    """
    import rasterio
    from rasterio.features import shapes
    from owslib.wms import WebMapService

    if crs is None:
        crs = gdf.crs

    # Connect to the WebMapService
    url = 'https://sgx.geodatenzentrum.de/wms_dgm200_inspire'
    wms = WebMapService(url, version='1.3.0')

    # Specify the layer and style, by reading it from the source
    layer = list(wms.contents)[0]  # 'EL.GridCoverage'
    style = list(wms[layer].styles.keys())[0]  # 'default'

    if crs not in wms[layer].crsOptions:
        crs_fallback = 'EPSG:4326'  # WGS 84
        logger.warning("Selected CRS '%s' not supported, using '%s' instead",
                       crs, crs_fallback)
        crs = crs_fallback

    # Convert polygon to a bounding box
    bbox = gdf.to_crs(crs).geometry.total_bounds

    # Calculate the width and height of the bounding box in meters
    bbox_m = gdf.to_crs('EPSG:25832').geometry.total_bounds
    width_m = bbox_m[2] - bbox_m[0]
    height_m = bbox_m[3] - bbox_m[1]

    # Calculate the number of pixels in each dimension
    # Data source has a resolution of 200m, so to be save use 100m
    size = (int(round(width_m / 100, 0)), int(round(height_m / 100, 0)))

    # Other available information
    # wms.identification.type
    # list(wms.contents)
    # wms[layer].boundingBoxWGS84

    # Select the image format from the given file extension
    format_ = 'image/'+ext
    if format_ not in wms.getOperationByName('GetMap').formatOptions:
        logger.warning("Extension '%s' not supported", ext)

    # Make the request
    logger.debug("Downloading: %s", wms.identification.title)
    response = wms.getmap(layers=[layer], styles=[style], bbox=bbox,
                          srs=str(crs), format=format_,
                          size=size, transparent=True)

    # Write the response to a file
    filepath = path + '.' + ext
    logger.info("Saving downloaded elevation data to %s", filepath)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        f.write(response.read())

    # Open the file with rasterio and plot it
    with rasterio.open(filepath) as src:
        image = src.read(1)  # read the image

    # Create a GeoDataFrame from the results
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(image, transform=src.transform))
        )
    gdf_elevation = gpd.GeoDataFrame.from_features(list(results), crs=crs)

    if show_plot:
        plot_geometries([gdf_elevation, gdf],
                        plt_kwargs=[
                            dict(column='raster_val', legend=True,
                                 legend_kwds=dict(label='Elevation [m]')),
                            dict(label='Area', alpha=0.5)],
                        title='Elevation data',
                        set_axis_off=True,
                        )

    return gdf_elevation


# Section with experimental / broken functions
def QGIS_test_python():
    """Test for running QGIS scripts from python."""
    import sys
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\bin')
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\qt5\bin')
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\qgis\bin')
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\qgis\python')
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\qgis\python\qgis')
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\qgis\python\qgis\core')
    sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\qgis\python\plugins')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39\DLLs')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39\lib')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39\lib\site-packages')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39\lib\site-packages\win32')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39\lib\site-packages\win32\lib')
    # sys.path.append(r'C:\Program Files\QGIS 3.28.1\apps\Python39\lib\site-packages\Pythonwin')
    import qgis
    import processing
    import qgis.core

    from qgis.analysis import QgsNativeAlgorithms


    sys.path.append([
        # 'C:/PROGRA~1/QGIS32~2.1/apps/qgis/./python',
        # 'C:/Users/Nettelstroth/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python',
        # 'C:/Users/Nettelstroth/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python/plugins',
        # 'C:/PROGRA~1/QGIS32~2.1/apps/qgis/./python/plugins',
        # 'C:\\Program Files\\QGIS 3.28.1\\bin\\python39.zip',
        # 'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39\\DLLs',
        # 'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39\\lib',
        # 'C:\\Program Files\\QGIS 3.28.1\\bin',
        # 'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39',
        'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39\\lib\\site-packages',
        'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39\\lib\\site-packages\\win32',
        'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39\\lib\\site-packages\\win32\\lib',
        'C:\\PROGRA~1\\QGIS32~2.1\\apps\\Python39\\lib\\site-packages\\Pythonwin',
        'C:/Users/Nettelstroth/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python',
        'C:\\Users/Nettelstroth/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python/plugins\\qgis2web',
        'C:\\Users\\Nettelstroth\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins',
        '.',
        'C:/Users/Nettelstroth/code_projects/SIZ145_FirstTin/Qgis'])

    processing.run(
        "native:setzfromraster",
        {'INPUT':'C:/Users/Nettelstroth/code_projects/SIZ145_FirstTin/python/alkis/pipes_result.geojson',
         'RASTER':'C:/Users/Nettelstroth/code_projects/SIZ145_FirstTin/Qgis/Layer/DGM/dgm200.utm32s.geotiff/dgm200/dgm200_utm32s.tif',
         'BAND':1,'NODATA':0,'SCALE':1,'OFFSET':0,

         'OUTPUT':'C:/Users/Nettelstroth/code_projects/bs-wärmeplan/src/Tiles/test.geojson'})


def qgis_drape(path_input_layer, path_input_raster, path_output):
    """Run QGIS algorithm 'drape (set Z value from raster)'.

    This algorithm sets the z value of every vertex in the feature
    geometry to a value sampled from a band within a raster layer.
    The raster values can optionally be scaled by a preset amount
    and an offset can be algebraically added.

    Afterwards in QGIS, the expression ``length3D($geometry)`` can be used
    to calculate the line lengths in 3-dimensional space.

    https://opensourceoptions.com/blog/pyqgis-calculate-geometry-and-field-values-with-the-qgis-python-api/


    path_input_layer (str):
        Path to e.g. a file like pipes_result.geojson

    path_input_raster (str):
        Path to a raster file containing the height information, e.g.
        a file like dgm200_utm32s.tif

    path_output (str):
        Path to the output file, e.g. pipes_result_draped.geojson

    """
    import subprocess

    subprocess.check_output([
        r'C:\Program Files\QGIS 3.28.1\apps\qgis\bin\qgis_process.exe',
        'run',
        'native:setzfromraster',
        '--distance_units=meters',
        '--area_units=m2',
        '--ellipsoid=EPSG:7019',
        f'--INPUT={path_input_layer}',
        f'--RASTER={path_input_raster}',
        '--BAND=1',
        '--NODATA=0',
        '--SCALE=1',
        '--OFFSET=0',
        f'--OUTPUT={path_output}',
        ])


if __name__ == '__main__':
    main()
