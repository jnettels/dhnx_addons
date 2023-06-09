"""Collection of generalized functions for ALKIS and OpenStreetMap data.

Special requirements:

conda install libpysal fiona osmnx

conda install tobler>=0.8.0 -c conda-forge

conda install h3-py  # optional dependency of tobler

conda install contextily -c conda-forge

dhnx: https://github.com/oemof/DHNx


Responsible for
"OSError: exception: access violation reading 0xFFFFFFFFFFFFFFFF"
pygeos-0.12.0 (comes with tobler-0.9.0)

TODO
Neue Quelle für Wärmebedarfe ohne Trinkwasser
http://energieberatung.ibs-hlk.de/eb_begr.htm

DIN 18599 Teil 10 für Trinkwarmwasser

Interesting blog post about 3D data
http://bbonczak.com/posts/multipatch-conversion/


General workflows:

ALKIS "FUNKTION" tags can be translated to OpenStreetMap "building" tags.
Then those are grouped into building types that have a defined specific
heat demand.


Other related projects:
    - https://simstadt.hft-stuttgart.de/


Common errors:

"Cannot mix incompatible Qt library (5.15.8) with this library (5.15.6)"

Qt is a library for graphical user interfaces.
This error can occur when you have separate conda environments with
different versions of the package "PyQt5". It should be solved by
making sure your new environments use the same version as your base
environment.
In the example above, we can run the command
"conda list qt"
to find the actual package name that causes the issue and then
"conda install qt-main==5.15.6"


"from rasterio._version import gdal_version, get_geos_version, get_proj_version
ImportError: DLL load failed while importing _version:
    Die angegebene Prozedur wurde nicht gefunden."

Solution: "import osgeo" before rasterio is imported by fiona or contextily.
See https://gis.stackexchange.com/a/450445/135438


"""
import os
import re
import io
import json
import logging
from pkg_resources import parse_version
import numpy as np
import shapely
if parse_version(shapely.__version__) >= parse_version("2.0"):
    # There is a weird dll import error that occurs either if osgeo is
    # imported or not, and it seems to be related to the shapely version
    import osgeo  # import before geopandas fixes issue with rasterio, fiona

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    from . import cbc_installer  # local import
except ImportError:
    import cbc_installer  # local import for running dhnx_addons.py

# Define the logging function
logger = logging.getLogger(__name__)

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
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'dhnx' can be installed with "
                   "pip install dhnx==0.0.3")
if parse_version(dhnx.__version__) != parse_version("0.0.3"):
    raise ImportError(f"Installed dhnx version ({dhnx.__version__}"
                      ") is different from the tested version (0.0.3)")

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


def main():
    """Run an example main method."""
    setup()
    # workflow_example_openstreetmap()
    workflow_example_openstreetmap(show_plot=False)


def setup(log_level='INFO'):
    """Set up the logger."""
    logger.setLevel(level=log_level.upper())  # Logger for this module
    # logging.getLogger('osmnx').setLevel(level='ERROR')
    # logging.getLogger('dhnx').setLevel(level='ERROR')
    logging.basicConfig(format='%(asctime)-15s %(levelname)-8s %(message)s')


def workflow_example_openstreetmap(
        gdf_area=None,
        gdf_prod=None,
        show_plot=True):
    """Run an example workflow from OpenStreetMap data."""
    if gdf_area is None:
        gdf_area = load_example_area()

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
    gdf_houses, df_load_ts_slice = lpagg_run(
        gdf_houses,
        sigma=3,
        E_th_col='e_th_total_kWh',
        result_folder='./lpagg_result',
        print_file='load.dat',
        # intervall='15 minutes',
        show_plot=show_plot,
        print_columns=['HOUR', 'E_th_RH_HH', 'E_th_TWE_HH', 'E_el_HH',
                       'E_th_RH_GHD', 'E_th_TWE_GHD', 'E_el_GHD'],
        house_type_replacements={
            'SFH': 'EFH',
            'MFH': 'MFH',
            'business': 'G1G',
            'other-heated-non-residential': 'G1G',
        },
        use_demandlib="auto",
        # use_demandlib=False,
        )
    save_path = './dhnx_result'

    save_geojson(gdf_houses, 'consumers', path=save_path, save_excel=True)
    save_geojson(gdf_prod, 'producers', path=save_path)

    gdf_streets = download_streets_from_osm(gdf_area)

    gdf_pipes, df_pipes = run_dhnx(
        gdf_streets, gdf_prod, gdf_houses,
        save_path=save_path,
        show_plot=show_plot,
        path_invest_data='invest_data',
        # df_load_ts_slice=None,
        df_load_ts_slice=df_load_ts_slice,
        col_p_th=None,
        # col_p_th='p_th_guess_kW',
        # col_p_th='P_heat_max',
        simultaneity=0.8,
        reset_index=False,
        method='boundary',
        solver=None,
        solver_cmdline_options={  # gurobi
            # 'MIPGapAbs': 1e-5,  # (absolute gap) default: 1e-10
            # 'MIPGap': 0.03,  # (0.2 = 20% gap) default: 0
            # 'TimeLimit': 60 * 1,  # (seconds of maximum runtime)
            # 'TimeLimit': 60 * 1 * 1,  # (seconds of maximum runtime) (gurobi)
            'TimeLimit': 60 * 5 * 1,  # (seconds of maximum runtime) (gurobi)
            'seconds': 60 * 10 * 1,  # (seconds of maximum runtime) (cbc)
            # 'TimeLimit': 60 * 60 * 6,  # (seconds of maximum runtime)
        },
        )

    if show_plot:
        plot_geometries([gdf_houses, gdf_prod, gdf_pipes], plot_basemap=True)

    print(gdf_houses)
    print(gdf_pipes)


def workflow_default(buildings, show_plot=True):
    """Run all the functions with default values in correct order.

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
    buildings = set_domestic_hot_water_from_DIN18599(buildings)
    buildings = separate_heating_and_DHW(buildings)
    buildings = guess_thermal_power_from_full_load_hours(buildings)
    buildings = set_n_persons_and_flats(buildings)
    log_statistics(buildings, show_plot=show_plot)

    gdf_hex = create_hexgrid(buildings, clip=False, show_plot=show_plot,
                             resolution=11, buffer_distance=100,
                             plot_basemap=True,
                             )

    return buildings


def load_example_area(crs='epsg:4647'):
    # Define a bounding box polygon from a list of lat/lon coordinates
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

def load_xml_geodata(file, layer=None, driver='GML', crs="EPSG:25833"):
    """Load (3D) geographical data from XML files (citygml).

    This should not be necessary, since gpd.read_file(file) should do the
    same. But somehow this approach works, using fiona directly, while
    geopandas fails for some xml files.
    """
    with fiona.open(file, 'r', driver=driver, layer=layer) as src:
        features = [feature for feature in src]
    gdf = gpd.GeoDataFrame.from_features([feature for feature in features],
                                         crs=crs)
    return gdf


def save_geojson(gdf, file, path='.', crs=None, type_errors='raise',
                 save_excel=False):
    """Save a gdf to geojson file at the given path.

    For large datasets, consider using save_sql() instead, which yields
    better performance in e.g. QGIS.

    crs "EPSG:4647" is recommended for good length calculation results.

    Parameters
    ----------
    gdf : GeoDataFrame
        Data to save.
    file : str
        File name to save to.
    path : str, optional
        File path to save to. The default is '.'.
    crs : pyproj.CRS, optional
        Coordinate reference system to use for the saved file.
        The default is None. In this case the crs of gdf is not changed.
    type_errors : str, optional
        The GeoPandas to_file() method can sometimes cause TypeErrors.
        If ‘raise’, then invalid parsing will raise an exception.
        If ‘coerce’, then invalid columns will be converted to string.
        The default is 'raise'.
    save_excel : bool, optional
        If True, also save an Excel file with the same name. The default
        is False.

    Returns
    -------
    None.

    """
    if not os.path.exists(path):
        os.makedirs(path)

    if crs is None:
        if gdf.crs is None:
            crs = "EPSG:4647"  # 'backup' crs
        else:
            crs = gdf.crs
    try:
        logger.info('Saving... %s', os.path.join(path, file+'.geojson'))
        if gdf.crs is not None and gdf.crs != crs:
            gdf.to_crs(crs=crs, inplace=True)

        gdf.to_file(os.path.join(path, file+'.geojson'), driver='GeoJSON',
                    crs=crs)
    except PermissionError:
        try:
            input("Please close QGIS to allow saving the file '{}.geojon'. "
                  "Then hit Enter, or CTRL+C to abort.\n".format(file))
            save_geojson(gdf, file, path=path, crs=crs,
                         type_errors=type_errors, save_excel=save_excel)
        except KeyboardInterrupt:
            logger.info('Saving %s.geojson skipped!', file)
    except ValueError as e:
        # This may be an error like:
        # "ValueError: Invalid field type <class 'numpy.int32'>"
        # Test which column causes the error
        for col in gdf.columns:
            if col in gdf.select_dtypes('geometry').columns:
                continue  # Do not mess with the geometry column
            try:
                json.dumps(gdf[col].tolist())  # Convert the column to JSON
            except TypeError as e2:
                if type_errors == 'raise':
                    raise TypeError(
                        f"Error when converting column '{col}' to JSON: "
                        f"'{e2}'. You may try the option save_geojson(..., "
                        "type_errors='coerce') to attempt fixing this "
                        "by forcing conversion to string") from e
                else:
                    logger.warning("Converting column '%s' to string to "
                                   "avoid error: %s", col, e2)
                    gdf[col] = gdf[col].astype(str)

        if type_errors == 'coerce':
            # All errors should have been fixed now, so try saving again
            # Pass type_errors='raise' to avoid getting caught in a loop
            save_geojson(gdf, file, path=path, crs=crs,
                         type_errors='raise', save_excel=save_excel)
    else:  # Execute when there is no error
        if save_excel:
            _save_excel(gdf, os.path.join(path, file+'.xlsx'))


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


def _save_excel(df, path):
    """Save (Geo-)DataFrame as Excel (alias function)."""
    save_excel(df, path)


def save_excel(df, path):
    """Save (Geo-)DataFrame as Excel file (without 'geometry' column).

    Can also be automatically called by save_geojson().

    Just a wrapper around 'df.to_excel(path)' that creates the directory
    and asks to close the file in case of permission error.
    """
    if 'geometry' in df.columns:
        df_save = df.drop(columns='geometry')
    else:
        df_save = df.copy()

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    try:
        logger.info('Saving... %s', path)
        df_save.to_excel(path)
    except PermissionError:
        input("Please close the file to allow saving! Then hit Enter.")
        save_excel(df_save, path)


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
        # 'BAUWERK_AX_VorratsbehaelterSpeicherbauwerk': '',
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
        # 'Schornstein im Gebäude': '',
        # 'Schornstein, Schlot, Esse': '',
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
        'Wassertutm': 'water_tower',
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
        buildings[col_building_osm].where(~mask, buildings[col_alkis_function],
                                          inplace=True)
        # Replace the values in "building_osm"
        buildings[[col_building_osm]] = buildings[[col_building_osm]].replace(
            {col_building_osm: alkis_to_osm_building})

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

    Args:
        gdf (GeoDataFrame): A GeoDataFrame of the buildings

        col_building_osm (str): Column name storing the building type

        col_heated (str): Column name used for an additional filter. Only
        heated buildings (column value == True) are assigned a building type.

        area_threshold (float): Area in m² that should separate houses
        and appartments

        discard_types (list): List of keys in the column 'col_building_osm'
        that will be discarded and replaced by the new types. Default
        is None, but recommendation is 'yes'. This generic building tag
        may make further processing problamatic otherwise.

        notna_columns (list, optional):
        List of names of columns that must not have
        missing (na) values. Common example: Entries for street name and house
        number are required for a building to be considered heated.
        For OpenStreetMap data, use e.g. ['addr:street', 'addr:housenumber'].

    """
    logger.info('Fill random osm building types')

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
        mask_notna = True

    if col_heated is not None:
        mask_heated = gdf[col_heated] == True
    else:
        mask_heated = [True]*len(gdf)

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
        'SFH': ['house', 'residential', 'detached', 'semidetached_house'],
        'MFH': ['apartments'],
        'business': ['commercial', 'office', 'retail'],
        'other-heated-non-residential':
            ['school', 'college', 'university', 'fire_station', 'government',
             'civic', 'industrial', 'public', 'hotel', 'hospital', 'museum',
             'sports_hall', 'kindergarten', 'warehouse',
             ],
        'non-heated':
            ['roof', 'tower', 'hangar', 'train_station', 'bridge', 'barn',
             'cemetery', 'mast', 'garage', 'garages', 'stadium', 'church',
             'chapel', 'parking', 'castle', 'hut', 'silo', 'storage_tank',
             'greenhouse', 'water_tower', 'grandstand', 'roof', 'stadium',
             'shed', 'service', 'carport'],
        }

    # The user needs to decide how to handle the building tag 'yes'
    # It is appended to whichever type they choose.
    if assign_key_yes in translate_dict.keys():
        translate_dict[assign_key_yes].append('yes')
    else:
        translate_dict[assign_key_yes] = ['yes']

    for b_type, b_list in translate_dict.items():
        gdf.loc[gdf[col_building_osm].isin(b_list), col_building_type] = b_type

    return gdf


def assign_random_construction_classification(
        gdf,
        col_construction_year='construction_year',
        col_refurbished_state='refurbished_state',
        refurbished_weights=[0.6, 0.3, 0.1],
        year_mu=None,
        year_sigma=None,
        ):
    """Assign random construction years and refurbished states to buildings.

    If construction years are known, consider using
    assign_construction_classification_from_arge() instead.

    Args:
        col_refurbished_state (str): If not None, this colun name is used
        to store the generated random states of refurbishment.

        refurbished_weights (list): List of three floats, defining the
        probabilities associated with the states 'not refurbished',
        'slightly refurbished' and 'mostly refurbished'

        col_construction_year(str): If not None, this column name is used
        to store the generated random year values.

        year_mu (int): If mean mu and sigma of year are given, a normal
        distribution with those properties is used to assign the random years.
        If not, a regular random distribution between 1900 and 2008 is used.

        year_sigma (int): (See year_mu)


    Columns 'refurbished_state' and 'construction_year
    Afterwards, set_heat_demand_from_construction_classification() can be used.
    """
    logger.info('Assign random construction classification')

    rng = np.random.default_rng(42)
    if col_construction_year is not None:
        if year_mu is None or year_sigma is None:
            gdf[col_construction_year] = rng.integers(low=1900, high=2008,
                                                      size=len(gdf))
        else:
            gdf[col_construction_year] = rng.normal(year_mu, year_sigma,
                                                    size=len(gdf))
            gdf[col_construction_year] = \
                gdf[col_construction_year].astype('int')

    if col_refurbished_state is not None:
        gdf[col_refurbished_state] = rng.choice(
            ['not refurbished', 'slightly refurbished', 'mostly refurbished'],
            size=len(gdf), p=refurbished_weights)

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
    For each building, draw a random refurbishement states based on its
    assigned probabilities.

    The original source defines single- and multi family homes (SFH, MFH)

    TODO: Allow additional building types. These are given the mean
    of the available source building types.

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
        df,
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
    gdf[col_refurbished_state] = df_selections.idxmax(axis=1)

    # Test the results
    # for building_type in ['SFH', 'MFH']:
    #     dfx = gdf[gdf[col_building_type] == building_type]
    #     for year in sorted(dfx[col_construction_year].unique()):
    #         dfy = dfx[dfx[col_construction_year] == year]
    #         hist = dfy[col_refurbished_state].value_counts()/len(dfy)
    #         print(building_type, year)
    #         print(hist)

    if fillna_value is not None:
        gdf[col_refurbished_state].fillna(fillna_value, inplace=True)

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
    # Merging assings the correct heat demand to each row in the DataFrame
    df_out = (pd.merge(df_in, df, how='left',
                       on=[col_building_type, col_refurbished_state,
                           col_construction_year])
              .set_index(df_in.index))

    if fillna_value is not None:
        df_out[col_spec_total].fillna(value=fillna_value, inplace=True)

    elif df_out[col_spec_total].isna().any() and warnings == 'raise':
        n = df_out[col_spec_total].isna().value_counts().get(True, default=0)
        logger.warning("%s buildings did not receive a heat demand, because "
                       "they did not match the criteria of the data source. "
                       "You may use the argument 'fillna_value' to fill in a "
                       "specific heat demand, or ignore this warning.", n)

    if col_heated is not None:
        # Set heat demand of non-heated buildings to zero
        df_out.loc[df_out[col_heated] == False, col_spec_total] = 0

    df_out[col_total] = (df_out[col_spec_total] * df_out['a_N']
                         ).round(decimals)

    return df_out


def process_src_data_arge(
        df, idx_name, col_names, val_name, year_min=1800,
        year_max=2050, aliases_SFH=None, aliases_MFH=None):
    """Process data from source ARGE."""
    df.index.set_names(idx_name, inplace=True)
    df.columns.set_names(col_names, inplace=True)

    # Apply the aliases by making copies of the correct columns
    if aliases_SFH is not None:
        for alias in aliases_SFH:
            df = pd.concat([df, df[['SFH']].rename(columns={'MFH': alias})],
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
        df_tmp.loc[df_tmp[col_heated] == False, col_spec_total] = 0

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
        col_building_type='building_osm',
        decimals=2):
    """Set residential domestic hot water energy demand from DIN 18599.

    Residential:
    qw b = max[16,5 - (A_NGF,WE,m · 0,05); 8,5] kWh/(m2 ∙ a)

    DIN V 18599-10:2016, page 17

    Non-residential:
    Demand for hot water for non-residential buildings
    DIN V 18599-10:2016, page 30

    Categories assigned to OpenStreetMap building type keys.

    """
    # DIN V 18599-10:2016, page 17
    mask = df[col_building_type].isin(
        ['SFH', 'MFH', 'house', 'residential', 'detached',
         'semidetached_house', 'apartments'])
    df.loc[mask, col_spec_DHW] = ((16.5 - df['a_NRF'] * 0.05)
                                  .clip(lower=8.5)
                                  .round(decimals))

    # DIN V 18599-10:2016, page 30
    df_non_residential = pd.DataFrame(
        columns=[col_building_type, col_spec_DHW],
        data=[
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
    df_tmp = (df[[col_building_type]]
              .merge(df_non_residential, on=col_building_type, how='left')
              .set_index(df.index)  # preserve index for correct update
              )

    try:
        geometry_col = df.geometry.name  # Store name of geometry column
    except AttributeError:
        geometry_col = None

    df.update(df_tmp, overwrite=False)  # Update function breaks GeoDataFrame

    if geometry_col is not None:
        df.set_geometry(geometry_col, inplace=True)  # Restore geometry column

    if col_heated is not None:
        # Set heat demand of non-heated buildings to zero
        df.loc[df[col_heated] == False, col_spec_DHW] = 0

    df[col_DHW] = (df[col_spec_DHW] * df['a_NRF']).round(decimals)

    return df


def set_domestic_hot_water_from_values(
        df, col_DHW='e_th_DHW_kWh', col_spec_DHW='e_th_spec_DHW',
        col_building_type='building_type',
        E_th_spec_DHW_SFH=10, E_th_spec_DHW_MFH=15, A_ref='a_N'):
    """Subtract a fixed specific domestic hot water heat from total heat.

    Unit: kWh/(m² * a) (annual energy per Area A_N defined in German EnEV)

    Applies only to the following 'building_type' options:
        - 'SFH'
        - 'MFH'
    """
    logger.info("Set fixed DHW energy demand")

    df.loc[df[col_building_type] == 'SFH', col_spec_DHW] = E_th_spec_DHW_SFH
    df.loc[df[col_building_type] == 'MFH', col_spec_DHW] = E_th_spec_DHW_MFH

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
        col_level_height=None,
        kind='mean'):
    """Calculate level height from building height and number of levels.

    TODO: Given enough input data, can provide average height for groups
    of buildings. Then those grouped averages can be used to fill the
    missing number of levels.
    """
    if not isinstance(col_levels, list):
        col_levels = [col_levels]
    breakpoint()

    _col_levels = [col for col in col_levels if col in gdf.columns]
    if len(_col_levels) > 0:
        levels = gdf[_col_levels].sum(axis='columns', min_count=1)

        df_level_height = (gdf[col_height]
                           .div(levels)
                           .replace(np.inf, np.nan)
                           )
        if col_level_height:
            gdf[col_level_height] = df_level_height
    else:
        df_level_height = pd.Series()

    if kind == 'mean':
        level_height = df_level_height.mean()

    """
    gdf.groupby(by='building_osm')[col_level_height].mean()
    gdf.groupby(by='building_type')[col_level_height].mean()
    gdf.groupby(by='baujahr')[col_level_height].mean()
    test = gdf.groupby(by=[ 'building_osm', 'baujahr'])[col_level_height].mean()
    test = gdf.groupby(by=[ 'building_type', 'baujahr'])[col_level_height].mean()
    test = gdf[gdf['heated']].groupby(by=[ 'building_osm', 'baujahr'])[col_level_height].count()
    """

    return level_height


def calculate_levels_from_height(
        gdf, col_height, col_levels='building:levels', level_height=3.5,
        limit=None, decimals=1):
    """Calculate the number of levels per building from the height.

    Given the height of a building in column 'col_height' and the
    height per level 'level_height', store the numver of levels in
    column 'col_levels'. Existing values are not overwritten.

    limit (int): Set a maximum limit for the number of levels
    decimals (int): Number of decimal places to round the result to

    """
    gdf.loc[gdf[col_levels].isna(), col_levels] = (
        gdf[col_height]
        .div(level_height)
        .clip(lower=1, upper=limit)
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
    """
    if not isinstance(col_levels, list):
        col_levels = [col_levels]

    _col_levels = [col for col in col_levels if col in gdf.columns]
    if len(_col_levels) > 0:
        make_columns_numeric(gdf, _col_levels, errors='raise')
        gdf[col_levels[0]].fillna(levels_default, inplace=True)
        levels = gdf[_col_levels].sum(axis='columns', min_count=1)

    else:  # None of col_levels are in gdf.columns
        logger.warning("Column '%s' (number of levels) not in DataFrame. "
                       "Setting number of levels to %s for each building",
                       col_levels[0], levels_default)
        gdf[col_levels[0]] = levels_default
        levels = gdf[col_levels[0]]

    # Calculate Bruttogrundfläche from polygon area and number of floors
    gdf['a_BGF'] = (gdf.area * levels).round(decimals)

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

    buildings['E_th_heat_kWh'].fillna(0, inplace=True)
    buildings['e_th_DHW_kWh'].fillna(0, inplace=True)

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
    gdf_tmp[col_building_type].fillna('unknown', inplace=True)
    # Set the subtype for each building. Treat the aliases as if they
    # belonged to the original categories
    for alias in ['SFH'] + aliases_SFH:
        gdf_tmp.loc[gdf_tmp[col_building_type] == alias, 'subtype'
                    ] = 'w/o cellar'
        gdf_tmp[col_building_type].replace(alias, 'SFH', inplace=True)
    for alias in ['MFH'] + aliases_MFH:
        gdf_tmp.loc[gdf_tmp[col_building_type] == 'MFH', 'subtype'
                    ] = '7-19 WE'
        gdf_tmp[col_building_type].replace(alias, 'MFH', inplace=True)
    for alias in ['business'] + aliases_business:
        gdf_tmp.loc[gdf_tmp[col_building_type] == alias, 'subtype'
                    ] = 'business'
        gdf_tmp[col_building_type].replace(alias, 'business', inplace=True)
    for alias in ['unknown'] + aliases_unknown:
        gdf_tmp.loc[gdf_tmp[col_building_type] == alias, 'subtype'
                    ] = 'unknown'
        gdf_tmp[col_building_type].replace(alias, 'unknown', inplace=True)

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
    """Merge the given 'digital terrain/elevation model' files into one.

    The DTM/DEM files must be in the xyz format.
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


def combine_alkis_and_osm_buildings(
        gdf_alkis,
        gdf_osm,
        columns=['addr:street', 'addr:housenumber', 'building:levels'],
        address_col=None):
    """Add the columns from gdf_osm to gdf_alkis where buildings intersect.

    This is useful for copying data from osm to an alkis dataset.
    """
    if columns is None:
        columns = gdf_osm.drop(columns=['geometry']).columns

    gdf_alkis = tobler.area_weighted.area_join(
        gdf_osm.to_crs(gdf_alkis.crs), gdf_alkis, columns)

    if address_col is not None:
        if ('addr:street' in gdf_alkis.columns
           and 'addr:housenumber' in gdf_alkis.columns):
            gdf_alkis[address_col] = \
                gdf_alkis['addr:street'] + ' ' + gdf_alkis['addr:housenumber']

    return gdf_alkis


def combine_buildings_and_parcels(
        buildings, parcels, columns=None):
    """Store information from parcels in buildings within those parcels.

    Join variables from parcels based on the largest intersection with
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
    logger.info('Combine buildings and parcels')

    if columns is None:
        geom = parcels.geometry.name
        columns = parcels.drop(columns=[geom]).columns

    buildings = tobler.area_weighted.area_join(parcels, buildings, columns)
    return buildings


def make_geographic_selection(
        buildings, gdf_selection, col_candidates='DISTRICT_HEATING',
        show_plot=True, drop=False):
    """Make a geographic selection of buildings.

    In column 'col_candidates' of GeoDataFrame 'buildings', only those within
    GeoDataFrame 'gdf_selection' and previously True will remain True,
    all other buildings will be set to False.
    """
    logger.info('Make geographic selection')
    if col_candidates is not None:
        if col_candidates not in buildings.columns:
            buildings[col_candidates] = True

    if len(gdf_selection) > 1:
        raise ValueError("Area selection can only be a single polygon")

    # "Within" only works if both gdf share the same coordinate reference
    gdf_selection.to_crs(crs=buildings.crs, inplace=True)
    mask1 = buildings.within(gdf_selection.loc[0, 'geometry'])
    n_dropped = mask1.value_counts().get(False, default=0)
    n_remain = mask1.value_counts().get(True, default=0)
    logger.info('Buildings discarded by area selection: %s', n_dropped)
    if n_remain == 0:
        logger.error('No buildings left after area selection!')

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
        parcel_text_alt=None, address_empty="", mask_skip=None):
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
        same parcel_number are kept.

        parcel_text_alt (str): Alternative column name to use for the parcel
        text definition, where the column 'parcel_text' is equal to the
        argument 'address_empty'

        address_empty (str): String to determine when to use 'parcel_text_alt'

        mask_skip (index):
            An index mask as e.g. returned by ``buildings['HEATED'] == False``.
            Where this mask is True, the objects are not merged with their
            neighbours. This allows to skip specific buildings.

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
    buildings = buildings.sort_values(by=sort_columns + ['area'],
                                      ascending=False)

    # Use libpysal magic
    # 1: Create "weights" from the buildings with the same parcel text
    regimes = buildings[parcel_text_tmp]
    w_block = libpysal.weights.block_weights(regimes, silence_warnings=True)
    # 2: Create "weights" from buildings that touch each other
    w_queen = libpysal.weights.Queen.from_dataframe(buildings,
                                                    silence_warnings=True)
    # 3: Get the intersetions of all buildings that touch in the same parcel
    weights = libpysal.weights.w_intersection(w_queen, w_block,
                                              silence_warnings=True)

    # Merge those selected buildings
    buildings_merged = buildings.dissolve(by=weights.component_labels,
                                          sort=False,  # better performance?
                                          aggfunc='first',
                                          as_index=True)

    # Drop unused columns:
    buildings_merged.drop('area', axis='columns', inplace=True)
    buildings_merged.drop(parcel_text_tmp, axis='columns', inplace=True)

    # buildings.set_index('OID').groupby(by='LAGEBEZTXT_parcel').groups
    # buildings.set_index('OID').groupby(by='LAGEBEZTXT_parcel').sum()
    return buildings_merged


def create_hexgrid(gdf_buildings, gdf_area=None, resolution=10, clip=False,
                   buffer_distance=200,
                   col_heat='e_th_total_kWh', figsize=(20, 10),
                   scheme='fisherjenkssampled', k=7,
                   show_plot=True,
                   method='interpolation',
                   save_path=None,
                   plot_basemap=False,
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
        DESCRIPTION. The default is 5.
    clip : TYPE, optional
        DESCRIPTION. The default is False.
    intensive_variables : TYPE, optional
        DESCRIPTION. The default is 'E_th_total_kWh'.

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

    """
    if gdf_area is None:
        gdf_area = gpd.GeoDataFrame(
            geometry=[gdf_buildings
                      .unary_union
                      .convex_hull
                      .buffer(buffer_distance)],
            crs=gdf_buildings.crs)
        if show_plot:
            plot_geometries([gdf_area, gdf_buildings],
                            title="'Buffered' area used for hexgrid")

    gdf_hex = tobler.util.h3fy(gdf_area,
                               resolution=resolution,
                               clip=clip)
    if gdf_hex.empty:
        raise ValueError("Choose a higher resolution than {} for the hexgrid".
                         format(resolution))

    # plot_geometries([
    #     gdf_area,
    #     gdf_hex,
    #     gdf_buildings,
    #     ])

    if method == 'interpolation':
        gdf_hex_interp = tobler.area_weighted.area_interpolate(
            source_df=gdf_buildings, target_df=gdf_hex,
            # intensive_variables=[col_heat],
            extensive_variables=[col_heat],
            allocate_total=True,
            )
    elif method == 'join':
        # This caused problems with the resulting heat demand sums
        gdf_hex_interp = tobler.area_weighted.area_join(
            source_df=gdf_buildings, target_df=gdf_hex,
            variables=[col_heat],
            )
        gdf_hex_interp.fillna(0, inplace=True)
    else:
        raise ValueError("Method {} not defined".format(method))

    logger.debug("Sum of heat demand: %s MWh in buildings, %s MWh in hexgrid",
                 round(gdf_buildings[col_heat].sum()/1000, 2),
                 round(gdf_hex_interp[col_heat].sum()/1000, 2),
                 )

    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)

        gdf_hex_interp_plot = gdf_hex_interp.copy()
        gdf_hex_interp_plot[col_heat] *= 1/1000  # from kWh to MWh

        try:
            gdf_hex_interp_plot.plot(
                column=col_heat,
                ax=ax,
                scheme=scheme,
                k=k,  # Number of classes
                alpha=0.5,
                legend=True,
                legend_kwds=dict(title='Wärmebedarf [MWh]'),
                )
        except ValueError as e:
            raise ValueError("Column '{}' is missing enough valid values "
                             "for plot".format(col_heat)) from e

        if plot_basemap:
            add_basemap(ax, crs=gdf_buildings.crs)
        gdf_buildings.plot(ax=ax)
        ax.axis('off')
        plt.show()

    if save_path is not None:
        save_geojson(gdf_hex_interp, file='buildings_hex', path=save_path)

    return gdf_hex_interp


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


def sort_from_north_to_south(gdf, col_id='ID', set_index=False):
    """Sort the objects in gdf from north to south.

    Insert an increasing index into colum 'col_id', if given.
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


# Section "OpenStreetMap downloads"
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
    gdf = ox.geometries_from_polygon(area.geometry[0], tags=tags)

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
    area = ox.geocode_to_gdf(places)
    if show_plot:  # Show plot in a projected CRS
        plot_geometries(area.to_crs('EPSG:4647'), plot_basemap=True)
    return area


def download_buildings_from_osm(
        gdf_polygon,
        building_keys=[
            'apartments',
            'commercial',
            'detached',
            'house',
            'industrial',
            'residential',
            'retail',
            'semidetached_house',
            'yes',
        ],
        crs=None,
        show_plot=False,
        dropna_tresh=None,
        rename_dict={'building': 'building_osm'},
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
                      .unary_union
                      .convex_hull],
            crs=gdf_polygon.crs)

    gdf_polygon.to_crs(epsg=4326, inplace=True)  # make it work with osmnx
    polygon = gdf_polygon.geometry[0]
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
    try:  # 'node' elements are unwanted point geometries
        gdf = gdf.loc[gdf["element_type"] != 'node'].copy()
    except KeyError:
        pass
    # Remove nodes column (that somehow make trouble for exporting .geojson)
    gdf.drop(columns=['nodes'], inplace=True, errors='ignore')

    if rename_dict:
        gdf.rename(columns=rename_dict, inplace=True)

    return gdf


def choose_random_thermal_load(gdf_buildings, low=10, high=50):
    """Choose a random maximum thermal power for each building.

    Set the column 'P_heat_max' to a random value between 10 and 50 kW
    for all houses. This column is required for dhnx.
    """
    np.random.seed(42)
    gdf_buildings['P_heat_max'] = \
        np.random.randint(low, high, size=len(gdf_buildings))
    return gdf_buildings


def download_streets_from_osm(
        gdf_polygon,
        highway_keys=[
            'residential',
            'service',
            'unclassified',
            'primary',
            'secondary',
            'tertiary',
            'footway',
            'steps',
            'pedestrian',
            # 'path',
            # 'track'
            # 'yes',
            ],
        crs="EPSG:3857",
        show_plot=False,
        ):
    """Download street network data from OpenStreetMap.

    Select the street types you want to consider as district heating routes

    For documentation on the key system for types of streets, see
    https://wiki.openstreetmap.org/wiki/Key:highway

    Args:
        gdf_polygon (GeoDataFrame): An object defining the selection area

        highway_keys (list): List of osm highway tags or 'True' for
        all highway objects (but can include points)

    """
    logger.info("Download street data from OpenStreetMap")
    streets = dict({'highway': highway_keys})

    if len(gdf_polygon) > 1:
        raise ValueError("Area selection can only be a single polygon")

    gdf_polygon.to_crs(epsg=4326, inplace=True)  # make it work with osmnx
    polygon = gdf_polygon.geometry[0]

    if show_plot:
        # Download the street network data from OpenStreetMap
        graph = ox.graph_from_polygon(polygon, network_type='drive_service')
        ox.plot_graph(graph)

    # Create GeoDataFrame object from the polygon
    gdf_lines_streets = ox.geometries_from_polygon(polygon, tags=streets)
    # Remove nodes column (that make somehow trouble for exporting .geojson)
    gdf_lines_streets.drop(columns=['nodes'], inplace=True)
    # Filter out e.g. 'polygon' types that would cause issues with dhnx
    accepted_types = ['LineString', 'MultiLineString']
    gdf_lines_streets = gdf_lines_streets.loc[
        gdf_lines_streets['geometry'].type.isin(accepted_types)].copy()

    if crs is not None:
        # Convert to target crs
        gdf_lines_streets.to_crs(crs=crs, inplace=True)
    return gdf_lines_streets


# Section 'Plotting functions'

def plot_geometries(
        gdf_list,
        title='',
        crs_default="EPSG:4647",
        plot_basemap=False,
        **fig_kwargs,
        ):
    """Plot the given list of geometry objects.

    Geometries can be GeoDataFrames, GeoSeries and shapely geometries.
    They are converted to a common crs. If taking the crs of the first
    entry in gdf_list fails, 'crs_default' is used instead.
    """
    fig_kwargs.setdefault('figsize', (20, 10))
    fig, ax = plt.subplots(**fig_kwargs)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if isinstance(gdf_list, gpd.GeoDataFrame):
        gdf_list = [gdf_list]  # Allow single GeoDataFrame as list

    try:
        crs_use = gdf_list[0].crs
    except Exception:
        crs_use = crs_default

    for gdf, color in zip(gdf_list, colors[:len(gdf_list)]):
        if not (isinstance(gdf, gpd.GeoDataFrame)
                or isinstance(gdf, gpd.GeoSeries)):
            # Assume that this is a shapely geometry that can be converted
            gdf = gpd.GeoDataFrame(geometry=[gdf], crs=crs_default)

        gdf.to_crs(crs=crs_use).plot(ax=ax, color=color)

    if plot_basemap:
        add_basemap(ax, crs=crs_use)

    plt.title(title)
    plt.show()


def plot_heated(gdf, col_heated='heated', **fig_kwargs):
    """Plot the buildings where col_heated is True."""
    logger.info('Plot map of column %s', col_heated)
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


def run_dhnx(gdf_lines_streets, gdf_poly_gen, gdf_poly_houses,
             save_path='./out', show_plot=True,
             path_invest_data='invest_data',
             df_load_ts_slice=None,
             bidirectional_pipes=False,
             simultaneity=1,
             reset_index=True,
             method='midpoint',
             col_p_th='P_heat_max',
             solver=None,
             solver_cmdline_options=None,
             ):
    """Run the dhnx (district heating networks) process.

    Parameters
    ----------
    gdf_lines_streets : TYPE
        DESCRIPTION.
    gdf_poly_gen : TYPE
        DESCRIPTION.
    gdf_poly_houses : TYPE
        DESCRIPTION.
    save_path : TYPE, optional
        DESCRIPTION. The default is './out'.
    show_plots : TYPE, optional
        DESCRIPTION. The default is True.
    path_invest_data : TYPE, optional
        DESCRIPTION. The default is 'invest_data'.
    df_load_ts_slice : DataFrame, optional
        A DataFrame with timeseries of the thermal load
        in kW for building. The column names (identifier for each building)
        need to match the index of gdf_poly_houses, to be able to match.
        The default is None.
    bidirectional_pipes : TYPE, optional
        DESCRIPTION. The default is False.
    simultaneity : TYPE, optional
        DESCRIPTION. The default is 1.
    reset_index : TYPE, optional
        DESCRIPTION. The default is True.
     : TYPE
        DESCRIPTION.

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
    if df_load_ts_slice is None and col_p_th is not None:
        # Specific column name for thermal power is required for DHNX
        if gdf_poly_houses[col_p_th].isna().any():
            raise ValueError("Each building connected to the district "
                             "heating grid needs to have an associated "
                             f"thermal power in column '{col_p_th}'")

        gdf_poly_houses['P_heat_max'] = gdf_poly_houses[col_p_th]

    if ((df_load_ts_slice is None) and
       ('P_heat_max' not in gdf_poly_houses.columns)):
        raise ValueError("The thermal load of each house in gdf_poly_houses "
                         "needs to be given via column 'P_heat_max' or via "
                         "a separate timeseries df_load_ts_slice")

    # process the geometry
    tn_input = dhnx.gistools.connect_points.process_geometry(
        lines=gdf_lines_streets,
        producers=gdf_poly_gen.copy(),
        consumers=gdf_poly_houses.copy(),
        method=method,
        reset_index=reset_index,
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
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    for filename, gdf in tn_input.items():
        try:
            save_geojson(gdf, file=filename, path=save_path)
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
            raise ValueError("Buildings GeoDataFrame and time series must"
                             "describe the same number of buildings")
        # Enforce string column names, because dhnx expects them
        df_load_ts_slice.columns = df_load_ts_slice.columns.astype(str)
        # Add the time series slice of the thermal load to the Network
        # (required for using settings["heat_demand"]="series") later on
        network.sequences['consumers']['heat_flow'] = df_load_ts_slice

    # check if ThermalNetwork is consistent
    network.is_consistent()

    # load the specification of the oemof-solph components
    invest_opt = dhnx.input_output.load_invest_options(path_invest_data)

    # Define which solver to use. This function checks if 'gurobi' or 'cbc'
    # are installed and installs 'cbc' if necessary
    if solver is None:
        solver = get_installed_solver()

    # Optionally, define some settings for the solver. Especially increasing
    # the solution tolerance with 'ratioGap' or setting a maximum runtime
    # in 'seconds' helps if large networks take too long to solve
    if solver_cmdline_options is None:
        solver_cmdline_options = {}

    settings = dict(
        solver=solver,
        solve_kw={
            'tee': True,  # print solver output
        },
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
        settings["frequence"] = pd.infer_freq(df_load_ts_slice.index)

    # perform the investment optimisation
    # try:
    network.optimize_investment(invest_options=invest_opt, **settings)
    # except Exception as e:
    #     logger.error(str(e))
    #     breakpoint()
    #     logger.error(str(e))

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
    gdf_pipes = gdf_pipes.join(results_edges, rsuffix='results_')

    gdf_pipes = apply_DN(gdf_pipes, save_path)  # Apply DN from capacity

    if show_plot:
        # plot output after processing the geometry
        _, ax = plt.subplots(figsize=(20, 10), dpi=300)
        # network.components['consumers'].plot(ax=ax, color='green', markersize=.5)
        # network.components['producers'].plot(ax=ax, color='red', markersize=.5)
        # network.components['forks'].plot(ax=ax, color='grey', markersize=.5)
        # gdf_pipes[gdf_pipes['capacity'] > 0].plot(ax=ax, color='blue',
        #                                           linewidth=0.1)
        gdf_plot = pd.concat([gdf_poly_gen, gdf_poly_houses],
                              keys=['Producer', 'Consumer'],
                              names=['role']
                              ).reset_index().to_crs(gdf_pipes.crs)
        gdf_plot.plot(ax=ax, column='role', legend=True)

        gdf_pipes[gdf_pipes['DN'] > 0].plot(ax=ax, column='DN',
                                            linewidth=2, legend=True,
                                            legend_kwds={'label':'DN'})
        # gdf_poly_gen.plot(ax=ax, color='orange', label='Producer')
        # gdf_poly_houses.plot(ax=ax, color='green', label='Consumer')

        # plt.legend()
        plt.title('Invested pipelines')
        plt.show()

    # Export results
    gdf_pipes = gdf_pipes[gdf_pipes['DN'] > 0]  # Keep only DN>0 in output
    save_geojson(gdf_pipes, file='pipes_result', path=save_path)
    save_excel(gdf_pipes, os.path.join(save_path, 'pipes_result.xlsx'))

    # Save info about pipes
    df_pipes = gdf_pipes.copy()
    df_pipes = df_pipes[df_pipes['DN'] > 0]
    df_pipes.replace(['DL', 'GL'], 'Verteilleitung', inplace=True)
    df_pipes.replace(['HL'], 'Hausanschlussleitung', inplace=True)
    df_pipes = (df_pipes.set_index('type')
                .groupby(['type', 'DN'])
                .sum(numeric_only=True))
    df_pipes = df_pipes[['length']]
    save_excel(df_pipes, os.path.join(save_path, 'WN_pipes.xlsx'))

    return gdf_pipes, df_pipes


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


def apply_DN(gdf_pipes=None, save_path=".", T_FF=80, T_RF=50):
    """Apply norm diameter of pipes from capacity."""
    df_DN = pd.DataFrame(
        {'Bezeichnung [DN]': [25, 32, 40, 50, 63, 75, 90, 110, 125,
                              160, 200, 250, 300, 350, 400, 500, 600]})

    df_DN['Innendurchmesser [m]'] = df_DN['Bezeichnung [DN]']/1000
    df_DN['Max delta p [Pa/m]'] = 100
    df_DN['Rauhigkeit [mm]'] = 0.01
    df_DN['T_Vorlauf [°C]'] = T_FF  # °C forward flow
    df_DN['T_Rücklauf [°C]'] = T_RF  # °C return flow
    df_DN['Temperaturniveau [°C]'] = (
        (df_DN['T_Vorlauf [°C]'] + df_DN['T_Rücklauf [°C]']) / 2)

    df_DN = calc_pipes_p_max(df_DN)
    if save_path is not None:
        df_DN.to_excel(os.path.join(save_path, 'DN_table.xlsx'))

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

            gdf_pipes.loc[idx, 'DN'] = df_DN.loc[index, "Bezeichnung [DN]"]

        elif capacity == 0:
            continue

    return gdf_pipes


def calc_pipes_p_max(df):
    """Calculate maximum capacity of district heating pipes."""
    df['v_max [m/s]'] = df.apply(
        lambda row: dhnx.optimization.precalc_hydraulic.v_max_bisection(
            d_i=row['Innendurchmesser [m]'],
            T_average=row['Temperaturniveau [°C]'],
            k=row['Rauhigkeit [mm]'],
            p_max=row['Max delta p [Pa/m]']), axis=1)

    df['Massenstrom [kg/s]'] = df.apply(
        lambda row: dhnx.optimization.precalc_hydraulic.calc_mass_flow(
            v=row['v_max [m/s]'], di=row['Innendurchmesser [m]'],
            T_av=row['Temperaturniveau [°C]'],
            ), axis=1)

    df['P_max [kW]'] = df.apply(
        lambda row: 0.001*dhnx.optimization.precalc_hydraulic.calc_power(
            T_vl=row['T_Vorlauf [°C]'],
            T_rl=row['T_Rücklauf [°C]'],
            mf=row['Massenstrom [kg/s]']
            ), axis=1)

    return df


# Section "lpagg" (load profile aggregator)
def find_TRY_regions(gdf, show_plot=False, buffer=2000, col_try='try_code'):
    """Find DWD typical reference year (TRY) region of objects in gdf."""
    import lpagg.misc

    # Find the TRY-region for each building
    # gdf_TRY = lpagg.misc.get_TRY_polygons_GeoDataFrame(col_try)
    gdf_TRY = lpagg.misc.get_TRY_polygons_GeoDataFrame()  # TODO lpagg=v0.15.3
    gdf_TRY.rename(columns={'TRY_code': col_try}, inplace=True)

    gdf = tobler.area_weighted.area_join(
        gdf_TRY.to_crs(gdf.crs), gdf, [col_try])

    if len(gdf[col_try].unique()) > 1:
        logger.info("Buildings are located in multiple TRY-regions.")

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
        point = gdf.to_crs(4326).unary_union.convex_hull.centroid
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
    # Print *_houses.xlsx file (in addition to .dat, which is always saved).
    # Be careful, can create large file sizes that take long to save!
    settings.setdefault('print_houses_xlsx', False)
    # Print peak thermal power to separate file
    settings.setdefault('print_P_max', False)
    settings.setdefault('print_GLF_stats', True)
    # Language used for certain plots ('de', or 'en')
    settings.setdefault('language', 'de')

    cfg = dict(settings=settings,
               houses=houses_dict)

    return df_lpagg, cfg


def lpagg_run(gdf, sigma=0, E_th_col='E_th_total_kWh', show_plot=True,
              **cfg_kwargs):
    """Replace the __main__.py script from the regular LPagg program."""
    import lpagg.agg

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

    # Now let the aggregator do its job
    agg_dict = lpagg.agg.aggregator_run(cfg)

    # Plot and print the results
    lpagg.agg.plot_and_print(agg_dict['weather_data'], cfg)

    # Postprocessing
    df_load_ts_slice = lpagg_get_max_power_slice(agg_dict['load_curve_houses'],
                                                 show_plot=show_plot)
    gdf = lpagg_merge_houses_and_load(gdf, agg_dict['P_max_houses'],
                                      E_th_col=E_th_col)

    return gdf, df_load_ts_slice


def lpagg_get_max_power_slice(df_load_ts, buffer=2, show_plot=True):
    """Get the time slice with the maximum thermal power from lpagg.

    This time slice can be used as input for dhnx.
    """
    from pandas.tseries.frequencies import to_offset

    freq = to_offset(pd.infer_freq(df_load_ts.index)) / pd.Timedelta('1 hours')

    # Take the sum of all the thermal energies (so drop electrical)
    df_load_ts = (df_load_ts
                  .stack(["house"])
                  .drop(columns=['W_TT'], level='energy', errors='ignore')
                  .sum(axis='columns')
                  .div(freq)  # convert kWh to kW
                  .unstack(["house"])
                  )
    df_load_ts.columns.set_names(None, inplace=True)

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
    df_load_ts_slice = pd.concat(slice_list)
    df_load_ts_slice = df_load_ts_slice.drop_duplicates().sort_index()

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


def qgis_drape():
    """Run QGIS algorithm 'drape (set Z value from raster)'.

    This algorithm sets the z value of every vertex in the feature
    geometry to a value sampled from a band within a raster layer.
    The raster values can optionally be scaled by a preset amount
    and an offset can be algebraically added.

    Afterwards in QGIS, the expression ``length3D($geometry)`` can be used
    to calculate the line lengths in 3-dimensional space.

    https://opensourceoptions.com/blog/pyqgis-calculate-geometry-and-field-values-with-the-qgis-python-api/

    """
    import subprocess

    subprocess.check_output([
        r'C:\Program Files\QGIS 3.28.1\apps\qgis\bin\qgis_process.exe',
        'run',
        'native:setzfromraster',
        '--distance_units=meters',
        '--area_units=m2',
        '--ellipsoid=EPSG:7019',
        r'--INPUT=C:\Users\Nettelstroth\code_projects\SIZ145_FirstTin\python\alkis\pipes_result.geojson',
        r'--RASTER=C:\Users\Nettelstroth\code_projects\SIZ145_FirstTin\Qgis\Layer\DGM\dgm200.utm32s.geotiff\dgm200\dgm200_utm32s.tif',
        '--BAND=1',
        '--NODATA=0',
        '--SCALE=1',
        '--OFFSET=0',
        r'--OUTPUT=C:\Users\Nettelstroth\code_projects\SIZ145_FirstTin\python\alkis\pipes_result_draped.geojson',
        ])


if __name__ == '__main__':
    main()
