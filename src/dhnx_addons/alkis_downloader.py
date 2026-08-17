# -*- coding: utf-8 -*-
"""
Experimental module to download ALKIS and LOD2 data for a given area
from public sources. Currently only Schleswig-Holstein is supported.
"""

import os
import requests
import geopandas as gpd
from shapely.geometry import box
from urllib.parse import urlparse
import zipfile
import re
import zlib
import shutil
import pandas as pd
import dhnx_addons
import logging

from . import dhnx_lib_3d

# Define the logging function
logger = logging.getLogger(__name__)


class AlkisDownloader:
    SH_ALKIS_OVERVIEW = "https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/single.php?file=ALKIS_SH_Massendownload.geojson&id=4"
    SH_LOD2_OVERVIEW = "https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/single.php?file=LOD2_SH_Massendownload.geojson&id=4"

    def __init__(self, gdf: gpd.GeoDataFrame, output_dir="data"):
        self.gdf = gdf
        self.output_dir = output_dir
        self._ensure_crs()
        self.files_alkis = []
        self.files_lod2 = []
        self.file_lod2_gpkg = os.path.join(self.output_dir, "lod2.gpkg")
        self.file_alkis_gpkg = os.path.join(self.output_dir, "alkis.gpkg")

    def _ensure_crs(self):
        if self.gdf.crs is None:
            raise ValueError("Input GeoDataFrame must have a CRS")
        if self.gdf.crs.to_epsg() != 4326:
            self.gdf = self.gdf.to_crs(4326)

    def run(self):
        bundesland = self._detect_bundesland()

        if bundesland != "Schleswig-Holstein":
            raise NotImplementedError(
                "Currently only Schleswig-Holstein is supported"
            )

        base_path = os.path.join(self.output_dir, "Schleswig-Holstein")
        os.makedirs(base_path, exist_ok=True)

        alkis_overview = self._download_overview(
            self.SH_ALKIS_OVERVIEW, base_path, "alkis"
        )
        lod2_overview = self._download_overview(
            self.SH_LOD2_OVERVIEW, base_path, "lod2"
        )

        self._process_dataset(alkis_overview, base_path, "alkis")
        self._process_dataset(lod2_overview, base_path, "lod2")

    def read_file_alkis_gpkg(self, layer=None):
        if not os.path.exists(self.file_alkis_gpkg):
            self.alkis_save_geopackage(self)

        return gpd.read_file(self.file_alkis_gpkg, layer=layer)

    def read_file_lod2_gpkg(self):
        if not os.path.exists(self.file_lod2_gpkg):
            self.lod2_save_geopackage(self)

        return gpd.read_file(self.file_lod2_gpkg)

    def lod2_remove_TerrainIntersetion(self):
        for file in self.files_lod2:
            dhnx_lib_3d.citygml_remove_TerrainIntersetion(file, file)

    def lod2_save_geopackage(self):
        gdf_lod2 = pd.concat(
            [
                dhnx_addons.load_xml_geodata(path, crs="EPSG:25832")
                for path in self.files_lod2
            ]
        )

        dhnx_addons.save_geopackage(gdf_lod2, self.file_lod2_gpkg)

        return self.file_lod2_gpkg

    def alkis_save_layer(self, layer, **kwargs):
        gdf_lod2 = pd.concat(
            [
                dhnx_addons.load_xml_geodata(
                    path, crs="EPSG:25832", layer=layer
                )
                for path in self.files_alkis
            ]
        )

        dhnx_addons.save_geopackage(
            gdf_lod2, self.file_alkis_gpkg, layer=layer, **kwargs
        )
        return self.file_alkis_gpkg

    def alkis_save_geopackage(self):
        self.alkis_save_layer(layer="AX_Gebaeude", mode="w")
        self.alkis_save_layer(
            layer="AX_GeoreferenzierteGebaeudeadresse", mode="a"
        )
        self.alkis_save_layer(layer="AX_Flurstueck", mode="a")

    def alkis_list_layers(self):
        if self.files_alkis > 0:
            return gpd.list_layers(self.files_alkis[0])
        else:
            return pd.DataFrame()

    def _detect_bundesland(self):
        sh_bbox = box(7.5, 53.3, 11.5, 55.1)

        if self.gdf.union_all().intersects(sh_bbox):
            return "Schleswig-Holstein"
        return "Unknown"

    def _download_overview(self, url, base_path, dataset_type):
        path = os.path.join(base_path, f"{dataset_type}_overview.geojson")

        if not os.path.exists(path):
            logger.info(f"Downloading {dataset_type} overview...")
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

        return gpd.read_file(path)

    def _process_dataset(self, overview_gdf, base_path, dataset_type):
        logger.info(f"Processing {dataset_type}...")

        if overview_gdf.crs.to_epsg() != 4326:
            overview_gdf = overview_gdf.to_crs(4326)

        intersecting = overview_gdf[
            overview_gdf.intersects(self.gdf.union_all())
        ]

        logger.info(f"{len(intersecting)} {dataset_type} tiles intersect")

        dataset_dir = os.path.join(base_path, dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)

        col_link = None
        if "data_link" in intersecting.columns:
            col_link = "data_link"
        elif "link_data" in intersecting.columns:
            col_link = "link_data"
        else:
            raise ValueError("No data link column found")

        for _, row in intersecting.iterrows():
            link = row.get(col_link)
            if not link:
                continue

            output_path = self._download_and_extract(link, dataset_dir)

            if output_path:
                if dataset_type == "alkis":
                    self.files_alkis.append(output_path)
                elif dataset_type == "lod2":
                    self.files_lod2.append(output_path)

    def _download_and_extract(self, url, dataset_dir):
        logger.info(f"Downloading from {url}...")

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                filename = self._get_filename_from_response(r, url)

                name, ext = os.path.splitext(filename)
                if ext == ".gz":
                    name, ext2 = os.path.splitext(name)
                    ext = ext2 + ext  # ".xml.gz"

                tile_dir = os.path.join(dataset_dir, name)
                os.makedirs(tile_dir, exist_ok=True)

                # skip if already extracted
                # if os.listdir(tile_dir):
                #     logger.info(f"Skipping already extracted: {name}")
                #     return

                filepath = os.path.join(dataset_dir, filename)

                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # extraction logic
            if filename.endswith(".zip"):
                raise NotImplementedError("Zip not yet implemented")
            elif filename.endswith(".gz"):
                output_path = self._extract_gz(filepath, tile_dir)
            else:
                output_path = os.path.join(tile_dir, filename)
                if not os.path.exists(output_path):
                    os.rename(filepath, output_path)

            return output_path

        except Exception as e:
            logger.info(f"Failed to process {url}: {e}")

    def _get_filename_from_response(self, response, url):
        cd = response.headers.get("Content-Disposition")

        if cd:
            match = re.search(r'filename="?([^"]+)"?', cd)
            if match:
                return match.group(1)

        # fallback (in case header is missing)
        return self._get_filename_from_url(url)

    def _extract_gz(self, filepath, target_dir):
        output_path = os.path.join(
            target_dir, os.path.splitext(os.path.basename(filepath))[0]
        )

        if not os.path.exists(output_path):
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)  # gzip mode

            with open(filepath, "rb") as f_in, open(
                output_path, "wb"
            ) as f_out:
                while True:
                    chunk = f_in.read(1024 * 1024)
                    if not chunk:
                        break

                    data = decompressor.decompress(chunk)
                    f_out.write(data)

                    if decompressor.eof:
                        break

        os.remove(filepath)

        return output_path
