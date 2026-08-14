# -*- coding: utf-8 -*-
"""Implement custom basemaps for map plotting.

They are designed to be used às a replacement for ``contextily`` like

``contextily.add_basemap(ax=ax, source=source, crs=crs, **kwargs)``


1. Custom WebMapService Basemap

Designed to downlaod and plot raster images from specific WMS providers.


2. Custom Vector Basemap

Currently is adapted to https://basisvisualisierung.niedersachsen.de
and the styles it provides.

Downloads the map elements as vector data and then plots the 'fill' and 'line'
elements (background, buildings, streets, ...) with matplotlib. This allows
saving maps as complete vector images, e.g. svg. Text is currently not
properly supported.

"""

import geopandas as gpd
import logging
import matplotlib
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)  # Create a logger for this module

try:
    import contextily
except ImportError as e:
    logger.exception(e)
    logger.warning(
        "Optional dependency 'contextily' can be installed with "
        "'conda install contextily -c conda-forge'"
    )

try:
    import mercantile
    import mapbox_vector_tile
    from vt2geojson.tools import (
        vt_bytes_to_geojson,
    )  # https://github.com/Amyantis/python-vt2geojson
except ImportError as e:
    logger.exception(e)
    logger.warning(
        "mercantile, mapbox_vector_tile and vt2geojson are "
        "required for a custom base map"
    )


def add_custom_wms_basemap(
    ax,
    source,
    crs,
    version="1.1.1",
    wms_layer=None,
    wms_resolution=0.1,
    attribution=None,
    attribution_size=8,
    **kwargs,
):
    """Add a WebMapService basemap from given source to plot axis."""
    from owslib.wms import WebMapService
    from PIL import Image
    import io

    xmin, xmax, ymin, ymax = ax.axis()
    width_m = xmax - xmin
    height_m = ymax - ymin

    # Calculate the number of pixels in each dimension
    size = (
        int(round(width_m * wms_resolution, 0)),
        int(round(height_m * wms_resolution, 0)),
    )

    # Create a requests session
    # ``headers`` may contain ``username`` and ``password`` for authentication
    wms = WebMapService(
        source, version=version, **kwargs.get("headers", dict())
    )

    if wms_layer is None:
        wms_layer = list(wms.contents)[0]

    if crs not in wms[wms_layer].crsOptions:
        crs_fallback = "EPSG:4326"  # WGS 84
        logger.warning(
            "Selected CRS '%s' not supported, using '%s' instead",
            crs,
            crs_fallback,
        )
        crs = crs_fallback

    img = wms.getmap(
        layers=[wms_layer],
        size=size,
        srs=crs,
        bbox=(xmin, ymin, xmax, ymax),
        format="image/jpeg",
    )

    image = Image.open(io.BytesIO(img.read()))
    ax.imshow(
        np.asarray(image),
        extent=(xmin, xmax, ymin, ymax),
        origin="upper",
        zorder=-1,
    )

    if attribution:
        contextily.plotting.add_attribution(
            ax, attribution, font_size=attribution_size
        )

    return


def add_custom_vector_basemap(
    ax,
    source,
    crs,
    zoom="auto",
    zoom_adjust=None,
    enabled_layer_types=["fill", "line"],
    attribution="© GeoBasis-DE / BKG CC BY 4.0",
    attribution_size=8,
    style_scale_size=1 / 6,
    style_override=None,
    **kwargs,
):
    # Calculate auto-zoom with the help of contextily
    xmin, xmax, ymin, ymax = ax.axis()
    w, e, s, n = contextily.plotting._reproj_bb(
        xmin, xmax, ymin, ymax, crs, "EPSG:4326"
    )

    auto_zoom = zoom == "auto"
    if auto_zoom:
        zoom = contextily.tile._calculate_zoom(w, s, e, n)
        print(zoom)
    if zoom_adjust:
        zoom += zoom_adjust

    basemap_gdfs = get_basemap_gdfs_from_vectortiles(
        ax, crs, zoom=zoom, **kwargs
    )

    # Load style
    style_src = source.split(".")[-1]  # color, classic, greyscale, ...
    # https://basisvisualisierung.niedersachsen.de/services/basiskarte/styles/vt-style-color.json
    # https://basisvisualisierung.niedersachsen.de/services/basiskarte/styles/vt-style-classic.json

    style_url = (
        "https://basisvisualisierung.niedersachsen.de/services/"
        f"basiskarte/styles/vt-style-{style_src}.json"
    )
    style_response = requests.get(style_url)
    style = style_response.json()
    # Plot layers according to style
    # breakpoint()
    for layer in style["layers"]:
        layer_id = layer.get("id")
        layer_type = layer.get("type")
        if layer_type not in enabled_layer_types:
            continue

        if style_override is not None and layer_id in style_override.keys():
            # E.g. {"Nicht öffentliches Gebäude 2D": {"minzoom": 14}}
            layer.update(style_override[layer_id])

        minzoom = layer.get("minzoom", 0)
        maxzoom = layer.get("maxzoom", 24)
        if not (minzoom <= zoom < maxzoom):
            # print(f"Skip {_id} due to zoom")
            continue

        source_layer = layer.get("source-layer")
        if source_layer not in basemap_gdfs.index.unique("layer"):
            # print(f"Skip {_id} (missing from geodata")
            continue

        gdf = basemap_gdfs.xs(source_layer, level="layer")

        class_filter = layer.get("filter")
        if class_filter:
            mask = evaluate_vector_style_filter(gdf, class_filter)
            gdf = gdf[mask]
        paint = layer.get("paint", {})
        layout = layer.get("layout", {})

        if layer_type == "fill":
            color = evaluate_style_value(
                paint.get("fill-color", "#cccccc"), zoom
            )

            alpha = evaluate_style_value(paint.get("fill-opacity", 1), zoom)

            # if isinstance(color, dict):
            #     try:
            #         color = color['stops'][0][1]
            #     except Exception:
            #         breakpoint()
            # if isinstance(alpha, list):
            #     alpha = 1  # pick a default

            try:
                if not gdf.empty:
                    gdf.plot(ax=ax, color=color, alpha=alpha, zorder=0)
            except Exception:
                breakpoint()
        elif layer_type == "line":
            color = evaluate_style_value(
                paint.get("line-color", "#000000"), zoom
            )

            width = evaluate_style_value(paint.get("line-width", 1), zoom)
            join = evaluate_style_value(paint.get("line-join", "bevel"), zoom)

            dasharray = evaluate_style_value(
                paint.get("line-dasharray", None), zoom
            )

            # For simplicity, use fixed width
            # if isinstance(width, list):
            #     width = 1  # pick a default

            if dasharray is None:
                linestyle = "solid"
            else:
                linestyle = (0, dasharray)

            try:
                if not gdf.empty:
                    gdf.plot(
                        ax=ax,
                        color=color,
                        linestyle=linestyle,
                        linewidth=width * style_scale_size,
                        joinstyle=join,
                        capstyle="butt",
                        zorder=0,
                    )
            except Exception:
                breakpoint()
        elif layer_type == "symbol":
            text_field = layout.get("text-field", "")
            text_color = paint.get("text-color", "#000000")
            text_size = layout.get("text-size", 12)
            text_halo_color = paint.get("text-halo-color", "#ffffff")
            text_halo_width = paint.get("text-halo-width", 0)

            # For simplicity, use fixed text size
            if isinstance(text_size, list):
                text_size = 10  # pick a default

            if not gdf.empty and text_field:
                # Extract text from the specified field
                if text_field.startswith("{") and text_field.endswith("}"):
                    col_name = text_field[1:-1]
                    if col_name in gdf.columns:
                        for idx, row in gdf.iterrows():
                            geom = row.geometry
                            if geom and hasattr(geom, "centroid"):
                                centroid = geom.centroid
                                text = str(row[col_name])
                                # Add halo if specified
                                if text_halo_width > 0:
                                    ax.text(
                                        centroid.x,
                                        centroid.y,
                                        text,
                                        fontsize=text_size,
                                        color=text_halo_color,
                                        ha="center",
                                        va="center",
                                        path_effects=[
                                            matplotlib.patheffects.withStroke(
                                                linewidth=text_halo_width * 2,
                                                foreground=text_halo_color,
                                            )
                                        ],
                                    )
                                ax.text(
                                    centroid.x,
                                    centroid.y,
                                    text,
                                    fontsize=text_size,
                                    color=text_color,
                                    ha="center",
                                    va="center",
                                )

    # Set axis limits to basemap bounds
    bounds_target = basemap_gdfs.total_bounds
    ax.set_xlim(bounds_target[0], bounds_target[2])
    ax.set_ylim(bounds_target[1], bounds_target[3])

    if attribution:
        contextily.plotting.add_attribution(
            ax, attribution, font_size=attribution_size
        )

    return


def get_basemap_gdfs_from_vectortiles(ax, target_crs, zoom=14, clip=True):
    """Download GeoDataFrames from vector tiles.

    Uses basisvisualisierung.niedersachsen.de
    """
    xmin, xmax, ymin, ymax = ax.axis()

    west, east, south, north = contextily.plotting._reproj_bb(
        xmin, xmax, ymin, ymax, target_crs, "EPSG:4326"
    )

    tiles = list(mercantile.tiles(west, south, east, north, zoom))

    base_url = (
        "https://basisvisualisierung.niedersachsen.de/"
        "services/basiskarte/v4/tiles/{z}/{x}/{y}.pbf"
    )

    gdf_list = []

    for tile in tiles:

        url = base_url.format(z=tile.z, x=tile.x, y=tile.y)

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                continue

            response.raise_for_status()
            pbf = response.content
            tile_data = mapbox_vector_tile.decode(pbf)

            for layer in tile_data.keys():

                features = vt_bytes_to_geojson(
                    pbf, tile.x, tile.y, tile.z, layer=layer
                )

                gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

                gdf["layer"] = layer
                gdf_list.append(gdf)

        except Exception as e:
            logging.warning(f"Tile error {tile}: {e}")
            continue

    if not gdf_list:
        return gpd.GeoDataFrame()

    gdf = pd.concat(gdf_list)
    gdf = gdf.set_index("layer", append=True).swaplevel()
    gdf = gdf.to_crs(target_crs)

    if clip:
        gdf = gpd.clip(gdf, (xmin, ymin, xmax, ymax))

    return gdf


def evaluate_vector_style_filter(gdf, filter_expr):
    """
    Recursively evaluate a Mapbox GL filter expression against a GeoDataFrame.
    Returns a boolean mask.
    """
    if not isinstance(filter_expr, list) or len(filter_expr) < 2:
        return pd.Series(True, index=gdf.index)

    op = filter_expr[0]

    if op == "all":
        mask = pd.Series(True, index=gdf.index)
        for sub_filter in filter_expr[1:]:
            sub_mask = evaluate_vector_style_filter(gdf, sub_filter)
            try:
                mask &= sub_mask
            except Exception:
                breakpoint()
        return mask
    elif op == "any":
        mask = pd.Series(False, index=gdf.index)
        for sub_filter in filter_expr[1:]:
            sub_mask = evaluate_vector_style_filter(gdf, sub_filter)
            mask |= sub_mask
        return mask
    elif op == "==":
        col, val = filter_expr[1], filter_expr[2]
        if col in gdf.columns:
            return gdf[col] == val
        else:
            return pd.Series(False, index=gdf.index)
    elif op == "!=":
        col, val = filter_expr[1], filter_expr[2]
        return gdf[col] != val
    elif op == "<":
        col, val = filter_expr[1], filter_expr[2]
        return gdf[col] < val
    elif op == "<=":
        col, val = filter_expr[1], filter_expr[2]
        return gdf[col] <= val
    elif op == ">":
        col, val = filter_expr[1], filter_expr[2]
        return gdf[col] > val
    elif op == ">=":
        col, val = filter_expr[1], filter_expr[2]
        return gdf[col] >= val
    elif op == "in":
        col = filter_expr[1]
        vals = filter_expr[2:]
        if col in gdf.columns:
            return gdf[col].isin(vals)
        else:
            return pd.Series(False, index=gdf.index)
    elif op == "!in":
        col = filter_expr[1]
        vals = filter_expr[2:]
        if col in gdf.columns:
            return ~gdf[col].isin(vals)
        else:
            return pd.Series(True, index=gdf.index)
    elif op == "has":
        col = filter_expr[1]
        if col in gdf.columns:
            return gdf[col].notna()
        else:
            return pd.Series(False, index=gdf.index)
    elif op == "!has":
        col = filter_expr[1]
        if col in gdf.columns:
            return gdf[col].isna()
        else:
            return pd.Series(True, index=gdf.index)
    else:
        # Unknown operator, return True to include all
        logging.warning(f"Unknown filter operator: {op}")
        return pd.Series(True, index=gdf.index)


def evaluate_style_value(value, zoom, row=None):
    """
    Minimal Mapbox-style expression evaluator.
    Supports:
      - interpolate (linear, zoom)
      - step (zoom)
      - match (property based)
      - get
      - literal
      - zoom
    """

    # LEGACY STOPS FORMAT
    if isinstance(value, dict) and "stops" in value:
        stops = value["stops"]

        for i in range(len(stops) - 1):
            z0, v0 = stops[i]
            z1, v1 = stops[i + 1]

            if zoom <= z0:
                return v0
            if z0 <= zoom <= z1:
                return v0  # color stops don't interpolate in this style

        return stops[-1][1]

    # Regular
    if not isinstance(value, list):
        return value

    op = value[0]

    # ZOOM
    if op == "zoom":
        return zoom

    # GET (property access)
    if op == "get":
        if row is None:
            return None
        return row.get(value[1])

    # LITERAL
    if op == "literal":
        return value[1]

    # INTERPOLATE (zoom only)
    # ["interpolate", ["linear"], ["zoom"], z0, v0, z1, v1, ...]
    if op == "interpolate":
        input_expr = value[2]

        if input_expr == ["zoom"]:
            stops = value[3:]

            for i in range(0, len(stops) - 2, 2):
                z0, v0 = stops[i], stops[i + 1]
                z1, v1 = stops[i + 2], stops[i + 3]

                if zoom <= z0:
                    return v0
                if z0 <= zoom <= z1:
                    if isinstance(v0, list):
                        # e.g. dasharray interpolation not supported
                        return v0
                    t = (zoom - z0) / (z1 - z0)
                    return v0 + t * (v1 - v0)

            return stops[-1]

    # STEP (zoom only)
    # ["step", ["zoom"], default, z1, v1, z2, v2]
    if op == "step":
        input_expr = value[1]
        if input_expr == ["zoom"]:
            default = value[2]
            stops = value[3:]

            result = default
            for i in range(0, len(stops), 2):
                stop_zoom = stops[i]
                stop_val = stops[i + 1]
                if zoom >= stop_zoom:
                    result = stop_val
                else:
                    break
            return result

    # MATCH
    # ["match", ["get","class"], "motorway", "#f00", ..., default]
    if op == "match":
        key_expr = value[1]
        key = evaluate_style_value(key_expr, zoom, row)
        pairs = value[2:-1]
        default = value[-1]

        for i in range(0, len(pairs), 2):
            if key == pairs[i]:
                return pairs[i + 1]
        return default

    return value
