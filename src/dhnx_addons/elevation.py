# -*- coding: utf-8 -*-
"""Download elevation data for a given geometry."""
import logging
import geopandas as gpd
import numpy as np
import shapely
import requests
import time

from .dhnx_addons import (
    plot_geometries,
    load_example_area,
    download_streets_from_osm,
    download_buildings_from_osm,
)

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    """Run tests that also show the usage."""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set debug logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Load test data
    gdf_poly = load_example_area()
    gdf_lines = download_streets_from_osm(gdf_poly)
    gdf_points = download_buildings_from_osm(gdf_poly)
    gdf_points.geometry = gdf_points.representative_point()

    # Test with points
    logger.info("Testing with points...")
    logger.debug(f"Points CRS: {gdf_points.crs}, count: {len(gdf_points)}")
    points_elevation = download_elevation_data(gdf_points, show_plot=True)
    if points_elevation is not None:
        logger.info(
            f"Points elevation range: "
            f"{points_elevation['elevation'].min():.1f} to "
            f"{points_elevation['elevation'].max():.1f} meters"
        )

    # Test with lines
    logger.info("Testing with lines...")
    logger.debug(f"Lines CRS: {gdf_lines.crs}, count: {len(gdf_lines)}")
    lines_elevation = download_elevation_data(
        gdf_lines, sampling_dist=0.05, show_plot=True
    )
    if lines_elevation is not None:
        logger.info(
            f"Lines elevation range: "
            f"{lines_elevation['elevation'].min():.1f} to "
            f"{lines_elevation['elevation'].max():.1f} meters"
        )

    # Test with polygons
    logger.info("Testing with polygons...")
    logger.debug(f"Polygons CRS: {gdf_poly.crs}, count: {len(gdf_poly)}")
    poly_elevation = download_elevation_data(
        gdf_poly, sampling_dist=0.05, show_plot=True
    )
    if poly_elevation is not None:
        logger.info(
            f"Polygons elevation range: "
            f"{poly_elevation['elevation'].min():.1f} to "
            f"{poly_elevation['elevation'].max():.1f} meters"
        )


def _get_point_elevations(gdf_points, dataset):
    """Get elevations for a set of points using the OpenTopoData API."""
    gdf_points = gdf_points.to_crs("EPSG:4326")  # (required for the API)
    elevations = []
    batch_size = 100
    for i in range(0, len(gdf_points), batch_size):
        batch = gdf_points.iloc[i:i + batch_size]
        locations = [(point.y, point.x) for point in batch.geometry]
        coords_str = "|".join([f"{lat},{lon}" for lat, lon in locations])
        url = (
            f"https://api.opentopodata.org/v1/{dataset}?locations={coords_str}"
        )

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            batch_elevations = [
                result["elevation"] for result in data["results"]
            ]
            elevations.extend(batch_elevations)
        else:
            if response.status_code == 429:  # Too many requests
                time.sleep(1)  # Wait before retrying
                continue
            raise Exception(
                f"API request failed: {response.status_code} - {response.text}"
            )

        time.sleep(0.5)  # Rate limiting

    return elevations


def _create_grid_cells(points, sampling_dist):
    """Create grid cells (rectangles) from a set of points.

    Args:
        points: List containing [x_grid, y_grid] from np.meshgrid
        sampling_dist: Grid spacing in degrees
    """
    x_grid, y_grid = points
    cells = []
    for i in range(len(y_grid) - 1):
        for j in range(len(x_grid) - 1):
            minx, miny = x_grid[j], y_grid[i]
            maxx, maxy = x_grid[j + 1], y_grid[i + 1]
            cells.append(shapely.geometry.box(minx, miny, maxx, maxy))
    return cells


def download_elevation_data(
    gdf,
    crs=None,
    sampling_dist=0.05,
    show_plot=False,
    col_elevation="elevation",
    dataset="eudem25m",
):
    """Download elevation data for a region defined by the geometry in gdf.

    Adapts its behavior based on the input geometry type:
    - Points/MultiPoints: Gets elevation for each point directly
    - Lines/MultiLines: Samples points along the line segments
    - Polygons/MultiPolygons: Creates a grid of rectangles within the area

    Args:
        gdf (GeoDataFrame): The geometries to get elevation data for.
        crs (str, optional): Coordinate reference system.
            Defaults to input GDF's CRS.
        sampling_dist (float, optional): Only relevant for Polygons. Is either:
            - A value between 0 and 1: Represents the sampling distance
              relative to the area's size (e.g., 0.05 = 5% of the area's
              width/height)
            - A value > 1: Absolute distance in meters between sampling points
            Defaults to 0.05 (5% of area size).
        dataset (str, optional): Name of dataset from OpenTopoData API.
            Options include 'aster30m', 'eudem25m'. Defaults to 'eudem25m'.
        show_plot (bool, optional): Whether to show a plot of the
            elevation data. Defaults to False.
        col_elevation (str, optional): Name of the elevation column in output.
            Defaults to 'elevation'.

    Returns:
        GeoDataFrame: Contains geometries with elevation data in meters:
            - For points: Original points with elevation values
            - For lines: Line segments with mean elevation
            - For polygons: Grid cells (rectangles) with mean elevation
        Returns None if the API request fails or no valid geometries are found.
    """
    if crs is None:
        crs = gdf.crs

    # Convert to WGS84 (required for the API)
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    bounds = gdf_wgs84.total_bounds

    # Calculate sampling distance
    if 0 < sampling_dist <= 1:
        bbox_meters = gdf.to_crs("EPSG:25832").total_bounds
        width_m = bbox_meters[2] - bbox_meters[0]
        height_m = bbox_meters[3] - bbox_meters[1]
        sampling_dist = min(width_m, height_m) * sampling_dist

    # rough conversion from meters to degrees
    sample_dist_deg = sampling_dist / 111000

    try:
        geom_type = gdf_wgs84.geometry.iloc[0].geom_type
        logger.debug(f"Processing geometry type: {geom_type}")

        if geom_type in ["Point", "MultiPoint"]:
            # For points, get elevation directly
            gdf_points = gdf_wgs84.representative_point()
            elevations = _get_point_elevations(gdf_points, dataset)
            gdf_wgs84[col_elevation] = elevations
            result_gdf = gdf_wgs84

        elif geom_type in ["LineString", "MultiLineString"]:
            # For lines, get elevation for each line segment
            # segments_gdf = gdf_wgs84.segmentize(1).explode()
            segments_gdf = gdf_wgs84.explode()
            # Get elevations for all points at once
            gdf_points = segments_gdf.representative_point()
            elevations = _get_point_elevations(gdf_points, dataset)
            segments_gdf[col_elevation] = elevations
            result_gdf = segments_gdf

        elif geom_type in ["Polygon", "MultiPolygon"]:
            # For polygons, create a grid of points and then rectangles
            x = np.arange(bounds[0], bounds[2], sample_dist_deg)
            y = np.arange(bounds[1], bounds[3], sample_dist_deg)
            xx, yy = np.meshgrid(x, y)

            # Create grid cells first
            grid_cells = _create_grid_cells([x, y], sample_dist_deg)
            cells_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")

            # Filter cells that intersect with the polygon
            polygon_union = gdf_wgs84.geometry.union_all()
            cells_in_poly = cells_gdf[cells_gdf.intersects(polygon_union)]

            if len(cells_in_poly) == 0:
                logger.error("No grid cells intersect with the given area")
                return None

            # Get elevations for cell centers
            cell_centers = cells_in_poly.to_crs("EPSG:25832").centroid
            gdf_points = cell_centers.to_crs("EPSG:4326")
            elevations = _get_point_elevations(gdf_points, dataset)

            # Assign elevations to cells
            cells_in_poly[col_elevation] = elevations
            result_gdf = cells_in_poly

        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

        # Convert result back to original CRS
        result_gdf = result_gdf.to_crs(crs)
        logger.debug(
            f"Successfully created elevation data for {len(result_gdf)} "
            "geometries"
        )

        if show_plot:
            plot_geometries(
                [result_gdf, gdf],
                plt_kwargs=[
                    dict(
                        column=col_elevation,
                        legend=True,
                        legend_kwds=dict(label="Elevation [m]"),
                    ),
                    dict(label="Input geometry", alpha=0.5),
                ],
                title="Elevation data",
                set_axis_off=True,
            )

        return result_gdf

    except Exception as e:
        logger.exception(f"Error processing elevation data: {str(e)}")
        return None


if __name__ == "__main__":
    main()
