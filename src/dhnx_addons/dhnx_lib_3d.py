"""Define function related to 3D geometry.


calculate_3d_building_areas()

Takes a GeoDataFrame with 3D building geometries (e.g. LoD2 CityGML) and
calculates the actual roof area of the building. This achieved
by converting each building geometry into a pyvista mesh.
This is useful for calculating photovoltaic installation potential.

These meshes can also be stored as e.g. '.stl' files for extraction and
further processing in other software.


citygml_remove_TerrainIntersetion()

Potentially fixes an issue with CityGML files that contain
'lod2TerrainIntersection' elements for each
building. This is a MultiLineString of the building footprint and e.g.
in QGIS only that footprint may be shown, instead of the 3D building.

"""
import os
import numpy as np
import logging
from functools import reduce
import shapely
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import dhnx_addons

# Define the logging function
logger = logging.getLogger(__name__)

try:
    import pyvista as pv  # pyvista>=0.44.0
except ImportError as e:
    logger.error(e)
    logger.warning("Package pyvista required for 3d object functions")

# Define the logging function
logger = logging.getLogger(__name__)


def calculate_3d_building_areas(
        gdf, slope_min=0, slope_max=90,
        update_geometry=False,
        col_area_roof_compare=None,
        plot_static_images=False,
        plot_slope_if_issue=False):
    """Calculate actual (tilted) area of building roof surfaces in gdf.

    All surfaces between (including) slope_min and slope_max (in degrees)
    are considered roof surfaces. This function adds the columns
    area_roof, slope_min, slope_max and slope_avg to gdf. The slope_* entries
    differ if the building consists of multiple roof surfaces with different
    slopes.

    It would be possible to exclude surfaces that are e.g. too small or
    that face north, in case of PV-yield estimations. But this is not yet
    implemented.
    """
    if slope_max is None:  # If no limit for slope is defined
        slope_max = 90  # Just use a very large angle

    # idx = gdf.index[5]
    # breakpoint()
    n_buildings = len(gdf)
    for idx in gdf.index:
        logger.debug("Calculate roof area of building index %s of %s",
                     idx, n_buildings-1)
        geometry = gdf.loc[idx].geometry
        if plot_static_images:
            gdf.loc[[idx]].plot()
            plt.show()
        # breakpoint()

        try:
            polygon = multipolygonz_to_pyvista(geometry)
        except ValueError as e:
            logger.error("Conversion from shapely to PyVista failed with "
                         "error '%s'", e)
            continue
        # polygon.plot()
        # polygon.plot(show_edges=True)
        # polygon.plot_normals(show_edges=True)

        # polygon = polygon.fill_holes(hole_size=999)

        # polygon.save(f'polygon_{idx}.stl')

        if plot_static_images:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(polygon, color="orange")
            plotter.show()
            plt.imshow(plotter.image)
            plt.show()

        # Calculate area statistics of polygon object
        polygon, stats = calc_polygon_area_stats(polygon, slope_min, slope_max)
        if stats["area_roof"] == 0:
            # Chances are that all faces point inwards. Flipping faces might
            # help with roof area calculation
            polygon = polygon.flip_faces()
            polygon, stats = calc_polygon_area_stats(
                polygon, slope_min, slope_max)

        for key, value in stats.items():
            gdf.loc[idx, key] = round(value, 3)

        # Perform a comparison, if the data already contained values for
        # the roof area
        if col_area_roof_compare is not None:
            area_roof_comp = gdf.loc[idx, col_area_roof_compare]
            area_roof = stats["area_roof"]
            if ((abs(area_roof_comp - area_roof)/area_roof_comp*100 > 1)
               and area_roof_comp > 1.0):  # Original area larger than 1m²
                logger.debug("Roof area difference: %.0fm² (%.1f%%) "
                             "Ref: %.0fm² Calc: %.0fm²",
                             (area_roof_comp - area_roof),
                             (area_roof_comp - area_roof)/area_roof_comp*100,
                             area_roof_comp, area_roof)
                # breakpoint()

                if plot_slope_if_issue:
                    # Create a plotter
                    plotter = pv.Plotter()
                    # Color polygon mesh based on the 'slope_deg' scalar field
                    plotter.add_mesh(polygon,
                                     # scalars="slope_deg",
                                     # scalars="z_height",
                                     scalars="condition_roof",
                                     # scalars="condition_ground",
                                     cmap="viridis", show_edges=True)
                    plotter.show()
                    # breakpoint()

        if update_geometry:
            geometry_new = convert_pyvista_to_shapely(polygon)
            gdf.loc[idx, gdf.geometry.name] = geometry_new

    return gdf

def calc_polygon_area_stats(polygon, slope_min=0, slope_max=90):
    # Sum up the area, depending on the slope of the surface
    # in a vectorized way for best performance
    polygon = polygon.compute_normals(cell_normals=True, point_normals=False)
    polygon = polygon.compute_cell_sizes(length=False, volume=False)
    areas = polygon.cell_data["Area"]
    z_height = np.array([cell.center[2] for cell in polygon.cell])
    z_height_min = min(z_height)

    surface_normals = polygon.cell_normals
    # Define the vertical vector
    vertical_vector = np.array([0, 0, 1])
    # Calculate the slope angles for all cells
    slopes = np.arccos(np.dot(surface_normals, vertical_vector))
    slopes_deg = np.degrees(slopes)

    # Apply the slope condition for the roofs:
    # Either by defining a degree range to include, or exclude 90° (walls)
    # The tolerance for the 90° (vertical) walls/roofs is very important.
    # A very strickt tolerance will declare some surfaces to roofs
    # that are at a slight angle from 90° and would be called a wall
    # by any observer. However, in the orginal CityGML, it seems that
    # those surfaces are typically counted as roofs.
    condition_roof = (
        (slopes_deg >= slope_min)
        # Option A) Be strict, allow only faces with expected normal
        # orientation that result in slopes < 90°C for roofs
        # & (slopes_deg <= slope_max)
        # Option B): Use slope_max input just as absolute difference to 90°,
        # i.e. allow slopes >90° to be identified as roofs, too.
        # This should not happen if all normals were calculated correctly(?)
        & (abs(90-slopes_deg) >= abs(90-slope_max))
        # & (~np.isclose(slopes_deg, 90, rtol=1e-1))  # Better subjective fit
        & (~np.isclose(slopes_deg, 90, rtol=1e-2))  # Better fit with CityGML definition
        & (~np.isclose(z_height, z_height_min, rtol=1e-2))
        )
    selected_areas = areas[condition_roof]
    selected_slopes = slopes_deg[condition_roof]

    # Sum the areas that meet the condition
    area_roof = np.sum(selected_areas)

    # Also calculate the ground area of the building
    # area_ground = np.sum(areas[(slopes_deg == 180)])
    condition_ground = np.isclose(z_height, z_height_min, rtol=1e-2)
    area_ground = np.sum(areas[condition_ground])

    area_walls = polygon.area - (area_ground + area_roof)
    stats = dict(area_ground=area_ground,
                 area_walls=area_walls,
                 area_roof=area_roof,
                 )
    if len(selected_slopes) > 0:
        stats['slope_min'] = np.min(selected_slopes)
        stats['slope_max'] = np.max(selected_slopes)
        stats['slope_avg'] = np.mean(selected_slopes)
    else:
        stats['slope_min'] = np.nan
        stats['slope_max'] = np.nan
        stats['slope_avg'] = np.nan

    # Add slope data as a scalar field for plotting later on
    polygon.cell_data.set_array(slopes_deg, 'slope_deg')
    polygon.cell_data.set_array(z_height, 'z_height')
    polygon.cell_data.set_array(condition_roof, 'condition_roof')
    polygon.cell_data.set_array(condition_ground, 'condition_ground')
    return polygon, stats


def extrude_3d_mesh(gdf, z=0):
    """Extrude a pyvista mesh from 2d polygons and a z height."""
    def points_2d_to_poly(points, z):
        """Convert a sequence of 2D or 3D coordinates to polydata."""
        faces = [len(points), *range(len(points))]
        if len(points[0]) == 3:  # If points are 3D
            # Replace existing z coordinate
            points = [(x, y, z) for x, y, _ in points]
        else:  # If points are 2D
            points = [(x, y, z) for x, y in points]  # Add z coordinate
        poly = pv.PolyData(points, faces=faces)
        return poly

    if isinstance(z, list):
        z_values = z
    elif isinstance(z, str):
        z_values = gdf[z]
    else:
        z_values = [z] * len(gdf)

    extruded_geometries = []
    for geometry, z in zip(gdf.geometry, z_values):
        # breakpoint()
        # Convert the polygon to a pyvista PolyData object
        polygon = points_2d_to_poly(list(geometry.exterior.coords), z)

        # Add all holes
        if len(geometry.interiors) > 0:
            for interior in geometry.interiors:
                polygon += points_2d_to_poly(list(interior.coords), z)

            # Triangulate poly with all three subpolygons supplying edges
            polygon = polygon.delaunay_2d(edge_source=polygon)

        # Extrude
        extruded = polygon.extrude((0, 0, z), capping=True)
        # extruded.plot()

        # Append the extruded geometry to the list
        extruded_geometries.append(extruded)

    # Replace the 'geometry' column with the extruded geometries
    return extruded_geometries


def convert_shapely_to_pyvista(geometry, clean=True):
    """Convert a shapely MultiPolygonZ into pyvista 3d mesh."""

    # Flatten points and create faces for PyVista
    all_points = []  # Store all points here
    faces = []  # Store face information here
    offset = 0  # Keep track of the index offset

    for geom in geometry.geoms:
        geom_points = list(geom.exterior.coords)
        num_points = len(geom_points)
        all_points.extend(geom_points)
        faces.extend([num_points] + list(range(offset, offset + num_points)))
        offset += num_points

    # Convert the points to a numpy array
    all_points_array = np.array(all_points)

    # Create a PyVista PolyData object
    polygon = pv.PolyData(all_points_array, faces)
    if clean:
        polygon = polygon.clean()
    # polygon.plot()
    # polygon.plot_normals()
    return polygon


def convert_pyvista_to_shapely(mesh):
    """Convert a pyvista 3d mesh into shapely MultiPolygonZ."""
    from shapely.geometry import MultiPolygon, Polygon

    # Extract the vertices from the mesh
    vertices = mesh.points

    # Extract the faces from the mesh
    faces = mesh.faces

    # Create a new Polygon Z for each face
    polygons = []
    i = 0
    while i < len(faces):
        n_points = faces[i]
        face_indices = faces[i + 1 : i + 1 + n_points]
        polygon_points = vertices[face_indices]
        polygon = Polygon(polygon_points)
        polygons.append(polygon)
        i += 1 + n_points

    # Create a MultiPolygon Z from the polygons
    multipolygon = MultiPolygon(polygons)

    return multipolygon


def convert_3d_mesh_to_polygonz(meshes):
    """Convert a list of pyvista 3d meshes into MultiPolygonZ.

    Not yet working correctly."""
    polygons = [convert_pyvista_to_shapely(mesh) for mesh in meshes]

    return polygons



def flatten_to_polygons(geom):
    if isinstance(geom, shapely.geometry.Polygon):
        return [geom]
    elif isinstance(geom, (shapely.geometry.MultiPolygon,
                           shapely.geometry.GeometryCollection)):
        return [p for g in geom.geoms for p in flatten_to_polygons(g)]
    else:
        return []


def plane_equal(a, b, atol=1e-6):
    normals_equal = (
        np.array_equal(a.cell_normals[0], b.cell_normals[0])
        or np.array_equal(a.cell_normals[0], -1 * b.cell_normals[0]))
    center_equal = np.array_equal(a.center, b.center)
    return normals_equal & center_equal


def agg_geoms_to_collection(gdf, col, agg_dict=None, agg_func_other='first'):
    """Aggregate geometries into GeometryCollection for groups in col.

    gml and shp files support the concept of BuildingPart objects,
    where the complete building is made up of individual parts.

    In many cases, we want the complete building to be one single object.
    This function allows to group all geometry by a column that identifies
    which parts belong to each other. Used names are e.g. 'gml_id' or
    'externRef', but you might have to investigate the source file for
    the right column name.

    The resulting geometries within the gdf_collection can be handed to
    multipolygonz_to_pyvista(), which will create a single geometry
    from all the building parts.
    """
    from shapely.geometry import GeometryCollection

    col_geom = gdf.geometry.name
    if agg_dict is None:
        agg_dict = dict()

    agg_dict[col_geom] = lambda g: GeometryCollection(g.to_list())
    for c in gdf.columns:
        if c != col_geom and agg_func_other is not None:
            agg_dict.setdefault(c, agg_func_other)

    gdf_collection = (
        gdf
        .groupby(col, as_index=False)
        .agg(agg_dict)
    ).set_geometry(col_geom).set_crs(gdf.crs)
    return gdf_collection


def multipolygonz_to_pyvista(multipolygonz):
    """Convert a Shapely MultiPolygonZ object to PyVista PolyData.

    The process is accounting for holes in the MultiPolygons, which need to
    be cut into the PolyData meshes. In the context of buildings, these
    holes often occur on roofs, e.g. when an additional structure is
    built on top of a roof surface.

    Cutting these holes should seemingly be solvable by the pyvista operation
    'exterior_plane.boolean_difference(interior_plane)'. However, it often
    throws errors or does not give the expected output.

    Reports related to broken boolean_difference()
    - https://github.com/pyvista/pyvista/issues/3290
    - https://github.com/pyvista/pyvista/discussions/2379

    A manual approach was necessary, where an exterior plane with n
    interiors (holes) is turned into n planes without holes, which are then
    merged into one complete plane (which now has holes).

    Args:
        multipolygonz (shapely.geometry.MultiPolygon): A MultiPolygonZ
        (or GeometryCollection) object.

    Returns:
        pyvista.PolyData: The converted PolyData object.

    """


    if not isinstance(multipolygonz,
                      (shapely.geometry.MultiPolygon,
                       shapely.geometry.collection.GeometryCollection,
                       )):
        raise ValueError("Input must be a shapely MultiPolygon or "
                         "GeometryCollection object, is {}"
                         .format(type(multipolygonz)))

    # Translate (shift) the complete geometry to the origin (0, 0).
    # This seemed to help with precision issues
    anchor = multipolygonz.centroid
    multipolygonz = shapely.affinity.translate(multipolygonz,
                                               xoff=-anchor.x,
                                               yoff=-anchor.y
                                               )

    polygon_list = flatten_to_polygons(multipolygonz)

    meshes = []
    for polygon in polygon_list:
        if not polygon.is_empty:
            # Create a mesh for the exterior
            ring = polygon.exterior

            exterior_coords = np.array(ring.coords).astype(np.float64)
            exterior_lines = pv.lines_from_points(exterior_coords, close=True)
            exterior_triangulated = points_to_mesh(exterior_coords,
                                                   # projection_mesh=None
                                                   )
            # Triangulation with projection is the saver option most of the
            # time, to create a valid surface
            if not np.isclose(exterior_triangulated.points,
                              exterior_lines.points,
                              atol=1e-6).all():
                # If the point of the projected surface do not match with
                # the original points, test triangulation without projection
                breakpoint()
                # exterior_triangulated = points_to_mesh(exterior_coords)

                # faces = [len(exterior_coords)] + list(range(len(exterior_coords)))
                # exterior_mesh = pv.PolyData(exterior_coords, faces, n_faces=1)
                # exterior_delaunay = exterior_mesh.delaunay_2d()

                # if (np.isclose(exterior_delaunay.area, exterior_mesh.area)):
                #     exterior_triangulated = exterior_delaunay

            # plotter = pv.Plotter(off_screen=False)
            # plotter.add_mesh(exterior_triangulated, color='green', show_edges=True)
            # plotter.add_mesh(exterior_lines, color='blue', line_width=3)
            # plotter.add_axes(interactive=True)
            # plotter.show()

            exterior_triangulated = cut_interior_with_rasterization(
                exterior_triangulated, polygon,
                # plot_debug=True,
                )

            # gml and shp files support the concept of BuildingPart objects,
            # where the complete building is made up of individual parts.
            # The issue: Each of those parts has all its walls, so touching
            # surfaces are duplicates that overlap. If we want to create
            # a valid (manifold) object of the complete building, the
            # duplicate surfaces need to be filtered out.
            try:
                filtered = [m for m in meshes
                            if not plane_equal(exterior_triangulated, m)]
            except Exception as e:
                logger.error(e)
                continue
                # breakpoint()
            if len(filtered) < len(meshes):
                meshes = filtered
            else:
                meshes.append(exterior_triangulated)

    # Combine all the polygon meshes
    combined_mesh = meshes[0].merge(meshes[1:]) if meshes else pv.PolyData()
    # combined_mesh.plot(show_edges=True)
    # combined_mesh.plot_normals(show_edges=True)
    # combined_mesh.flip_faces().plot_normals(show_edges=True)
    # breakpoint()

    # If automatic cleaning fixes a non-closed the geometry, use it
    if not combined_mesh.is_manifold:
        combined_mesh2 = combined_mesh.clean(tolerance=0.1, absolute=True)
        combined_mesh2 = combined_mesh2.fill_holes(hole_size=1)
        combined_mesh2 = combined_mesh2.triangulate()
        # combined_mesh2.plot(show_edges=True)
        if combined_mesh2.is_manifold:
            combined_mesh = combined_mesh2

    # Triangulation seems to make sure that the floor's normal points down
    combined_mesh = combined_mesh.triangulate()
    # combined_mesh.plot(show_edges=True)
    # combined_mesh.plot_normals(show_edges=True)

    # Optionally test if the volume is closed ("watertight")
    custom_cleaning = False
    if not combined_mesh.is_manifold and custom_cleaning:
        sep_grid = combined_mesh.separate_cells()
        unique, counts = np.unique(sep_grid.points, axis=0, return_counts=True)
        points_occuring_once = unique[counts == 1]

        # Points must be shared by two or more cells, otherwise they
        # are probably a "loose" point, sticking out of the surface.
        # This prevents the surface from being closed and allowing the
        # correct surface normal calculation
        for point_occuring_once in points_occuring_once:
            plotter = pv.Plotter(off_screen=False)
            plotter.add_mesh(combined_mesh, color='green', show_edges=True)
            plotter.add_mesh(point_occuring_once, color='red',
                              render_points_as_spheres=True, point_size=20,)
            plotter.add_axes(interactive=True)
            plotter.show()

            combined_mesh, ridx = combined_mesh.remove_points(
                [combined_mesh.find_closest_point(point_occuring_once)])
            # combined_mesh.plot(show_edges=True)

            # Just for good measure, also try filling holes:
            combined_mesh = combined_mesh.fill_holes(hole_size=100)

    # Similarily to "loose points", a single edge might also stick out of
    # the building surface. We need to identify the cell this edge creates
    # and remove that.
    boundary_edges = combined_mesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False)
    if boundary_edges.n_points != 0 or boundary_edges.n_cells != 0:
        # plotter = pv.Plotter(off_screen=False)
        # plotter.add_mesh(combined_mesh, color='green', show_edges=True)
        # plotter.add_mesh(boundary_edges, color='red', line_width=6)
        # plotter.add_axes(interactive=True)
        # plotter.show()

        if len(boundary_edges.points) == 2:
            line = boundary_edges.points
            arrays = [combined_mesh.point_cell_ids(
                combined_mesh.find_closest_point(point)) for point in line]

            # Find common values across all arrays
            cell_ids = reduce(np.intersect1d, arrays)
            if len(cell_ids) == 1:
                combined_mesh = combined_mesh.remove_cells(cell_ids[0])
        else:
            # breakpoint()
            pass

    # Extract the non-manifold edges (where the mesh is not closed)
    non_manifold_edges = combined_mesh.extract_feature_edges(
        boundary_edges=False, non_manifold_edges=True,
        feature_edges=False, manifold_edges=False)

    # Translate (shift) the complete geometry back to initial point
    combined_mesh = combined_mesh.translate((anchor.x, anchor.y, 0))

    return combined_mesh


def normals_point_inwards(mesh, threshold=0.9):
    # Ensure face normals exist
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)

    centroid = mesh.center
    centers = mesh.cell_centers().points
    normals = mesh.cell_normals

    # Vector from interior to face
    vectors = centers - centroid

    # Normalize for safety
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    dots = np.einsum("ij,ij->i", normals, vectors)

    # normals pointing inward → dot < 0
    inward_ratio = np.mean(dots < 0)

    return inward_ratio > threshold


def points_to_mesh(points, projection_mesh="self"):
    # Projection into a single plane before triangulation is important to
    # prevent issues from precision errors
    exterior_lines = pv.lines_from_points(points, close=True)
    faces = [len(points)] + list(range(len(points)))
    mesh = pv.PolyData(points, faces)
    mesh.points = mesh.points.astype(np.float64)
    if projection_mesh is not None:
        if projection_mesh == "self":
            projection_mesh = mesh

        projection_mesh.points = projection_mesh.points.astype(np.float64)
        mesh_triangulated = (
            mesh
            .project_points_to_plane(
                normal=projection_mesh.cell_normals[0],
                origin=projection_mesh.points[0],
                )
            .triangulate()
            )
    else:
        mesh_triangulated = mesh.triangulate()

    # Force the points in the triangulated mesh to their original positions
    mesh_triangulated.points = mesh.points
    # print(np.dot(projection_mesh.cell_normals[0], mesh_triangulated.cell_normals[0]))
    # print(mesh_triangulated.cell_normals)

    # if not np.isclose(mesh.area, mesh_triangulated.area):
    #     print(mesh.area - mesh_triangulated.area)
    #     breakpoint()
    #     plotter = pv.Plotter(off_screen=False)
    #     plotter.add_mesh(mesh, color='blue', line_width=3)
    #     plotter.add_mesh(exterior_lines, color='red', show_edges=True)
    #     # plotter.add_mesh(mesh_triangulated, color='green', show_edges=True)
    #     plotter.add_axes(interactive=True)
    #     plotter.show()

    #     debug_triangulation(mesh_triangulated.points, mesh_triangulated)



    return mesh_triangulated


def cut_interior_with_rasterization(
        exterior_triangulated, polygon, plot_debug=False):
    if len(list(polygon.interiors)) == 0:
        return exterior_triangulated

    z = exterior_triangulated.center[2]  # Temporary uniform height
    polygon = shapely.force_3d(shapely.force_2d(polygon), z=z)
    grid_size = find_optimal_grid_size_for_multipolygon(polygon,
                                                        # plot_debug=True,
                                                        )
    rasterized_multipolygon = rasterize_multipolygon_with_grid(
        polygon, grid_size, z=z)
    if rasterized_multipolygon.is_empty:
        breakpoint()
    interpolated_multipolygon = interpolate_boundaries_for_multipolygon(
        rasterized_multipolygon, polygon, grid_size)

    if plot_debug:
        gdf = gpd.GeoDataFrame({'geometry': [polygon]})
        rasterized_gdf = gpd.GeoDataFrame(
            {'geometry': [rasterized_multipolygon]})
        interpolated_gdf = gpd.GeoDataFrame(
            {'geometry': [interpolated_multipolygon]})
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        gdf.plot(ax=ax[0], color="blue", edgecolor="black")
        ax[0].set_title('Original Multipolygon')
        rasterized_gdf.plot(ax=ax[1], color="green", edgecolor="black")
        ax[1].set_title('Rasterized Multipolygon')
        interpolated_gdf.plot(ax=ax[2], color="red", edgecolor="black")
        ax[2].set_title('Interpolated Multipolygon')
        plt.show()

    # Combine the rasterized and interpolated polygons into a pyvista mesh
    meshes = []
    for polygon_cell in interpolated_multipolygon.geoms:
        # Create a mesh for the exterior
        ring = polygon_cell.exterior
        mesh_coords = np.array(ring.coords)
        mesh_triangulated = points_to_mesh(mesh_coords)
        # Normals of planes must point in same direction
        if (np.dot(exterior_triangulated.cell_normals[0],
                   mesh_triangulated.cell_normals[0])) != 1:
            # mesh_triangulated.flip_normals()
            mesh_triangulated.flip_faces(inplace=True)

        meshes.append(mesh_triangulated)

    # Combine all the polygon meshes
    combined_mesh = meshes[0].merge(meshes[1:]) if meshes else pv.PolyData()

    # Compare the (projected into 2D plane!) area of input and output
    if not np.isclose(polygon.area, combined_mesh.area):
        logger.error("Mismatch of areas in rasterization of interior")
        # breakpoint()

    # Restore z-height information (and possibly tilt the surface)
    # combined_mesh = combined_mesh.project_points_to_plane(
        # normal=exterior_triangulated.cell_normals[0],
        # origin=exterior_triangulated.points[0],
        # normal=exterior_triangulated.cell_normals.mean(axis=0),
        # origin=exterior_triangulated.center,
        # )
    # The height information can be restored by projection onto the original
    # surface, which is defined by a point and a normal vector. However,
    # the triangulated exterior surface can in fact have different normal
    # vectors. Here a separate reference plane with one "mean" normal vector
    # is contructed from all the points of the surface.
    plane_reference = pv.lines_from_points(exterior_triangulated.points,
                                           close=True).delaunay_2d()
    combined_mesh = custom_plane_projection(combined_mesh, plane_reference)
    # combined_mesh = custom_plane_projection(combined_mesh, exterior_triangulated)
    # pv.PolyData(exterior_triangulated.points.delauny_2d()).plot()

    # plot_meshes(exterior_triangulated, combined_mesh, plane_reference)
    # breakpoint()
    # exterior_triangulated.plot(show_edges=True)
    # combined_mesh.plot(show_edges=True)
    # exterior_triangulated.plot_normals()
    # combined_mesh.plot_normals()
    # exterior_triangulated.plot_normals(show_edges=True)
    return combined_mesh

def find_optimal_grid_size_for_multipolygon(multipolygon, min_grid_size=0.0001,
                                            plot_debug=False):
    """Function to find optimal grid size for MultiPolygon."""
    if isinstance(multipolygon, shapely.geometry.Polygon):
        multipolygon = shapely.geometry.MultiPolygon([multipolygon])

    minx, miny, maxx, maxy = multipolygon.bounds
    grid_size = min(maxx-minx, maxy-miny) / 2
    while grid_size > min_grid_size:

        if grid_size >= 20:
            step = 10
        elif grid_size >= 2:
            step = 1
        elif grid_size >= 0.2:
            step = 0.1
        else:
            step = 0.01

        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)

        violates_condition = False
        for poly in multipolygon.geoms:
            for x in x_coords:
                for y in y_coords:
                    candidate = shapely.geometry.box(x, y, x + grid_size, y + grid_size)
                    if poly.intersects(candidate):
                        if plot_debug:
                            dhnx_addons.plot_geometries([poly, candidate], title=grid_size)
                        if any(hole.within(candidate) for hole in poly.interiors):
                            violates_condition = True
                            break
                if violates_condition:
                    break
            if violates_condition:
                break

        if not violates_condition:
            return grid_size

        grid_size -= step

    if grid_size < min_grid_size:
        breakpoint()

    return grid_size


def rasterize_multipolygon_with_grid(multipolygon, grid_size, z=None):
    """Rasterize the multipolygon using the optimal grid size."""
    minx, miny, maxx, maxy = multipolygon.bounds
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    polygons = []
    for x in x_coords:
        for y in y_coords:
            candidate = shapely.geometry.box(x, y, x + grid_size, y + grid_size)
            if z is not None:
                candidate = shapely.force_3d(candidate, z=z)
            if multipolygon.intersects(candidate):
                polygons.append(candidate)

    result_multipolygon = shapely.geometry.MultiPolygon(polygons)
    return result_multipolygon

def interpolate_boundaries_for_multipolygon(
        rasterized_multipolygon, multipolygon, grid_size):
    """interpolate the boundaries of the rasterized multipolygon."""
    if isinstance(multipolygon, shapely.geometry.Polygon):
        multipolygon = shapely.geometry.MultiPolygon([multipolygon])

    interpolated_polygons = []

    for square in rasterized_multipolygon.geoms:
        intersects_exterior = any(poly.exterior.intersects(square) for poly in multipolygon.geoms)
        intersects_hole = any(hole.intersects(square) for poly in multipolygon.geoms for hole in poly.interiors)

        if intersects_exterior or intersects_hole:
            # Adjust the boundary squares by intersecting them with the multipolygon
            interpolated_square = square.intersection(multipolygon)
            # Only add the square if it's a polygon
            if isinstance(interpolated_square, shapely.geometry.Polygon):
                interpolated_polygons.append(interpolated_square)
            elif isinstance(interpolated_square, shapely.geometry.MultiPolygon):
                interpolated_polygons.extend(interpolated_square.geoms)
        else:
            # Keep the interior squares as they are, if they are polygons
            if isinstance(square, shapely.geometry.Polygon):
                interpolated_polygons.append(square)

    # Create a MultiPolygon from the list of polygons
    result_multipolygon = shapely.geometry.MultiPolygon(interpolated_polygons)

    return result_multipolygon

def custom_plane_projection(plane_a, plane_b, atol=1e-3):
    """Project points from plane A onto B along the normal vector of A.

    This is different from pyvista's project_points_to_plane(), which
    projects along the normal vector of B.
    """
    # Get the normal of Plane A
    # normal_a = plane_a.cell_normals.mean(axis=0)
    # normal_a = plane_a.cell_normals[0]
    normal_a = mean_without_outliers(plane_a.cell_normals)

    # Get the coefficients of the plane equation for Plane B (Ax + By + Cz + D = 0)
    # normal_b = plane_b.cell_normals[0]
    normal_b = mean_without_outliers(plane_b.cell_normals)
    # point_on_b = plane_b.center
    point_on_b = plane_b.points[0]
    D = -np.dot(normal_b, point_on_b)

    # Project each point
    new_points = []
    for point in plane_a.points:
        # Line equation: point + t * normal_a
        # Plane equation: normal_b . (point + t * normal_a) + D = 0
        # Solve for t
        t = -(np.dot(normal_b, point) + D) / np.dot(normal_b, normal_a)
        projected_point = point + t * normal_a
        new_points.append(projected_point)

    plane_a_new = plane_a.copy()
    plane_a_new.points = np.array(new_points)

    # As a separate step, force the close points from both planes to
    # actually be equal
    for i, point_on_a in enumerate(plane_a_new.points):
        closest, = np.where(np.isclose(point_on_a, plane_b.points, atol=atol
                                       ).all(axis=1))
        if len(closest) > 0:
            plane_a_new.points[i] = plane_b.points[closest[0]]

    return plane_a_new

def mean_without_outliers(data, std_dev_threshold=1):
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)

    # Create a mask for data within the threshold
    mask = np.all(np.abs(data - means) <= std_dev_threshold * std_devs, axis=1)

    # Filter the data and compute the mean
    filtered_data = data[mask]
    mean_filtered = np.mean(filtered_data, axis=0)
    while np.isnan(mean_filtered).any():
        mean_filtered = mean_without_outliers(
            data, std_dev_threshold=std_dev_threshold+0.1)
    return mean_filtered


def citygml_remove_TerrainIntersetion(file_in, file_out):
    """Fix an issue with the CityGML files.

    They contain 'lod2TerrainIntersection' elements for each
    building (a MultiLineString of the building footprint).
    Both my python installation with gdal 3.7.0 and QGIS 3.32.0 load
    only these MultiLineStrings from the gml file, not the 3d-building.
    As a workaround, removing the 'lod2TerrainIntersection' elements fixes
    the issue.
    """
    from lxml import etree as ET  # Keeps the original namespaces on write

    # Load and parse the GML file
    tree = ET.parse(file_in)
    root = tree.getroot()

    # Define the file path and namespaces
    # namespaces = {
    #     'xsi': r"http://www.w3.org/2001/XMLSchema-instance",
    #     'gml': r"http://www.opengis.net/gml",
    #     'bldg': r"http://www.opengis.net/citygml/building/2.0",
    #     'gen': r"http://www.opengis.net/citygml/generics/2.0",
    #     'core': r"http://www.opengis.net/citygml/1.0"
    # }
    namespaces = dict([node for _, node in ET.iterparse(
        file_in, events=['start-ns'])])

    def remove_lod2TerrainIntersections(building, namespaces):
        """Remove all <bldg:lod2TerrainIntersection> elements from building."""
        lod2TerrainIntersections = building.findall(
            './/bldg:lod2TerrainIntersection', namespaces)
        for lod2TerrainIntersection in lod2TerrainIntersections:
            try:
                building.remove(lod2TerrainIntersection)
            except ValueError:
                # This can fail if the found lod2TerrainIntersection is
                # assiciated with a BuildingPart object of the current
                # building. We check those separately
                pass

    # Find all buildings and their roof surfaces
    buildings = root.findall('.//bldg:Building', namespaces)
    if len(buildings) == 0:
        logger.info("Skipping file (no buildings): %s", file_in)
        return

    building_ids = [building.attrib['{http://www.opengis.net/gml}id']
                    for building in buildings]

    # Iterate over the buildings
    for building_id in building_ids:
        building = root.find(f".//bldg:Building[@gml:id='{building_id}']",
                             namespaces)
        # breakpoint()
        if building is not None:
            # Remove <bldg:lod2TerrainIntersection> elements from the building
            remove_lod2TerrainIntersections(building, namespaces)

            # Some buildings consist of BuildingPart objects
            building_parts = building.findall('.//bldg:BuildingPart', namespaces)
            for building_part in building_parts:
                remove_lod2TerrainIntersections(building_part, namespaces)

    # Save the modified XML tree to a new GML file
    if not os.path.exists(os.path.dirname(file_out)):
        os.makedirs(os.path.dirname(file_out))

    tree.write(file_out, encoding='UTF-8', xml_declaration=True)
    logger.info("Modified file saved at: %s", file_out)

    # gdf = dhnx_addons.load_xml_geodata(file_path, crs="EPSG:25832")
    # gdf_edit = dhnx_addons.load_xml_geodata(new_file_path, crs="EPSG:25832")
    # breakpoint()
    return
