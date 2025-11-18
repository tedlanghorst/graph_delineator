from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage as ndi
from numpy import ceil, floor
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Geod


MIN_PIXELS = 5
MERIT_RES = 0.000833333
PIXEL_AREA = MERIT_RES * MERIT_RES

def _get_largest_polygon(geom):
    """Extract largest polygon from a geometry (handles MultiPolygon)"""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def get_pixel_area_m2(gauge, grid):
    g = Geod(ellps="WGS84")
    lon, lat = gauge['lng'], gauge['lat']
    
    # Four corners of pixel around gauge
    x0, y0 = lon, lat
    x1 = lon + grid.affine[0]
    y1 = lat + grid.affine[4]   # negative
    
    area, _ = g.polygon_area_perimeter(
        [x0, x1, x1, x0], 
        [y0, y0, y1, y1]
    )
    pixel_area_m2 = abs(area)
    
    return pixel_area_m2


def split_catchment_raster(
    gauge: dict,
    basin: int,
    catchment_poly: Polygon,
    flow_dir_path: Path,
    flow_acc_path: Path,
) -> Optional[Polygon]:
    """
    Performs detailed pixel-scale raster-based delineation using pysheds.

    Since gauges are already snapped to MERIT polylines, we skip the snapping step
    and use the exact gauge location for delineation.

    Args:
        gauge_id: Gauge identifier
        basin: Pfafstetter basin code (tells us which raster files to open)
        lat: Latitude of gauge (already snapped to river)
        lng: Longitude of gauge (already snapped to river)
        catchment_poly: Shapely polygon of the catchment to split
        flow_dir_path: Path to flow direction rasters

    Returns:
        gauge_polygon: catchment for the gauge, split at gauge point if needed.

    """
    from pysheds.grid import Grid

    # Get bounding box and adjust to pixel boundaries
    bounds = catchment_poly.bounds
    bounds_list = [float(i) for i in bounds]

    # MERIT-Hydro has 3 arcsecond resolution (1/1200 degree)
    halfpix = MERIT_RES / 2  # Half pixel width

    # Adjust bounds to pixel centers
    bounds_list[0] = floor(bounds_list[0] * 1200) / 1200 - halfpix
    bounds_list[1] = floor(bounds_list[1] * 1200) / 1200 - halfpix
    bounds_list[2] = ceil(bounds_list[2] * 1200) / 1200 + halfpix
    bounds_list[3] = ceil(bounds_list[3] * 1200) / 1200 + halfpix
    bounding_box = tuple(bounds_list)

    # Load flow direction raster (windowed reading)
    flow_file = str(flow_dir_path / f"{basin}.tif")
    grid = Grid.from_raster(flow_file, window=bounding_box, nodata=0)
    fdir = grid.read_raster(flow_file, window=bounding_box, nodata=0)

    facc_file = str(flow_acc_path / f"{basin}.tif")
    facc = grid.read_raster(facc_file, window=bounding_box, nodata=0)

    mymask = grid.rasterize([catchment_poly])

    # Mask land outside catchment
    fdir[mymask == 0] = 0
    facc[mymask == 0] = 0

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # ESRI flow direction standard

    # Get the minimum flow (in pixels) of the mainstem of this polygon.
    min_area_m2 = gauge['min_area_km2'] * 1E6 * 0.9
    pixel_area_m2 = get_pixel_area_m2(gauge, grid)
    facc_thresh = max(min_area_m2 / pixel_area_m2, 100)

    x_snap, y_snap = grid.snap_to_mask(facc > facc_thresh, (gauge['lng'], gauge['lat']))

    # Delineate watershed using pysheds
    try:
        catch = grid.catchment(
            fdir=fdir,
            x=x_snap,
            y=y_snap,
            dirmap=dirmap,
            xytype="coordinate",
            recursionlimit=15000,
        )
        grid.clip_to(catch)
        clipped_catch = grid.view(catch, dtype=np.uint8)
    except Exception as e:
        print(f"Error during catchment delineation for gauge {gauge['id']}: {e}")
        return None, None

    # Convert raster to polygon
    shapes = grid.polygonize(clipped_catch)

    # Collect all polygons from pysheds
    shapely_polygons = []
    for shape, value in shapes:
        poly_coords = [[p[0], p[1]] for p in shape["coordinates"][0]]
        shapely_polygon = Polygon(poly_coords)
        shapely_polygons.append(shapely_polygon)

    # Dissolve multiple polygons if necessary
    if len(shapely_polygons) > 1:
        result_polygon = unary_union(shapely_polygons)
        result_polygon = _get_largest_polygon(result_polygon)
    elif len(shapely_polygons) == 1:
        result_polygon = shapely_polygons[0]
    else:
        print(f"Warning: No polygon generated for gauge {gauge['id']}")
        return None, None
    
    # By definition, pysheds.catchment finds the area UPSTREAM of the pour point.
    # result_polygon IS the upstream polygon.
    upstream_poly = result_polygon

    # The remainder is the downstream portion.
    downstream_poly = catchment_poly.difference(upstream_poly)
    
    # Validate the split based on pixel counts.
    downstream_pixels = downstream_poly.area / PIXEL_AREA
    
    # Check for slivers.
    # Only skip split if DOWNSTREAM portion is tiny (gauge at bottom of catchment).
    if downstream_pixels < MIN_PIXELS:
        # Gauge is at the very bottom - it gets the entire catchment.
        return catchment_poly, False
    
    # Check if we are an internal tributary (graph leaf) of the catchment
    catch_edge = catch & ~ndi.binary_erosion(catch)
    significant_edge_flow = catch_edge & (facc > (facc_thresh*0.95))
    # Label the clusters (Connected Components)
    # structure=np.ones((3,3)) allows diagonal connections (8-connectivity)
    _, num_features = ndi.label(significant_edge_flow, structure=np.ones((3,3)))
    is_leaf = num_features > 1
    
    # Return the delineated upstream polygon.
    return upstream_poly, is_leaf
