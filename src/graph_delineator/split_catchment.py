from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ceil, floor
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import wkb


def _get_largest_polygon(geom):
    """Extract largest polygon from a geometry (handles MultiPolygon)"""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def split_catchment_raster(
    gauge_id: str,
    basin: int,
    lat: float,
    lng: float,
    catchment_poly: Polygon,
    flow_dir_path: Path,
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
        accum_dir_path: Path to flow accumulation rasters
    
    Returns:
        gauge_polygon: Upstream portion of split catchment
    """
    from pysheds.grid import Grid
    
    # Get bounding box and adjust to pixel boundaries
    bounds = catchment_poly.bounds
    bounds_list = [float(i) for i in bounds]
    
    # MERIT-Hydro has 3 arcsecond resolution (1/1200 degree)
    halfpix = 0.000416667  # Half pixel width
    
    # Adjust bounds to pixel centers
    bounds_list[0] = floor(bounds_list[0] * 1200) / 1200 - halfpix
    bounds_list[1] = floor(bounds_list[1] * 1200) / 1200 - halfpix
    bounds_list[2] = ceil(bounds_list[2] * 1200) / 1200 + halfpix
    bounds_list[3] = ceil(bounds_list[3] * 1200) / 1200 + halfpix
    bounding_box = tuple(bounds_list)
    
    # Load flow direction raster (windowed reading)
    flow_file = str(flow_dir_path / f"{basin}.tif")
    try:
        grid = Grid.from_raster(flow_file, window=bounding_box, nodata=0)
    except Exception as e:
        print(f"Error loading flow direction for gauge {gauge_id}: {e}")
        return None
    
    # Rasterize catchment polygon to create mask
    # Convert to simple polygon (no holes, single part)
    hexpoly = catchment_poly.wkb_hex
    poly = wkb.loads(hexpoly, hex=True)
    poly = _get_largest_polygon(poly)
    filled_poly = Polygon(poly.exterior.coords)
    multi_poly = MultiPolygon([filled_poly])
    polygon_list = list(multi_poly.geoms)
    
    mymask = grid.rasterize(polygon_list)
    
    # Load and mask flow direction
    fdir = grid.read_raster(flow_file, window=bounding_box, nodata=0)
    m, n = grid.shape
    for i in range(m):
        for j in range(n):
            if int(mymask[i, j]) == 0:
                fdir[i, j] = 0
    
    # Use gauge location directly (already snapped to river)
    lng_snap = lng
    lat_snap = lat
    
    # Delineate watershed using pysheds
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # ESRI flow direction standard
    
    try:
        catch = grid.catchment(
            fdir=fdir,
            x=lng_snap,
            y=lat_snap,
            dirmap=dirmap,
            xytype="coordinate",
            recursionlimit=15000,
        )
        grid.clip_to(catch)
        clipped_catch = grid.view(catch, dtype=np.uint8)
    except Exception as e:
        print(f"Error during catchment delineation for gauge {gauge_id}: {e}")
        return None
    
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
        print(f"Warning: No polygon generated for gauge {gauge_id}")
        return None
    
    return result_polygon