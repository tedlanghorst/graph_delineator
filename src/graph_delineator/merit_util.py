import geopandas as gpd
import pandas as pd
from pathlib import Path


def load_merit_basin_data(
    basin_id: int, catchment_dir: Path, river_dir: Path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load MERIT-Basins catchment and river data for a single Pfafstetter basin.

    Args:
        basin_id: Pfafstetter level 2 basin code (e.g., 11, 42, 73)
        catchment_dir: Base directory containing MERIT-Basins data
        river_dir: Base directory containing MERIT-Basins river data

    Returns:
        (catchments_gdf, rivers_gdf)
    """
    # Construct paths following MERIT naming convention
    basin_subdir = f"pfaf_{basin_id}_MERIT_Hydro_v07_Basins_v01"

    cat_file = (
        catchment_dir
        / basin_subdir
        / f"cat_pfaf_{basin_id}_MERIT_Hydro_v07_Basins_v01.shp"
    )
    riv_file = (
        river_dir / basin_subdir / f"riv_pfaf_{basin_id}_MERIT_Hydro_v07_Basins_v01.shp"
    )

    if not cat_file.exists():
        raise FileNotFoundError(f"Catchment file not found: {cat_file}")
    if not riv_file.exists():
        raise FileNotFoundError(f"River file not found: {riv_file}")

    print(f"  Loading basin {basin_id}...")
    catchments = gpd.read_file(cat_file)
    rivers = gpd.read_file(riv_file)

    return catchments, rivers


def identify_basins_for_gauges(
    gauges_gdf: gpd.GeoDataFrame, megabasins_path: Path
) -> dict[int, list[str]]:
    """
    Identify which Pfafstetter basins contain the gauges.

    Args:
        gauges_gdf: GeoDataFrame with gauge points
        megabasins_path: Path to megabasins shapefile

    Returns:
        Dict mapping basin_id -> list of gauge_ids in that basin
    """
    megabasins = gpd.read_file(megabasins_path)

    # Spatial join to find which basin each gauge is in
    gauges_with_basins = gpd.sjoin(
        gauges_gdf, megabasins, how="left", predicate="within"
    )

    # Group by basin (assuming megabasins has a 'PFAF_ID' or similar field)
    # Adjust field name based on your shapefile
    basin_field = None
    for field in ["PFAF_ID", "pfaf_id", "basin_id", "BASIN_ID", "HYBAS_ID"]:
        if field in megabasins.columns:
            basin_field = field
            break

    if basin_field is None:
        # Try to infer from first two digits of any ID field
        print(
            "Warning: Could not find basin ID field, attempting to use 'basin' column from gauges"
        )
        if "basin" in gauges_with_basins.columns:
            basin_gauge_map = (
                gauges_with_basins.groupby("basin")["id"].apply(list).to_dict()
            )
            return {int(k): v for k, v in basin_gauge_map.items()}
        else:
            raise ValueError(
                "Could not identify basin field in megabasins shapefile and no 'basin' column in gauges"
            )

    basin_gauge_map = (
        gauges_with_basins.groupby(basin_field)["id"].apply(list).to_dict()
    )

    # Convert to int and handle any NaN basins
    result = {}
    for basin, gauge_list in basin_gauge_map.items():
        if pd.notna(basin):
            result[int(basin)] = gauge_list

    return result
