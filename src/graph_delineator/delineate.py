"""
Simplified Watershed Delineation Module
========================================

Store all data in the graph to avoid synchronization issues.
Each node contains its polygon, river geometry, and all attributes.
"""

import networkx as nx
import geopandas as gpd
import pandas as pd
import pyproj
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

MERIT_RES = 0.000833333  # 3 arc second resolution in degrees

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class DelineationResult:
    """Container for delineation results"""

    graph: nx.DiGraph
    outlet_id: str

    def to_geodataframes(
        self, crs="EPSG:4326"
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Export graph to GeoDataFrames for visualization/saving"""
        subbasins = self._export_subbasins(crs)
        gauges = self._export_gauges(crs)
        return subbasins, gauges

    def _export_subbasins(self, crs) -> gpd.GeoDataFrame:
        records = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("polygon") is not None:
                records.append(
                    {
                        "id": node_id,
                        "geometry": data["polygon"],
                        "area_km2": data.get("area_km2", 0),
                        "uparea_km2": data.get("uparea_km2", 0),
                        "node_type": data.get("node_type", "unknown"),
                        "is_gauge": data.get("is_gauge", False),
                        "nextdown": list(self.graph.successors(node_id))[0]
                        if self.graph.out_degree(node_id) > 0
                        else None,
                    }
                )
        return gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame()

    def _export_gauges(self, crs) -> gpd.GeoDataFrame:
        records = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("is_gauge", False):
                records.append(
                    {
                        "id": node_id,
                        "geometry": Point(data["lng"], data["lat"]),
                        "lat": data["lat"],
                        "lng": data["lng"],
                        "area_km2": data.get("area_km2", 0),
                        "uparea_km2": data.get("uparea_km2", 0),
                    }
                )
        return gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame()


# =============================================================================
# GAUGE UTILS
# =============================================================================


def load_gauges(csv_file: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load gauge locations from CSV.

    CSV must have columns: id, lat, lng
    Optional column: basin (Pfafstetter basin code)
    """
    df = pd.read_csv(csv_file)

    required_cols = ["id", "lat", "lng", "outlet_id"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    geometry = [Point(lng, lat) for lng, lat in zip(df["lng"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

    return gdf


def get_megabasin_outlet_dict(
    gauges_gdf: gpd.GeoDataFrame, megabasins_path: Path
) -> dict[int, dict[str, list[str]]]:
    """
    Identify which Pfafstetter basins contain the gauges.

    Args:
        gauges_gdf: GeoDataFrame with gauge points (must have 'id' column)
        megabasins_path: Path to megabasins shapefile

    Returns:
        Dict mapping {megabasin_id: {outlet_id: [gauge_ids]}}
    """
    megabasins = gpd.read_file(megabasins_path)
    gauges_with_basins = gpd.sjoin(
        gauges_gdf, megabasins, how="left", predicate="within"
    )

    megabasin_dict = {}
    for megabasin, group in gauges_with_basins.groupby("BASIN")[["outlet_id", "id"]]:
        megabasin_dict[megabasin] = (
            group.groupby("outlet_id")["id"].apply(list).to_dict()
        )

    return megabasin_dict


def get_outlet_comids(
    gauges: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame
) -> set:
    """
    Find all COMIDs in the watershed network containing these gauges.

    Uses network traversal to find all upstream catchments from the outlet.
    """
    joined = gpd.sjoin(gauges, catchments.reset_index(), how="left", predicate="within")
    gauge_comids = set(joined["COMID"].dropna().unique())

    if not gauge_comids:
        return set()

    # Build network graph
    G = nx.DiGraph()
    for comid in rivers.index:
        nextdown = rivers.loc[comid, "NextDownID"]
        if nextdown != "0" and nextdown in rivers.index:
            G.add_edge(comid, nextdown)

    # Find all upstream catchments
    network_comids = set()
    for start_comid in gauge_comids:
        if start_comid in G:
            # Add all ancestors (upstream)
            ancestors = {int(a) for a in nx.ancestors(G, start_comid)}
            network_comids.update(ancestors)

            # Add all descendants (downstream)
            descendants = {int(d) for d in nx.descendants(G, start_comid)}
            network_comids.update(descendants)

            # Add the node itself
            network_comids.add(int(start_comid))

    return network_comids


# =============================================================================
# GRAPH BUILDING WITH GEOMETRIES
# =============================================================================


def build_graph_with_geometries(
    catchments: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame
) -> nx.DiGraph:
    """
    Build graph with all geometries stored in nodes.

    Each node contains:
    - polygon: catchment geometry
    - river_geom: river line geometry
    - area_km2: catchment area
    - length_km: river length
    - original_comid: original catchment ID
    """
    G = nx.DiGraph()

    # Project to equal area for accurate area calculation
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)

    for comid in catchments.index:
        if comid not in rivers.index:
            continue

        catchment = catchments.loc[comid]
        river = rivers.loc[comid]

        # Calculate area in km²
        poly_projected = transform(transformer.transform, catchment.geometry)
        area_km2 = poly_projected.area / 1e6

        # Add node with complete data
        G.add_node(
            str(comid),  # Use string IDs for consistency
            polygon=catchment.geometry,
            river_geom=river.geometry,
            area_km2=area_km2,
            length_km=river["lengthkm"],
            node_type="original",
            is_gauge=False,
            original_comid=comid,
            nextdown=str(river["NextDownID"]) if river["NextDownID"] != "0" else None,
        )

    # Add edges based on river network
    for node_id in G.nodes():
        nextdown = G.nodes[node_id].get("nextdown")
        if nextdown and nextdown in G.nodes:
            G.add_edge(node_id, nextdown)

    return G


# =============================================================================
# GAUGE INSERTION WITH POLYGON SPLITTING
# =============================================================================


def insert_gauge_into_graph(
    G: nx.DiGraph,
    gauge_id: str,
    lat: float,
    lng: float,
    split_polygon: Polygon,
    remainder_polygon: Polygon,
    original_node_id: str,
) -> bool:
    """
    Insert a gauge by splitting a node's polygon.

    The gauge gets the upstream polygon, the original node keeps the downstream remainder.
    """
    if original_node_id not in G.nodes:
        print(f"Warning: Node {original_node_id} not in graph")
        return False

    original_data = G.nodes[original_node_id]

    # Calculate areas
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    gauge_area = transform(transformer.transform, split_polygon).area / 1e6
    remainder_area = transform(transformer.transform, remainder_polygon).area / 1e6

    # Split the river geometry at the gauge point
    gauge_point = Point(lng, lat)
    river_geom = original_data["river_geom"]

    # Find split point on river
    if river_geom:
        distance_along = river_geom.project(gauge_point)
        total_length = river_geom.length
        split_ratio = distance_along / total_length if total_length > 0 else 0.5

        # Split river into two segments
        if split_ratio > 0 and split_ratio < 1:
            coords = list(river_geom.coords)
            # Interpolate split point
            split_point = river_geom.interpolate(distance_along)

            # Create upstream and downstream river segments
            upstream_river = LineString([coords[0], split_point.coords[0]])
            downstream_river = LineString([split_point.coords[0], coords[-1]])
        else:
            upstream_river = river_geom
            downstream_river = river_geom
    else:
        upstream_river = None
        downstream_river = None
        split_ratio = 0.5

    # Get network connections
    predecessors = list(G.predecessors(original_node_id))

    # Create gauge node with upstream polygon
    G.add_node(
        gauge_id,
        polygon=split_polygon,
        river_geom=upstream_river,
        area_km2=gauge_area,
        length_km=original_data["length_km"] * split_ratio,
        node_type="gauge",
        is_gauge=True,
        lat=lat,
        lng=lng,
        original_comid=original_data["original_comid"],
    )

    # Update original node with downstream polygon
    G.nodes[original_node_id].update(
        {
            "polygon": remainder_polygon,
            "river_geom": downstream_river,
            "area_km2": remainder_area,
            "length_km": original_data["length_km"] * (1 - split_ratio),
            "node_type": "split_downstream",
        }
    )

    # Rewire edges
    # Remove old edges
    for pred in predecessors:
        G.remove_edge(pred, original_node_id)

    # Add new edges: predecessors -> gauge -> original
    for pred in predecessors:
        G.add_edge(pred, gauge_id)
    G.add_edge(gauge_id, original_node_id)

    return True


def insert_all_gauges(
    G: nx.DiGraph, gauges_gdf: gpd.GeoDataFrame, flow_dir_path: Path, basin_id: int
) -> Dict[str, Any]:
    """
    Insert all gauges into the graph.
    """
    from .split_catchment import split_catchment_raster

    gauge_info = {}

    # Find which catchment each gauge falls in
    catchment_polys = []
    node_ids = []
    for node_id, data in G.nodes(data=True):
        if data.get("polygon"):
            catchment_polys.append(data["polygon"])
            node_ids.append(node_id)

    catchments_gdf = gpd.GeoDataFrame(
        {"node_id": node_ids}, geometry=catchment_polys, crs=gauges_gdf.crs
    )

    # Spatial join
    gauges_with_nodes = gpd.sjoin(
        gauges_gdf, catchments_gdf, how="left", predicate="within"
    )

    for _, gauge in gauges_with_nodes.iterrows():
        if pd.isna(gauge.get("node_id")):
            print(f"  Warning: Gauge {gauge['id']} not in any catchment")
            continue

        node_id = gauge["node_id"]
        gauge_id = gauge["id"]

        # Get the catchment polygon
        original_poly = G.nodes[node_id]["polygon"]

        # Split using raster method
        gauge_polygon = split_catchment_raster(
            gauge_id=gauge_id,
            basin=basin_id,
            lat=gauge["lat"],
            lng=gauge["lng"],
            catchment_poly=original_poly,
            flow_dir_path=flow_dir_path,
        )

        if gauge_polygon is None:
            print(f"  Warning: Could not split catchment for gauge {gauge_id}")
            continue

        # Calculate remainder
        remainder_polygon = original_poly.difference(gauge_polygon)
        remainder_polygon = remainder_polygon.buffer(-MERIT_RES / 2).buffer(
            MERIT_RES / 2
        )  # Opening op. to clean up small errors

        # Insert into graph
        success = insert_gauge_into_graph(
            G,
            gauge_id,
            gauge["lat"],
            gauge["lng"],
            gauge_polygon,
            remainder_polygon,
            node_id,
        )

        if success:
            gauge_info[gauge_id] = {
                "original_node": node_id,
                "gauge_polygon": gauge_polygon,
                "remainder_polygon": remainder_polygon,
            }

    print(f"  Inserted {len(gauge_info)} gauges")

    return gauge_info


# =============================================================================
# NETWORK CONSOLIDATION
# =============================================================================
def consolidate_graph(
    G: nx.DiGraph, max_area: float = 500, preserve_gauges: bool = True
) -> List[Dict[str, Any]]:
    """
    Consolidate small catchments by merging nodes.
    """
    merge_history = []

    def can_merge(node_id):
        """Check if node can be merged"""
        if preserve_gauges and G.nodes[node_id].get("is_gauge", False):
            return False
        return True

    def merge_nodes(source, target, merge_type):
        """Merge source node into target node"""
        if source not in G.nodes or target not in G.nodes:
            return None

        source_data = G.nodes[source]
        target_data = G.nodes[target]

        # Merge polygons
        if source_data.get("polygon") and target_data.get("polygon"):
            merged_polygon = unary_union(
                [source_data["polygon"], target_data["polygon"]]
            ).buffer(MERIT_RES).buffer(-MERIT_RES) # Closure op. to fill holes.
            if isinstance(merged_polygon, MultiPolygon):
                merged_polygon = max(merged_polygon.geoms, key=lambda p: p.area)
            target_data["polygon"] = merged_polygon

        # Merge river geometries (keep the longer one or combine)
        if source_data.get("river_geom") and target_data.get("river_geom"):
            # For simplicity, keep the target's river geometry
            # Could be improved to actually merge the lines
            pass

        # Sum areas
        target_data["area_km2"] = target_data.get("area_km2", 0) + source_data.get(
            "area_km2", 0
        )
        target_data["length_km"] = max(
            target_data.get("length_km", 0), source_data.get("length_km", 0)
        )

        # Update node type
        if target_data.get("node_type") == "original":
            target_data["node_type"] = "merged"

        # Rewire edges
        for pred in list(G.predecessors(source)):
            if pred != target:
                G.add_edge(pred, target)

        for succ in list(G.successors(source)):
            if succ != target:
                G.add_edge(target, succ)

        # Remove source node
        G.remove_node(source)

        return {
            "source": source,
            "target": target,
            "merge_type": merge_type,
            "merged_area": source_data.get("area_km2", 0),
        }

    # Calculate stream orders
    calculate_stream_orders(G)

    # Iterative consolidation
    max_iterations = 100
    for iteration in range(max_iterations):
        merges_this_round = []

        # Find small leaf nodes
        leaf_nodes = [
            n
            for n in G.nodes()
            if G.out_degree(n) == 0 or G.nodes[n].get("strahler_order", 1) == 1
        ]

        for leaf in leaf_nodes:
            if not can_merge(leaf):
                continue

            if G.nodes[leaf].get("area_km2", 0) >= max_area:
                continue

            # Find best merge candidate (usually downstream neighbor)
            successors = list(G.successors(leaf))
            if successors:
                target = successors[0]
                combined_area = G.nodes[leaf].get("area_km2", 0) + G.nodes[target].get(
                    "area_km2", 0
                )

                if combined_area < max_area:
                    merge_info = merge_nodes(leaf, target, "leaf_downstream")
                    if merge_info:
                        merges_this_round.append(merge_info)

        merge_history.extend(merges_this_round)

        if not merges_this_round:
            break

        # Recalculate stream orders after merging
        calculate_stream_orders(G)

    print(f"  Completed {len(merge_history)} merges in {iteration + 1} iterations")

    # Recalculate upstream areas
    calculate_upstream_areas(G)

    return merge_history


def calculate_stream_orders(G: nx.DiGraph):
    """Calculate Strahler and Shreve stream orders"""
    # Sort nodes topologically (upstream to downstream)
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXError:
        # Graph has cycles, use all nodes
        sorted_nodes = list(G.nodes())

    for node in sorted_nodes:
        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Headwater node
            G.nodes[node]["strahler_order"] = 1
            G.nodes[node]["shreve_order"] = 1
        else:
            # Calculate Shreve (sum of upstream orders)
            shreve = sum(G.nodes[p].get("shreve_order", 1) for p in predecessors)
            G.nodes[node]["shreve_order"] = shreve

            # Calculate Strahler
            upstream_orders = [
                G.nodes[p].get("strahler_order", 1) for p in predecessors
            ]
            max_order = max(upstream_orders)
            count_max = upstream_orders.count(max_order)

            if count_max > 1:
                G.nodes[node]["strahler_order"] = max_order + 1
            else:
                G.nodes[node]["strahler_order"] = max_order


def calculate_upstream_areas(G: nx.DiGraph):
    """Calculate cumulative upstream area for each node"""
    # Sort nodes topologically (upstream to downstream)
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXError:
        sorted_nodes = list(G.nodes())

    for node in sorted_nodes:
        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Headwater - upstream area is just its own area
            G.nodes[node]["uparea_km2"] = G.nodes[node].get("area_km2", 0)
        else:
            # Sum all upstream areas plus own area
            upstream_total = sum(G.nodes[p].get("uparea_km2", 0) for p in predecessors)
            G.nodes[node]["uparea_km2"] = upstream_total + G.nodes[node].get(
                "area_km2", 0
            )


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def delineate_watershed(
    gauges_csv: str,
    merit_dirs: Dict[str, Path],
    max_area: float = 500,
    consolidate: bool = True,
    preserve_gauges: bool = True,
) -> Dict[str, DelineationResult]:
    """
    Main function to delineate watersheds.

    Args:
        gauges_csv: Path to CSV with columns: id, lat, lng, outlet_id
        merit_dirs: Dictionary with paths to MERIT data
        max_area: Maximum area for consolidation (km²)
        consolidate: Whether to merge small catchments
        preserve_gauges: Keep gauge nodes during consolidation

    Returns:
        Dictionary mapping outlet_id to DelineationResult
    """
    print("\n" + "=" * 80)
    print("GRAPH-BASED WATERSHED DELINEATION")
    print("=" * 80)

    all_gauges = load_gauges(gauges_csv)
    megabasins_outlets = get_megabasin_outlet_dict(all_gauges, merit_dirs["megabasins"])

    results = {}

    for megabasin_id, outlet_dict in megabasins_outlets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing Basin {megabasin_id} (contains {len(outlet_dict)} outlets)")
        print(f"{'=' * 60}")

        # Load basin data
        mega_catchments, mega_rivers = load_basin_data(megabasin_id, merit_dirs)

        # Process each outlet in this basin
        for outlet_id, outlet_gauge_ids in outlet_dict.items():
            print(f"Processing outlet {outlet_id} ({len(outlet_gauge_ids)} gauges)")

            gauges = all_gauges[all_gauges["id"].isin(outlet_gauge_ids)]

            comids = get_outlet_comids(gauges, mega_catchments, mega_rivers)

            catchments = mega_catchments.loc[mega_catchments.index.isin(comids)]
            rivers = mega_rivers.loc[mega_rivers.index.isin(comids)]

            # Build graph with all geometries
            G = build_graph_with_geometries(catchments, rivers)

            # Insert gauges
            gauge_info = insert_all_gauges(
                G, gauges, merit_dirs["flow_dir"], megabasin_id
            )

            # Trim network at outlet if it's a gauge
            if outlet_id in G.nodes:
                trim_at_outlet(G, outlet_id)

            # Consolidate if requested
            if consolidate:
                merge_history = consolidate_graph(
                    G, max_area=max_area, preserve_gauges=preserve_gauges
                )
                # print(f"  Consolidation: {len(merge_history)} merges")

            # Store result
            results[outlet_id] = DelineationResult(graph=G, outlet_id=outlet_id)

            print('')

    print(f"\n{'=' * 80}")
    print(f"COMPLETE: Processed {len(results)} watersheds")
    print(f"{'=' * 80}")

    return results


def trim_at_outlet(G: nx.DiGraph, outlet_id: str):
    """Remove all nodes downstream of the outlet"""
    if outlet_id not in G.nodes:
        return

    # Find all downstream nodes
    downstream = list(nx.descendants(G, outlet_id))

    if downstream:
        # Trim any nodes downstream of the lowest gauge.
        for node in downstream:
            G.remove_node(node)

    # Mark as outlet
    G.nodes[outlet_id]["is_outlet"] = True


def load_basin_data(
    basin_id: int, merit_dirs: Dict[str, Path]
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load MERIT catchment and river data for a basin"""
    catchment_dir = Path(merit_dirs["catchment"])
    river_dir = Path(merit_dirs["river"])

    basin_str = f"{basin_id:02d}"
    basin_subdir = f"pfaf_{basin_str}_MERIT_Hydro_v07_Basins_v01"

    cat_file = (
        catchment_dir
        / basin_subdir
        / f"cat_pfaf_{basin_str}_MERIT_Hydro_v07_Basins_v01.shp"
    )
    riv_file = (
        river_dir
        / basin_subdir
        / f"riv_pfaf_{basin_str}_MERIT_Hydro_v07_Basins_v01.shp"
    )

    catchments = gpd.read_file(cat_file).set_index("COMID").set_crs('EPSG:4326')
    rivers = gpd.read_file(riv_file).set_index("COMID").set_crs('EPSG:4326')

    return catchments, rivers


# =============================================================================
# EXPORT AND VISUALIZATION
# =============================================================================


def save_results(results: Dict[str, DelineationResult], output_dir: str):
    """Save all watershed results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for outlet_id, result in results.items():
        outlet_dir = output_path / f"outlet_{outlet_id}"
        outlet_dir.mkdir(exist_ok=True)

        # Export to GeoDataFrames
        subbasins, gauges = result.to_geodataframes()

        # Save files
        if not subbasins.empty:
            subbasins.to_file(outlet_dir / "subbasins.gpkg", driver="GPKG")
        if not gauges.empty:
            gauges.to_file(outlet_dir / "gauges.gpkg", driver="GPKG")

        print(f"  Saved {outlet_id} to {outlet_dir}")


# # =============================================================================
# # MAIN ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     # Example usage
#     merit_dirs = {
#         "catchment": Path("/path/to/catchments"),
#         "river": Path("/path/to/rivers"),
#         "flow_dir": Path("/path/to/flow_direction"),
#         "accum": Path("/path/to/accumulation"),
#         "megabasins": Path("/path/to/megabasins.shp"),
#     }

#     results = delineate_watershed(
#         gauges_csv="gauges.csv",
#         merit_dirs=merit_dirs,
#         max_area=500,
#         consolidate=True,
#         preserve_gauges=True,
#     )

#     # Save results
#     save_results(results, "output")

#     # Example: Access specific watershed
#     if "outlet_001" in results:
#         result = results["outlet_001"]
#         print(f"\nOutlet 001: {result.graph.number_of_nodes()} nodes")

#         # Export to GeoDataFrames for visualization
#         subbasins, gauges = result.to_geodataframes()
#         print(f"  Subbasins: {len(subbasins)}")
#         print(f"  Gauges: {len(gauges)}")
