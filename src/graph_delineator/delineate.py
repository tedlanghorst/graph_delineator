"""
Simplified Watershed Delineation Module
========================================

Store all data in the graph to avoid synchronization issues.
Each node contains its polygon, river geometry, and all attributes.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
import geopandas as gpd
import pandas as pd
import pyproj
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


MERIT_RES = 0.000833333  # 3 arc second resolution in degrees
TINY_AREA_THRESHOLD_KM2 = 0.5 

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


def get_pfaf1_outlet_dict(gauges_gdf: gpd.GeoDataFrame) -> dict[int, dict[str, list[str]]]:
    """
    Identify which Pfafstetter basins contain the gauges.

    Args:
        gauges_gdf: GeoDataFrame with gauge points (must have 'id' column)

    Returns:
        Dict mapping {megabasin_id: {outlet_id: [gauge_ids]}}
    """
    gauges_gdf['pfaf1'] = (gauges_gdf['COMID']//1E7).astype(int)

    megabasin_dict = {}
    for megabasin, group in gauges_gdf.groupby("pfaf1")[["outlet_id", "id"]]:
        megabasin_dict[megabasin] = (
            group.groupby("outlet_id")["id"].apply(list).to_dict()
        )

    return megabasin_dict


def get_network_comids(
    outlet_comid: int, rivers: gpd.GeoDataFrame) -> set:
    """
    Find all COMIDs in the watershed network containing this outlet.

    Uses recursive network traversal to find all upstream catchments from the outlet.
    """

    def addnode(B: list, node_id):
        """Recursively assemble list of upstream unit catchments."""
        B.append(node_id)
        for up_field in ["up1", "up2", "up3", "up4"]:
            up = rivers[up_field].loc[node_id]
            if up != 0:
                addnode(B, up)

    network_comids = []
    addnode(network_comids, outlet_comid)

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

def add_leaf_gauge_to_graph(
    G: nx.DiGraph,
    gauge_id: str,
    lat: float,
    lng: float,
    split_polygon: Polygon,
    remainder_polygon: Polygon,
    original_node_id: str,
) -> bool:
    """
    Inserts a new gauge as a LEAF node (tributary) flowing into an existing node.

    - The new gauge node [gauge_id] gets the 'split_polygon' and its area.
    - The [original_node_id] keeps the 'remainder_polygon' and its reduced area.
    - Flow from other predecessors into [original_node_id] is NOT interrupted.
    - A new edge [gauge_id] -> [original_node_id] is created.
    """
    if original_node_id not in G.nodes:
        print(f"Warning: Node {original_node_id} not in graph")
        return False

    if gauge_id in G.nodes:
        print(f"Error: Node {gauge_id} already exists in graph")
        return False

    original_data = G.nodes[original_node_id]

    # Calculate areas
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    gauge_area = transform(transformer.transform, split_polygon).area / 1e6
    remainder_area = transform(transformer.transform, remainder_polygon).area / 1e6

    # --- No river geometry or length splitting is performed ---
    # The new gauge is a separate tributary, so it doesn't split the
    # original node's river segment. We assume its river geometry
    # and length are not derived from the original node.
    leaf_river_geom = None  # Or could be passed as an argument if available
    leaf_length_km = 0.0  # Or passed as an argument

    # Create the new gauge (leaf) node
    G.add_node(
        gauge_id,
        polygon=split_polygon,
        river_geom=leaf_river_geom,
        area_km2=gauge_area,
        length_km=leaf_length_km,
        node_type="leaf_gauge",
        is_gauge=True,
        lat=lat,
        lng=lng,
        original_comid=original_data.get("original_comid"),
    )

    # Update original node with the remainder polygon and area
    # We DO NOT modify its river_geom or length_km.
    G.nodes[original_node_id].update(
        {
            "polygon": remainder_polygon,
            "area_km2": remainder_area,
            "node_type": "tributary_junction",  # Mark as modified
        }
    )

    # Rewire edges: Add the new leaf as a predecessor
    # We DO NOT touch existing predecessors.
    G.add_edge(gauge_id, original_node_id)

    return True


def insert_all_gauges(
    G: nx.DiGraph, gauges: gpd.GeoDataFrame, flow_dir_path: Path, flow_acc_path: Path, pfaf2_id: int
) -> Dict[str, Any]:
    """
    Insert all gauges into the graph.
    If a gauge is at the catchment outlet, it replaces the node.
    Otherwise, it splits the node.
    """
    from .split_catchment import split_catchment_raster

    gauge_info = {}
    # Track node replacements: old_node_id -> new_gauge_id
    node_replacements = {}
    
    # COMID is what we assigned during matchups. 
    # node_id links this gauge into the graph and will change as we split and merge polys. 
    gauges['node_id'] = gauges['COMID'].astype(str)

    # Process from upstream (0.0) to downstream (1.0)
    gauges = gauges.sort_values(['node_id', 'position'])

    for _, gauge in gauges.iterrows():
        gauge_id = str(gauge["id"])

        # Follow the replacement chain to find current node
        original_node_id = gauge["node_id"]
        current_node_id = original_node_id
        while current_node_id in node_replacements:
            current_node_id = node_replacements[current_node_id]

        # Get the current polygon (may have been split already)
        current_polygon = G.nodes[current_node_id]["polygon"]

        gauge_polygon, is_leaf = split_catchment_raster(
            gauge=gauge.to_dict(),
            basin=pfaf2_id,
            catchment_poly=current_polygon,
            flow_dir_path=flow_dir_path,
            flow_acc_path=flow_acc_path,
        )

        is_leaf = False

        
        if gauge_polygon is None:
            # Failed, reason will be logged by fn
            continue

        if gauge_polygon == current_polygon:
            # No split needed. Convert the node to a gauge
            convert_node_to_gauge(G, gauge_id, gauge["lat"], gauge["lng"], current_node_id)
            # Track that this node was replaced
            node_replacements[current_node_id] = gauge_id
            gauge_info[gauge_id] = {
                "original_node": original_node_id, 
                "current_node": current_node_id, 
                "method": "replace",
            }
            continue

        # Calculate remainder
        remainder_polygon = current_polygon.difference(gauge_polygon)
        remainder_polygon = remainder_polygon.buffer(-MERIT_RES / 2).buffer(MERIT_RES / 2)

        if is_leaf:
            add_leaf_gauge_to_graph(
                G,
                gauge_id,
                gauge['lat'],
                gauge['lng'],
                gauge_polygon,
                remainder_polygon,
                current_node_id
            )
        else:
            insert_gauge_into_graph(
                G,
                gauge_id,
                gauge["lat"],
                gauge["lng"],
                gauge_polygon,
                remainder_polygon,
                current_node_id,
            )
            gauge_info[gauge_id] = {"original_node": original_node_id, "current_node": current_node_id, "method": "split"}

    return gauge_info

def convert_node_to_gauge(
    G: nx.DiGraph,
    gauge_id: str,
    lat: float,
    lng: float,
    original_node_id: str,
):
    """
    Converts an existing node to a gauge by updating its
    attributes and relabeling its ID in the graph.
    """
    # 1. Update the attributes in-place
    G.nodes[original_node_id].update({
        "node_type": "gauge",
        "is_gauge": True,
        "lat": lat,
        "lng": lng,
    })
    
    # 2. Relabel the node (NetworkX handles all edge rewiring)
    mapping = {original_node_id: gauge_id}
    nx.relabel_nodes(G, mapping, copy=False)



# =============================================================================
# NETWORK CONSOLIDATION
# =============================================================================
def consolidate_graph(
    G: nx.DiGraph,
    target_area: float = 500,
    preserve_gauges: bool = True,
    max_iterations: int = 100
) -> List[Dict[str, Any]]:
    """
    Consolidate river network (DiGraph) by merging small catchments into neighbors
    while targeting a specific catchment area.

    Rules:
      • Small nodes (< target_area) merge ONLY downstream (upward in the network).
      • All upstream siblings must be able to merge together into their common downstream node.
      • Prioritizes merges that get closest to target_area without exceeding it.
      • Gauge nodes can receive merges (upstream → gauge allowed).
      • Gauge nodes cannot be merged into another node.
      • Merging respects flow direction and rewires edges safely.
    """
    merge_history = []

    def can_be_source(n):
        """Node can be merged INTO another node."""
        if preserve_gauges and G.nodes[n].get("is_gauge", False):
            return False
        return True

    def merge_nodes(source, target, reason):
        """Merge `source` node INTO `target` node."""
        if source not in G or target not in G:
            return None

        s = G.nodes[source]
        t = G.nodes[target]

        # --- Merge polygons ---
        if s.get("polygon") is not None and t.get("polygon") is not None:
            merged_polygon = unary_union([s["polygon"], t["polygon"]]) \
                                .buffer(MERIT_RES).buffer(-MERIT_RES)
            t["polygon"] = merged_polygon

        # --- Merge area + length ---
        t["area_km2"] = t.get("area_km2", 0) + s.get("area_km2", 0)
        t["length_km"] = max(t.get("length_km", 0), s.get("length_km", 0))

        # --- Update node type ---
        if t.get("node_type") == "original":
            t["node_type"] = "merged"

        # --- Rewire incoming edges ---
        for pred in list(G.predecessors(source)):
            if pred != target:
                G.add_edge(pred, target)

        # --- Rewire outgoing edges ---
        for succ in list(G.successors(source)):
            if succ != target:
                G.add_edge(target, succ)

        # --- Remove old node ---
        G.remove_node(source)

        return {
            "source": source,
            "target": target,
            "merged_area": s.get("area_km2", 0),
            "reason": reason,
        }

    for _ in range(max_iterations):
        # Group nodes by their downstream neighbor
        downstream_groups = {}
        
        for n in list(G.nodes):
            if not can_be_source(n):
                continue
            
            area_n = G.nodes[n].get("area_km2", 0)
            if area_n >= target_area:
                continue
            
            succs = list(G.successors(n))
            if len(succs) == 1:  # Node has exactly one downstream neighbor
                downstream = succs[0]
                if downstream not in downstream_groups:
                    downstream_groups[downstream] = []
                downstream_groups[downstream].append(n)
        
        if not downstream_groups:
            break
        
        # Find the best merge: the one that gets closest to target_area without exceeding
        best_merge = None
        best_distance = float('inf')
        
        for downstream, upstream_nodes in downstream_groups.items():
            # Check if ALL upstream nodes + downstream node fit within target_area
            total_area = G.nodes[downstream].get("area_km2", 0)
            total_area += sum(G.nodes[n].get("area_km2", 0) for n in upstream_nodes)
            
            if total_area <= target_area:
                # Calculate how close this gets us to the target
                distance = target_area - total_area
                
                if distance < best_distance:
                    best_distance = distance
                    best_merge = (downstream, upstream_nodes, total_area)
        
        if best_merge is None:
            break
        
        # Perform the best merge
        downstream, upstream_nodes, total_area = best_merge
        merges_this_round = []
        
        for upstream in upstream_nodes:
            info = merge_nodes(upstream, downstream, "upstream_group_to_downstream")
            if info:
                merges_this_round.append(info)
        
        merge_history.extend(merges_this_round)


    # Recompute stream order after structural change
    calculate_stream_orders(G)

    # Recompute upstream accumulated areas
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
    gauges_path: str,
    merit_dirs: Dict[str, Path],
    max_area: float = 500,
    consolidate: bool = True,
    preserve_gauges: bool = True,
    output_dir: Path = None,
    save_plots: bool = False,
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
    gauges_path = Path(gauges_path)
    if output_dir is None:
        output_dir = (gauges_path.parent / gauges_path.stem).resolve()

    print("\n" + "=" * 80)
    print("GRAPH-BASED WATERSHED DELINEATION")
    print(f"Data output directory: {output_dir}")
    print("=" * 80)

    
    all_gauges = load_gauges(gauges_path)
    pfaf1_outlets = get_pfaf1_outlet_dict(all_gauges)

    for pfaf1_id, outlet_dict in pfaf1_outlets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing Pfafstetter Basin {pfaf1_id} (contains {len(outlet_dict)} outlets)")
        print(f"{'=' * 60}")

        # Load basin data
        pfaf1_catchments, pfaf1_rivers = load_basin_data(pfaf1_id, merit_dirs)

        # Process each outlet in this basin
        for outlet_id, outlet_gauge_ids in outlet_dict.items():

            # # TESTING
            # if len(outlet_gauge_ids) < 5 or len(outlet_gauge_ids) > 30:
            #     continue

            print(f"Processing outlet {outlet_id} ({len(outlet_gauge_ids)} gauges)")

            gauges = all_gauges[all_gauges["id"].isin(outlet_gauge_ids)]
            gauges = set_gauge_area_range(gauges, pfaf1_rivers)

            final_comid = gauges[gauges['id'] == outlet_id]['COMID'].item()
            comids = get_network_comids(final_comid,  pfaf1_rivers)

            catchments = pfaf1_catchments.loc[pfaf1_catchments.index.isin(comids)]
            rivers = pfaf1_rivers.loc[pfaf1_rivers.index.isin(comids)]

            # Build graph with all geometries
            G = build_graph_with_geometries(catchments, rivers)

            # Insert gauges
            pfaf2_id = int(final_comid // 1E6)
            gauge_info = insert_all_gauges(
                G, gauges, merit_dirs["flow_dir"], merit_dirs["flow_acc"], pfaf2_id
            )

            # Trim network at outlet if it's a gauge
            if outlet_id in G.nodes:
                trim_at_outlet(G, outlet_id)

            # Consolidate if requested
            if consolidate:
                # starting_nodes = len(G.nodes)
                merge_history = consolidate_graph(
                    G, max_area=max_area, preserve_gauges=preserve_gauges
                )
                # ending_nodes = len(G.nodes)
                # print(f"  Consolidated graph from {starting_nodes} to {ending_nodes} nodes")

            # Store result
            result = DelineationResult(graph=G, outlet_id=outlet_id)
            save_result(outlet_id, result, output_dir, save_plots)


    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"{'=' * 80}")


def set_gauge_area_range(gauges, rivers):
    gauges = gauges.copy()
    gauges['min_area_km2'] = 0.0
    gauges['max_area_km2'] = 0.0

    for i, g in gauges.iterrows():
        min_area = rivers.loc[rivers['NextDownID'] == g['COMID'], 'uparea'].sum()
        gauges.loc[i, 'min_area_km2'] = min_area
        gauges.loc[i, 'max_area_km2'] = rivers.loc[g['COMID'], 'uparea']
    
    return gauges



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
    basins_dir = Path(merit_dirs["basins"])

    cat_file = (
        basins_dir / f"cat_pfaf_{int(basin_id)}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    )
    riv_file = (
        basins_dir / f"riv_pfaf_{int(basin_id)}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    )

    catchments = gpd.read_file(cat_file).set_index("COMID").set_crs('EPSG:4326')
    rivers = gpd.read_file(riv_file).set_index("COMID").set_crs('EPSG:4326')

    return catchments, rivers


def save_result(outlet_id: str, result: DelineationResult, output_dir: Path, save_plots: bool):
    """Save all watershed results"""

    # Export to GeoDataFrames
    subbasins, gauges = result.to_geodataframes()

    if not subbasins.empty:
        subbasins_dir = output_dir / 'subbasins'
        subbasins_dir.mkdir(parents=True, exist_ok=True)
        # subbasins.to_file(subbasins_dir / f"{outlet_id}.gpkg", driver='GPKG')
        subbasins.to_parquet(subbasins_dir / f"{outlet_id}_subbasins.parquet")
    if not gauges.empty:
        gauges_dir = output_dir / 'gauges'
        gauges_dir.mkdir(parents=True, exist_ok=True)
        # gauges.to_file(gauges_dir / f"{outlet_id}.gpkg", driver='GPKG')
        gauges.to_parquet(gauges_dir / f"{outlet_id}_gauges.parquet")

    if save_plots and not subbasins.empty and not gauges.empty:
        ids = pd.concat([
            subbasins.loc[subbasins['is_gauge'], 'id'],
            gauges['id']
        ]).unique()

        cmap = plt.get_cmap('tab20')
        id_to_color = {id_val: cmap(i % cmap.N) for i, id_val in enumerate(ids)}

        sub_color = subbasins[subbasins['is_gauge']]['id'].map(id_to_color)
        gauge_color = gauges['id'].map(id_to_color)

        # import numpy as np
        # subbasins['log_area'] = np.log10(subbasins['uparea_km2'])

        fig, ax = plt.subplots(figsize=(6,6))
        subbasins.plot(color='grey', ax=ax)
        subbasins[subbasins['is_gauge']].plot(color=sub_color, ax=ax, edgecolor='black', linewidth=0.5)
        gauges.plot(color=gauge_color, ax=ax, edgecolor='black', linewidth=0.8)

        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"{outlet_id}.png", dpi=300)
        plt.close(fig)

# # =============================================================================
# # MAIN ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     # Example usage
#     merit_dirs = {
#         "basins": Path("/path/to/catchments_and_centerlines"),
#         "flow_dir": Path("/path/to/flow_direction"),
#         "accum": Path("/path/to/accumulation"),
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
