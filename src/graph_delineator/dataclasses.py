import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from dataclasses import dataclass


@dataclass
class GaugeSplit:
    """Information about how a gauge split a catchment"""
    gauge_id: str
    original_comid: str
    snapped_point: Point  # Gauge location (already on river)
    split_ratio: float  # 0-1, distance along river segment
    gauge_polygon: object  # Upstream polygon from raster delineation
    remainder_polygon: object  # Downstream polygon
    upstream_node: str  # Node upstream of split
    downstream_node: str  # Node downstream of split
    lat: float  # Original gauge lat (already snapped)
    lng: float  # Original gauge lng (already snapped)


@dataclass
class DelineationResult:
    """Container for delineation results"""
    graph: nx.DiGraph
    subbasins: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    gauges: gpd.GeoDataFrame
    gauge_splits: dict[str, GaugeSplit]
    merge_history: list[dict]  # List of merge operations
    
    def __repr__(self):
        return (f"DelineationResult(\n"
                f"  nodes={self.graph.number_of_nodes()}, "
                f"  edges={self.graph.number_of_edges()},\n"
                f"  gauges={len(self.gauges)}, "
                f"  subbasins={len(self.subbasins)}\n"
                f")")