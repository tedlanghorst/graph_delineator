import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from dataclasses import dataclass
from typing import Tuple



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

