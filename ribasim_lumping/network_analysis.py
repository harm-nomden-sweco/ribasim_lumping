# pylint: disable=missing-function-docstring
from pydantic import BaseModel
from pathlib import Path
from typing import List, Union, Optional, Any, Tuple
import datetime
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pandas as pd
import numpy as np
import os
import dfm_tools as dfmt
import xarray as xr
import xugrid as xu
import networkx as nx

from .utils.read_simulation_data_utils import (
    get_data_from_simulations_set,
    get_simulation_names_from_dir,
    combine_data_from_simulations_sets,
)
from .utils.generate_basins_areas import create_basins_based_on_split_node_ids


class NetworkAnalysis(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""

    his_data: xr.Dataset = None
    map_data: xr.Dataset = None
    dhydro_edges: gpd.GeoDataFrame = None
    dhydro_nodes: gpd.GeoDataFrame = None
    network_graph: nx.DiGraph = None
    areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    confluence_points: gpd.GeoDataFrame = None
    bifurcation_points: gpd.GeoDataFrame = None
    weirs: gpd.GeoDataFrame = None
    pumps: gpd.GeoDataFrame = None
    laterals: gpd.GeoDataFrame = None
    culverts: gpd.GeoDataFrame = None
    crs: int = 28992

    class Config:
        arbitrary_types_allowed = True

    def add_data_from_simulations_set(
        self,
        set_name: str,
        simulations_dir: Path,
        simulations_names: List[str] = None,
        simulations_ts: Union[List, pd.DatetimeIndex] = [-1],
    ) -> Tuple[xr.Dataset, xu.UgridDataset]:
        """ "receives his- and map-data
        - from d-hydro simulations with names: simulation_names
        - within directory: simulations_dir
        - at timestamps: simulations_ts"""
        if simulations_names is None:
            if not Path(simulations_dir).exists():
                raise ValueError(
                    f"Directory D-Hydro calculations does not exist: {simulations_dir}"
                )
            self.simulation_names = get_simulation_names_from_dir(simulations_dir)

        his_data, map_data = get_data_from_simulations_set(
            set_name=set_name,
            simulations_dir=simulations_dir,
            simulations_names=simulations_names,
            simulations_ts=simulations_ts,
        )
        self.his_data = combine_data_from_simulations_sets(self.his_data, his_data)
        self.map_data = combine_data_from_simulations_sets(
            self.map_data, map_data, xugrid=True
        )
        return self.his_data, self.map_data

    def get_network_data(self):
        """Gives fast access to nodes, edges, confluences, bifurcations, weirs, pumps, laterals"""
        if self.map_data is None:
            raise ValueError("D-Hydro simulation map-data is not read")
        if self.his_data is None:
            raise ValueError("D-Hydro simulation his-data is not read")
        self.get_nodes()
        self.get_edges()
        self.get_confluence_points()
        self.get_bifurcation_points()
        self.get_weirs()
        self.get_pumps()
        self.get_laterals()
        print(
            "network locations read: nodes/edges/confluences/bifurcations/weirs/pumps/laterals"
        )

    def create_basins_based_on_split_node_ids(
        self,
        split_node_ids: List[int],
        areas: gpd.GeoDataFrame,
        nodes: gpd.GeoDataFrame = None,
        edges: gpd.GeoDataFrame = None,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        if nodes is None:
            nodes = self.dhydro_nodes
        if edges is None:
            edges = self.dhydro_edges
        basins, areas = create_basins_based_on_split_node_ids(
            nodes=self.dhydro_nodes,
            edges=self.dhydro_edges,
            split_node_ids=split_node_ids,
            areas=areas,
        )
        self.areas_gdf = areas
        self.basins_gdf = basins
        return self.basins_gdf, self.areas_gdf

    def get_nodes(self) -> gpd.GeoDataFrame:
        """calculate nodes dataframe"""
        self.dhydro_nodes = (
            self.map_data["mesh1d_node_id"]
            .ugrid.to_geodataframe()
            .reset_index()
            .set_crs(self.crs)
        )
        return self.dhydro_nodes

    def get_edges(self) -> gpd.GeoDataFrame:
        """calculate edges dataframe"""
        edges = (
            self.map_data["mesh1d_q1"][-1][-1]
            .ugrid.to_geodataframe()
            .reset_index()
            .set_crs(self.crs)
            .drop(columns=["condition", "mesh1d_q1"])
        )
        edges_nodes = self.map_data["mesh1d_edge_nodes"]
        edges_nodes = np.column_stack(edges_nodes.data)
        edges_nodes = pd.DataFrame(
            {"start_node_no": edges_nodes[0] - 1, "end_node_no": edges_nodes[1] - 1}
        )
        self.dhydro_edges = edges.merge(
            edges_nodes, how="inner", left_index=True, right_index=True
        )
        return self.dhydro_edges

    def get_confluence_points(self) -> gpd.GeoDataFrame:
        """calculate confluence points based on finding multiple inflows"""
        c = self.dhydro_edges.end_node_no.value_counts()
        self.confluence_points = self.dhydro_nodes[
            self.dhydro_nodes.index.isin(c.index[c.gt(1)])
        ]
        return self.confluence_points

    def get_bifurcation_points(self) -> gpd.GeoDataFrame:
        """calculate split points based on finding multiple outflows"""
        d = self.dhydro_edges.start_node_no.value_counts()
        self.bifurcation_points = self.dhydro_nodes[
            self.dhydro_nodes.index.isin(d.index[d.gt(1)])
        ]
        return self.bifurcation_points

    def get_weirs(self) -> gpd.GeoDataFrame:
        weirs = gpd.GeoDataFrame(
            data={"weirgen": self.his_data["weirgens"]},
            geometry=gpd.points_from_xy(
                self.his_data["weirgen_geom_node_coordx"].data[::2],
                self.his_data["weirgen_geom_node_coordy"].data[::2],
            ),
            crs=self.crs,
        )
        self.weirs = weirs.sjoin(self.dhydro_nodes[["mesh1d_nNodes", "geometry"]]).drop(
            columns=["index_right"]
        )
        return self.weirs

    def get_pumps(self) -> gpd.GeoDataFrame:
        pumps = gpd.GeoDataFrame(
            data={"pumps": self.his_data["pumps"]},
            geometry=gpd.points_from_xy(
                self.his_data["pump_geom_node_coordx"].data[::2],
                self.his_data["pump_geom_node_coordy"].data[::2],
            ),
            crs=self.crs,
        )
        self.pumps = pumps.sjoin(self.dhydro_nodes[["mesh1d_nNodes", "geometry"]]).drop(
            columns=["index_right"]
        )
        return self.pumps

    def get_laterals(self) -> gpd.GeoDataFrame:
        laterals = gpd.GeoDataFrame(
            data={"lateral": self.his_data["lateral"]},
            geometry=gpd.points_from_xy(
                self.his_data["lateral_geom_node_coordx"],
                self.his_data["lateral_geom_node_coordy"],
            ),
            crs=self.crs,
        )
        self.laterals = laterals.sjoin(
            self.dhydro_nodes[["mesh1d_nNodes", "geometry"]]
        ).drop(columns=["index_right"])
        return self.laterals

    def get_qh_relations_weirs(self, weirs_ids: List[str] = None):
        # TODO: get QH relations from selected locations
        return weirs_ids

    def get_qh_relation_edges_around_node(
        self, nodes_ids: List[str] = None, nodes_locations: List[Point] = None
    ):
        # TODO: get QH relations from selected locations
        return nodes_ids
