# pylint: disable=missing-function-docstring
import os
from pathlib import Path
from typing import List, Union, Optional, Any, Tuple, Dict
import shutil
import datetime
from pydantic import BaseModel, Field
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import polylabel
import pandas as pd
import numpy as np
import dfm_tools as dfmt
import xarray as xr
import xugrid as xu
import networkx as nx

from .utils.read_simulation_data_utils import (
    get_data_from_simulations_set,
    get_simulation_names_from_dir,
    combine_data_from_simulations_sets,
)
from .utils.generate_basins_areas import create_basins_using_split_nodes
from .utils.get_dhydro_network_objects import get_dhydro_network_objects
from .utils.general_functions import find_nearest_nodes


class RibasimLumpingNetwork(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""

    name: str
    his_data: xu.UgridDataset = None
    map_data: xu.UgridDataset = None
    edges_gdf: gpd.GeoDataFrame = None
    nodes_gdf: gpd.GeoDataFrame = None
    network_graph: nx.DiGraph = None
    areas_gdf: gpd.GeoDataFrame = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    confluences_gdf: gpd.GeoDataFrame = None
    bifurcations_gdf: gpd.GeoDataFrame = None
    weirs_gdf: gpd.GeoDataFrame = None
    pumps_gdf: gpd.GeoDataFrame = None
    laterals_gdf: gpd.GeoDataFrame = None
    culverts_gdf: gpd.GeoDataFrame = None
    split_nodes: gpd.GeoDataFrame = None
    ribasim_edges_gdf: gpd.GeoDataFrame = None
    basin_connections_gdf: gpd.GeoDataFrame = None
    boundaries_gdf: gpd.GeoDataFrame = None
    boundary_basin_connections_gdf: gpd.GeoDataFrame = None
    splitnodes_moved_gdf: gpd.GeoDataFrame = None
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
        """receives his- and map-data
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
        """Extracts nodes, edges, confluences, bifurcations, weirs, pumps, laterals from his/map"""
        results = get_dhydro_network_objects(self.map_data, self.his_data, self.crs)
        self.nodes_gdf = results[0]
        self.edges_gdf = results[1]
        self.weirs_gdf = results[2]
        self.pumps_gdf = results[3]
        self.laterals_gdf = results[4]
        self.confluences_gdf = results[5]
        self.bifurcations_gdf = results[6]


    def get_node_ids_from_type(
        self,
        bifurcations: bool = False,
        confluences: bool = False,
        weirs: bool = False,
        pumps: bool = False,
        laterals: bool = False,
    ):
        """receive node_ids from bifurcations, confluences, weirs, pumps and/or laterals"""
        split_node_ids = []
        if bifurcations and self.bifurcations_gdf is not None:
            split_node_ids += list(self.bifurcations_gdf.mesh1d_nNodes.values)
        if confluences and self.confluences_gdf is not None:
            split_node_ids += list(self.confluences_gdf.mesh1d_nNodes.values)
        if weirs and self.weirs_gdf is not None:
            split_node_ids += list(self.weirs_gdf.mesh1d_nNodes.values)
        if pumps and self.pumps_gdf is not None:
            split_node_ids += list(self.pumps_gdf.mesh1d_nNodes.values)
        if laterals and self.laterals_gdf is not None:
            split_node_ids += list(self.laterals_gdf.mesh1d_nNodes.values)
        return split_node_ids


    def add_split_nodes_based_on_node_ids(
        self,
        split_node_ids: List[int],
    ) -> gpd.GeoDataFrame:
        """add split_node_ids using a list of node_ids"""
        split_nodes = self.nodes_gdf[
            self.nodes_gdf.mesh1d_nNodes.isin(split_node_ids)
        ].reset_index(drop=True)
        # check whether all split_node_ids are present
        missing = set(split_node_ids).difference(split_nodes.mesh1d_nNodes.values)
        if len(missing):
            print(f"Selected split_node_ids not present in network: {missing}")
        self.split_nodes = split_nodes
        return self.split_nodes


    def add_split_nodes_based_on_locations(
        self, split_nodes: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """add split_node_ids using geodataframe (shapefile/geojson)"""
        nearest_node_ids = find_nearest_nodes(
            search_locations=split_nodes,
            nodes=self.nodes_gdf,
            id_column="mesh1d_nNodes",
        )
        self.split_nodes = self.add_split_nodes_based_on_node_ids(nearest_node_ids)
        return self.split_nodes


    @property
    def split_node_ids(self):
        if self.split_nodes is None:
            return None
        return list(self.split_nodes.mesh1d_nNodes.values)


    def create_basins_based_on_split_nodes(
        self,
        split_nodes: gpd.GeoDataFrame = None,
        split_node_ids: List[int] = None,
        areas: gpd.GeoDataFrame = None,
        nodes: gpd.GeoDataFrame = None,
        edges: gpd.GeoDataFrame = None,
    ) -> Tuple[gpd.GeoDataFrame]:
        if split_nodes is not None:
            self.split_nodes = split_nodes
        elif split_node_ids is not None:
            self.add_split_nodes_based_on_node_ids(split_node_ids)
        elif self.split_nodes is None:
            raise ValueError("no split_nodes or split_node_ids provided")

        if nodes is not None:
            self.nodes_gdf = nodes
        if edges is not None:
            self.edges_gdf = edges
        if areas is not None:
            self.areas_gdf = areas
        self.areas_gdf = self.areas_gdf[["geometry"]].explode(index_parts=False)
        results_basins = create_basins_using_split_nodes(
                nodes=self.nodes_gdf,
                edges=self.edges_gdf,
                split_nodes=self.split_nodes,
                areas=self.areas_gdf,
            )
        self.basin_areas_gdf, self.basins_gdf, self.areas_gdf, self.nodes_gdf, \
            self.edges_gdf, self.split_nodes = results_basins
        return results_basins


    def export_to_geopackage(
        self,
        output_dir: Union[Path, str],
    ):
        dir_output = Path(output_dir, self.name)
        if not dir_output.exists():
            dir_output.mkdir()

        gpkg_path = Path(dir_output, "ribasim_network.gpkg")
        qgz_path = Path(dir_output, "ribasim_network.qgz")
        qgz_path_stored_dir = os.path.abspath(os.path.dirname(__file__))
        qgz_path_stored = Path(qgz_path_stored_dir, "assets\\ribasim_network.qgz")

        gdfs = dict(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            confluences=self.confluences_gdf,
            bifurcations=self.bifurcations_gdf,
            weirs=self.weirs_gdf,
            pumps=self.pumps_gdf,
            laterals=self.laterals_gdf,
            areas=self.areas_gdf,
            basin_areas=self.basin_areas_gdf,
            split_nodes=self.split_nodes,
            basins=self.basins_gdf,
            basin_connections=self.basin_connections_gdf,
            ribasim_edges=self.ribasim_edges_gdf,
            boundaries=self.boundaries_gdf,
            boundary_basin_connections=self.boundary_basin_connections_gdf,
            splitnodes_moved = self.splitnodes_moved_gdf
        )
        for gdf_name, gdf in gdfs.items():
            if gdf is None:
                gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry", crs=self.crs
                ).to_file(gpkg_path, layer=gdf_name, driver="GPKG")
            else:
                gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")
        if not qgz_path.exists():
            shutil.copy(qgz_path_stored, qgz_path)


    # def get_qh_relations_weirs(self, weirs_ids: List[str] = None):
    #     # TODO: get QH relations from selected locations
    #     return weirs_ids


    # def get_qh_relation_edges_around_node(
    #     self, nodes_ids: List[str] = None, nodes_locations: List[Point] = None
    # ):
    #     # TODO: get QH relations from selected locations
    #     return nodes_ids
