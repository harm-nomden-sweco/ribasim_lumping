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
    areas_gdf: gpd.GeoDataFrame
    his_data: xu.UgridDataset = None
    map_data: xu.UgridDataset = None
    edges_gdf: gpd.GeoDataFrame = None
    nodes_gdf: gpd.GeoDataFrame = None
    edges_q_df: pd.DataFrame = None
    nodes_h_df: pd.DataFrame = None
    network_graph: nx.DiGraph = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    confluences_gdf: gpd.GeoDataFrame = None
    bifurcations_gdf: gpd.GeoDataFrame = None
    weirs_gdf: gpd.GeoDataFrame = None
    uniweirs_gdf: gpd.GeoDataFrame = None
    pumps_gdf: gpd.GeoDataFrame = None
    culverts_gdf: gpd.GeoDataFrame = None
    split_nodes: gpd.GeoDataFrame = None
    ribasim_edges_gdf: gpd.GeoDataFrame = None
    crs: int = 28992

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.areas_gdf = self.areas_gdf[["geometry"]].explode(index_parts=False)

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
        """Extracts nodes, edges, confluences, bifurcations, weirs, pumps from his/map"""
        results = get_dhydro_network_objects(self.map_data, self.his_data, self.crs)
        self.nodes_gdf, self.nodes_h_df, self.edges_gdf, self.edges_q_df, \
            self.weirs_gdf, self.uniweirs_gdf, self.pumps_gdf, self.confluences_gdf, \
            self.bifurcations_gdf = results


    def get_qh_relation_node_edge(self, node_no:int, edge_no:int, set:str=None):
        h_x = self.nodes_h_df.loc[node_no]
        q_x = self.edges_q_df.loc[edge_no]
        qh_x = q_x.merge(h_x, how='outer', left_index=True, right_index=True)
        if set is None:
            return qh_x
        else:
            return qh_x.loc[set]
        

    def get_split_nodes_based_on_type(
        self,
        bifurcations: bool = False,
        confluences: bool = False,
        weirs: bool = False,
        uniweirs: bool = False,
        pumps: bool = False,
    ):
        """receive node_ids from bifurcations, confluences, weirs and/or pumps"""
        list_objects_gdf = []
        if bifurcations and self.bifurcations_gdf is not None:
            list_objects_gdf.append(self.bifurcations_gdf)
        if confluences and self.confluences_gdf is not None:
            list_objects_gdf.append(self.confluences_gdf)
        if weirs and self.weirs_gdf is not None:
            list_objects_gdf.append(self.weirs_gdf)
        if uniweirs and self.uniweirs_gdf is not None:
            list_objects_gdf.append(self.uniweirs_gdf)
        if pumps and self.pumps_gdf is not None:
            list_objects_gdf.append(self.pumps_gdf)
        split_nodes_objects = pd.concat(list_objects_gdf)
        return split_nodes_objects


    def add_split_nodes(
        self,
        bifurcations: bool = False,
        confluences: bool = False,
        weirs: bool = False,
        uniweirs: bool = False,
        pumps: bool = False,
        split_node_ids_to_include: List[int] = [],
        split_node_ids_to_exclude: List[int] = [],
    ) -> gpd.GeoDataFrame:
        # get split_nodes based on type
        split_nodes_objects = self.get_split_nodes_based_on_type(
            bifurcations=bifurcations,
            confluences=confluences,
            weirs=weirs,
            uniweirs=uniweirs,
            pumps=pumps,
        )
        # add additional split_node extracting them from nodes_gdf based on id
        split_nodes = self.nodes_gdf[
            self.nodes_gdf.mesh1d_nNodes.isin(split_node_ids_to_include)
        ].reset_index(drop=True)
        # combine split_nodes
        split_nodes = pd.concat([split_nodes_objects, split_nodes])
        # exclude split_nodes based on id
        split_nodes = split_nodes[
            ~split_nodes['mesh1d_nNodes'].isin(split_node_ids_to_exclude)
        ]
        # check whether all split_node_ids are present in list
        missing = set(split_node_ids_to_include).difference(
            split_nodes.mesh1d_nNodes.values
        )
        if len(missing):
            print(f" - Selected split_node_ids not present in network: {missing}")
        self.split_nodes = split_nodes
        return split_nodes


    def add_split_nodes_based_on_locations(
        self, split_nodes: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """add split_node_ids using geodataframe (shapefile/geojson)"""
        nearest_node_ids = find_nearest_nodes(
            search_locations=split_nodes,
            nodes=self.nodes_gdf,
            id_column="mesh1d_nNodes",
        )
        self.split_nodes = self.add_split_nodes(nearest_node_ids)
        return self.split_nodes


    @property
    def split_node_ids(self):
        if self.split_nodes is None:
            return None
        return list(self.split_nodes.mesh1d_nNodes.values)


    def create_basins_based_on_split_nodes(self) -> Tuple[gpd.GeoDataFrame]:
        if self.split_nodes is None:
            raise ValueError('no split_nodes defined: use .add_split_nodes()')
        if self.nodes_gdf is None or self.edges_gdf is None:
            raise ValueError('no nodes and/or edges defined: add d-hydro simulation results')
        if self.areas_gdf is None:
            raise ValueError('no areas defined: add areas-geodataframe (drainage areas)')

        results_basins = create_basins_using_split_nodes(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            split_nodes=self.split_nodes,
            areas=self.areas_gdf,
            crs=self.crs,
        )
        self.basin_areas_gdf, self.basins_gdf, self.areas_gdf, self.nodes_gdf, \
            self.edges_gdf, self.split_nodes = results_basins
        return results_basins

    # def create_connections_between_basins(self) -> Tuple[gpd.GeoDataFrame]:


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
            uniweirs=self.uniweirs_gdf,
            pumps=self.pumps_gdf,
            areas=self.areas_gdf,
            basin_areas=self.basin_areas_gdf,
            split_nodes=self.split_nodes,
            basins=self.basins_gdf,
            ribasim_edges=self.ribasim_edges_gdf
        )
        print('Exporting:')
        for gdf_name, gdf in gdfs.items():
            if gdf is None:
                print(f' - {gdf_name} (not available)')
                gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry", crs=self.crs
                ).to_file(gpkg_path, layer=gdf_name, driver="GPKG")
            else:
                print(f' - {gdf_name}')
                gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")
        if not qgz_path.exists():
            shutil.copy(qgz_path_stored, qgz_path)

