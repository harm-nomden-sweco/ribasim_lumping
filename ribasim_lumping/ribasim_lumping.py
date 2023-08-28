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
import ribasim

from .utils.read_simulation_data_utils import (
    get_data_from_simulations_set,
    get_simulation_names_from_dir,
    combine_data_from_simulations_sets,
)
from .utils.generate_basins_areas import create_basins_and_connections_using_split_nodes
from .utils.read_dhydro_network_objects import get_dhydro_network_objects
from .utils.general_functions import find_nearest_nodes
from .utils.generate_ribasim_model import generate_ribasimmodel


class RibasimLumpingNetwork(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""

    name: str
    dhydro_dir: Path
    results_dir: Path
    areas_gdf: gpd.GeoDataFrame = None
    his_data: xu.UgridDataset = None
    map_data: xu.UgridDataset = None
    boundary_data: Dict = None
    edges_gdf: gpd.GeoDataFrame = None
    nodes_gdf: gpd.GeoDataFrame = None
    edges_q_df: pd.DataFrame = None
    nodes_h_df: pd.DataFrame = None
    network_graph: nx.DiGraph = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    stations_gdf: gpd.GeoDataFrame = None
    pumps_gdf: gpd.GeoDataFrame = None
    weirs_gdf: gpd.GeoDataFrame = None
    orifices_gdf: gpd.GeoDataFrame = None
    bridges_gdf: gpd.GeoDataFrame = None
    culverts_gdf: gpd.GeoDataFrame = None
    uniweirs_gdf: gpd.GeoDataFrame = None
    confluences_gdf: gpd.GeoDataFrame = None
    bifurcations_gdf: gpd.GeoDataFrame = None
    split_nodes: gpd.GeoDataFrame = None
    basin_connections_gdf: gpd.GeoDataFrame = None
    boundaries_gdf: gpd.GeoDataFrame = None
    boundary_basin_connections_gdf: gpd.GeoDataFrame = None
    split_node_type_conversion: Dict = None
    split_node_id_conversion: Dict = None
    ribasim_model: ribasim.Model = None
    crs: int = 28992

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.areas_gdf is not None:
            self.areas_gdf = self.areas_gdf.explode(index_parts=False)
            if self.areas_gdf.crs is None:
                self.areas_gdf = self.areas_gdf[['geometry']].set_crs(self.crs)
            else:
                self.areas_gdf = self.areas_gdf[['geometry']].to_crs(self.crs)

    def add_data_from_simulations_set(
        self,
        set_name: str,
        simulations_dir: Path,
        simulations_names: List[str],
        simulation_output_dir: str,
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

        his_data, map_data, boundary_data = get_data_from_simulations_set(
            set_name=set_name,
            simulations_dir=simulations_dir,
            simulations_names=simulations_names,
            simulation_output_dir=simulation_output_dir,
            simulations_ts=simulations_ts,
        )
        self.his_data = combine_data_from_simulations_sets(self.his_data, his_data)
        self.map_data = combine_data_from_simulations_sets(self.map_data, map_data, xugrid=True)

        if self.boundary_data is None:
            self.boundary_data = {f'{set_name}': boundary_data}
        else:
            self.boundary_data[set_name] = boundary_data
        return self.his_data, self.map_data, self.boundary_data


    def get_network_data(self):
        """Extracts nodes, edges, confluences, bifurcations, weirs, pumps from his/map"""
        results = get_dhydro_network_objects(self.map_data, self.his_data, self.boundary_data, self.crs)

        self.nodes_gdf, self.nodes_h_df, self.edges_gdf, self.edges_q_df, \
            self.stations_gdf, self.pumps_gdf, self.weirs_gdf, self.orifices_gdf, \
            self.bridges_gdf, self.culverts_gdf, self.uniweirs_gdf, \
            self.confluences_gdf, self.bifurcations_gdf, self.boundaries_gdf = results


    def get_qh_relation_node_edge(self, node_no: int, edge_no: int, set: str = None):
        h_x = self.nodes_h_df.loc[node_no]
        q_x = self.edges_q_df.loc[edge_no]
        qh_x = q_x.merge(h_x, how='outer', left_index=True, right_index=True)
        if set is None:
            return qh_x
        else:
            return qh_x.loc[set]
        

    def get_qh_relation_split_node_basin(self, basin_id: int, split_node_id: int, set: str = None):
        node_no = self.basins_gdf[self.basins_gdf.basin==basin_id].mesh1d_nNodes.iloc[0]
        edge_no = self.split_nodes[self.split_nodes.mesh1d_node_id==split_node_id].mesh1d_nEdges.iloc[0]
        display(edge_no)
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
        stations: bool = False,
        pumps: bool = False,
        weirs: bool = False,
        orifices: bool = False,
        bridges: bool = False,
        culverts: bool = False,
        uniweirs: bool = False,
    ):
        """receive node_ids from bifurcations, confluences, weirs and/or pumps"""
        list_objects = [
            bifurcations, confluences, stations, pumps, 
            weirs, orifices, bridges, culverts, uniweirs
        ]
        list_gdfs = [
            self.bifurcations_gdf,
            self.confluences_gdf,
            self.stations_gdf,
            self.pumps_gdf,
            self.weirs_gdf,
            self.orifices_gdf,
            self.bridges_gdf,
            self.culverts_gdf,
            self.uniweirs_gdf
        ]
        split_nodes = gpd.GeoDataFrame(
            columns=['mesh1d_node_id', 'mesh1d_nEdges', 'geometry', 'object_type'],
            geometry='geometry',
            crs=self.crs
        )
        for gdf_name, gdf in zip(list_objects, list_gdfs):
            if gdf_name and gdf is not None:
                split_nodes = pd.concat([split_nodes, gdf])
        return split_nodes


    def add_split_nodes(
        self,
        bifurcations: bool = False,
        confluences: bool = False,
        stations: bool = False,
        pumps: bool = False,
        weirs: bool = False,
        orifices: bool = False,
        bridges: bool = False,
        culverts: bool = False,
        uniweirs: bool = False,
        edges: bool = False,
        structures_ids_to_include: List[int] = [],
        structures_ids_to_exclude: List[int] = [],
        node_ids_to_include: List[int] = [],
        node_ids_to_exclude: List[int] = [],
        edge_ids_to_include: List[int] = [],
        edge_ids_to_exclude: List[int] = [],
    ) -> gpd.GeoDataFrame:
        """receive node id's of splitnodes 
        by choosing which structures to use as splitnodes locations 
        and including or excluding specific nodes as splitnode
        returns splitnodes"""
        # get split_nodes based on type
        split_nodes_structures = self.get_split_nodes_based_on_type(
            bifurcations=bifurcations,
            confluences=confluences,
            stations=stations,
            pumps=pumps,
            weirs=weirs,
            orifices=orifices,
            bridges=bridges,
            culverts=culverts,
            uniweirs=uniweirs,
        )
        # include split_nodes with node_id
        all_structures = self.get_split_nodes_based_on_type(
            stations=True, pumps=True, weirs=True, orifices=True, 
            bridges=True, culverts=True, uniweirs=True,
        )
        structures_to_include = all_structures[
            all_structures.mesh1d_node_id.isin(structures_ids_to_include)
        ]
        split_nodes = pd.concat([split_nodes_structures, structures_to_include])
        # exclude split_nodes with node_id
        split_nodes = split_nodes[
            ~split_nodes.mesh1d_node_id.isin(structures_ids_to_exclude)
        ]

        # add additional split_node extracting them from nodes_gdf based on id
        nodes_to_include = self.nodes_gdf[
            self.nodes_gdf.mesh1d_nNodes.isin(node_ids_to_include)
        ].reset_index(drop=True)
        nodes_to_include['object_type'] = 'manual'
        nodes_to_include['projection_x'] = nodes_to_include.geometry.x
        nodes_to_include['projection_y'] = nodes_to_include.geometry.y
        # combine split_nodes
        split_nodes = pd.concat([split_nodes, nodes_to_include])
        # exclude split_nodes based on id
        split_nodes = split_nodes[
            ~split_nodes.mesh1d_nNodes.isin(node_ids_to_exclude)
        ]
        # drop duplicates
        split_nodes = split_nodes.drop_duplicates()

        # include/exclude edge centers
        if edges or len(edge_ids_to_include) > 1:
            if edges:
                additional_split_nodes = self.edges_gdf.copy()
                if len(edge_ids_to_exclude):
                    additional_split_nodes[
                        ~additional_split_nodes.mesh1d_nEdges.isin(edge_ids_to_exclude)
                    ]
            elif len(edge_ids_to_include):
                additional_split_nodes = self.edges_gdf[self.edges_gdf.mesh1d_nEdges.isin(edge_ids_to_include)]
            additional_split_nodes.geometry = additional_split_nodes.geometry.apply(lambda g: g.centroid)
            additional_split_nodes['object_type'] = 'edge'
            additional_split_nodes['mesh1d_nNodes'] = -1
            additional_split_nodes = additional_split_nodes.rename(columns={"mesh1d_edge_x": "projection_x", "mesh1d_edge_y": "projection_y"})
            additional_split_nodes = additional_split_nodes.drop(['start_node_no', 'end_node_no', 'basin'], axis=1, errors='ignore')

            split_nodes = pd.concat([split_nodes, additional_split_nodes])
            split_nodes = split_nodes.drop_duplicates(subset='mesh1d_nEdges', keep="first")

        # check whether all split_node_ids are present in list
        missing = list(set(node_ids_to_include).difference(
            split_nodes.mesh1d_nNodes.values
        ))
        if len(missing):
            print(f" - Selected split_node_ids not present in network: {missing}")
        split_nodes['mesh1d_nEdges'] = split_nodes['mesh1d_nEdges'].fillna(-1).astype(int)
        split_nodes['mesh1d_nNodes'] = split_nodes['mesh1d_nNodes'].fillna(-1).astype(int)
        split_nodes = split_nodes.drop_duplicates(subset=['mesh1d_nEdges', 'mesh1d_nNodes'])
        self.split_nodes = split_nodes.reset_index(drop=True)
        print(f"{len(split_nodes)} split locations")
        for obj_type in self.split_nodes.object_type.unique():
            print(f" - {obj_type}: {len(self.split_nodes[self.split_nodes['object_type']==obj_type])}")
        return self.split_nodes


    def add_split_nodes_based_on_locations(
        self, 
        split_nodes: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """add split_node_ids using geodataframe (shapefile/geojson)"""
        nearest_node_ids = find_nearest_nodes(
            search_locations=split_nodes,
            nodes=self.nodes_gdf,
            id_column="mesh1d_nNodes",
        )["mesh1d_nNodes"].values
        self.split_nodes = self.add_split_nodes(nearest_node_ids)
        return self.split_nodes


    @property
    def split_node_ids(self):
        if self.split_nodes is None:
            return None
        return list(self.split_nodes.mesh1d_nNodes.values)


    def create_basins_and_connections_based_on_split_nodes(self) -> Tuple[gpd.GeoDataFrame]:
        if self.split_nodes is None:
            raise ValueError('no split_nodes defined: use .add_split_nodes()')
        if self.nodes_gdf is None or self.edges_gdf is None:
            raise ValueError('no nodes and/or edges defined: add d-hydro simulation results')
        if self.areas_gdf is None:
            print("no areas defined, will not generate basin_areas")
        if self.boundaries_gdf is None:
            print("no boundaries defined, will not generate boundaries and boundaries_basin_connections")

        results_basins = create_basins_and_connections_using_split_nodes(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            split_nodes=self.split_nodes,
            areas=self.areas_gdf,
            boundaries=self.boundaries_gdf,
            crs=self.crs,
        )
        self.basin_areas_gdf, self.basins_gdf, self.areas_gdf, self.nodes_gdf, \
            self.edges_gdf, self.split_nodes, self.network_graph, \
                self.basin_connections_gdf, self.boundary_basin_connections_gdf = results_basins
        return results_basins


    def generate_ribasim_model(
            self, 
            split_node_type_conversion: Dict = None, 
            split_node_id_conversion: Dict = None
        ):
        self.split_node_type_conversion = split_node_type_conversion
        self.split_node_id_conversion = split_node_id_conversion

        ribasim_model = generate_ribasimmodel(
            basins=self.basins_gdf, 
            split_nodes=self.split_nodes, 
            boundaries=self.boundaries_gdf, 
            basin_connections=self.basin_connections_gdf, 
            boundary_basin_connections=self.boundary_basin_connections_gdf,
            split_node_type_conversion=split_node_type_conversion, 
            split_node_id_conversion=split_node_id_conversion
        )
        self.ribasim_model = ribasim_model
        return ribasim_model


    def export_to_geopackage(
        self,
        results_dir: Union[Path, str] = None,
    ):
        if results_dir is None:
            results_dir = self.results_dir
        dir_output = Path(results_dir, self.name)
        if not dir_output.exists():
            dir_output.mkdir()

        gpkg_path = Path(dir_output, "ribasim_network.gpkg")
        qgz_path = Path(dir_output, "ribasim_network.qgz")
        qgz_path_stored_dir = os.path.abspath(os.path.dirname(__file__))
        qgz_path_stored = Path(qgz_path_stored_dir, "assets\\ribasim_network.qgz")

        gdfs_orig = dict(
            areas=self.areas_gdf,
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            stations=self.stations_gdf,
            pumps=self.pumps_gdf,
            weirs=self.weirs_gdf,
            orifices=self.orifices_gdf,
            bridges=self.bridges_gdf,
            culverts=self.culverts_gdf,
            uniweirs=self.uniweirs_gdf,
            basin_areas=self.basin_areas_gdf,
            split_nodes=self.split_nodes,
            basins=self.basins_gdf,
            basin_connections=self.basin_connections_gdf,
            boundaries=self.boundaries_gdf,
            boundary_basin_connections=self.boundary_basin_connections_gdf,
        )
        gdfs_none = dict()
        gdfs = dict()
        for gdf_name, gdf in gdfs_orig.items():
            if gdf is None:
                gdfs_none[gdf_name] = gdf
            else:
                gdfs[gdf_name] = gdf

        print(f'Exporting to geopackage:')
        print(f' - ', end="", flush=True)
        for gdf_name, gdf in gdfs.items():
            print(f'{gdf_name}, ', end="", flush=True)
            gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        print("")
        print(" - not available: ", end="", flush=True)
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=self.crs)
        for gdf_name, gdf in gdfs_none.items():
            print(f'{gdf_name}, ', end="", flush=True)
            empty_gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        if not qgz_path.exists():
            shutil.copy(qgz_path_stored, qgz_path)
        print("")
        print(f'Export location: {qgz_path}')


def create_ribasim_lumping_network(**kwargs):
    return RibasimLumpingNetwork(**kwargs)


