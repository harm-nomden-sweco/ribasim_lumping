# pylint: disable=missing-function-docstring
import sys
import os
from pathlib import Path
from typing import List, Union, Tuple, Dict
import shutil
from pydantic import BaseModel
import geopandas as gpd
import pandas as pd
import xarray as xr
import xugrid as xu
import networkx as nx
from .utils.read_simulation_data_utils import (
    get_data_from_simulations_set,
    get_simulation_names_from_dir,
    combine_data_from_simulations_sets,
)
from .utils.generate_ribasim_network_plus_areas import generate_ribasim_network_using_split_nodes
from .utils.read_dhydro_network_locations import get_dhydro_data_from_simulation, get_dhydro_files
from .utils.general_functions import find_nearest_nodes
from .utils.generate_ribasim_model import generate_ribasimmodel
from .utils.export_load_splitnodes import (
    write_structures_to_excel,
    read_structures_from_excel,
)

sys.path.append("..\\..\\ribasim\\python\\ribasim")
import ribasim


class RibasimLumpingNetwork(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""

    name: str
    base_dir: Path
    dhydro_basis_dir: Path
    dhydro_results_dir: Path
    results_dir: Path
    his_data: xu.UgridDataset = None
    map_data: xu.UgridDataset = None
    network_data: xr.Dataset = None
    areas_gdf:gpd.GeoDataFrame = None
    branches_gdf: gpd.GeoDataFrame = None
    network_nodes_gdf: gpd.GeoDataFrame = None
    edges_gdf: gpd.GeoDataFrame = None
    nodes_gdf: gpd.GeoDataFrame = None
    structures_gdf: gpd.GeoDataFrame = None
    stations_gdf: gpd.GeoDataFrame = None
    pumps_gdf: gpd.GeoDataFrame = None
    weirs_gdf: gpd.GeoDataFrame = None
    orifices_gdf: gpd.GeoDataFrame = None
    bridges_gdf: gpd.GeoDataFrame = None
    culverts_gdf: gpd.GeoDataFrame = None
    uniweirs_gdf: gpd.GeoDataFrame = None
    boundaries_gdf: gpd.GeoDataFrame = None
    laterals_gdf: gpd.GeoDataFrame = None
    boundaries_data: Dict = None
    laterals_data: Dict = None
    edges_q_df: pd.DataFrame = None
    nodes_h_df: pd.DataFrame = None
    network_graph: nx.DiGraph = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    confluences_gdf: gpd.GeoDataFrame = None
    bifurcations_gdf: gpd.GeoDataFrame = None
    split_nodes: gpd.GeoDataFrame = None
    basin_connections_gdf: gpd.GeoDataFrame = None
    boundary_basin_connections_gdf: gpd.GeoDataFrame = None
    split_node_type_conversion: Dict = None
    split_node_id_conversion: Dict = None
    ribasim_model: ribasim.Model = None
    basis_set_names: List[str] = []
    basis_simulations_names: List[str] = []
    set_names: List[str] = []
    simulations_names: List[List] = []
    simulations_output_dirs: List[str] = []
    simulations_ts: List[Union[List, pd.DatetimeIndex]] = []
    crs: int = 28992

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Config:
        arbitrary_types_allowed = True

    def read_areas(self, areas_file_path: Path, areas_id_column: str):
        areas_gdf = gpd.read_file(areas_file_path)
        self.areas_gdf = areas_gdf[[areas_id_column, "geometry"]]
        print(f" - areas ({len(areas_gdf)}x)")

    def add_basis_simulation_data_set(self, set_name: str, simulation_name: str = None):
        """Extracts nodes, edges, confluences, bifurcations, weirs, pumps from his/map"""
        basis_simulation_path = Path(self.dhydro_basis_dir, set_name, simulation_name)
        results = get_dhydro_data_from_simulation(basis_simulation_path, crs=self.crs)

        self.network_data = results.get('network_data', None)
        self.branches_gdf = results.get('branches_gdf', None)
        self.network_nodes_gdf = results.get('network_nodes_gdf', None)
        self.edges_gdf = results.get('edges_gdf', None)
        self.nodes_gdf = results.get('nodes_gdf' , None)
        self.structures_gdf = results.get('structures_gdf' , None)
        if results['structures_dict'] is not None:
            self.weirs_gdf = results['structures_dict'].get('weir', None)
            self.uniweirs_gdf = results['structures_dict'].get('universalWeir', None)
            self.pumps_gdf = results['structures_dict'].get('pump', None)
            self.orifices_gdf = results['structures_dict'].get('orifice', None)
            self.bridges_gdf = results['structures_dict'].get('bridge', None)
            self.culverts_gdf = results['structures_dict'].get('culvert', None)
        self.boundaries_gdf = results.get('boundaries_gdf', None)
        self.laterals_gdf = results.get('laterals_gdf', None)

        if self.boundaries_data is None:
            self.boundaries_data = {set_name: results.get('boundaries_data', None)}
        else:
            self.boundaries_data[set_name] = results.get('boundaries_data', None)
        if self.laterals_data is None:
            self.laterals_data = {set_name: results.get('laterals_data', None)}
        else:
            self.laterals_data[set_name] = results.get('laterals_data', None)
        return results

    def add_simulation_data_from_set(
        self,
        set_name: str,
        simulations_names: List[str],
        simulations_ts: Union[List, pd.DatetimeIndex] = [-1],
    ):
        """receives his- and map-data. calculations should be placed in dhydro_results_dir
        - set_name
        - within directory: simulations_dir
        - at timestamps: simulations_ts"""
        simulations_dir = Path(self.dhydro_results_dir, set_name)
        if not simulations_dir.exists():
            raise ValueError(
                f"Directory D-Hydro calculations does not exist: {simulations_dir}"
            )
        if self.his_data is not None:
            if set_name in self.his_data.set:
                print(
                    f'    x set_name "{set_name}" already taken. data not overwritten. change set_name'
                )
                return self.his_data, self.map_data, self.boundary_data

        self.set_names.append(set_name)
        self.simulations_names.append(simulations_names)
        self.simulations_ts.append(simulations_ts)

        his_data, map_data = get_data_from_simulations_set(
            set_name=set_name,
            simulations_dir=simulations_dir,
            simulations_names=simulations_names,
            simulations_ts=simulations_ts,
        )
        # self.files.append(files)
        self.his_data = combine_data_from_simulations_sets(self.his_data, his_data)
        self.map_data = combine_data_from_simulations_sets(self.map_data, map_data, xugrid=True)

        # if self.boundary_data is None:
        #     self.boundary_data = {f"{set_name}": boundary_data}
        # else:
        #     self.boundary_data[set_name] = boundary_data
        return self.his_data, self.map_data #, self.boundary_data

    def get_qh_relation_node_edge(self, node_no: int, edge_no: int, set: str = None):
        h_x = self.nodes_h_df.loc[node_no]
        q_x = self.edges_q_df.loc[edge_no]
        qh_x = q_x.merge(h_x, how="outer", left_index=True, right_index=True)
        if set is None:
            return qh_x
        else:
            return qh_x.loc[set]

    def get_qh_relation_split_node_basin(
        self, basin_id: int, split_node_id: int, set: str = None
    ):
        node_no = self.basins_gdf[self.basins_gdf.basin == basin_id].node_no.iloc[
            0
        ]
        edge_no = self.split_nodes[
            self.split_nodes.mesh1d_node_id == split_node_id
        ].edge_no.iloc[0]
        h_x = self.nodes_h_df.loc[node_no]
        q_x = self.edges_q_df.loc[edge_no]
        qh_x = q_x.merge(h_x, how="outer", left_index=True, right_index=True)
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
            bifurcations,
            confluences,
            stations,
            pumps,
            weirs,
            orifices,
            bridges,
            culverts,
            uniweirs,
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
            self.uniweirs_gdf,
        ]
        split_nodes = gpd.GeoDataFrame(
            columns=["node_id", "name", "branchid", "geometry", "object_type"],
            geometry="geometry",
            crs=self.crs,
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
        structures_ids_to_include: List[str] = [],
        structures_ids_to_exclude: List[str] = [],
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
            stations=True,
            pumps=True,
            weirs=True,
            orifices=True,
            bridges=True,
            culverts=True,
            uniweirs=True,
        )
        structures_to_include = all_structures[
            all_structures.node_id.isin(structures_ids_to_include)
        ]
        split_nodes = pd.concat([split_nodes_structures, structures_to_include])
        # exclude split_nodes with node_id
        split_nodes = split_nodes[~split_nodes.node_id.isin(structures_ids_to_exclude)]

        # include/exclude edge centers
        if edges or len(edge_ids_to_include) > 1:
            if edges:
                additional_split_nodes = self.edges_gdf.copy()
                if len(edge_ids_to_exclude):
                    additional_split_nodes[
                        ~additional_split_nodes.edge_no.isin(edge_ids_to_exclude)
                    ]
            elif len(edge_ids_to_include):
                additional_split_nodes = self.edges_gdf[
                    self.edges_gdf.edge_no.isin(edge_ids_to_include)
                ]
            additional_split_nodes.geometry = additional_split_nodes.geometry.apply(
                lambda g: g.centroid
            )
            additional_split_nodes["object_type"] = "edge"
            additional_split_nodes["node_no"] = -1
            additional_split_nodes = additional_split_nodes.rename(columns={
                "mesh1d_edge_x": "projection_x",
                "mesh1d_edge_y": "projection_y",
            })
            additional_split_nodes = additional_split_nodes.drop(
                ["start_node_no", "end_node_no", "basin"], axis=1, errors="ignore"
            )
            split_nodes = pd.concat([split_nodes, additional_split_nodes])
            split_nodes = split_nodes.drop_duplicates(subset="edge_no", keep="first")

        split_nodes["node_no"] = -1

        self.split_nodes = split_nodes.reset_index(drop=True)
        print(f"{len(split_nodes)} split locations")
        for obj_type in self.split_nodes.object_type.unique():
            print(f" - {obj_type}: {len(self.split_nodes[self.split_nodes['object_type']==obj_type])}")
        return self.split_nodes

    def add_split_nodes_based_on_locations(
        self, split_nodes: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """add split_node_ids using geodataframe (shapefile/geojson)"""
        nearest_node_ids = find_nearest_nodes(
            search_locations=split_nodes,
            nodes=self.nodes_gdf,
            id_column="node_no",
        )["node_no"].values
        self.split_nodes = self.add_split_nodes(nearest_node_ids)
        return self.split_nodes

    @property
    def split_node_ids(self):
        if self.split_nodes is None:
            return None
        return list(self.split_nodes.node_no.values)

    def generate_ribasim_network(
        self,
    ) -> Dict:
        if self.split_nodes is None:
            raise ValueError("no split_nodes defined: use .add_split_nodes()")
        if self.nodes_gdf is None or self.edges_gdf is None:
            raise ValueError(
                "no nodes and/or edges defined: add d-hydro simulation results"
            )
        if self.areas_gdf is None:
            print("no areas defined, will not generate basin_areas")
        if self.boundaries_gdf is None:
            print(
                "no boundaries defined, will not generate boundaries and boundaries_basin_connections"
            )

        results = generate_ribasim_network_using_split_nodes(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            split_nodes=self.split_nodes,
            areas=self.areas_gdf,
            boundaries=self.boundaries_gdf,
            crs=self.crs,
        )
        self.basin_areas_gdf = results['basin_areas']
        self.basins_gdf = results['basins']
        self.areas_gdf = results['areas']
        self.nodes_gdf = results['nodes']
        self.edges_gdf = results['edges']
        self.split_nodes = results['split_nodes']
        self.network_graph = results['network_graph']
        self.basin_connections_gdf = results['basin_connections']
        self.boundary_basin_connections_gdf = results['boundary_connections']
        return results

    def generate_ribasim_model(
        self,
        split_node_type_conversion: Dict = None,
        split_node_id_conversion: Dict = None,
    ):
        self.split_node_type_conversion = split_node_type_conversion
        self.split_node_id_conversion = split_node_id_conversion

        ribasim_model = generate_ribasimmodel(
            basins=self.basins_gdf,
            split_nodes=self.split_nodes.copy(),
            boundaries=self.boundaries_gdf,
            basin_connections=self.basin_connections_gdf,
            boundary_basin_connections=self.boundary_basin_connections_gdf,
            split_node_type_conversion=split_node_type_conversion,
            split_node_id_conversion=split_node_id_conversion,
        )
        self.ribasim_model = ribasim_model
        return ribasim_model


    def export_to_geopackage(self, results_dir: Union[Path, str] = None):
        if results_dir is None:
            results_dir = self.results_dir
        if not Path(results_dir).exists():
            results_dir.mkdir()
        gpkg_path = Path(results_dir, "ribasim_network.gpkg")
        qgz_path = Path(results_dir, "ribasim_network.qgz")

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

        print(f"Exporting to geopackage:")
        print(" - available: ", end="", flush=True)
        for gdf_name, gdf in gdfs.items():
            print(f"{gdf_name}, ", end="", flush=True)
            display(gdf.head())
            gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        print("")
        print(" - not available: ", end="", flush=True)
        empty_gdf = gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=self.crs
        )
        for gdf_name, gdf in gdfs_none.items():
            print(f"{gdf_name}, ", end="", flush=True)
            empty_gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        if not qgz_path.exists():
            qgz_path_stored_dir = os.path.abspath(os.path.dirname(__file__))
            qgz_path_stored = Path(qgz_path_stored_dir, "assets\\ribasim_network.qgz")
            shutil.copy(qgz_path_stored, qgz_path)
        print("")
        print(f"Export location: {qgz_path}")


    def export_structures_to_excel(
        self,
        results_dir: Union[Path, str] = None,
    ):
        if results_dir is None:
            results_dir = Path(self.results_dir, self.name)

        write_structures_to_excel(
            pumps=self.pumps_gdf,
            weirs=self.weirs_gdf,
            orifices=self.orifices_gdf,
            bridges=self.bridges_gdf,
            culverts=self.culverts_gdf,
            uniweirs=self.uniweirs_gdf,
            split_nodes=self.split_nodes,
            split_node_type_conversion=self.split_node_type_conversion,
            split_node_id_conversion=self.split_node_id_conversion,
            results_dir=results_dir,
        )

    def import_structures_from_excel(
        self,
        excel_path: Union[Path, str],
    ):
        (
            structures_excel,
            structures_ids_to_include_as_splitnode,
            split_node_id_conversion,
        ) = read_structures_from_excel(excel_path)

        return structures_ids_to_include_as_splitnode, split_node_id_conversion


def create_ribasim_lumping_network(**kwargs):
    return RibasimLumpingNetwork(**kwargs)
