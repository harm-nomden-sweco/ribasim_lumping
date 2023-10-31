# pylint: disable=missing-function-docstring
import sys
import os
from pathlib import Path
from typing import List, Union, Dict
import shutil
import matplotlib.pyplot as plt
import subprocess
from pydantic import BaseModel
import geopandas as gpd
import pandas as pd
import xarray as xr
import xugrid as xu
import networkx as nx
from .utils.general_functions import find_file_in_directory
from .dhydro.read_dhydro_network import get_dhydro_volume_based_on_basis_simulations
from .dhydro.read_dhydro_simulations import add_dhydro_basis_network, add_dhydro_simulation_data
from .ribasim_utils.generate_split_nodes import add_split_nodes_based_on_selection
from .ribasim_utils.generate_ribasim_network import generate_ribasim_network_using_split_nodes
from .ribasim_utils.export_load_split_nodes import (
    write_structures_to_excel,
    read_structures_from_excel,
)
from .ribasim_utils.generate_ribasim_model import generate_ribasim_model
from .ribasim_utils.generate_ribasim_model_preprocessing import preprocessing_ribasim_model_tables
from .ribasim_utils.generate_ribasim_model_tables import generate_ribasim_model_tables

sys.path.append("..\\..\\ribasim\\python\\ribasim")
import ribasim


class RibasimLumpingNetwork(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""
    name: str
    base_dir: Path
    dhydro_basis_dir: Path
    dhydro_results_dir: Path
    results_dir: Path
    areas_gdf:gpd.GeoDataFrame = None
    his_data: xu.UgridDataset = None
    map_data: xu.UgridDataset = None
    network_data: xr.Dataset = None
    volume_data: xr.Dataset = None
    network_graph: nx.DiGraph = None
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
    boundaries_data: pd.DataFrame = None
    laterals_data: pd.DataFrame = None
    simulation_code: str = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    split_nodes: gpd.GeoDataFrame = None
    basin_connections_gdf: gpd.GeoDataFrame = None
    boundary_connections_gdf: gpd.GeoDataFrame = None
    split_node_type_conversion: Dict = None
    split_node_id_conversion: Dict = None
    nodes_h_df: pd.DataFrame = None
    nodes_a_df: pd.DataFrame = None
    nodes_v_df: pd.DataFrame = None
    basins_h_df: pd.DataFrame = None
    basins_a_df: pd.DataFrame = None
    basins_v_df: pd.DataFrame = None
    basins_outflows: pd.DataFrame = None
    node_bedlevel: pd.DataFrame = None
    node_targetlevel: pd.DataFrame = None
    ribasim_model: ribasim.Model = None
    basis_source_types: List[str] = []
    basis_set_names: List[str] = []
    basis_model_dirs: List[Path] = []
    basis_simulations_names: List[str] = []
    source_types: List[str] = []
    set_names: List[str] = []
    model_dirs: List[Path] = []
    simulations_names: List[List] = []
    simulations_output_dirs: List[str] = []
    simulations_ts: List[Union[List, pd.DatetimeIndex]] = []
    crs: int = 28992

    class Config:
        arbitrary_types_allowed = True

    def read_areas(self, areas_file_path: Path, areas_id_column: str):
        areas_gdf = gpd.read_file(areas_file_path)
        self.areas_gdf = areas_gdf[[areas_id_column, "geometry"]]
        print(f" - areas ({len(areas_gdf)}x)")

    def add_basis_network(
        self, 
        source_type: str, 
        model_dir: Path,
        set_name: str, 
        simulation_name: str,
        dhydro_volume_tool_bat_file: Path, 
        dhydro_volume_tool_force: bool = False,
        dhydro_volume_tool_increment: float = 0.1
    ):
        results = None
        if source_type == 'dhydro':
            results = add_dhydro_basis_network(
                model_dir=model_dir, 
                set_name=set_name, 
                simulation_name=simulation_name,
                volume_tool_bat_file=dhydro_volume_tool_bat_file, 
                volume_tool_force=dhydro_volume_tool_force,
                volume_tool_increment=dhydro_volume_tool_increment
            )
        
        self.basis_source_types.append(source_type)
        self.basis_set_names.append(set_name)
        self.basis_model_dirs.append(model_dir)
        self.basis_simulations_names.append(simulation_name)

        if results is not None:
            self.network_data, self.branches_gdf, self.network_nodes_gdf, self.edges_gdf, \
                self.nodes_gdf, self.boundaries_gdf, self.laterals_gdf, self.weirs_gdf, \
                self.uniweirs_gdf, self.pumps_gdf, self.orifices_gdf, self.bridges_gdf, \
                self.culverts_gdf, self.boundaries_data, self.laterals_data, self.volume_data = results
        return results

    def add_simulation_set(
        self,
        set_name: str,
        model_dir: Path,
        simulation_names: List[str],
        simulation_ts: Union[List, pd.DatetimeIndex] = [-1],
        source_type: str = 'dhydro',
    ):
        results = None
        if source_type == 'dhydro':
            results = add_dhydro_simulation_data(
                set_name=set_name,
                model_dir=model_dir,
                simulation_names=simulation_names,
                simulation_ts=simulation_ts,
                set_names=self.set_names,
                model_dirs=self.model_dirs,
                simulations_names=self.simulations_names,
                simulations_ts=self.simulations_ts,
                his_data=self.his_data,
                map_data=self.map_data
            )
            self.set_names, self.model_dirs, self.simulations_names, self.simulations_ts, \
                self.his_data, self.map_data = results
            self.source_types.append(source_type)
            self.set_names.append(set_name)
            self.model_dirs.append(model_dir)
            self.simulations_names.append(simulation_names)
            self.simulations_ts.append(simulation_ts)
        else:
            print(f"  x for this source type ({source_type}) no model type is added")
        return results

    def add_split_nodes(
        self,
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
        self.split_nodes  = add_split_nodes_based_on_selection(
            stations=stations,
            pumps=pumps,
            weirs=weirs,
            orifices=orifices,
            bridges=bridges,
            culverts=culverts,
            uniweirs=uniweirs,
            edges=edges,
            structures_ids_to_include=structures_ids_to_include,
            structures_ids_to_exclude=structures_ids_to_exclude,
            edge_ids_to_include=edge_ids_to_include,
            edge_ids_to_exclude=edge_ids_to_exclude,
            list_gdfs=[
                self.stations_gdf, 
                self.pumps_gdf, 
                self.weirs_gdf, 
                self.orifices_gdf, 
                self.bridges_gdf, 
                self.culverts_gdf,
                self.uniweirs_gdf,
                self.edges_gdf
            ]
        )
        return self.split_nodes

    @property
    def split_node_ids(self):
        if self.split_nodes is None:
            return None
        return list(self.split_nodes.node_no.values)

    def generate_ribasim_lumping_model(
        self,
        simulation_code: str,
        set_name: str,
        split_node_type_conversion: Dict,
        split_node_id_conversion: Dict,
        starttime: str = None,
        endtime: str = None,
    ):
        self.generate_ribasim_lumping_network(
            simulation_code=simulation_code,
            split_node_type_conversion=split_node_type_conversion,
            split_node_id_conversion=split_node_id_conversion,
        )
        ribasim_model = self.generate_ribasim_model_complete(
            set_name=set_name,
            starttime=starttime,
            endtime=endtime
        )
        return ribasim_model

    def generate_ribasim_lumping_network(
        self,
        simulation_code: str,
        split_node_type_conversion: Dict,
        split_node_id_conversion: Dict,
    ) -> Dict:
        self.simulation_code = simulation_code
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
        self.split_node_type_conversion = split_node_type_conversion
        self.split_node_id_conversion = split_node_id_conversion

        results = generate_ribasim_network_using_split_nodes(
            nodes=self.nodes_gdf,
            edges=self.edges_gdf,
            split_nodes=self.split_nodes,
            areas=self.areas_gdf,
            boundaries=self.boundaries_gdf,
            split_node_type_conversion=split_node_type_conversion,
            split_node_id_conversion=split_node_id_conversion,
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
        self.boundary_connections_gdf = results['boundary_connections']
        # Export to geopackage
        self.export_to_geopackage(simulation_code=simulation_code)
        return results

    def generate_ribasim_model_complete(
        self, 
        set_name: str,
        starttime: str = None,
        endtime: str = None
    ):
        if set_name not in self.basis_set_names:
            raise ValueError(f'set_name {set_name} not in available set_names')

        # preprocessing data to input for tables
        basins_outflows, node_h, node_a, node_v, basin_h, basin_a, basin_v, node_bedlevel, node_targetlevel, orig_bedlevel = \
            preprocessing_ribasim_model_tables(
                map_data=self.map_data, 
                volume_data=self.volume_data, 
                nodes=self.nodes_gdf, 
                weirs=self.weirs_gdf, 
                pumps=self.pumps_gdf, 
                basins=self.basins_gdf, 
                split_nodes=self.split_nodes, 
                basin_connections=self.basin_connections_gdf, 
                boundary_connections=self.boundary_connections_gdf,
            )
        self.nodes_gdf["bedlevel"] = orig_bedlevel.copy()
        self.nodes_h_df = node_h.copy()
        self.nodes_a_df = node_a.copy()
        self.nodes_v_df = node_v.copy()
        self.basins_h_df = basin_h.copy()
        self.basins_a_df = basin_a.copy()
        self.basins_v_df = basin_v.copy()
        self.basins_outflows = basins_outflows.copy()
        self.node_bedlevel = node_bedlevel.copy()
        self.node_targetlevel = node_targetlevel.copy()

        # generate ribasim model tables
        tables = generate_ribasim_model_tables(
            basin_h=basin_h, 
            basin_a=basin_a, 
            basins=self.basins_gdf, 
            basin_areas=self.basin_areas_gdf,
            boundaries=self.boundaries_gdf, 
            boundaries_data=self.boundaries_data, 
            split_nodes=self.split_nodes,
            basins_outflows=basins_outflows,
            set_name=set_name,
        )

        # generate ribasim model
        ribasim_model = generate_ribasim_model(
            simulation_code=self.simulation_code,
            basins=self.basins_gdf.copy(),
            split_nodes=self.split_nodes.copy(),
            boundaries=self.boundaries_gdf.copy(),
            basin_connections=self.basin_connections_gdf.copy(),
            boundary_connections=self.boundary_connections_gdf.copy(),
            tables=tables,
            starttime=starttime,
            endtime=endtime,
        )
        self.ribasim_model = ribasim_model

        # Export ribasim model
        ribasim_model.write(Path(self.results_dir, self.simulation_code))
        print(f"Export location: {Path(self.results_dir, self.simulation_code)}")
        return ribasim_model

    def export_to_geopackage(self, simulation_code: str, results_dir: Union[Path, str] = None):
        if results_dir is None:
            results_dir = self.results_dir
        results_network_dir = Path(results_dir, simulation_code)
        if not Path(results_network_dir).exists():
            Path(results_network_dir).mkdir()
        gpkg_path = Path(results_network_dir, "ribasim_network.gpkg")
        qgz_path = Path(results_network_dir, "ribasim_network.qgz")

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
            boundary_connections=self.boundary_connections_gdf,
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
            gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        print("")
        print(" - not available: ", end="", flush=True)
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=self.crs)
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

    def plot_basin_waterlevels_based_on_node_nos(self, set_name: str, basins_nos: List[int]):

        for basin_no in basins_nos:
            basin_node_no = self.basins_gdf[self.basins_gdf['basin']==basin_no].node_no.values[0]

            fig, ax = plt.subplots()
            nodes_basin = self.nodes_gdf.groupby(by='basin').get_group(basin_no)
            nodes_basin['bedlevel'].plot(ax=ax, linewidth=4, color='black')

            basin_node_nos = list(nodes_basin.node_no.values)
            node_h_new = self.nodes_h_df[basin_node_nos].loc[set_name].T
            node_h_new[node_h_new.columns[-1:5:-1]].plot(ax=ax)
            node_h_new[node_h_new.columns[4:6]].plot(ax=ax, color='lightgrey', linestyle='--')
            node_h_new[node_h_new.columns[3]].plot(ax=ax, linewidth=4, linestyle='--')
            node_h_new[node_h_new.columns[1:3]].plot(ax=ax, color='lightgrey', linestyle='--')
            node_h_new[node_h_new.columns[0]].rename('lowestlevel').plot(ax=ax, linewidth=4)
            
            ax.axvline(basin_node_no, linestyle='--')
            plt.legend(loc='upper left', bbox_to_anchor=(-0.4,1))
            ax.text(
                0.95, 0.95, f'Basin {basin_no}',
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, fontsize=15
            )

    def read_ribasim_results(self, simulation_code: str):
        simulation_path = Path(self.results_dir, simulation_code)
        def read_arrow_file(name: str):
            arrow_file = Path(simulation_path, 'results', f"{name}.arrow")
            return pd.read_feather(arrow_file)
        basin_df = read_arrow_file('basin')
        control_df = read_arrow_file('control')
        flow_df = read_arrow_file('flow')
        return basin_df, control_df, flow_df

def create_ribasim_lumping_network(**kwargs):
    return RibasimLumpingNetwork(**kwargs)
