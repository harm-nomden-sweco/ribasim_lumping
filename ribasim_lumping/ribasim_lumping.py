import os
import shutil
import sys
from contextlib import closing
from pathlib import Path
from sqlite3 import connect
from typing import Dict, List, Union, Tuple

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
import contextily as cx
from pydantic import BaseModel
from shapely.geometry import Point

from .utils.general_functions import (
    read_geom_file, 
    snap_to_network, 
    split_edges_by_split_nodes,
    log_and_remove_duplicate_geoms,
    assign_unassigned_areas_to_basin_areas,
    split_edges_by_dx,
)
from .dhydro.read_dhydro_simulations import add_dhydro_basis_network, add_dhydro_simulation_data
from .hydamo.read_hydamo_network import add_hydamo_basis_network
from .ribasim_network_generator.generate_split_nodes import add_split_nodes_based_on_selection
from .ribasim_network_generator.generate_ribasim_network import generate_ribasim_network_using_split_nodes
from .ribasim_network_generator.export_load_split_nodes import write_structures_to_excel, read_structures_from_excel
from .ribasim_model_generator.generate_ribasim_model import generate_ribasim_model
from .ribasim_model_generator.generate_ribasim_model_preprocessing import preprocessing_ribasim_model_tables
from .ribasim_model_generator.generate_ribasim_model_tables import generate_ribasim_model_tables
from .ribasim_model_results.ribasim_results import read_ribasim_model_results

import ribasim


class RibasimLumpingNetwork(BaseModel):
    """class to select datapoints from different simulations at certain timestamps"""
    name: str
    base_dir: Path
    results_dir: Path
    path_ribasim_executable: Path = None
    dhydro_basis_dir: Path = None,
    dhydro_results_dir: Path = None,
    areas_gdf:gpd.GeoDataFrame = None
    drainage_areas_gdf: gpd.GeoDataFrame = None
    supply_areas_gdf: gpd.GeoDataFrame = None
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
    boundaries_data: pd.DataFrame = None
    boundaries_timeseries_data: pd.DataFrame = None
    laterals_gdf: gpd.GeoDataFrame = None
    laterals_data: pd.DataFrame = None
    simulation_code: str = None
    simulation_path: Path = None
    basin_areas_gdf: gpd.GeoDataFrame = None
    basins_gdf: gpd.GeoDataFrame = None
    split_nodes: gpd.GeoDataFrame = None
    basin_connections_gdf: gpd.GeoDataFrame = None
    boundary_connections_gdf: gpd.GeoDataFrame = None
    split_node_type_conversion: Dict = None
    split_node_id_conversion: Dict = None
    nodes_h_df: pd.DataFrame = None
    nodes_h_basin_df: pd.DataFrame = None
    nodes_a_df: pd.DataFrame = None
    nodes_v_df: pd.DataFrame = None
    basins_h_df: pd.DataFrame = None
    basins_a_df: pd.DataFrame = None
    basins_v_df: pd.DataFrame = None
    basins_nodes_h_relation: gpd.GeoDataFrame = None
    edge_q_df: pd.DataFrame = None
    weir_q_df: pd.DataFrame = None
    uniweir_q_df: pd.DataFrame = None
    orifice_q_df: pd.DataFrame = None
    culvert_q_df: pd.DataFrame = None
    bridge_q_df: pd.DataFrame = None
    pump_q_df: pd.DataFrame = None
    basins_outflows: pd.DataFrame = None
    node_bedlevel: pd.DataFrame = None
    node_targetlevel: pd.DataFrame = None
    method_boundaries: int = 1
    method_laterals: int = 1
    laterals_areas_data: pd.DataFrame = None
    laterals_drainage_per_ha: pd.Series = None
    method_initial_waterlevels: int = 1
    initial_waterlevels_set_name: str = ""
    initial_waterlevels_timestep: int = 0
    initial_waterlevels_areas_id_column: str = ""
    initial_waterlevels_outside_areas: float = 0.0
    ribasim_model: ribasim.Model = None
    basis_source_types: List[str] = []
    basis_set_names: List[str] = []
    basis_set_start_months: List[int] = []
    basis_set_start_days: List[int] = []
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

    def add_basis_network(
            self, 
            source_type: str,
            set_name: str = None,
            dhydro_model_dir: Path = '.', 
            dhydro_simulation_name: str = 'default',
            dhydro_volume_tool_bat_file: Path = '.', 
            dhydro_volume_tool_force: bool = False,
            dhydro_volume_tool_increment: float = 0.1,
            hydamo_network_file: Path = 'network.gpkg',
            hydamo_network_gpkg_layer: str = 'network',
            ribasim_input_boundary_file: Path = 'boundary.gpkg',
            ribasim_input_boundary_gpkg_layer: str = 'boundary',
            ribasim_input_split_nodes_file: Path = 'split_nodes.gpkg',
            ribasim_input_split_nodes_gpkg_layer: Path = 'split_nodes',
            hydamo_split_nodes_bufdist: float = 10.0,
            hydamo_boundary_bufdist: float = 10.0,
            hydamo_write_results_to_gpkg: bool = False,
            hydamo_split_network_dx: float = None,
        ) -> Tuple[gpd.GeoDataFrame]:
        """
        Add (detailed) base network which will used to derive Ribasim network. Source type can either be "dhydro" or
        "hydamo". 
        If "dhydro", provide arguments for D-HYDRO volume tool (arguments dhydro_volume_*) and model simulation
        (model_dir, simulation_name). All necessary information (like boundaries and split nodes) will be derived from
        the D-HYDRO model data.
        If "hydamo", provide arguments for paths to files containing network geometries (hydamo_network_file) and
        in case of geopackage also the layer name (hydamo_network_gpkg_layer). HyDAMO does not contain information for
        boundaries and split nodes, so these need to be provided additionally as Ribasim input.

        Args:
            source_type (str):                          Source type of network. Options are "dhydro" or "hydamo"
            set_name (str):                             Name of set of settings (e.g. summer/winter). Will be ignored in hydamo.
            dhydro_model_dir (Path):                    Directory path to D-HYDRO model
            dhydro_simulation_name (str):               Name of D-HYDRO model simulation
            dhydro_volume_tool_bat_file (Path):         Path to D-HYDRO volume tool batch file
            dhydro_volume_tool_force (bool):            Set to True to force recalculation of D-HYDRO volume. Use False to read pre-calculated 
                                                        volume in simulation results
            dhydro_volume_tool_increment (float):       Increment argument (in m) for D-HYDRO volume tool
            dhydro_volume_tool_bat_file (Path):         Path to D-HYDRO volume tool batch file
            hydamo_network_file (str):                  Path to HyDAMO file containing network geometries
            hydamo_network_gpkg_layer (str):            Layer name of HyDAMO network file in case it is a geopackage
            ribasim_input_boundary_file (str):          Path to file containing Ribasim input boundary geometries (additionally needed when using HyDAMO data)
            ribasim_input_boundary_gpkg_layer (str):    Layer name of Ribasim input boundary file in case it is a geopackage
            ribasim_input_split_nodes_file (str):       Path to file containing Ribasim input split nodes geometries (additionally needed when using HyDAMO data)
            ribasim_input_split_nodes_gpkg_layer (str): Layer name of Ribasim input split nodes file in case it is a geopackage
            hydamo_split_nodes_bufdist (float):         Buffer distance (in meter) to snap split nodes to HyDAMO network
            hydamo_boundary_bufdist (float):            Buffer distance (in meter) to snap boundaries to HyDAMO network start/end nodes
            hydamo_write_results_to_gpkg (bool):        Write intermediate (HyDAMO) results (split nodes, edges, nodes, boundaries) to geopackages
            hydamo_split_network_dx (float):            Distance (in meter) to split up hydamo network (hydroobjects) into smaller sections. Default None
                                                        meaning no splitting is performed. The actual lengths of splitted sections will be close to given distance
                                                        but not exactly the same to prevent generating very small sections

        Returns:
            gpd.Geo:                                    Network
        """
        results = None
        if source_type == "dhydro":
            if set_name is None or not isinstance(set_name, str):
                raise ValueError("set_name for dhydro should be a string")
            results = add_dhydro_basis_network(
                set_name=set_name,
                model_dir=dhydro_model_dir,
                simulation_name=dhydro_simulation_name,
                volume_tool_bat_file=dhydro_volume_tool_bat_file, 
                volume_tool_force=dhydro_volume_tool_force,
                volume_tool_increment=dhydro_volume_tool_increment
            )
            self.basis_model_dirs.append(dhydro_model_dir)
            self.basis_simulations_names.append(dhydro_simulation_name)
            if results is not None:
                self.network_data, self.branches_gdf, self.network_nodes_gdf, self.edges_gdf, \
                    self.nodes_gdf, self.boundaries_gdf, self.laterals_gdf, self.weirs_gdf, \
                    self.uniweirs_gdf, self.pumps_gdf, self.orifices_gdf, self.bridges_gdf, \
                    self.culverts_gdf, self.boundaries_data, self.laterals_data, self.volume_data = results
        
        elif source_type == 'hydamo':
            # add network from HyDAMO files
            results = add_hydamo_basis_network(
                hydamo_network_file=hydamo_network_file,
                hydamo_network_gpkg_layer=hydamo_network_gpkg_layer,
            )
            if results is not None:
                self.branches_gdf, self.edges_gdf, self.nodes_gdf = results
            
            # add Ribasim input boundaries and split nodes from files
            self.add_split_nodes_from_file(
                split_nodes_file_path=ribasim_input_split_nodes_file,
                layer_name=ribasim_input_split_nodes_gpkg_layer,
            )
            self.add_boundaries_from_file(
                boundary_file_path=ribasim_input_boundary_file,
                layer_name=ribasim_input_boundary_gpkg_layer,
            )
            
            # Split up hydamo edges with given distance as approximate length of new edges
            if hydamo_split_network_dx is not None:
                self.edges_gdf = split_edges_by_dx(
                    edges=self.edges_gdf, 
                    dx=hydamo_split_network_dx,
                )

            # Because the split node and boundary locations won't always nicely match up with HyDAMO network, we need to adjust this
            # by snapping split nodes and boundaries to network nodes/edges
            self.split_nodes = snap_to_network(
                snap_type='split_node',
                points=self.split_nodes, 
                edges=self.edges_gdf, 
                nodes=self.nodes_gdf, 
                buffer_distance=hydamo_split_nodes_bufdist,
            )  
            self.boundaries_gdf = snap_to_network(
                snap_type='boundary',
                points=self.boundaries_gdf, 
                edges=self.edges_gdf, 
                nodes=self.nodes_gdf, 
                buffer_distance=hydamo_boundary_bufdist,
            )

            # split edges by split node locations so we end up with an network where split nodes are only located on nodes (and not edges)
            self.split_nodes, self.edges_gdf, self.nodes_gdf = split_edges_by_split_nodes(
                self.split_nodes, 
                self.edges_gdf, 
                buffer_distance=0.1,  # some small buffer to be sure but should actually not be necessary because of previous snap actions
            )

            # reset indexes of gdfs, rename some columns and do some automatic corrections
            for gdf in [self.split_nodes, self.edges_gdf, self.nodes_gdf, self.boundaries_gdf]:
                gdf.reset_index(drop=True, inplace=True)
                gdf.rename(columns={'split_type': 'object_type', 'Type': 'quantity', 'boundary_id': 'boundary', 'branch_id': 'code'}, inplace=True)
            if 'split_node' not in self.split_nodes.columns:
                self.split_nodes['split_node'] = self.split_nodes.index
            if 'quantity' in self.boundaries_gdf.columns:
                self.boundaries_gdf['quantity'] = self.boundaries_gdf['quantity'].replace({'FlowBoundary': 'dischargebnd', 'LevelBoundary': 'waterlevelbnd'})
            else:
                self.boundaries_gdf['quantity'] = 'waterlevelbnd'  # add default bnd type if not present in gdf
            
            # remove non-snapped split nodes and boundaries
            split_nodes_not_on_network = self.split_nodes.loc[(self.split_nodes['edge_no'] == -1) & (self.split_nodes['node_no'] == -1)]
            print(f"Remove non-snapped split nodes from dataset ({len(split_nodes_not_on_network)})")
            self.split_nodes = self.split_nodes.loc[[i not in split_nodes_not_on_network.index for i in self.split_nodes.index]]
            boundaries_not_on_network = self.boundaries_gdf.loc[self.boundaries_gdf['node_no'] == -1]
            print(f"Remove non-snapped boundaries from dataset ({len(boundaries_not_on_network)})")
            self.boundaries_gdf = self.boundaries_gdf.loc[[i not in boundaries_not_on_network.index for i in self.boundaries_gdf.index]]

            if hydamo_write_results_to_gpkg:
                self.split_nodes.to_file(Path(self.results_dir, 'split_nodes.gpkg'), layer='split_nodes', index=False)
                split_nodes_not_on_network.to_file(Path(self.results_dir, 'split_nodes_not_on_network.gpkg'), layer='split_nodes', index=False)
                self.edges_gdf.to_file(Path(self.results_dir, 'edges.gpkg'), layer='edges', index=False)
                self.nodes_gdf.to_file(Path(self.results_dir, 'nodes.gpkg'), layer='nodes', index=False)
                self.boundaries_gdf.to_file(Path(self.results_dir, 'boundaries.gpkg'), layer='boundaries', index=False)
                boundaries_not_on_network.to_file(Path(self.results_dir, 'boundaries_not_on_network.gpkg'), layer='boundaries', index=False)
        else:
            raise ValueError(f'Unknown source_type {source_type}. Only "dhydro" and "hydamo" are allowed')
        
        self.basis_source_types.append(source_type)

        return results

    def add_simulation_set(
        self,
        set_name: str,
        model_dir: Path,
        simulation_names: List[str],
        simulation_ts: Union[List, pd.DatetimeIndex] = [-1],
        source_type: str = 'dhydro',
    ):
        if source_type == 'dhydro':
            self.his_data, self.map_data = add_dhydro_simulation_data(
                set_name=set_name,
                model_dir=model_dir,
                simulation_names=simulation_names,
                simulation_ts=simulation_ts,
                his_data=self.his_data,
                map_data=self.map_data
            )
            self.source_types.append(source_type)
            self.set_names.append(set_name)
            self.model_dirs.append(model_dir)
            self.simulations_names.append(simulation_names)
            self.simulations_ts.append(simulation_ts)
        else:
            print(f"  x for this source type ({source_type}) no model type is added")
        return self.his_data, self.map_data
        
    def add_areas_from_file(
            self, 
            areas_file_path: Path, 
            areas_code_column: str = None, 
            layer_name: str = None, 
            crs: int = 28992
        ):
        """
        Add discharge unit areas (e.g. "afwateringseenheden") from file. In case of geopackage, provide layer_name.
        Overwrites previously defined areas

        Args:
            areas_file_path (Path):     Path to file containing areas geometries
            areas_code_column (str):    (optional) Id column for areas. If not provided, first column will be used for id
            layer_name (str):           Layer name in geopackage. Needed when file is a geopackage
            crs (int):                  (optional) CRS EPSG code. Default 28992 (RD New) 
        """
        print('Reading areas from file...')
        areas_gdf = read_geom_file(
            filepath=areas_file_path, 
            layer_name=layer_name, 
            crs=crs,
            explode_geoms=False,
        )
        areas_code_column = areas_code_column if areas_code_column is not None else list(areas_gdf.columns)[0]
        self.areas_gdf = areas_gdf[[areas_code_column, "geometry"]]
        self.areas_gdf.rename(columns={areas_code_column: 'area_code'}, inplace=True)
        print(f" - areas ({len(self.areas_gdf)}x)")
    
    def add_drainage_areas_from_file(
            self, 
            drainage_areas_file_path: Path,
            layer_name: str = None, 
            crs: int = 28992
        ):
        """
        Add drainage areas (e.g. "afvoergebieden") from file. In case of geopackage, provide layer_name.
        Overwrites previously defined drainage areas

        Args:
            drainage_areas_file_path (Path):    Path to file containing supply areas geometries
            layer_name (str):                   Layer name in geopackage. Needed when file is a geopackage
            crs (int):                          (optional) CRS EPSG code. Default 28992 (RD New) 
        """
        print('Reading drainage areas from file...')
        self.drainage_areas_gdf = read_geom_file(
            filepath=drainage_areas_file_path, 
            layer_name=layer_name, 
            crs=crs
        )
        print(f" - drainage areas ({len(self.drainage_areas_gdf)}x)")

    def add_supply_areas_from_file(
            self, 
            supply_areas_file_path: Path,
            layer_name: str = None, 
            crs: int = 28992
        ):
        """
        Add supply areas (e.g. "afvoergebieden") from file. In case of geopackage, provide layer_name.
        Overwrites previously defined supply areas

        Args:
            supply_areas_file_path (Path):      Path to file containing supply areas geometries
            layer_name (str):                   Layer name in geopackage. Needed when file is a geopackage
            crs (int):                          (optional) CRS EPSG code. Default 28992 (RD New) 
        """
        print('Reading supply areas from file...')
        self.supply_areas_gdf = read_geom_file(
            filepath=supply_areas_file_path, 
            layer_name=layer_name, 
            crs=crs
        )
        print(f" - supply areas ({len(self.supply_areas_gdf)}x)")
    
    def add_boundaries_from_file(
            self, 
            boundary_file_path: Path,
            layer_name: str = None,
            crs: int = 28992
        ):
        """
        Add Ribasim boundaries from file. In case of geopackage, provide layer_name. 
        Overwrites previously defined boundaries

        Args:
            boundary_file_path (Path):  Path to file containing boundary geometries
            layer_name (str):           Layer name in geopackage. Needed when file is a geopackage
            crs (int):                  (optional) CRS EPSG code. Default 28992 (RD New)     
        """
        print('Reading boundaries from file...')
        self.boundaries_gdf = read_geom_file(
            filepath=boundary_file_path, 
            layer_name=layer_name, 
            crs=crs,
            remove_z_dim=True
        )
        self.boundaries_gdf = log_and_remove_duplicate_geoms(self.boundaries_gdf, colname='boundary_id')
    
    def read_areas_laterals_timeseries(
            self, 
            areas_laterals_path: Path, 
            sep: str = ',', 
            index_col: int = 0, 
            dayfirst=False
        ):
       
        self.laterals_areas_data = pd.read_csv(
            areas_laterals_path, 
            index_col=index_col, 
            sep=sep, 
            parse_dates=True, 
            dayfirst=dayfirst
        )

    def read_boundaries_timeseries_data(
            self, 
            boundaries_timeseries_path: Path, 
            skiprows=0, 
            sep=",", 
            index_col=0
        ):
        
        boundary_csv_data = pd.read_csv(
            boundaries_timeseries_path, 
            sep=sep,
            skiprows=skiprows, 
            index_col=index_col, 
            parse_dates=True
        )
        boundary_csv_data = boundary_csv_data.interpolate()
        self.boundaries_timeseries_data = boundary_csv_data
    
    def export_or_update_all_ribasim_structures_specs(self, structure_specs_dir_path: Path):
        for structure_type in ['pump', 'weir', 'orifice', 'culvert']:
            structure_specs_path = Path(structure_specs_dir_path, f"ribasim_{structure_type}s_specs.xlsx")
            self.export_or_update_ribasim_structure_specs(
                structure_type=structure_type,
                structure_specs_path=structure_specs_path
            )

    def export_or_update_ribasim_structure_specs(self, structure_type: str, structure_specs_path: Path):
        if structure_type not in ['pump', 'weir', 'orifice', 'culvert']:
            raise ValueError(f" x not able export/update structure type {structure_type}")
        gdfs = dict(
            pump=self.pumps_gdf,
            weir=self.weirs_gdf,
            orifice=self.orifices_gdf,
            culvert=self.culverts_gdf
        )
        structures_columns = dict(
            pump=['capacity'],
            weir=['crestwidth'],
            orifice=['crestwidth'],
            culvert=['crestwidth']
        )
        control_headers = ["upstream_upperlimit", "upstream_setpoint", "upstream_lowerlimit", 
                           "downstream_upperlimit", "downstream_setpoint", "downstream_lowerlimit"]
        sets_columns = dict(
            pump=['startlevelsuctionside', 'stoplevelsuctionside', 'startleveldeliveryside', 'stopleveldeliveryside'],
            weir=['crestlevel'] + control_headers,
            orifice=['crestlevel'] + control_headers,
            culvert=['crestlevel'] + control_headers
        )
        gdf = gdfs[structure_type]
        general_columns = ['structure_id']
        structure_columns = structures_columns[structure_type]
        set_columns = sets_columns[structure_type]
        all_columns_second_level = general_columns + structure_columns + set_columns

        if not structure_specs_path.exists():
            print(f" x ribasim_input_{structure_type} ('{structure_specs_path}') DOES NOT exist. network.{structure_type}s_gdf will be EXPORTED.")
            gdf_export = gdf.drop(columns=[col for col in gdf.columns if col[1] not in all_columns_second_level])
            gdf_export.to_excel(structure_specs_path)
            return gdf

        print(f" x ribasim_input_{structure_type} ('{structure_specs_path}') DOES exist. network.{structure_type}s_gdf will be UPDATED.")
        gdf_input = pd.read_excel(structure_specs_path, header=[0,1], index_col=0)
        for set_name in self.basis_set_names:
            if set_name in gdf_input.columns.get_level_values(level=0):
                gdf_input[set_name] = gdf_input[set_name].replace('', np.nan)
        for col in gdf.columns:
            if col[1] in all_columns_second_level:
                gdf[col] = gdf_input[col]
        return gdf

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
        """
        Set split nodes geodataframe in network object. Overwrites previously defined split nodes
        """
        self.split_nodes  = add_split_nodes_based_on_selection(
            stations=stations,
            pumps=pumps,
            weirs=weirs,
            uniweirs=uniweirs,
            orifices=orifices,
            culverts=culverts,
            bridges=bridges,
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
    
    def add_split_nodes_from_file(
            self, 
            split_nodes_file_path: Path, 
            layer_name: str = None, 
            crs: int = 28992
        ):
        """
        Add split nodes in network object from file. Overwrites previously defined split nodes

        Args:
            split_nodes_file_path (Path):   Path to file containing split node geometries
            layer_name (str):               Layer name in geopackage. Needed when file is a geopackage
            crs (int):                      (optional) CRS EPSG code. Default 28992 (RD New) 
        """
        print('Reading split nodes from file...')
        self.split_nodes = read_geom_file(
            filepath=split_nodes_file_path, 
            layer_name=layer_name, 
            crs=crs,
            remove_z_dim=True    
        )
        self.split_nodes = log_and_remove_duplicate_geoms(self.split_nodes, colname='split_node')

    def set_split_nodes(self, split_nodes_gdf: gpd.GeoDataFrame):
        """
        Set split nodes geodataframe in network object. Overwrites previously defined split nodes
        """
        self.split_nodes = split_nodes_gdf


    # def add_hydamo_split_nodes_boundaries(self, model_dir: Path = None, areas_id_column: str = None):
    #     if model_dir is None:
    #         model_dir = self.hydamo_basis_dir
    #     areas_gpkg_path = Path(model_dir, "areas.gpkg")
    #     areas_gpkg_layer = "areas"
    #     self.read_areas(
    #         areas_gpkg_path=areas_gpkg_path, 
    #         areas_gpkg_layer=areas_gpkg_layer, 
    #         areas_id_column=areas_id_column
    #     )
    #     ribasim_input_gpkg_path = Path(model_dir, "ribasim_input.gpkg")
    #     split_nodes = gpd.read_file(ribasim_input_gpkg_path, layer="split_nodes")
    #     boundaries = gpd.read_file(ribasim_input_gpkg_path, layer="boundaries")


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
            use_laterals_for_basin_area: bool = False,
            remove_isolated_basins: bool = False,
        ) -> Dict:
        """
        Generate ribasim_lumping network. This function generates all 

        Args:
            simulation_code (str):              give name for ribasim-simulation
            split_node_type_conversion (Dict):  dictionary general conversion of object-type to ribasim-type (e.g. weir: TabulatedRatingCurve)
            split_node_id_conversion (Dict):    dictionary specific conversion of split-node-id to ribasim-type (e.g. KST01234: Outlet)
            use_laterals_for_basin_area (bool): use standard lateral inflow per second per area applied to basin_areas.
            remove_isolated_basins (bool):      
        """
        # first some checks
        self.simulation_code = simulation_code
        self.simulation_path = Path(self.results_dir, simulation_code)
        if self.split_nodes is None:
            raise ValueError("no split_nodes defined: use .add_split_nodes(), .add_split_nodes_from_file() or .set_split_nodes()")
        if self.nodes_gdf is None or self.edges_gdf is None:
            raise ValueError(
                "no nodes and/or edges defined: add d-hydro simulation results or hydamo network"
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
            laterals=self.laterals_gdf,
            use_laterals_for_basin_area=use_laterals_for_basin_area,
            remove_isolated_basins=remove_isolated_basins,
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

        # assign areas to basin areas which have no edge within it so it is not assigned to any basin area
        # also update the basin and basin area code in edges and nodes
        results = assign_unassigned_areas_to_basin_areas(
            self.areas_gdf,
            self.basin_areas_gdf, 
            self.drainage_areas_gdf,
            # self.edges_gdf,
        )
        self.areas_gdf = results['areas']
        self.basin_areas_gdf = results['basin_areas']
        if 'edges' in results.keys():
            self.edges_gdf = results['edges']

        # Export to geopackage
        self.export_to_geopackage(simulation_code=simulation_code)
        return results


    def generate_ribasim_model_complete(
        self, 
        set_name: str,
        dummy_model: bool = False,
        saveat: int = None,
        interpolation_lines: int = 5,
        database_gpkg: str = 'database.gpkg',
        results_dir: str = 'results',
        results_subgrid: bool = False
    ):
        if set_name not in self.basis_set_names:
            raise ValueError(f'set_name {set_name} not in available set_names')
        
        # preprocessing data to input for tables
        basins_outflows, node_h_basin, node_h_node, node_a, node_v, basin_h, basin_a, basin_v, \
            node_bedlevel, node_targetlevel, orig_bedlevel, basins_nodes_h_relation, \
                edge_q_df, weir_q_df, uniweir_q_df, \
                    orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df = \
                        preprocessing_ribasim_model_tables(
                            dummy_model=dummy_model,
                            map_data=self.map_data, 
                            his_data=self.his_data,
                            volume_data=self.volume_data, 
                            nodes=self.nodes_gdf, 
                            weirs=self.weirs_gdf,
                            uniweirs=self.uniweirs_gdf,
                            pumps=self.pumps_gdf, 
                            culverts=self.culverts_gdf,
                            orifices=self.orifices_gdf,
                            basins=self.basins_gdf, 
                            split_nodes=self.split_nodes, 
                            basin_connections=self.basin_connections_gdf, 
                            boundary_connections=self.boundary_connections_gdf,
                            interpolation_lines=interpolation_lines,
                            set_names=self.basis_set_names
                        )
        
        self.nodes_gdf["bedlevel"] = orig_bedlevel
        self.nodes_h_df = node_h_node
        self.nodes_h_basin_df = node_h_basin
        self.nodes_a_df = node_a
        self.nodes_v_df = node_v
        self.basins_h_df = basin_h
        self.basins_a_df = basin_a
        self.basins_v_df = basin_v
        self.basins_nodes_h_relation = basins_nodes_h_relation
        self.edge_q_df = edge_q_df
        self.weir_q_df = weir_q_df
        self.uniweir_q_df = uniweir_q_df
        self.orifice_q_df = orifice_q_df
        self.culvert_q_df = culvert_q_df
        self.bridge_q_df = bridge_q_df
        self.pump_q_df = pump_q_df
        self.basins_outflows = basins_outflows
        self.node_bedlevel = node_bedlevel
        self.node_targetlevel = node_targetlevel

        basin_h_initial = None
        if not dummy_model:
            if self.method_initial_waterlevels == 1:
                basin_h_initial = basin_h.loc[set_name].loc["targetlevel"]
                # raise ValueError('method initial waterlevels = 1 not yet implemented')
            elif self.method_initial_waterlevels == 2:
                ind_initial_h = 0
                for i_set_name, i_simulations_names, i_sim_ts in zip(self.set_names, self.simulations_names, self.simulations_ts):
                    if i_set_name != set_name:
                        ind_initial_h += len(i_sim_ts)
                    else:
                        for i_ts, ts in enumerate(i_sim_ts):
                            if i_ts == self.initial_waterlevels_timestep:
                                break
                            ind_initial_h += 1
                        break
                basin_h_initial = basin_h.loc[set_name].iloc[ind_initial_h + 2 + interpolation_lines*2]
            elif self.method_initial_waterlevels == 3:
                raise ValueError('method initial waterlevels = 3 not yet implemented')
            else:
                raise ValueError('method initial waterlevels not 1, 2 or 3')

        # generate ribasim model tables
        tables = generate_ribasim_model_tables(
            dummy_model=dummy_model,
            basin_h=basin_h, 
            basin_a=basin_a, 
            basins=self.basins_gdf, 
            basin_areas=self.basin_areas_gdf,
            areas=self.areas_gdf,
            basins_nodes_h_relation=self.basins_nodes_h_relation,
            laterals=self.laterals_gdf,
            laterals_data=self.laterals_data,
            boundaries=self.boundaries_gdf, 
            boundaries_data=self.boundaries_data,
            split_nodes=self.split_nodes,
            basins_outflows=basins_outflows,
            set_name=set_name,
            method_boundaries=self.method_boundaries,
            boundaries_timeseries_data=self.boundaries_timeseries_data,
            method_laterals=self.method_laterals,
            laterals_areas_data=self.laterals_areas_data,
            laterals_drainage_per_ha=self.laterals_drainage_per_ha,
            basin_h_initial=basin_h_initial,
            saveat=saveat,
            edge_q_df=edge_q_df, 
            weir_q_df=weir_q_df, 
            uniweir_q_df=uniweir_q_df, 
            orifice_q_df=orifice_q_df, 
            culvert_q_df=culvert_q_df, 
            bridge_q_df=bridge_q_df, 
            pump_q_df=pump_q_df,
        )

        # generate ribasim model
        ribasim_model = generate_ribasim_model(
            simulation_filepath=Path(self.results_dir, self.simulation_code),
            basins=self.basins_gdf.copy(),
            split_nodes=self.split_nodes.copy(),
            boundaries=self.boundaries_gdf.copy(),
            basin_connections=self.basin_connections_gdf.copy(),
            boundary_connections=self.boundary_connections_gdf.copy(),
            tables=tables,
            database_gpkg=database_gpkg,
            results_dir=results_dir,
        )
        self.ribasim_model = ribasim_model
        
        # Export ribasim model
        if self.simulation_path is None:
            self.simulation_path = Path(self.results_dir, self.simulation_code)
        # check for timestep (saveat)
        if saveat is not None:
            ribasim_model.solver = ribasim.Solver(saveat=saveat)
        if results_subgrid:
            ribasim_model.results.subgrid = True

        ribasim_model.write(Path(self.simulation_path, "ribasim.toml"))
        with open(Path(self.simulation_path, "run_ribasim_model.bat"), 'w') as f:
            f.write(f"{str(self.path_ribasim_executable)} ribasim.toml\n")
            f.write(f"pause")

        print(f"Export location: {Path(self.results_dir, self.simulation_code)}")
        # export ribasim_network
        self.export_to_geopackage(simulation_code=self.simulation_code)
        return ribasim_model


    def export_to_geopackage(
            self, 
            simulation_code: str, 
            results_dir: 
            Union[Path, str] = None
        ):
        
        if results_dir is None:
            results_dir = self.results_dir
        results_network_dir = Path(results_dir, simulation_code)
        if not Path(results_network_dir).exists():
            Path(results_network_dir).mkdir()
        gpkg_path = Path(results_network_dir, "ribasim_network.gpkg")
        qgz_path = Path(results_network_dir, "ribasim_network.qgz")

        gdfs_orig = dict(
            areas=self.areas_gdf,
            branches=self.branches_gdf,
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
            laterals=self.laterals_gdf,
            boundaries=self.boundaries_gdf,
            boundary_connections=self.boundary_connections_gdf,
            node_h=self.nodes_h_df,
            node_a=self.nodes_a_df,
            node_v=self.nodes_v_df,
            basin_h=self.basins_h_df,
            basin_a=self.basins_a_df,
            basin_v=self.basins_v_df,
            basins_nodes_h_relation=self.basins_nodes_h_relation
        )
        gdfs_none = dict()
        gdfs = dict()
        for gdf_name, gdf in gdfs_orig.items():
            if gdf is None:
                gdfs_none[gdf_name] = gdf
            elif "geometry" not in gdf.columns:
                if gdf.columns.name is not None:
                    column = gdf.columns.name
                    gdf = gdf.stack()
                    gdf.name = "data"
                    gdf = gdf.reset_index().reset_index().sort_values(by=[column, "index"]).reset_index(drop=True).drop(columns="index")
                gdf["geometry"] = Point(0,0)
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=28992)
                gdfs[gdf_name] = gdf
            else:
                gdfs[gdf_name] = gdf

        print(f"Exporting to geopackage:")
        print(" - available: ", end="", flush=True)
        for gdf_name, gdf in gdfs.items():
            print(f"{gdf_name}, ", end="", flush=True)
            gdf_copy = gdf.copy()
            if isinstance(gdf_copy.columns, pd.MultiIndex):
                gdf_copy.columns = ['__'.join(col).strip('__') for col in gdf_copy.columns.values]
            gdf_copy.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        print("")
        print(" - not available: ", end="", flush=True)
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=self.crs)
        for gdf_name, gdf in gdfs_none.items():
            print(f"{gdf_name}, ", end="", flush=True)
            empty_gdf.to_file(gpkg_path, layer=gdf_name, driver="GPKG")

        # dfs_orig = dict(
        # )
        # write to database using sqlite3
        # with closing(connect(gpkg_path)) as connection:
        #     for df_orig_name, df_orig in dfs_orig.items():
        #         sql = "INSERT INTO gpkg_contents (table_name, data_type, identifier) VALUES (?, ?, ?)"
        #         if df_orig is None:
        #             continue
        #         df_orig.to_sql(df_orig_name, connection, index=False, if_exists="replace")
        #         with closing(connection.cursor()) as cursor:
        #             cursor.execute(sql, (df_orig_name, "attributes", df_orig_name))
        #     connection.commit()
        
        if not qgz_path.exists():
            qgz_path_stored_dir = os.path.abspath(os.path.dirname(__file__))
            qgz_path_stored = Path(qgz_path_stored_dir, "assets\\ribasim_network.qgz")
            shutil.copy(qgz_path_stored, qgz_path)
        print("")
        print(f"Export location: {qgz_path}")

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        if self.basin_areas_gdf is not None:
            cmap = matplotlib.colors.ListedColormap(np.random.rand(len(self.basin_areas_gdf)*2, 3))
            self.basin_areas_gdf.plot(ax=ax, column='basin_node_id', cmap=cmap, alpha=0.35, zorder=1)
            self.basin_areas_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.25, label='basin_areas', zorder=1)
        elif self.areas_gdf is not None:
            cmap = matplotlib.colors.ListedColormap(np.random.rand(len(self.areas_gdf)*2, 3))
            self.areas_gdf.plot(ax=ax, column='area_code', cmap=cmap, alpha=0.35, zorder=1)
            self.areas_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, label='areas', zorder=1)
        # if self.ribasim_model is not None:
        #     self.ribasim_model.plot(ax=ax)
        if self.edges_gdf is not None:
            self.edges_gdf.plot(ax=ax, linewidth=1, color='blue', label='hydro-objecten', zorder=2)
        if self.split_nodes is not None:
            self.split_nodes.plot(ax=ax, color='black', label='split_nodes', zorder=3)
        if self.boundaries_gdf is not None:
            self.boundaries_gdf.plot(ax=ax, color='red', marker='s', label='boundary', zorder=3)
        ax.axis('off')
        ax.legend(prop=dict(size=10), loc ="lower right", bbox_to_anchor=(1.4, 0.0))
        cx.add_basemap(ax, crs=self.areas_gdf.crs)
        return fig, ax

    def export_structures_to_excel(self, results_dir: Union[Path, str] = None):
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


    def import_structures_from_excel(self, excel_path: Union[Path, str]):
        (
            structures_excel,
            structures_ids_to_include_as_splitnode,
            split_node_id_conversion,
        ) = read_structures_from_excel(excel_path)

        return structures_ids_to_include_as_splitnode, split_node_id_conversion


    def plot_basin_waterlevels_for_basins(
            self, 
            set_name: str, 
            basin_node_ids: 
            List[int]
        ):
        """A plot will be generated showing the bed level and waterlevels along node_no (x-axis)
        input selected set_name and basin node_ids"""
        for basin_node_id in basin_node_ids:
            basin_node_no = self.basins_gdf[self.basins_gdf.basin_node_id==basin_node_id].node_no.values[0]

            fig, ax = plt.subplots(figsize=(9,6))
            nodes_basin = self.nodes_gdf.groupby(by='basin_node_id').get_group(basin_node_id)
            basin_node_nos = list(nodes_basin.node_no.values)
            node_h_new = self.nodes_h_df[basin_node_nos].loc[set_name].T

            ax.axvline(basin_node_no, linestyle='--')

            node_h_new[node_h_new.columns[-1:5:-1]].plot(ax=ax, style='o')
            node_h_new[node_h_new.columns[5:3:-1]].plot(ax=ax, color='lightgrey', style='o')
            node_h_new[node_h_new.columns[3]].plot(ax=ax, linewidth=4, style='o')
            node_h_new[node_h_new.columns[2:0:-1]].plot(ax=ax, color='lightgrey', style='o')
            node_h_new[node_h_new.columns[0]].rename('lowestlevel').plot(ax=ax, style='o')
            node_h_new["bedlevel"].plot(ax=ax, linewidth=4, color='black', linestyle='--')
            
            plt.legend(loc='upper left', bbox_to_anchor=(-0.4,1))
            ax.text(
                0.95, 0.95, f'Basin {basin_node_id}',
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, fontsize=15
            )


    def read_ribasim_results(self, simulation_code: str):
        simulation_path = Path(self.results_dir, simulation_code)
        ribasim_results = read_ribasim_model_results(
            simulation_path=simulation_path
        )
        return ribasim_results


def create_ribasim_lumping_network(**kwargs):
    return RibasimLumpingNetwork(**kwargs)
