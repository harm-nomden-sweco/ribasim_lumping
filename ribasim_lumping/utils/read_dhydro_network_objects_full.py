import os
import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Dict, List
import hydrolib.core.dflowfm as hcdfm
from pathlib import Path
from typing import List, Union, Tuple
from shapely.geometry import Point, LineString


def get_dhydro_network_objects_full(dhydro_dir: str, simulation_name: str, crs):
    simulation_path = dhydro_dir / simulation_name
    # simulation_path = os.path.join(dhydro_dir, simulation_name)

    structures_file, net_file, lateral_file = get_dhydro_files(simulation_path) 

    # read files from dhydro input
    netcdf_data, df_structures =  get_objectdata_from_dhydro_input(net_file=net_file, structures_file=structures_file, 
                                                                   simulation_path=simulation_path)
    
    boundary_data, lateral_data = get_timeseriesdata_from_dhydro_input(lateral_file=lateral_file,simulation_path=simulation_path)

    # read nodes from dhydro input
    nodes_gdf = get_nodes_dhydro_netcdf(netcdf_data, crs)
    edges_gdf = get_edges_dhydro_netcdf(netcdf_data, nodes_gdf, crs)
    branches_gdf = get_branches_dhydro_netcdf(netcdf_data, crs)

    weirs_gdf, culverts_gdf, uniweirs_gdf, pumps_gdf, orifices_gdf, bridges_gdf = get_structures_dhydro(df_structures)

    boundaries_gdf = None
    # edges_gdf = None
    edges_q_df = None
    nodes_h_df = None
    stations_gdf = None

    return nodes_gdf, nodes_h_df, edges_gdf, edges_q_df, branches_gdf, \
        stations_gdf, pumps_gdf, weirs_gdf, orifices_gdf, \
            bridges_gdf, culverts_gdf, uniweirs_gdf, boundaries_gdf
    

def get_dhydro_files(simulation_path: str):
    mdu_file = ""
    net_file = "FlowFM\input\FlowFM_net.nc"
    lateral_file = "FlowFM\input\FlowFM_lateral_sources.bc"
    structures_file = "FlowFM\input\structures.ini"

    return structures_file, net_file, lateral_file

def get_objectdata_from_dhydro_input(
    net_file,
    structures_file,
    simulation_path: Path,
):
    """ "read dhydro input data:
    - from simulation (names)
    - from dhydro folder (dir)
    Returns: dataframe of structures and boundary data
    """
    print("D-HYDRO-files analysed:")

    # get netcdf
    # net_file = "FlowFM_net.nc"
    netfilepath = simulation_path / net_file
    netcdf_data = hcdfm.net.models.Network.from_file(netfilepath)

    # get structure file
    # structures_file = "structures.ini"
    structuresfilepath = simulation_path / structures_file
    m = hcdfm.structure.models.StructureModel(structuresfilepath)
    df_structures = pd.DataFrame([f.__dict__ for f in m.structure])
    df_structures = df_structures.drop(columns=['comments'])

    return netcdf_data, df_structures

def get_timeseriesdata_from_dhydro_input(
    lateral_file,
    simulation_path,
):
    """ "read dhydro input data:
    - from simulation (names)
    - from dhydro folder (dir)
    Returns: dataframe of structures and boundary data
    """

    # get boundary1dconditions data 
    boundary_data = None
    for root, dirs, files in os.walk(simulation_path):
        for file in files:
            if file.endswith("boundaryconditions1d.bc"):
                filepath = root + os.sep + file
                forcingmodel_object = hcdfm.ForcingModel(filepath)
                boundary_data = pd.DataFrame([forcing.dict() for forcing in forcingmodel_object.forcing])
                # convert dictionary with boundary type to columns
                boundary_data = pd.concat([boundary_data.drop(['quantityunitpair'], axis=1), pd.DataFrame.from_records(boundary_data['quantityunitpair'])[0].apply(pd.Series)], axis=1)
    if boundary_data is None:
        print(" * simulation does not contain boundary file (ending with 'boundaryconditions1d.bc'")

    # get laterals timeseries
    # lateral_file = "FlowFM_lateral_sources.bc"
    lateralfilepath = simulation_path / lateral_file

    forcingmodel_object = hcdfm.ForcingModel(lateralfilepath)
    lateral_data = pd.DataFrame([forcing.dict() for forcing in forcingmodel_object.forcing])


    return boundary_data, lateral_data

def get_structures_dhydro(df_structures):
    """return dataframes of all structures"""
    weir_columns = ['id','branchid','chainage','crestlevel','crestwidth','allowedflowdir','corrcoeff', 'usevelocityheight']
    weir_gdf = df_structures.loc[df_structures['type'] == 'weir'][weir_columns].set_index('id')
    print(" - weirs", end="", flush=True)

    culvert_columns = ['id','branchid','chainage','leftlevel','rightlevel','length','inletlosscoeff','outletlosscoeff','bedfrictiontype', 'bedfriction']
    culvert_gdf = df_structures.loc[df_structures['type'] == 'culvert'][culvert_columns].set_index('id')
    print(" culverts", end="", flush=True)

    uniweir_columns = ['id','branchid','chainage','crestlevel','allowedflowdir','numlevels','yvalues','zvalues','dischargecoeff']
    umiweir_gdf = df_structures.loc[df_structures['type'] == 'universalWeir'][uniweir_columns].set_index('id')
    print(" universal weirs", end="", flush=True)

    pump_columns = ['id','branchid','chainage','orientation','controlside', 'numstages', 'capacity', 'startlevelsuctionside', 'stoplevelsuctionside',
                     'startleveldeliveryside', 'stopleveldeliveryside', 'numreductionlevels', 'head', 'reductionfactor']
    pump_gdf = df_structures.loc[df_structures['type'] == 'pump'][pump_columns].set_index('id')
    print(" pumps", end="", flush=True)

    orifice_columns = ['id','branchid','chainage','crestlevel','crestwidth','allowedflowdir','corrcoeff', 'usevelocityheight']
    orifice_gdf = df_structures.loc[df_structures['type'] == 'weir'][orifice_columns].set_index('id')
    print(" orifices", end="", flush=True)

    bridge_columns = ['id','branchid','chainage','length','inletlosscoeff','outletlosscoeff']
    bridge_gdf = df_structures.loc[df_structures['type'] == 'bridge'][bridge_columns].set_index('id')
    print(" bridges", end="", flush=True)

    return weir_gdf, culvert_gdf, umiweir_gdf, pump_gdf, orifice_gdf, bridge_gdf

def get_nodes_dhydro_netcdf(netcdf_data, crs) -> gpd.GeoDataFrame:
    """calculate nodes dataframe"""
    print(" - nodes")
    nodes_df = pd.DataFrame({'node_id':netcdf_data._mesh1d.mesh1d_node_id,'X':netcdf_data._mesh1d.mesh1d_node_x, 'Y':netcdf_data._mesh1d.mesh1d_node_y})
    nodes_df['index_node'] = nodes_df.index
    nodes_df['geometry'] = list(zip(nodes_df['X'], nodes_df['Y']))
    nodes_df['geometry'] = nodes_df['geometry'].apply(Point)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry='geometry',crs=crs)
    return nodes_gdf


def get_edges_dhydro_netcdf(netcdf_data, nodes_gdf, crs) -> gpd.GeoDataFrame:
    """calculate edges dataframe"""
    print(" - edges")

    edges_df = pd.DataFrame({'branch_id':netcdf_data._mesh1d.mesh1d_edge_branch_id,
                             'X':netcdf_data._mesh1d.mesh1d_edge_x, 
                             'Y':netcdf_data._mesh1d.mesh1d_edge_y, 
                             'from_node': netcdf_data._mesh1d.mesh1d_edge_nodes[:,0], 
                             'to_node': netcdf_data._mesh1d.mesh1d_edge_nodes[:,1]})
    edges_df['geometry']=''
    
    edges_gdf = edges_df.merge(nodes_gdf, how="inner", left_on="from_node", right_on="index_node", suffixes=["","_from"])
    edges_gdf = edges_gdf.merge(nodes_gdf, how="inner", left_on="to_node", right_on="index_node", suffixes=["","_to"])

    edges_gdf['geometry'] = edges_gdf.apply(
    lambda row: LineString([row['geometry_from'], row['geometry_to']]), 
    axis=1
    )
    edges_gdf = gpd.GeoDataFrame(edges_gdf, geometry='geometry', crs=crs)
    edges_gdf['index_edge'] = edges_gdf.index
    edges_gdf=edges_gdf[['index_edge','branch_id','geometry','from_node','to_node']]

    return edges_gdf

def get_branches_dhydro_netcdf(netcdf_data, crs)-> gpd.GeoDataFrame:
    """calculate branches dataframe"""
    branch_keys = [b for b in netcdf_data._mesh1d.branches.keys()]
    branch_geom = [b.geometry for b in netcdf_data._mesh1d.branches.values()]
    branches_df = pd.DataFrame({'branch_id':branch_keys,
                             'branch_geom':branch_geom, 
                             })
    
    branches_df['geometry'] = branches_df.apply(
    lambda row: LineString(row['branch_geom']), 
    axis=1
    )
    branches_gdf = gpd.GeoDataFrame(branches_df, geometry='geometry', crs=crs)

    return branches_gdf



