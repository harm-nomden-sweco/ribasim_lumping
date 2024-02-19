from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import numpy as np
import ribasim
from shapely.geometry import LineString
import networkx as nx

from ..ribasim_network_generator.generate_ribasim_network import create_graph_based_on_nodes_edges
from ..utils.general_functions import split_edges_by_split_nodes


def generate_ribasim_nodes_static(
    boundaries: gpd.GeoDataFrame, 
    split_nodes: gpd.GeoDataFrame, 
    basins: gpd.GeoDataFrame, 
):
    """Generate Ribasim Nodes"""
    # Ribasim Nodes Static
    nodes = pd.concat([
        boundaries.rename(columns={"boundary_node_id": "node_id"}), 
        split_nodes.rename(columns={"split_node_node_id": "node_id", "split_node_id": "name"}),
        basins.rename(columns={"basin_node_id": "node_id"}),
    ])

    print(f"nodes ({len(nodes)}x), ", end="", flush=True)
    ribasim_nodes_static = gpd.GeoDataFrame(
        data=nodes,
        geometry='geometry',
        crs=split_nodes.crs
    )
    ribasim_nodes_static = ribasim_nodes_static.set_index("node_id")
    ribasim_nodes_static = ribasim_nodes_static[["ribasim_type", "name", "geometry"]]
    ribasim_nodes_static = ribasim_nodes_static.rename(columns={"ribasim_type": "type"})

    if ~ribasim_nodes_static.empty:
        ribasim_nodes = ribasim.Node(df=ribasim_nodes_static)
    else:
        ribasim_nodes = None
    
    return ribasim_nodes


def generate_ribasim_edges(
    basin_connections: gpd.GeoDataFrame, 
    boundary_connections: gpd.GeoDataFrame
):
    """generate ribasim edges between nodes, using basin connections and boundary-basin connections"""
    edges = pd.concat([
        basin_connections[["from_node_id", "to_node_id", "geometry"]], 
        boundary_connections[["from_node_id", "to_node_id", "geometry"]], 
    ], ignore_index=True)

    print(f"edges ({len(edges)}x), ", end="", flush=True)

    edges["edge_type"] = "flow"
    ribasim_edges_static = gpd.GeoDataFrame(
        data=edges,
        geometry='geometry',
        crs=basin_connections.crs
    )
    if ribasim_edges_static.empty:
        ribasim_edges = None
    else:
        ribasim_edges = ribasim.Edge(df=ribasim_edges_static)
    return ribasim_edges


def generate_ribasim_basins(
    basin_profile: pd.DataFrame,
    basin_time: pd.DataFrame,
    basin_state: pd.DataFrame,
    basin_subgrid: pd.DataFrame
):
    """Generate settings for Ribasim Basins:
    static: node_id, drainage, potential_evaporation, infiltration, precipitation, urban_runoff
    profile: node_id, level, area, storage
    """
    if basin_profile.empty or basin_time.empty:
        print(f"basins (--)", end="", flush=True)
        return ribasim.Basin()
    print(f"basins ({len(basin_state)}x)", end="", flush=True)
    return ribasim.Basin(profile=basin_profile, time=basin_time, state=basin_state, subgrid=basin_subgrid)


def generate_ribasim_level_boundaries(
        level_boundary_static: gpd.GeoDataFrame,
        level_boundary_time: pd.DataFrame,
):
    """generate ribasim level boundaries for all level boundary nodes
    static: node_id, level"""
    if level_boundary_time is not None:
        print('level')
        return ribasim.LevelBoundary(time=level_boundary_time)
    elif level_boundary_static is None or level_boundary_static.empty:
        print(f"boundaries (--", end="", flush=True)
        return ribasim.LevelBoundary()
    print(f"boundaries ({len(level_boundary_static)}x)", end="", flush=True)
    return ribasim.LevelBoundary(static=level_boundary_static)


def generate_ribasim_flow_boundaries(
        flow_boundary_static: gpd.GeoDataFrame, 
        flow_boundary_time: pd.DataFrame):
    """generate ribasim flow boundaries for all flow boundary nodes
    static: node_id, flow_rate"""
    print("flow_boundaries ", end="", flush=True)
    if flow_boundary_time is not None:
        return ribasim.FlowBoundary(time=flow_boundary_time)
    elif flow_boundary_static is None or flow_boundary_static.empty:
        print("   x no flow boundaries")
        return ribasim.FlowBoundary()
    return ribasim.FlowBoundary(static=flow_boundary_static)


def generate_ribasim_pumps(pump_static: gpd.GeoDataFrame):
    """generate ribasim pumps for all pump nodes
    static: node_id, flow_rate""" 
    print("pumps ", end="", flush=True)
    if pump_static is None or pump_static.empty:
        print("   x no pumps")
        return ribasim.Pump()
    return ribasim.Pump(static=pump_static)


def generate_ribasim_outlets(outlet_static: gpd.GeoDataFrame):
    """generate ribasim outlets for all outlet nodes
    static: node_id, flow_rate"""
    print("outlets ", end="", flush=True)
    if outlet_static is None or outlet_static.empty:
        print("   x no outlets", end="", flush=True)
        return ribasim.Outlet()
    return ribasim.Outlet(static=outlet_static)


def generate_ribasim_tabulatedratingcurves(
    tabulated_rating_curve_static: pd.DataFrame
):
    """generate ribasim tabulated rating using dummyvalues for level and flow_rate
    static: node_id, level, flow_rate"""
    print("tabulatedratingcurve ", end="", flush=True)
    if tabulated_rating_curve_static is None or tabulated_rating_curve_static.empty:
        print("   x no tabulated rating curve")
        return ribasim.TabulatedRatingCurve()
    return ribasim.TabulatedRatingCurve(static=tabulated_rating_curve_static)


def generate_ribasim_manningresistances(manningresistance_static: gpd.GeoDataFrame):
    """generate ribasim manning resistances
    static: node_id, length, manning_n, profile_width, profile_slope"""
    print("manningresistances ", end="", flush=True)
    if manningresistance_static is None or manningresistance_static.empty:
        print("   x no manningresistance")
        return ribasim.ManningResistance()
    return ribasim.ManningResistance(static=manningresistance_static)
    

def generate_fractional_flows():
    return ribasim.FractionalFlow()


def generate_linear_resistances():
    return ribasim.LinearResistance()


def generate_terminals():
    return ribasim.Terminal()


def generate_discrete_controls():
    return ribasim.DiscreteControl()


def generate_pid_controls():
    return ribasim.PidControl()


def generate_users():
    return ribasim.User()


def generate_allocations():
    return ribasim.Allocation()


def generate_solvers():
    return ribasim.Solver()


def generate_loggings():
    return ribasim.Logging()


def generate_ribasim_model(
    simulation_filepath: Path,
    basins: gpd.GeoDataFrame = None, 
    split_nodes: gpd.GeoDataFrame = None, 
    boundaries: gpd.GeoDataFrame = None, 
    basin_connections: gpd.GeoDataFrame = None, 
    boundary_connections: gpd.GeoDataFrame = None, 
    tables: Dict = None,
    database_gpkg: str = 'database.gpkg',
    results_dir: str = 'results'
):
    """generate ribasim model from ribasim nodes and edges and
    optional input; ribasim basins, level boundary, flow_boundary, pump, tabulated rating curve and manning resistance """
    
    print("Generate ribasim model: ", end="", flush=True)
    
    ribasim_nodes = generate_ribasim_nodes_static(
        boundaries=boundaries, 
        split_nodes=split_nodes, 
        basins=basins,
    )

    ribasim_edges = generate_ribasim_edges(
        basin_connections=basin_connections,
        boundary_connections=boundary_connections
    )
    
    ribasim_basins = generate_ribasim_basins(
        basin_profile=tables['basin_profile'],
        basin_time=tables['basin_time'], 
        basin_state=tables['basin_state'],
        basin_subgrid=tables['basin_subgrid']
    )

    ribasim_level_boundaries = generate_ribasim_level_boundaries(
        level_boundary_static=tables['level_boundary_static'],
        level_boundary_time=tables['level_boundary_time']

    )

    ribasim_flow_boundaries = generate_ribasim_flow_boundaries(
        flow_boundary_static=tables['flow_boundary_static'],
        flow_boundary_time=tables['flow_boundary_time']
    )

    ribasim_pumps = generate_ribasim_pumps(
        pump_static=tables['pump_static']
    )

    ribasim_outlets = generate_ribasim_outlets(
        outlet_static=tables['outlet_static']
    )

    ribasim_tabulated_rating_curve = generate_ribasim_tabulatedratingcurves(
        tabulated_rating_curve_static=tables['tabulated_rating_curve_static'], 
    )

    ribasim_manning_resistance = generate_ribasim_manningresistances(
        manningresistance_static=tables['manningresistance_static'], 
    )

    fractions_flows = generate_fractional_flows()

    linear_resistances = generate_linear_resistances()

    terminals = generate_terminals()

    discrete_controls = generate_discrete_controls()

    pid_controls = generate_pid_controls()

    users = generate_users()

    allocations = generate_allocations()

    solvers = generate_solvers()

    loggings = generate_loggings()

    starttime = tables['basin_time']["time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    endtime = tables['basin_time']["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")

    print("")
    network = ribasim.Network(
        node=ribasim_nodes,
        edge=ribasim_edges,
        filepath=simulation_filepath
    )
    ribasim_model = ribasim.Model(
        # modelname=simulation_code,
        network=network,
        basin=ribasim_basins,
        level_boundary=ribasim_level_boundaries,
        flow_boundary=ribasim_flow_boundaries,
        pump=ribasim_pumps,
        outlet=ribasim_outlets,
        tabulated_rating_curve=ribasim_tabulated_rating_curve,
        manning_resistance=ribasim_manning_resistance,
        fractional_flow=fractions_flows,
        linear_resistance=linear_resistances,
        terminal=terminals,
        discrete_control=discrete_controls,
        pid_control=pid_controls,
        user=users,
        allocation=allocations,
        solver=solvers,
        logging=loggings,
        starttime=starttime,
        endtime=endtime,
    )

    # add database name and results folder
    ribasim_model.database = database_gpkg
    ribasim_model.results_dir = results_dir
    return ribasim_model


def get_direct_downstream_structures_for_nodes(
        structures: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,    
        nodes: gpd.GeoDataFrame,
        split_nodes: gpd.GeoDataFrame = None,
    ) -> gpd.GeoDataFrame:
    """
    Get structure directly downstream of nodes within the same basin based on shortest path. 
    Use this function to get the node no and the shortest path length to that node for weir and/or pump structures.
    
    Basin code is retrieved from edges.
    The edges will be split on the structure locations before determining the most direct downstream structure.
    Supply split nodes optionally to update structure point geometries with snapped split nodes to avoid duplication

    Parameters
    ----------
    structures (gpd.GeoDataFrame):               
        GeoDataFrame containing all structure geometries. Should include 'code' column
    edges (gpd.GeoDataFrame):
        GeoDataFrame containing edges. Should include a 'basin' column with the associated basin code
    nodes (gpd.GeoDataFrame):
        GeoDataFrame containing nodes (that are compliant with edges). These will be returned as a result with info of downstream structure
    split_nodes (gpd.GeoDataFrame):
        (optional) snapped split nodes to edges/nodes. Should include 'split_node_id' column which contains code that is used in structures geodataframe
    
    Returns
    -------
    GeoDataFrames with nodes with 2 columns added containing direct downstream structure node no and the path length from node to that structure
    """

    structures, edges, nodes_orig = structures.copy(), edges.copy(), nodes.copy()
    # update structure point locations with already snapped points of split nodes
    if split_nodes is not None:
        structures.geometry = [split_nodes.loc[split_nodes['split_node_id'] == c]['geometry'].values[0]
                        if c in split_nodes['split_node_id'] 
                        else g 
                        for c, g in zip(structures['code'], structures.geometry)]
    # split edges on structure locations. this will also regenerate the edges and nodes (but preserves basin in edges)
    structures, edges, nodes = split_edges_by_split_nodes(structures, edges=edges)
    structures.columns = [c.lower() for c in structures.columns]
    # initialize new columns
    nodes['downstream_structure_code'] = np.nan
    nodes['downstream_structure_node_no'] = np.nan
    nodes['downstream_structure_path_length'] = np.nan
    # go over each basin individually to restrict the network size
    basin_nrs = np.sort(edges['basin'].unique())
    basin_nrs = basin_nrs[basin_nrs >= 1]
    for basin_nr in basin_nrs:
        # select nodes, edges and structures within basin
        _edges = edges.loc[edges['basin'] == basin_nr].copy()
        _nodes = nodes.loc[np.isin(nodes['node_no'], np.unique(np.concatenate([_edges['from_node'], _edges['to_node']])))].copy()
        _structures = structures.loc[np.isin(structures['node_no'], _nodes['node_no'])].copy()

        if _structures.empty:
            continue  # skip if no structures are within basin

        # create graph
        _graph = create_graph_based_on_nodes_edges(_nodes, _edges, add_edge_length_as_weight=True)

        # get shortest paths between nodes (keeping in mind direction of edges in graph) and the lengths of those paths
        paths = dict(nx.all_pairs_dijkstra_path(_graph))
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(_graph))

        # select for each node the direct downstream structure (the one with the shortest path to node)
        structure_nodes = {k: v for k, v in zip(_structures['node_no'].values, _structures['code'].values)}
        store = {}
        for k, vps in paths.items():
            length_to_structures = {vp: path_lengths[k][vp] for vp in vps.keys() if vp in structure_nodes.keys()}

            # select the structure and store together with structure code and distance from node to that structure
            if k in structure_nodes.keys():
                store[k] = (k, structure_nodes[k], 0)  # node is a structure
            elif len(length_to_structures.keys()) == 0:
                store[k] = None  # there is no downstream structure within basin for node
            else:
                n = list(length_to_structures.keys())[np.argmin(length_to_structures.values())]
                store[k] = (n, structure_nodes[n], length_to_structures[n])

        # save in nodes geodataframe
        for k, v in store.items():
            if v is not None:
                nodes.loc[nodes['node_no'] == k, 'downstream_structure_node_no'] = v[0]
                nodes.loc[nodes['node_no'] == k, 'downstream_structure_code'] = v[1]
                nodes.loc[nodes['node_no'] == k, 'downstream_structure_path_length'] = v[2]
    
    # update original nodes geodataframe with downstream structure info using spatial join with max 5 cm buffer
    nodes_updated = nodes_orig.copy()
    nodes_updated = nodes_updated.sjoin_nearest(
        nodes[['downstream_structure_node_no', 'downstream_structure_code', 'downstream_structure_path_length', 'geometry']], 
        how='left', 
        max_distance=0.05
    )

    return nodes_updated
    