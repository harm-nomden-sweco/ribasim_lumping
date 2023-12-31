import geopandas as gpd
import pandas as pd
from typing import List, Union, Optional, Any, Tuple, Dict
import ribasim
from typing import Dict
from shapely.geometry import LineString


def generate_ribasim_nodes_static(
    boundaries: gpd.GeoDataFrame, 
    split_nodes: gpd.GeoDataFrame, 
    basins: gpd.GeoDataFrame, 
):
    """Generate Ribasim Nodes"""
    print("nodes ", end="", flush=True)
    # Ribasim Nodes Static
    ribasim_nodes_static = gpd.GeoDataFrame(
        data=pd.concat([
            boundaries.rename(columns={"boundary_node_id": "node_id"}), 
            split_nodes.rename(columns={"split_node_node_id": "node_id", "split_node_id": "name"}),
            basins.rename(columns={"basin_node_id": "node_id"}),
        ]),
        geometry='geometry',
        crs=split_nodes.crs
    )
    ribasim_nodes_static = ribasim_nodes_static.set_index("node_id")
    ribasim_nodes_static = ribasim_nodes_static[["geometry", "ribasim_type", "name"]]
    ribasim_nodes_static = ribasim_nodes_static.rename(columns={"ribasim_type": "type"})
    if ~ribasim_nodes_static.empty:
        ribasim_nodes = ribasim.Node(static=ribasim_nodes_static)
    else:
        ribasim_nodes = None
    
    return ribasim_nodes


def generate_ribasim_edges(
    basin_connections: gpd.GeoDataFrame, 
    boundary_connections: gpd.GeoDataFrame
):
    """generate ribasim edges between nodes, using basin connections and boundary-basin connections"""
    print("edges ", end="", flush=True)
    edges = pd.concat([
        basin_connections[["from_node_id", "to_node_id", "geometry"]], 
        boundary_connections[["from_node_id", "to_node_id", "geometry"]], 
    ], ignore_index=True)
    edges["edge_type"] = "flow"

    ribasim_edges_static = gpd.GeoDataFrame(
        data=edges,
        geometry='geometry',
        crs=basin_connections.crs
    )
    if ribasim_edges_static.empty:
        ribasim_edges = None
    else:
        ribasim_edges = ribasim.Edge(static=ribasim_edges_static)
    return ribasim_edges


def generate_ribasim_basins(
    basin_profile: pd.DataFrame,
    basin_time: pd.DataFrame,
    basin_state: pd.DataFrame,
):
    """Generate settings for Ribasim Basins:
    static: node_id, drainage, potential_evaporation, infiltration, precipitation, urban_runoff
    profile: node_id, level, area, storage
    """
    print("basins ", end="", flush=True)
    if basin_profile.empty or basin_time.empty:
        print("   x no basins")
        return None

    return ribasim.Basin(profile=basin_profile, time=basin_time, state=basin_state)


def generate_ribasim_level_boundaries(level_boundary_static: gpd.GeoDataFrame):
    """generate ribasim level boundaries for all level boundary nodes
    static: node_id, level"""
    print("boundaries ", end="", flush=True)
    if level_boundary_static is None or level_boundary_static.empty:
        print("   x no level boundaries")
        return None
    return ribasim.LevelBoundary(static=level_boundary_static)


def generate_ribasim_flow_boundaries(flow_boundary_static: gpd.GeoDataFrame):
    """generate ribasim flow boundaries for all flow boundary nodes
    static: node_id, flow_rate"""
    print("flow_boundaries ", end="", flush=True)
    if flow_boundary_static is None or flow_boundary_static.empty:
        return None
    return ribasim.LevelBoundary(static=flow_boundary_static)


def generate_ribasim_pumps(pump_static: gpd.GeoDataFrame):
    """generate ribasim pumps for all pump nodes
    static: node_id, flow_rate""" 
    print("pumps ", end="", flush=True)
    if pump_static is None or pump_static.empty:
        return None
    return ribasim.Pump(static=pump_static)


def generate_ribasim_outlets(outlet_static: gpd.GeoDataFrame):
    """generate ribasim outlets for all outlet nodes
    static: node_id, flow_rate"""
    print("outlets ", end="", flush=True)
    if outlet_static is None or outlet_static.empty:
        return None
    return ribasim.Outlet(static=outlet_static)


def generate_ribasim_tabulatedratingcurves(
    tabulated_rating_curve_static: pd.DataFrame
):
    """generate ribasim tabulated rating using dummyvalues for level and discharge
    static: node_id, level, discharge"""
    print("tabulatedratingcurve ", end="", flush=True)
    if tabulated_rating_curve_static is None or tabulated_rating_curve_static.empty:
        return None
    return ribasim.TabulatedRatingCurve(static=tabulated_rating_curve_static)


def generate_ribasim_manningresistances(manningresistance_static: gpd.GeoDataFrame):
    """generate ribasim manning resistances
    static: node_id, length, manning_n, profile_width, profile_slope"""
    print("manningresistances ", end="", flush=True)
    if manningresistance_static is None or manningresistance_static.empty:
        return None
    return ribasim.ManningResistance(static=manningresistance_static)
    

def generate_ribasim_model(
    simulation_code: str = "ribasim_model",
    basins: gpd.GeoDataFrame = None, 
    split_nodes: gpd.GeoDataFrame = None, 
    boundaries: gpd.GeoDataFrame = None, 
    basin_connections: gpd.GeoDataFrame = None, 
    boundary_connections: gpd.GeoDataFrame = None, 
    tables: Dict = None,
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
        basin_state=tables['basin_state']
    )

    ribasim_level_boundaries = generate_ribasim_level_boundaries(
        level_boundary_static=tables['level_boundary_static']
    )

    ribasim_flow_boundaries = generate_ribasim_flow_boundaries(
        flow_boundary_static=tables['flow_boundary_static']
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

    starttime = tables['basin_time']["time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    endtime = tables['basin_time']["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")

    print("")
    ribasim_model = ribasim.Model(
        modelname=simulation_code,
        node=ribasim_nodes,
        edge=ribasim_edges,
        basin=ribasim_basins,
        level_boundary=ribasim_level_boundaries,
        flow_boundary=ribasim_flow_boundaries,
        pump=ribasim_pumps,
        outlet=ribasim_outlets,
        tabulated_rating_curve=ribasim_tabulated_rating_curve,
        manning_resistance=ribasim_manning_resistance,
        starttime=starttime,
        endtime=endtime,
    )
    return ribasim_model
