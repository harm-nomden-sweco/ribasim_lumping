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
    split_node_type_conversion: Dict, 
    split_node_id_conversion: Dict
):
    """Generate Ribasim Nodes"""

    print(" - create Ribasim nodes")
    # Basins
    basins["type"] = "Basin"
    basins["name"] = basins["basin"].apply(lambda x: f"Basin{str(x)}")

    # Boundaries
    boundary_conversion = {
        "dischargebnd": "FlowBoundary", 
        "waterlevelbnd": "LevelBoundary"
    }
    boundaries["type"] = boundaries["quantity"].replace(boundary_conversion)

    # Split nodes
    if not split_nodes[split_nodes.status==False].empty:
        print(f"   * {len(split_nodes[split_nodes.status==False])} split_nodes resulting in no_split (removed)")
        split_nodes = split_nodes[split_nodes.status==True]

    split_nodes["type"] = "TabulatedRatingCurve" 
    split_nodes_conversion = {
        "weir": "TabulatedRatingCurve",
        "uniweir": "TabulatedRatingCurve",
        "pump": "Pump",
        "culvert":"ManningResistance",
        "manual": "ManningResistance",
        "orifice": "TabulatedRatingCurve",
        "boundary_connection": "ManningResistance"
    }
    if isinstance(split_node_type_conversion, Dict):
        for key, value in split_node_type_conversion.items():
            split_nodes_conversion[key] = value
    split_nodes["type"] = split_nodes["split_type"].replace(split_nodes_conversion)

    if isinstance(split_node_id_conversion, Dict):
        for key, value in split_node_id_conversion.items():
            if len(split_nodes[split_nodes["node_id"] == key]) == 0:
                print(f"   * split_node type conversion id={key} (type={value}) does not exist")
            split_nodes.loc[split_nodes["node_id"] == key, "type"] = value

    # Ribasim Nodes Static
    ribasim_nodes_static = gpd.GeoDataFrame(
        data=pd.concat([
            basins, 
            boundaries, 
            split_nodes.rename(columns={"split_node_id": "name"})
        ]),
        geometry='geometry',
        crs=split_nodes.crs
    )

    ribasim_nodes_static = ribasim_nodes_static.set_index("node_id")
    ribasim_nodes_static = ribasim_nodes_static[["geometry", "type", "name"]]
    if ~ribasim_nodes_static.empty:
        ribasim_nodes = ribasim.Node(
            static=ribasim_nodes_static
        )
    else:
        ribasim_nodes = None
    
    return ribasim_nodes, ribasim_nodes_static


def generate_ribasim_edges(
    basin_connections: gpd.GeoDataFrame, 
    boundary_connections: gpd.GeoDataFrame
):
    """generate ribasim edges between nodes, using basin connections and boundary-basin connections"""
    print(" - create Ribasim edges")
    ribasim_edges_static = gpd.GeoDataFrame(
        data=pd.concat([
            basin_connections[["from_node_id", "to_node_id", "geometry"]], 
            boundary_connections[["from_node_id", "to_node_id", "geometry"]], 
        ], ignore_index=True),
        geometry='geometry',
        crs=basin_connections.crs
    )
    if ribasim_edges_static.empty:
        ribasim_edges = None
    else:
        ribasim_edges = ribasim.Edge(static=ribasim_edges_static)
    return ribasim_edges


def generate_ribasim_basins(
    basins_profile: pd.DataFrame,
    basins_static: pd.DataFrame
):
    """Generate settings for Ribasim Basins:
    static: node_id, drainage, potential_evaporation, infiltration, precipitation, urban_runoff
    profile: node_id, level, area, storage
    """
    print(" - generate settings Ribasim Basins")
    if basins_profile.empty or basins_static.empty:
        print("   x no basins")
        return None
    return ribasim.Basin(profile=basins_profile, static=basins_static)


def generate_ribasim_level_boundaries(level_boundary_static: gpd.GeoDataFrame):
    """generate ribasim level boundaries for all level boundary nodes
    static: node_id, level"""
    print(" - create Ribasim level boundaries")
    if level_boundary_static is None or level_boundary_static.empty:
        print("   x no level boundaries")
        return None
    return ribasim.LevelBoundary(static=level_boundary_static)


def generate_ribasim_flow_boundaries(flow_boundary_static: gpd.GeoDataFrame):
    """generate ribasim flow boundaries for all flow boundary nodes
    static: node_id, flow_rate"""
    print(" - create Ribasim flow boundaries")
    if flow_boundary_static is None or flow_boundary_static.empty:
        print("   x no flow boundaries")
        return None
    return ribasim.LevelBoundary(static=flow_boundary_static)


def generate_ribasim_pumps(pump_static: gpd.GeoDataFrame):
    """generate ribasim pumps for all pump nodes
    static: node_id, flow_rate""" 
    print(" - create Ribasim pumps")
    if pump_static is None or pump_static.empty:
        print("   x no pumps")
        return None
    return ribasim.Pump(static=pump_static)


def generate_ribasim_outlets(outlet_static: gpd.GeoDataFrame):
    """generate ribasim outlets for all outlet nodes
    static: node_id, flow_rate"""
    print(" - create Ribasim outlets")
    if outlet_static is None or outlet_static.empty:
        print("   x no outlets")
        return None
    return ribasim.Outlet(static=outlet_static)


def generate_ribasim_tabulatedratingcurves(
    tabulated_rating_curve_static: pd.DataFrame
):
    """generate ribasim tabulated rating using dummyvalues for level and discharge
    static: node_id, level, discharge"""
    print(" - create Ribasim tabulated rating curve")
    if tabulated_rating_curve_static.empty:
        print("   x no tabulated rating curves")
        return None
    return ribasim.TabulatedRatingCurve(static=tabulated_rating_curve_static)


def generate_ribasim_manningresistances(manningresistance_static: gpd.GeoDataFrame):
    """generate ribasim manning resistances
    static: node_id, length, manning_n, profile_width, profile_slope"""
    print(" - create Ribasim manning resistances")
    if manningresistance_static is None or manningresistance_static.empty:
        print("   x no manning resistances")
        return None
    return ribasim.ManningResistance(static=manningresistance_static)
    

def generate_ribasim_network_model(
    basins: gpd.GeoDataFrame = None, 
    split_nodes: gpd.GeoDataFrame = None, 
    boundaries: gpd.GeoDataFrame = None, 
    basin_connections: gpd.GeoDataFrame = None, 
    boundary_connections: gpd.GeoDataFrame = None, 
    split_node_type_conversion: Dict = None, 
    split_node_id_conversion: Dict = None
):
    """generate ribasim model from ribasim nodes and edges and
    optional input; ribasim basins, level boundary, flow_boundary, pump, tabulated rating curve and manning resistance """
    print("Generate ribasim model:")
    ribasim_nodes = generate_ribasim_nodes_static(
        boundaries=boundaries, 
        split_nodes=split_nodes, 
        basins=basins, 
        split_node_type_conversion=split_node_type_conversion, 
        split_node_id_conversion=split_node_id_conversion
    )
    ribasim_edges = generate_ribasim_edges(
        basin_connections=basin_connections,
        boundary_connections=boundary_connections
    )
    
    ribasim_basins = generate_ribasim_basins(
        basins_profile=basins_profile,
        basins_static=basins_static
    )
    ribasim_level_boundaries = generate_ribasim_level_boundaries(
        level_boundary_static=level_boundary_static
    )
    ribasim_flow_boundaries = generate_ribasim_flow_boundaries(
        flow_boundary_static=flow_boundary_static
    )
    ribasim_pumps = generate_ribasim_pumps(
        pump_static=pump_static
    )
    ribasim_outlets = generate_ribasim_outlets(
        outlet_static=outlet_static
    )
    ribasim_tabulated_rating_curve = generate_ribasim_tabulatedratingcurves(
        tabulated_rating_curve_static=tabulated_rating_curve_static, 
    )
    ribasim_manning_resistance = generate_ribasim_manningresistances(
        manningresistance_static=manningresistance_static, 
    )

    ribasim_model = ribasim.Model(
        modelname="ribasim_model",
        node=ribasim_nodes,
        edge=ribasim_edges,
        basin=ribasim_basins,
        level_boundary=ribasim_level_boundaries,
        flow_boundary=ribasim_flow_boundaries,
        pump=ribasim_pumps,
        outlet=ribasim_outlets,
        tabulated_rating_curve=ribasim_tabulated_rating_curve,
        manning_resistance=ribasim_manning_resistance,
        starttime="2020-01-01 00:00:00",
        endtime="2021-01-01 00:00:00",
    )
    return ribasim_model
