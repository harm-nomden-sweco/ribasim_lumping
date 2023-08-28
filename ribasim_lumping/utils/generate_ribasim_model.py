import geopandas as gpd
import pandas as pd
from typing import List, Union, Optional, Any, Tuple, Dict
import ribasim
from typing import Dict
from shapely.geometry import LineString


def generate_ribasim_nodes(
    basins: gpd.GeoDataFrame = None, 
    split_nodes: gpd.GeoDataFrame = None, 
    boundaries: gpd.GeoDataFrame = None, 
    split_node_type_conversion: Dict = None, 
    split_node_id_conversion: Dict = None
) -> Tuple[gpd.GeoDataFrame]:
    """use basins, splintnodes and boundaries to generate ribasim nodes
    define splitnodetypes per nodetype or use default 
    (pump as Pump, manual and culvert as ManningResistance, all others as TabulatedRatingcurve)
    returns ribasim node and boundaries, splitnodes and ribasim nodes as gdf's with ribsasim node-id column"""
    print(" - create Ribasim nodes")
    basins_gdf =basins.copy()
    basins_gdf['node_id'] = basins_gdf['basin'] 
    basins_gdf['type'] = 'Basin'

    if boundaries is None:
        boundaries_gdf = None
        len_boundaries = 0
    else:
        boundaries_gdf = boundaries.copy()
        boundaries_gdf['node_id'] = boundaries_gdf['boundary_id'] + len(basins) +1
        boundary_conversion = {
            'dischargebnd': 'FlowBoundary', 
            'waterlevelbnd': 'LevelBoundary'
        }
        boundaries_gdf['type'] = boundaries_gdf['quantity'].replace(boundary_conversion)
        len_boundaries = len(boundaries)

    splitnodes_gdf = split_nodes.copy()
    splitnodes_gdf.insert(0, 'splitnode_id', range(len(splitnodes_gdf)))
    splitnodes_gdf['node_id'] = splitnodes_gdf['splitnode_id'] + len(basins) + len_boundaries +1
    splitnodes_gdf['type'] = 'TabulatedRatingCurve' 

    split_nodes_conversion = {
        'weir': 'TabulatedRatingCurve',
        'uniweir': 'TabulatedRatingCurve',
        'pump': 'Pump',
        'culvert':'ManningResistance',
        'manual': 'ManningResistance',
        'orifice' : 'TabulatedRatingCurve'
    }
    if isinstance(split_node_type_conversion, Dict):
        for key, value in split_node_type_conversion.items():
            split_nodes_conversion[key] = value
    splitnodes_gdf['type'] = splitnodes_gdf['split_type'].replace(split_nodes_conversion)

    if isinstance(split_node_id_conversion, Dict):
        for key, value in split_node_id_conversion.items():
            if len(splitnodes_gdf[splitnodes_gdf['mesh1d_node_id'] == key]) == 0:
                print(f" * split_node type conversion id={key} (type={value}) does not exist")
            splitnodes_gdf.loc[splitnodes_gdf['mesh1d_node_id'] == key, 'type'] = value

    # concat nodes
    ribasim_node_gdf = pd.concat([basins_gdf, boundaries_gdf,splitnodes_gdf]).set_crs(split_nodes.crs)
    ribasim_node_gdf = ribasim_node_gdf.set_index('node_id')
    ribasim_node_gdf = ribasim_node_gdf[['geometry', 'type']]

    if not ribasim_node_gdf.empty:
        node = ribasim.Node(static=ribasim_node_gdf)
    else:
        node=None
    
    return node, boundaries_gdf, splitnodes_gdf, ribasim_node_gdf


def generate_ribasim_edges(
    basins: gpd.GeoDataFrame = None, 
    split_nodes_gdf: gpd.GeoDataFrame = None, 
    basin_connections: gpd.GeoDataFrame = None, 
    boundary_basin_connections: gpd.GeoDataFrame = None
):
    """generate ribasim edges between nodes, using basin connections and boundary-basin connections"""
    print(" - create Ribasim edges")
    basin_connections_gdf = basin_connections[['mesh1d_node_id', 'mesh1d_nEdges', 'basin_in','basin_out','geometry']]

    # (1) nodes
    # merge to find splitnode id
    basin_conn_node_gdf = basin_connections_gdf.merge(
        split_nodes_gdf[['splitnode_id','mesh1d_node_id', 'node_id']], 
        left_on='mesh1d_node_id', 
        right_on='mesh1d_node_id'
    )

    # split connections in the connections upstream and downstream of splitnode
    # add node ID's 
    basin_conn_node_gdf_us = basin_conn_node_gdf.copy()
    basin_conn_node_gdf_us['geometry'] = basin_conn_node_gdf_us.geometry.apply(lambda x: LineString([x.coords[0], x.coords[1]]))
    basin_conn_node_gdf_us['from_node_id'] = basin_conn_node_gdf_us['basin_out'] + 1
    basin_conn_node_gdf_us['to_node_id'] = basin_conn_node_gdf_us['node_id']

    basin_conn_node_gdf_ds = basin_conn_node_gdf.copy()
    basin_conn_node_gdf_ds['geometry'] = basin_conn_node_gdf.geometry.apply(lambda x: LineString([x.coords[1], x.coords[2]]))
    basin_conn_node_gdf_ds['from_node_id'] = basin_conn_node_gdf_ds['node_id']
    basin_conn_node_gdf_ds['to_node_id'] = basin_conn_node_gdf_ds['basin_in'] + 1

    # (1) edges
    # merge to find splitnode id
    basin_conn_edge_gdf = basin_connections_gdf.merge(
        split_nodes_gdf[['splitnode_id','mesh1d_nEdges', 'node_id']], 
        left_on='mesh1d_nEdges', 
        right_on='mesh1d_nEdges'
    )

    # split connections in the connections upstream and downstream of splitnode
    # add node ID's 
    basin_conn_edge_gdf_us = basin_conn_edge_gdf.copy()
    basin_conn_edge_gdf_us['geometry'] = basin_conn_edge_gdf_us.geometry.apply(lambda x: LineString([x.coords[0], x.coords[1]]))
    basin_conn_edge_gdf_us['from_node_id'] = basin_conn_edge_gdf_us['basin_out'] + 1
    basin_conn_edge_gdf_us['to_node_id'] = basin_conn_edge_gdf_us['node_id']

    basin_conn_edge_gdf_ds = basin_conn_edge_gdf.copy()
    basin_conn_edge_gdf_ds['geometry'] = basin_conn_edge_gdf.geometry.apply(lambda x: LineString([x.coords[1], x.coords[2]]))
    basin_conn_edge_gdf_ds['from_node_id'] = basin_conn_edge_gdf_ds['node_id']
    basin_conn_edge_gdf_ds['to_node_id'] = basin_conn_edge_gdf_ds['basin_in'] + 1

    # boundary basin connections - add node ID's`
    if boundary_basin_connections is None:
        boundary_basin_connections_ds = pd.DataFrame()
        boundary_basin_connections_us = pd.DataFrame()
    else:
        boundary_basin_connections = boundary_basin_connections[['boundary_id', 'basin','geometry','boundary_location']].copy()

        boundary_basin_connections_us = boundary_basin_connections.loc[boundary_basin_connections['boundary_location'] == 'upstream'].copy()
        boundary_basin_connections_us['from_node_id'] = boundary_basin_connections_us['boundary_id']  + len(basins) +1
        boundary_basin_connections_us['to_node_id'] = boundary_basin_connections_us['basin'] + 1

        boundary_basin_connections_ds = boundary_basin_connections.loc[boundary_basin_connections['boundary_location'] == 'downstream'].copy()
        boundary_basin_connections_ds['from_node_id'] = boundary_basin_connections_ds['basin'] + 1
        boundary_basin_connections_ds['to_node_id'] = boundary_basin_connections_ds['boundary_id'] + len(basins) + 1

    # Setup the edges:
    ribasim_edges = pd.concat([
        basin_conn_node_gdf_ds, 
        basin_conn_node_gdf_us, 
        basin_conn_edge_gdf_ds, 
        basin_conn_edge_gdf_us, 
        boundary_basin_connections_us, 
        boundary_basin_connections_ds,
    ]) 
    ribasim_edges = ribasim_edges[['from_node_id','to_node_id','geometry']].reset_index(drop=True)
    ribasim_edges['from_node_id'].astype(int)

    if not ribasim_edges.empty:
        edge = ribasim.Edge(static=ribasim_edges)
    else:
        edge=None
    return edge


def generate_ribasim_basins(
    ribasim_node_gdf: gpd.GeoDataFrame = None, 
    dummyvalue: float = 5.5
):
    """generate ribasim basins using nodes and dummyvalue as basinproperties"""
    print(" - create Ribasim basin")
    profile_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='Basin'].index.values
        }
    )
    profile_data['storage'] = 0.0
    profile_data['area'] = 0.0
    profile_data['level'] = 0.0
    profile_data2 = profile_data.copy()
    profile_data['storage'] = 1000.0
    profile_data2['area'] = 1000.0
    profile_data2['level'] = 1.0
    profile_data = pd.concat([profile_data, profile_data2]).sort_values(by=['node_id', 'level']).reset_index(drop=True)

    static_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='Basin'].index.values
        }
    )
    static_data['drainage'] = dummyvalue
    static_data['potential_evaporation'] = dummyvalue
    static_data['infiltration'] = dummyvalue
    static_data['precipitation'] = dummyvalue
    static_data['urban_runoff'] = dummyvalue

    
    if not static_data.empty:
        basin = ribasim.Basin(profile=profile_data, static=static_data)
    else:
        basin=None
        print("   x no basins")
    return basin


def generate_ribasium_tabulatedratingcurves(
    ribasim_node_gdf: gpd.GeoDataFrame = None,
    dummyvalue: float = 5.5
):
    """generate ribasim tabulated rating using dummyvalues for level and discharge"""
    print(" - create Ribasim tabulated rating curve")
    static_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='TabulatedRatingCurve'].index
        }
    )
    static_data['level'] = dummyvalue
    static_data['discharge'] = dummyvalue

    if not static_data.empty:
        tabulated_rating_curve = ribasim.TabulatedRatingCurve(static=static_data)
    else:
        tabulated_rating_curve=None
        print("   x no tabulated rating curves")
    
    return tabulated_rating_curve


def generate_ribasim_level_boundaries(
    boundaries_gdf: gpd.GeoDataFrame = None,
    dummyvalue: float =5.5
):
    """generate ribasim level boundaries for all level boundary nodes using dummyvalue as level"""
    print(" - create Ribasim level boundaries")
    if boundaries_gdf is None:
        print("   x no level boundaries")
        return None
    
    static_boundary = pd.DataFrame(
        data={
            "node_id": boundaries_gdf.loc[boundaries_gdf['quantity']=='waterlevelbnd']['node_id']
        }
    )
    static_boundary['level'] = dummyvalue

    if static_boundary.empty:
        level_boundary=None
        print("   x no level boundaries")
    else:
        level_boundary = ribasim.LevelBoundary(static=static_boundary)
    return level_boundary


def generate_ribasim_flow_boundaries(
    boundaries_gdf: gpd.GeoDataFrame = None,
    dummyvalue: float = 5.5
):
    """generate ribasim flow boundaries for all flow boundary nodes using dummyvalue as flow_rate"""
    print(" - create Ribasim flow boundaries")
    if boundaries_gdf is None:
        print("   x no flow boundaries")
        return None

    static_boundary = pd.DataFrame(
        data={
            "node_id": boundaries_gdf.loc[boundaries_gdf['quantity']=='dischargebnd']['node_id']
        }
    )
    static_boundary['flow_rate'] = dummyvalue

    if static_boundary.empty:
        flow_boundary=None
        print("   x no flow boundaries")
    else:
        flow_boundary = ribasim.FlowBoundary(static=static_boundary)
    return flow_boundary


def generate_ribasim_pumps(ribasim_node_gdf: gpd.GeoDataFrame = None):
    """generate ribasim pumps for all pump nodes""" 
    print(" - create Ribasim pumps")
    static_pump = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='Pump'].index
        }
    )
    static_pump['flow_rate'] = 0.0

    if not static_pump.empty:
        pump = ribasim.Pump(static=static_pump)
    else:
        pump=None
        print("   x no pumps")
    return pump


def generate_ribasim_manningresistances(
    ribasim_node_gdf: gpd.GeoDataFrame = None,
    dummyvalue: float = 5.5
):
    """generate ribasim manning resistances"""
    print(" - create Ribasim manning resistances")
    static_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='ManningResistance'].index
        }
    )

    static_data['length'] = dummyvalue
    static_data['manning_n'] = dummyvalue
    static_data['profile_width'] = dummyvalue
    static_data['profile_slope'] = dummyvalue

    if not static_data.empty:
        manning_resistance = ribasim.ManningResistance(static= static_data)
    else:
        manning_resistance=None
        print("   x no manning resistances")
    return manning_resistance


def generate_ribasimmodel(
    basins = None, 
    split_nodes = None, 
    boundaries = None, 
    basin_connections = None, 
    boundary_basin_connections = None, 
    split_node_type_conversion = None, 
    split_node_id_conversion = None
):
    """generate ribasim model from ribasim nodes and edges and
    optional input; ribasim basins, level boundary, flow_boundary, pump, tabulated rating curve and manning resistance """
    print("Generate ribasim model:")
    node, boundaries_gdf, splitnodes_gdf, ribasim_node_gdf = \
        generate_ribasim_nodes(basins, split_nodes, boundaries, split_node_type_conversion, split_node_id_conversion)
    edge = generate_ribasim_edges(basins, splitnodes_gdf, basin_connections, boundary_basin_connections)
    basin = generate_ribasim_basins(ribasim_node_gdf, dummyvalue=5.5)
    level_boundary = generate_ribasim_level_boundaries(boundaries_gdf)
    flow_boundary = generate_ribasim_flow_boundaries(boundaries_gdf)
    pump = generate_ribasim_pumps(ribasim_node_gdf)
    tabulated_rating_curve = generate_ribasium_tabulatedratingcurves(ribasim_node_gdf, dummyvalue=5.5)
    manning_resistance = generate_ribasim_manningresistances(ribasim_node_gdf, dummyvalue=5.5)

    ribasim_model = ribasim.Model(
        modelname="ribasim_model",
        node=node,
        edge=edge,
        basin=basin,
        level_boundary=level_boundary,
        flow_boundary=flow_boundary,
        pump=pump,
        tabulated_rating_curve=tabulated_rating_curve,
        manning_resistance=manning_resistance,
        starttime="2020-01-01 00:00:00",
        endtime="2021-01-01 00:00:00",
    )
    return ribasim_model
