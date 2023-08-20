import pandas as pd
import ribasim
from shapely.geometry import LineString

def generate_ribasim_nodes(basins=None, split_nodes=None, boundaries=None):
    print(" - create Ribasim nodes")
    basins_gdf =basins.copy()
    basins_gdf['node_id'] = basins_gdf['basin'] + 1
    basins_gdf['type'] = 'Basin'

    boundaries_gdf = boundaries.copy()
    boundaries_gdf['node_id'] = boundaries_gdf['boundary_id'] + len(basins) +1
    boundarynodetypes = {
        'dischargebnd': 'FlowBoundary', 
        'waterlevelbnd': 'LevelBoundary'
    }
    boundaries_gdf['type'] = boundaries_gdf['type'].replace(boundarynodetypes)

    splitnodes_gdf = split_nodes.copy()
    splitnodes_gdf.insert(0, 'splitnode_id', range(len(splitnodes_gdf)))
    splitnodes_gdf['node_id'] = splitnodes_gdf['splitnode_id'] + len(basins) + len(boundaries) +1
    splitnodes_gdf['type'] = 'TabulatedRatingCurve' 
    splitnodetypes = {
        'weir': 'TabulatedRatingCurve', 
        'uniweir': 'TabulatedRatingCurve' ,
        'pump': 'Pump', 
        'weir': 'TabulatedRatingCurve', 
        'culvert':'ManningResistance', 
        'manual': 'ManningResistance',
        'orifice' : 'TabulatedRatingCurve'
    }
    splitnodes_gdf['type'] = splitnodes_gdf['type'].replace(splitnodetypes)

    # concat nodes
    ribasim_node_gdf = pd.concat([basins_gdf, boundaries_gdf,splitnodes_gdf]).set_crs(split_nodes.crs)
    ribasim_node_gdf = ribasim_node_gdf.set_index('node_id')
    ribasim_node_gdf = ribasim_node_gdf[['geometry', 'type']]
    node = ribasim.Node(static=ribasim_node_gdf)
    return boundaries_gdf, splitnodes_gdf, ribasim_node_gdf, node


def generate_ribasim_edges(basins=None, split_nodes_gdf=None, basin_connections = None, boundary_basin_connections = None):
    print(" - create Ribasim edges")
    basin_connections_gdf = basin_connections[['mesh1d_node_id', 'basin_in','basin_out','geometry']]

    # merge to find splitnode id
    basin_connections_gdf = basin_connections_gdf.merge(split_nodes_gdf[['splitnode_id','mesh1d_node_id', 'node_id']], left_on='mesh1d_node_id', right_on='mesh1d_node_id')

    # split connections in the connections upstream and downstream of splitnode
    # add node ID's 
    basin_connections_gdf_us = basin_connections_gdf.copy()
    basin_connections_gdf_us['geometry'] = basin_connections_gdf_us.geometry.apply(lambda x: LineString([x.coords[0], x.coords[1]]))
    basin_connections_gdf_us['from_node_id'] = basin_connections_gdf_us['basin_out'] + 1
    basin_connections_gdf_us['to_node_id'] = basin_connections_gdf_us['node_id']

    basin_connections_gdf_ds = basin_connections_gdf.copy()
    basin_connections_gdf_ds['geometry'] = basin_connections_gdf.geometry.apply(lambda x: LineString([x.coords[1], x.coords[2]]))
    basin_connections_gdf_ds['from_node_id'] = basin_connections_gdf_ds['node_id']
    basin_connections_gdf_ds['to_node_id'] = basin_connections_gdf_ds['basin_in'] + 1

    # boundary basin connections - add node ID's
    boundary_basin_connections = boundary_basin_connections[['boundary_id', 'basin','geometry','boundary_location']].copy()

    boundary_basin_connections_us = boundary_basin_connections.loc[boundary_basin_connections['boundary_location'] == 'upstream'].copy()
    boundary_basin_connections_us['from_node_id'] = boundary_basin_connections_us['boundary_id']  + len(basins) +1
    boundary_basin_connections_us['to_node_id'] = boundary_basin_connections_us['basin'] + 1

    boundary_basin_connections_ds = boundary_basin_connections.loc[boundary_basin_connections['boundary_location'] == 'downstream'].copy()
    boundary_basin_connections_ds['from_node_id'] = boundary_basin_connections_ds['basin'] + 1
    boundary_basin_connections_ds['to_node_id'] = boundary_basin_connections_ds['boundary_id'] + len(basins) + 1

    # Setup the edges:
    ribasim_edges = pd.concat([basin_connections_gdf_ds, basin_connections_gdf_us,boundary_basin_connections_us, boundary_basin_connections_ds]) 
    ribasim_edges = ribasim_edges[['from_node_id','to_node_id','geometry']].reset_index()
    ribasim_edges['from_node_id'].astype(int)

    edge = ribasim.Edge(static=ribasim_edges)
    return edge


def generate_ribasim_basins(ribasim_node_gdf, dummyvalue=5.5):
    print(" - create Ribasim basin")
    profile_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='Basin'].index.values.tolist()
        }
    )
    profile_data['storage'] = dummyvalue
    profile_data['area'] = dummyvalue
    profile_data['level'] = dummyvalue

    static_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='Basin'].index.values.tolist()
        }
    )
    static_data['drainage'] = dummyvalue
    static_data['potential_evaporation'] = dummyvalue
    static_data['infiltration'] = dummyvalue
    static_data['precipitation'] = dummyvalue
    static_data['urban_runoff'] = dummyvalue

    basin = ribasim.Basin(profile=profile_data, static=static_data)
    return basin


def generate_ribasium_tabulatedratingcurves(ribasim_node_gdf=None, dummyvalue=5.5):
    print(" - create Ribasim tabulated rating curve")
    static_data = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='TabulatedRatingCurve'].index
        }
    )
    static_data['level'] = dummyvalue
    static_data['discharge'] = dummyvalue

    tabulated_rating_curve = ribasim.TabulatedRatingCurve(static=static_data)
    return tabulated_rating_curve


def generate_ribasim_level_boundaries(boundaries_gdf=None, dummyvalue=5.5):
    print(" - create Ribasim level boundaries")
    static_boundary = pd.DataFrame(
        data={
            "node_id": boundaries_gdf.loc[boundaries_gdf['quantity']=='waterlevelbnd']['node_id']
        }
    )
    static_boundary['level'] = dummyvalue

    level_boundary = ribasim.LevelBoundary(static=static_boundary)
    return level_boundary


def generate_ribasim_flow_boundaries(boundaries_gdf=None, dummyvalue=5.5):
    print(" - create Ribasim flow boundaries")
    static_boundary = pd.DataFrame(
        data={
            "node_id": boundaries_gdf.loc[boundaries_gdf['quantity']=='dischargebnd']['node_id']
        }
    )
    static_boundary['flow_rate'] = dummyvalue

    flow_boundary = ribasim.FlowBoundary(static=static_boundary)
    return flow_boundary


def generate_ribasim_pumps(ribasim_node_gdf=None):
    print(" - create Ribasim pumps")
    static_pump = pd.DataFrame(
        data={
            "node_id": ribasim_node_gdf.loc[ribasim_node_gdf['type']=='Pump'].index
        }
    )
    static_pump['flow_rate'] = 0.0
    pump = ribasim.Pump(static=static_pump)
    return pump


def generate_ribasim_manningresistances(ribasim_node_gdf, dummyvalue=5.5):
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

    manning_resistance = ribasim.ManningResistance(static= static_data)
    return manning_resistance


def generate_ribasim_model(basins=None, split_nodes=None, boundaries=None, basin_connections=None, boundary_basin_connections=None):
    print("Generate ribasim model:")
    boundaries_gdf, splitnodes_gdf, ribasim_node_gdf, node = generate_ribasim_nodes(basins, split_nodes, boundaries)
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
