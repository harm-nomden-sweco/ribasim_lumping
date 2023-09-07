import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Dict, List
from .general_functions import create_objects_gdf
import hydrolib.core.dflowfm as hcdfm


def get_dhydro_network_objects(map_data, his_data, boundary_data, crs):
    """Extracts nodes, edges, confluences, bifurcations, weirs, pumps, laterals from his/map"""
    if map_data is None:
        raise ValueError("D-Hydro simulation map-data is not read")
    if his_data is None:
        raise ValueError("D-Hydro simulation his-data is not read")
    print("D-HYDRO-network analysed:")
    nodes, nodes_h = get_nodes_dhydro_network(map_data, crs)
    edges, edges_q = get_edges_dhydro_network(map_data, crs)
    stations = get_stations_dhydro_network(his_data, edges, crs)
    pumps = get_pumps_dhydro_network(his_data, edges, crs)
    weirs = get_weirs_dhydro_network(his_data, edges, crs)
    orifices = get_orifices_dhydro_network(his_data, edges, crs)
    bridges = get_bridges_dhydro_network(his_data, edges, crs)
    culverts = get_culverts_dhydro_network(his_data, edges, crs)
    uniweirs = get_uniweirs_dhydro_network(his_data, edges, crs)
    confluences = get_confluences_dhydro_network(nodes, edges)
    bifurcations = get_bifurcations_dhydro_network(nodes, edges)
    boundaries = get_boundaries(boundary_data, nodes)
    return nodes, nodes_h, edges, edges_q, stations, pumps, weirs, \
        orifices, bridges, culverts, uniweirs, confluences, bifurcations, boundaries


def get_nodes_dhydro_network(map_data, crs) -> gpd.GeoDataFrame:
    """calculate nodes dataframe"""
    print(" - nodes and waterlevels")
    nodes_gdf = (
        map_data["mesh1d_node_id"]
        .ugrid.to_geodataframe()
        .reset_index()
        .set_crs(crs)
    ).drop(columns=['mesh1d_node_x', 'mesh1d_node_y'])
    nodes_gdf["mesh1d_node_id"] = nodes_gdf["mesh1d_node_id"].astype(str).apply(lambda r: r[2:-1].strip())
    nodes_h_df = map_data.mesh1d_s1.to_dataframe()[['mesh1d_s1']]
    nodes_h_df = nodes_h_df.reorder_levels(['mesh1d_nNodes', 'set', 'condition'])
    return nodes_gdf, nodes_h_df


def get_edges_dhydro_network(map_data, crs) -> gpd.GeoDataFrame:
    """calculate edges dataframe"""
    print(" - edges and discharges")
    edges = (
        map_data["mesh1d_q1"][-1][-1]
        .ugrid.to_geodataframe()
        .reset_index()
        .set_crs(crs)
        .drop(columns=["condition", "mesh1d_q1", "set"])
    )
    edges_nodes = map_data["mesh1d_edge_nodes"]
    edges_nodes = np.column_stack(edges_nodes.data)
    edges_nodes = pd.DataFrame(
        {"start_node_no": edges_nodes[0] - 1, "end_node_no": edges_nodes[1] - 1}
    )
    edges_gdf = edges.merge(
        edges_nodes, how="inner", left_index=True, right_index=True
    )
    edges_q_df = map_data.mesh1d_q1.to_dataframe()[['mesh1d_q1']]
    edges_q_df = edges_q_df.reorder_levels(['mesh1d_nEdges', 'set', 'condition'])
    return edges_gdf, edges_q_df


def get_stations_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get stations from dhydro_model"""
    print(" - stations", end="", flush=True)
    if 'stations' not in his_data:
        return None
    stations_gdf = create_objects_gdf(
        data={"mesh1d_node_id": his_data.stations},
        xcoor=his_data.station_geom_node_coordx,
        ycoor=his_data.station_geom_node_coordy,
        edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
        crs=crs,
    )
    stations_gdf['object_type'] = 'station'
    return stations_gdf


def get_pumps_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get pumps from dhydro_model"""
    print(" / pumps", end="", flush=True)
    if 'pumps' not in his_data:
        return None
    pumps_gdf = None
    if 'pump_input_geom_node_coordx' in his_data.variables:
        pumps_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.pumps},
            xcoor=his_data.pump_input_geom_node_coordx,
            ycoor=his_data.pump_input_geom_node_coordy,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    elif 'pump_geom_node_coordx' in his_data.variables:
        pumps_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.pumps},
            xcoor=(his_data.pump_geom_node_coordx[0::2]+his_data.pump_geom_node_coordx[1::2])/2.0,
            ycoor=(his_data.pump_geom_node_coordy[0::2]+his_data.pump_geom_node_coordy[1::2])/2.0,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    if pumps_gdf is None:
        return None
    pumps_gdf['object_type'] = 'pump'
    return pumps_gdf


def get_weirs_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get weirs from dhydro_model"""
    print(" / weirs", end="", flush=True)
    if 'weirgens' not in his_data:
        return None
    weirs_gdf = None
    if 'weir_input_geom_node_coordx' in his_data.variables:
        weirs_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.weirgens},
            xcoor=his_data.weir_input_geom_node_coordx,
            ycoor=his_data.weir_input_geom_node_coordy,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    elif 'weirgen_geom_node_coordx' in his_data.variables:
        weirs_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.weirgens},
            xcoor=(his_data.weirgen_geom_node_coordx[0::2]+his_data.weirgen_geom_node_coordx[1::2])/2.0,
            ycoor=(his_data.weirgen_geom_node_coordy[0::2]+his_data.weirgen_geom_node_coordy[1::2])/2.0,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    if weirs_gdf is None:
        return None
    weirs_gdf['object_type'] = 'weir'
    return weirs_gdf


def get_orifices_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get orifices from dhydro_model"""
    print(" / orifices", end="", flush=True)
    if 'orifice' not in his_data:
        return None
    orifices_gdf = None
    if 'orifice_input_geom_node_coordx' in his_data.variables:
        orifices_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.orifice},
            xcoor=his_data.orifice_input_geom_node_coordx,
            ycoor=his_data.orifice_input_geom_node_coordx,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    elif 'orifice_geom_node_coordx' in his_data.variables:
        orifices_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.orifice},
            xcoor=(his_data.orifice_geom_node_coordx[0::2]+his_data.orifice_geom_node_coordx[1::2])/2.0,
            ycoor=(his_data.orifice_geom_node_coordy[0::2]+his_data.orifice_geom_node_coordy[1::2])/2.0,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    if orifices_gdf is None:
        return None
    orifices_gdf['object_type'] = 'orifice'
    return orifices_gdf


def get_bridges_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get bridges from dhydro_model"""
    print(" / bridges", end="", flush=True)
    if 'bridge' not in his_data:
        return None
    bridges_gdf = None
    if 'bridge_input_geom_node_coordx' in his_data.variables:
        bridges_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.bridge},
            xcoor=his_data.bridge_input_geom_node_coordx,
            ycoor=his_data.bridge_input_geom_node_coordy,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    elif 'bridge_geom_node_coordx' in his_data.variables:
        bridges_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.bridge},
            xcoor=(his_data.bridge_geom_node_coordx[0::2]+his_data.bridge_geom_node_coordx[1::2])/2.0,
            ycoor=(his_data.bridge_geom_node_coordy[0::2]+his_data.bridge_geom_node_coordy[1::2])/2.0,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    if bridges_gdf is None:
        return None
    bridges_gdf['object_type'] = 'bridge'
    return bridges_gdf


def get_culverts_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get culverts from dhydro_model"""
    print(" / culverts", end="", flush=True)
    if 'culvert' not in his_data:
        return None
    culverts_gdf = None
    if 'culvert_input_geom_node_coordx' in his_data.variables:
        culverts_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.culvert},
            xcoor=his_data.culvert_input_geom_node_coordx,
            ycoor=his_data.culvert_input_geom_node_coordy,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    elif 'culvert_geom_node_coordx' in his_data.variables:
        culverts_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.culvert},
            xcoor=(his_data.culvert_geom_node_coordx[0::2]+his_data.culvert_geom_node_coordx[1::2])/2.0,
            ycoor=(his_data.culvert_geom_node_coordy[0::2]+his_data.culvert_geom_node_coordy[1::2])/2.0,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    if culverts_gdf is None:
        return None
    culverts_gdf['object_type'] = 'culvert'
    return culverts_gdf


def get_uniweirs_dhydro_network(his_data, edges_gdf, crs) -> gpd.GeoDataFrame:
    """Get weirs from dhydro_model"""
    print(" / uniweirs", end="", flush=True)
    if 'universalWeirs' not in his_data:
        return None
    uniweirs_gdf = None
    if 'uniweir_input_geom_node_coordx' in his_data.variables:
        uniweirs_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.universalWeirs},
            xcoor=his_data.uniweir_input_geom_node_coordx,
            ycoor=his_data.uniweir_input_geom_node_coordy,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    elif 'uniweir_geom_node_coordx' in his_data.variables:
        uniweirs_gdf = create_objects_gdf(
            data={"mesh1d_node_id": his_data.universalWeirs},
            xcoor=(his_data.uniweir_geom_node_coordx[0::2]+his_data.uniweir_geom_node_coordx[1::2])/2.0,
            ycoor=(his_data.uniweir_geom_node_coordy[0::2]+his_data.uniweir_geom_node_coordy[1::2])/2.0,
            edges_gdf=edges_gdf[["mesh1d_nEdges", "geometry"]],
            crs=crs,
        )
    if uniweirs_gdf is None:
        return None
    uniweirs_gdf['object_type'] = 'uniweir'
    return uniweirs_gdf


def get_confluences_dhydro_network(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    """calculate confluence points based on finding multiple inflows"""
    print(" / confluences", end="", flush=True)
    c = edges_gdf.end_node_no.value_counts()
    confluences_gdf = nodes_gdf[
        nodes_gdf.index.isin(c.index[c.gt(1)])
    ].reset_index(drop=True)
    confluences_gdf.object_type = 'confluence'
    return confluences_gdf


def get_bifurcations_dhydro_network(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    """calculate split points based on finding multiple outflows"""
    print(" / bifurcations", end="", flush=True)
    d = edges_gdf.start_node_no.value_counts()
    bifurcations_gdf = nodes_gdf[
        nodes_gdf.index.isin(d.index[d.gt(1)])
    ].reset_index(drop=True)
    bifurcations_gdf.object_type = 'bifurcation'
    return bifurcations_gdf

def get_boundaries(
    boundary_data: gpd.GeoDataFrame = None,
    nodes: gpd.GeoDataFrame = None
) -> gpd.GeoDataFrame:
    """merge boundary data with nodes
    returns: geodataframe with boundaries"""
    print(" / boundaries")
    boundary_data = list(boundary_data.values())[0]

    # merge boundary data with nodes
    boundaries_gdf = nodes.merge(boundary_data, left_on = 'mesh1d_node_id', right_on = 'name')
    boundaries_gdf = boundaries_gdf.drop(columns=['offset','factor','vertpositionindex','name', 'comments','datablock'])
    boundaries_gdf.insert(0, 'boundary_id', range(len(boundaries_gdf)))
    if boundaries_gdf.empty:
        return None
    return boundaries_gdf

