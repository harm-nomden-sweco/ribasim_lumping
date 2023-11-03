from pathlib import Path
from typing import List, Union

import fiona
import geopandas as gpd
import pandas as pd
import xarray as xr
import xugrid as xu

from .preprocess_hydamo_dataset import (
    add_basin_code_from_network_to_nodes_and_edges,
    connect_endpoints_by_buffer, create_graph_based_on_nodes_edges,
    create_nodes_and_edges_from_hydroobjects, get_outlet_nodes,
    replace_nodes_perpendicular_on_edges)


def read_hydamo_gpkg(hydamo_file, layername, object_type = None):
    if object_type is None:
        object_type = layername
    hydamo_layers = fiona.listlayers(hydamo_file)
    if layername in hydamo_layers:
        data = gpd.read_file(hydamo_file, layer=layername)
        data['object_type'] = object_type
        print(f'{object_type} ({len(data)})', end=' ')
    else:
        data = None
        print(f'{object_type} (None)', end=' ')
    
    return data

def get_edges_nodes_from_hydroobject(hydroobject_gdf):
    nodes, edges = create_nodes_and_edges_from_hydroobjects(hydroobject_gdf) 

    return nodes, edges

def create_network_from_edges_nodes(nodes, edges):
    network_graph = create_graph_based_on_nodes_edges(nodes, edges)
    return network_graph

def add_hydamo_basis_network(
    hydamo_basis_dir,
    crs: int = 28992,
):
    hydamo_file = Path(hydamo_basis_dir,"hydamo.gpkg")

    print(f'read data {hydamo_file}')

    stuw_gdf = read_hydamo_gpkg(hydamo_file, layername= 'stuw')  
    duikersifonhevel_gdf = read_hydamo_gpkg(hydamo_file, layername= 'duikersifonhevel')
    gemaal_gdf = read_hydamo_gpkg(hydamo_file, layername= 'gemaal')
    peilgebiedvigerend_gdf = read_hydamo_gpkg(hydamo_file, layername= 'peilgebiedvigerend')
    peilgebiedpraktijk_gdf = read_hydamo_gpkg(hydamo_file, layername= 'peilgebiedpraktijk')
    afvoergebiedaanvoergebied_gdf = read_hydamo_gpkg(hydamo_file, layername= 'afvoergebiedaanvoergebied')
    afsluiter_gdf = read_hydamo_gpkg(hydamo_file, layername= 'afsluiter')
    sluis_gdf = read_hydamo_gpkg(hydamo_file, layername= 'sluis')
    hydroobject_gdf = read_hydamo_gpkg(hydamo_file, layername= 'hydroobject')

    pumps_gdf = read_hydamo_gpkg(hydamo_file, layername= 'gemaal', object_type='pump')
    if pumps_gdf is not None:
        pumps_gdf['structure_id']=pumps_gdf['code']
        pumps_gdf=pumps_gdf[['structure_id', 'geometry', 'object_type']]
    weirs_gdf = read_hydamo_gpkg(hydamo_file, layername= 'stuw', object_type='weir')
    if weirs_gdf is not None:
        weirs_gdf['structure_id']=weirs_gdf['code']
        weirs_gdf=weirs_gdf[['structure_id', 'geometry', 'object_type']]
    culverts_gdf = read_hydamo_gpkg(hydamo_file, layername= 'duikersifonhevel', object_type='culvert')
    if culverts_gdf is not None:
        culverts_gdf['structure_id']=culverts_gdf['code']
        culverts_gdf=culverts_gdf[['structure_id', 'geometry', 'object_type']]
    sluices_gdf = read_hydamo_gpkg(hydamo_file, layername= 'sluis', object_type='sluice')
    if sluices_gdf is not None:
        sluices_gdf['structure_id']=sluices_gdf['code']
        sluices_gdf=sluices_gdf[['structure_id', 'geometry', 'object_type']]

    nodes_gdf, edges_gdf = get_edges_nodes_from_hydroobject(hydroobject_gdf)
    network_graph = create_network_from_edges_nodes(nodes=nodes_gdf, edges=edges_gdf)
    


    return stuw_gdf, duikersifonhevel_gdf, gemaal_gdf, peilgebiedvigerend_gdf, peilgebiedpraktijk_gdf,\
        afvoergebiedaanvoergebied_gdf, afsluiter_gdf, sluis_gdf, hydroobject_gdf,\
        weirs_gdf, pumps_gdf, culverts_gdf, sluices_gdf, \
        nodes_gdf, edges_gdf, network_graph

