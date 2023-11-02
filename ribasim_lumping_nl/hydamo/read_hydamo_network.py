from pathlib import Path
from typing import List, Union

import geopandas as gpd
import pandas as pd
import xarray as xr
import xugrid as xu


def add_hydamo_basis_network(
    hydamo_basis_dir,
    crs: int = 28992,
):

    hydamo_file = Path(hydamo_basis_dir,"hydamo.gpkg")

    weirs_gdf = gpd.read_file(hydamo_file, layer='stuw')
    weirs_gdf['object_type'] = 'stuw'
    culverts_gdf = gpd.read_file(hydamo_file, layer='duikersifonhevel')
    culverts_gdf['object_type'] = 'duikersifonhevel'

    hydroobject_gdf = gpd.read_file(hydamo_file, layer='hydroobject')
    hydroobject_gdf['object_type'] = 'hydroobject'

    return weirs_gdf, culverts_gdf, hydroobject_gdf



    # return network_data, branches_gdf, network_nodes_gdf, edges_gdf, \
    #     nodes_gdf, boundaries_gdf, laterals_gdf, weirs_gdf, \
    #     uniweirs_gdf, pumps_gdf, orifices_gdf, bridges_gdf, \
    #     culverts_gdf