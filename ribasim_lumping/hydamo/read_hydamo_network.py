from pathlib import Path
from ..utils.general_functions import read_geom_file, generate_nodes_from_edges
from shapely.geometry import LineString, Point
from typing import Tuple
import geopandas as gpd


def add_hydamo_basis_network(
    hydamo_gpkg_file: Path = 'hydamo.gpkg',
    hydamo_hydroobject_gpkg_layer: str = 'hydroobject',
    hydamo_weir_gpkg_layer: str = 'stuw',
    hydamo_culvertsiphon_gpkg_layer: str = 'duikersifonhevel',
    hydamo_closing_gpkg_layer: str = 'afsluitmiddel',
    hydamo_pumpingstation_gpkg_layer: str = 'gemaal',
    hydamo_pump_gpkg_layer: str = 'pomp',
    hydamo_sluice_gpkg_layer: str = 'sluis',
    crs: int = 28992,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load HyDAMO data from geopackage

    Args:
        hydamo_gpkg_file (Path):                Path to geopackage containing hydamo objects (hydroobjects, weirs, pumps, etc.)
        hydamo_hydroobject_gpkg_layer (str):    Layer name in geopackage for hydroobject. Defaults to 'hydroobject'
        hydamo_weir_gpkg_layer (str):           Layer name in geopackage for weirs. Defaults to 'stuw'
        hydamo_culvertsiphon_gpkg_layer (str):  Layer name in geopackage for culverts and siphons. Defaults to 'duikersifonhevel'
        hydamo_closing_gpkg_layer (str):        Layer name in geopackage for closings. Defaults to 'afsluitmiddel'
        hydamo_pumpingstation_gpkg_layer (str): Layer name in geopackage for pumping stations. Defaults to 'gemaal'
        hydamo_pump_gpkg_layer (str):           Layer name in geopackage for pumps (which are part of a pumping station). Defaults to 'pomp'
        hydamo_sluice_gpkg_layer (str):         Layer name in geopackage for sluices. Defaults to 'sluis'
        crs (int):                              CRS EPSG code. Defaults to 28992 (RD New)
    
    Returns:
        Tuple containing GeoDataFrames with branches, weirs, culverts/siphons, pumps, edges, nodes
    """

    print('Reading HyDAMO objects from geopackage...')
    mapping = {
        'hydroobject': hydamo_hydroobject_gpkg_layer,
        'weir': hydamo_weir_gpkg_layer,
        'culvertsiphon': hydamo_culvertsiphon_gpkg_layer,
        'closing': hydamo_closing_gpkg_layer,
        'pumpingstation': hydamo_pumpingstation_gpkg_layer,
        'pump': hydamo_pump_gpkg_layer,
        'sluice': hydamo_sluice_gpkg_layer,
    }
    store = {}
    for k, layer in mapping.items():
        if layer is None:
            store[k] = gpd.GeoDataFrame()
        else:
            store[k] = read_geom_file(
                filepath=hydamo_gpkg_file, 
                layer_name=layer, 
                crs=crs, 
                remove_z_dim=True
            )
            print(f' - {k} ({len(store[k])}x features)')

    # to variables to be returned as results
    branches_gdf = store['hydroobject'].copy()
    weirs_gdf = store['weir'].copy()
    culverts_gdf = store['culvertsiphon'].copy()
    pumps_gdf = store['pumpingstation'].copy()
    
    # TODO: add logic to connect/process information of closing, pumpingstation and sluice tables
    # so all information is contained in weirs, culverts and pumps tables.

    # add edges and nodes based on branches
    edges_gdf = branches_gdf.copy()  # Edges are the same as branches in HyDAMO
    edges_gdf = edges_gdf.rename(columns={'code': 'branch_id'})[['branch_id', 'geometry']]
    edges_gdf, nodes_gdf = generate_nodes_from_edges(edges_gdf)

    return branches_gdf, weirs_gdf, culverts_gdf, pumps_gdf, edges_gdf, nodes_gdf
