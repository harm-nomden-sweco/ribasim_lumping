from pathlib import Path
from ..utils.general_functions import read_geom_file, generate_nodes_from_edges
from shapely.geometry import LineString, Point
from typing import Tuple
import geopandas as gpd


def add_hydamo_basis_network(
    hydamo_network_file: Path = 'network.gpkg',
    hydamo_network_gpkg_layer: str = None,
    crs: int = 28992,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load network data from HyDAMO files

    Args:
        hydamo_network_file (Path):         Path to file containing network geometries (hydroobjects)
        hydamo_network_gpkg_layer (str):    Layer name in geopackage. Needed when file is a geopackage
        crs (int):                          (optional) CRS EPSG code. Default 28992 (RD New)
    
    Returns:
        Tuple containing GeoDataFrames with branches, edges nodes
    """

    print('Reading network from HyDAMO files...')
    branches_gdf = read_geom_file(
        filepath=hydamo_network_file, 
        layer_name=hydamo_network_gpkg_layer, 
        crs=crs, 
        remove_z_dim=True
    )

    # Edges are the same as branches in HyDAMO
    edges_gdf = branches_gdf.copy()
    edges_gdf = edges_gdf.rename(columns={'code': 'branch_id'})[['branch_id', 'geometry']]
    edges_gdf, nodes_gdf = generate_nodes_from_edges(edges_gdf)

    return branches_gdf, edges_gdf, nodes_gdf
