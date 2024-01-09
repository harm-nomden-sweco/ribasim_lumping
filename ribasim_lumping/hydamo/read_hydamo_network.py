from pathlib import Path
from ..utils.general_functions import read_geom_file, generate_nodes_from_edges
from shapely.geometry import LineString, Point


def add_hydamo_basis_network(
    hydamo_network_file: Path = 'network.gpkg',
    hydamo_network_gpkg_layer: str = None,
    boundary_file: Path = None,
    boundary_gpkg_layer: str = None,
    crs: int = 28992,
):
    """
    Load HyDAMO network
    """

    print('Reading network from HyDAMO network file...')
    branches_gdf = read_geom_file(filepath=hydamo_network_file, layer_name=hydamo_network_gpkg_layer, crs=crs)
    branches_gdf = branches_gdf.explode()  # explode to transform multi-part geoms to single
    branches_gdf.geometry = [LineString([c[:2] for c in g.coords]) for g in branches_gdf.geometry.values]  # remove possible Z dimension

    # Edges are the same as branches in HyDAMO
    edges_gdf = branches_gdf.copy()
    edges_gdf = edges_gdf.rename(columns={'code': 'branch_id'})[['branch_id', 'geometry']]
    edges_gdf, nodes_gdf = generate_nodes_from_edges(edges_gdf)

    if boundary_file is not None:
        print('Reading boundaries from boundary file...')
        boundaries_gdf = read_geom_file(filepath=boundary_file, layer_name=boundary_gpkg_layer, crs=crs)
        boundaries_gdf = boundaries_gdf.explode()  # explode to transform multi-part geoms to single
        boundaries_gdf.geometry = [Point(g.coords[0][:2]) for g in boundaries_gdf.geometry.values]  # remove possible Z dimension
    else:
        boundaries_gdf = None

    return branches_gdf, edges_gdf, nodes_gdf, boundaries_gdf
