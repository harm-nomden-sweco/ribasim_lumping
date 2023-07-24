import geopandas as gpd
from shapely.ops import nearest_points


def find_nearest_nodes(
    search_locations: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, id_column: str
) -> gpd.GeoDataFrame:
    nearest_node_ids = []
    for index, row in search_locations.iterrows():
        point = row.geometry
        multipoint = nodes.drop(index, axis=0).geometry.unary_union
        _, nearest_geom = nearest_points(point, multipoint)
        nearest_node = nodes.loc[nodes["geometry"] == nearest_geom]
        nearest_node_ids.append(nearest_node[id_column].iloc[0])
    return nearest_node_ids
