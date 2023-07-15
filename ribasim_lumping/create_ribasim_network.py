from pydantic import BaseModel
import geopandas as gpd
import xugrid as xu
import xarray as xr

# TODO: create builder for all ribasim objects

# class BasinCreator(BaseModel):
#     grid: xr.Dataset = None
#     nodes: gpd.GeoDataFrame = None
#     edges: gpd.GeoDataFrame = None
#     split_points: gpd.GeoDataFrame = None
