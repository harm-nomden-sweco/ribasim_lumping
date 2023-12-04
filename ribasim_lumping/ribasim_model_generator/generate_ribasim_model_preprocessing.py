import scipy
import pandas as pd
import numpy as np
import geopandas as gpd
import math


def extract_bed_level_surface_storage(volume_data, nodes):
    increment = volume_data.increment.data[0]

    zlevels = volume_data.bedlevel.to_dataframe().T
    zlevels.columns.name = "node_no"
    zlevels.index = [0]
    bedlevel = zlevels.iloc[[0], nodes.node_no]
    bedlevel_T = bedlevel.T
    bedlevel_T.columns = ["bedlevel"]

    if "bedlevel" not in nodes.columns:
        nodes = nodes.merge(bedlevel_T, left_on="node_no", right_index=True)
    basins_bedlevels = nodes[["basin", "bedlevel"]].groupby(by="basin").min()
    bedlevel = nodes[["basin"]].merge(basins_bedlevels, left_on="basin", right_index=True)
    bedlevel = bedlevel[["bedlevel"]].T

    surface_df = volume_data.surface.to_dataframe().unstack()
    surface_df = surface_df.replace(0.0, np.nan).ffill(axis=1)
    surface_df.index.name = "node_no"
    surface_df = surface_df["surface"].T
    surface_df = pd.concat([bedlevel - 0.01, surface_df]).reset_index(drop=True)
    surface_df.iloc[0] = 0

    for i in range(1, len(surface_df)):
        zlevels = pd.concat([zlevels, bedlevel_T.T + increment * i])
    zlevels = zlevels.reset_index(drop=True)
    z_range = np.arange(np.floor(zlevels.min().min()), np.ceil(zlevels.max().max())+0.01, increment)

    node_surface_df = pd.DataFrame(index=z_range, columns=surface_df.columns)
    node_surface_df.index.name = "zlevel"
    for col in node_surface_df.columns:
        df_data_col = pd.DataFrame(index=zlevels[col], data=surface_df[col].values, columns=[col])[col]
        node_surface_df[col] = np.interp(z_range, zlevels[col].values, df_data_col.values)

    node_storage_df = ((node_surface_df + node_surface_df.shift(1))/2.0 * increment).cumsum()
    # node_storage_df = (node_surface_df * increment).cumsum()
    node_bedlevel = bedlevel
    node_bedlevel.index = ["bedlevel"]
    node_bedlevel.index.name = "condition"

    orig_bedlevel = bedlevel_T.copy()
    orig_bedlevel.columns = ["bedlevel"]
    return node_surface_df, node_storage_df, node_bedlevel, orig_bedlevel


def get_waterlevels_table_from_simulations(map_data):
    node_h_df1 = map_data.mesh1d_s1.to_dataframe().unstack().mesh1d_s1
    old_index = node_h_df1.index.copy()
    node_h_df = pd.concat([
        node_h_df1[col].sort_values().reset_index(drop=True)
        for col in node_h_df1.columns
    ], axis=1)
    node_h_df.index = old_index
    return node_h_df


def get_discharges_table_from_simulations(map_data):
    return map_data.mesh1d_q1.to_dataframe().unstack().mesh1d_q1


def get_discharges_table_structures_from_simulations(his_data):
    if "weirgen_discharge" in his_data.keys():
        weir_q_df = his_data.weirgen_discharge.to_dataframe().unstack().weirgen_discharge
    else:
        weir_q_df = None
    if "uniweir_discharge" in his_data.keys():
        uniweir_q_df = his_data.uniweir_discharge.to_dataframe().unstack().uniweir_discharge
    else:
        uniweir_q_df = None
    if "orifice_discharge" in his_data.keys():
        orifice_q_df = his_data.orifice_discharge.to_dataframe().unstack().orifice_discharge
    else:
        orifice_q_df = None
    if "culvert_discharge" in his_data.keys():
        culvert_q_df = his_data.culvert_discharge.to_dataframe().unstack().culvert_discharge
    else:
        culvert_q_df = None
    if "bridge_discharge" in his_data.keys():
        bridge_q_df = his_data.bridge_discharge.to_dataframe().unstack().bridge_discharge
    else:
        bridge_q_df = None
    if "pump_structure_discharge" in his_data.keys():
        pump_q_df = his_data.pump_structure_discharge.to_dataframe().unstack().pump_structure_discharge
    else:
        pump_q_df = None
    return weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df


def get_basins_outflows_including_settings(split_nodes, basin_connections, boundary_connections, weirs, pumps):
    gdfs_list = [weirs, pumps]
    gdfs_columns_list = [
        ["structure_id", "crestlevel", "crestwidth"],
        ["structure_id", "orientation", "controlside", "numstages", "capacity", "startlevelsuctionside", "stoplevelsuctionside"]
    ]

    basins_split_nodes = split_nodes[["split_node", "split_node_id", "split_node_node_id", "ribasim_type"]]
    basins_outflows1 = basin_connections[
        basin_connections["connection"]=="basin_to_split_node"
    ]

    basins_outflows1 = (
        basins_outflows1[["basin", "split_node"]]
        .merge(basins_split_nodes, how="left", on="split_node")
    )
    for gdf, gdf_columns in zip(gdfs_list, gdfs_columns_list):
        if gdf is None:
            basins_outflows1[gdf_columns] = np.nan
        else:
            basins_outflows1 = (
                basins_outflows1
                .merge(gdf[gdf_columns], 
                    how="left",
                    left_on="split_node_id", 
                    right_on="structure_id")
            )
    basins_outflows1 = (
        basins_outflows1
        .sort_values(by="basin")
        .reset_index(drop=True)
        .drop(columns=["structure_id"])
    )
    basins_outflows2 = boundary_connections[boundary_connections.connection=="basin_to_split_node"]
    basins_outflows2 = basins_outflows2[["basin", "split_node", "split_node_id"]]
    basins_outflows2["ribasim_type"] = "ManningResistance"
    basins_outflows = pd.concat([basins_outflows1, basins_outflows2])
    basins_outflows["targetlevel"] = basins_outflows["crestlevel"].fillna(basins_outflows["stoplevelsuctionside"])
    basins_outflows["split_node_node_id"] = basins_outflows["split_node_node_id"].fillna(-1).astype(int)
    return basins_outflows.reset_index(drop=True)


def get_target_levels_nodes_using_weirs_pumps(nodes, basins_outflows, name_column="targetlevel"):
    basins_outflows_grouped = basins_outflows.groupby(by="basin")
    basins_outflows_crest_levels = pd.DataFrame()
    for basin in basins_outflows_grouped.groups:
        basin_outflows = basins_outflows_grouped.get_group(basin)
        if len(basin_outflows) > 1:
            basin_outflows = basin_outflows.groupby("basin").min()
        basins_outflows_crest_levels = pd.concat([basins_outflows_crest_levels, basin_outflows])
    node_crestlevel = nodes[["node_no", "basin"]].merge(
        basins_outflows_crest_levels[["basin", name_column]], 
        on="basin"
    ).set_index("node_no")[[name_column]]
    node_crestlevel.index.name = "node_no"
    node_crestlevel.columns.name = "condition"
    return node_crestlevel.T


def generate_node_waterlevels_table(node_h_df, node_bedlevel, node_targetlevel, interpolation_lines):
    if node_targetlevel[node_targetlevel.columns].isnull().all(axis=1).targetlevel:
        node_targetlevel.rename(index={"targetlevel": -interpolation_lines-1}, inplace=True)

    node_nan = node_targetlevel.copy()
    node_nan.loc[:, :] = np.nan
    # create x additional interpolation lines between targetlevel and bedlevel
    node_nan_bedlevel = pd.concat([node_nan]*interpolation_lines)
    node_nan_bedlevel.index = range(-interpolation_lines*2 - 2, -interpolation_lines - 2)
    # create x additional interpolation lines between lowest backwatercurve and targetlevel
    node_nan_targetlevel = pd.concat([node_nan]*interpolation_lines)
    node_nan_targetlevel.index = range(-interpolation_lines - 1, -1)

    node_basis = pd.concat([
        node_bedlevel, node_nan_bedlevel, 
        node_targetlevel, node_nan_targetlevel
    ])
    node_basis.index.name = "condition"

    node_h = pd.DataFrame()
    for set_name in node_h_df.index.get_level_values(0).unique():
        node_h_set = pd.concat([
            pd.concat([
                node_basis, node_h_df.loc["winter"]
            ])
        ], keys=[set_name], names=["set"]).interpolate(axis=0)
        node_h = pd.concat([node_h, node_h_set])
    
    # check for increasing water level, if not then equal to previous
    # for i in range(len(node_h)-1):
    #     node_h[node_h.diff(1) <= 0.0001] = node_h.shift(1) + 0.0001
    node_h.columns.name = "node_no"
    return node_h


def generate_surface_storage_for_nodes(node_h, node_surface_df, node_storage_df, orig_bedlevel, decimals=3):
    
    def translate_waterlevels_to_surface_storage(nodes_h, curve, orig_bedlevel):
        nodes_x = nodes_h.copy()
        nodes_h_new = nodes_h.copy()
        for col in nodes_x.columns:
            level_correction_bedlevel = orig_bedlevel.bedlevel.loc[col] - 0.01
            curve.loc[curve.index <= level_correction_bedlevel, col] = 0.0
            nodes_h_new.loc[nodes_x[col] < level_correction_bedlevel, col] = math.floor(level_correction_bedlevel*10)/10.0
            interp_func = scipy.interpolate.interp1d(
                curve.index, 
                curve[col].to_numpy(), 
                kind="linear", 
                fill_value="extrapolate"
            )
            nodes_x[col] = np.round(interp_func(nodes_h_new[col].to_numpy()), decimals=decimals)
        return nodes_x, nodes_h_new

    node_h_new = pd.DataFrame()
    node_a = pd.DataFrame()
    node_v = pd.DataFrame()
    node_a.columns.name = "node_no"
    node_v.columns.name = "node_no"

    for set_name in node_h.index.get_level_values(0).unique():
        a_set, h_set = translate_waterlevels_to_surface_storage(node_h.loc[set_name], node_surface_df, orig_bedlevel)
        node_a_set = pd.concat([a_set], keys=[set_name], names=["set"])
        node_a = pd.concat([node_a, node_a_set])
        v_set, h_set = translate_waterlevels_to_surface_storage(node_h.loc[set_name], node_storage_df, orig_bedlevel)
        node_v_set = pd.concat([v_set], keys=[set_name], names=["set"])
        node_v = pd.concat([node_v, node_v_set])

        node_h_set = pd.concat([h_set], keys=[set_name], names=["set"])
        node_h_new = pd.concat([node_h_new, node_h_set])
    
    return node_h, node_a, node_v


def generate_waterlevels_for_basins(basins, node_h):
    basin_h = basins[["basin_node_id", "node_no"]].set_index("basin_node_id")
    basins_ids = basin_h.index
    basin_h = node_h.T.loc[basin_h["node_no"]]
    basin_h.index = basins_ids
    return basin_h.T


def generate_surface_storage_for_basins(node_a, node_v, nodes):
    nodes = nodes[["node_no", "basin_node_id"]]

    basin_a = pd.DataFrame()
    basin_v = pd.DataFrame()

    for set_name in node_a.index.get_level_values(0).unique():
        basin_a_set = node_a.loc[set_name].T.merge(nodes, how="inner", left_index=True, right_on="node_no").drop(columns=["node_no"])
        basin_a_set = basin_a_set.groupby(by="basin_node_id").sum().T
        basin_a_set = pd.concat([basin_a_set], keys=[set_name], names=["set"])
        basin_a = pd.concat([basin_a, basin_a_set])

        basin_v_set = node_v.loc[set_name].T.merge(nodes, how="inner", left_index=True, right_on="node_no").drop(columns=["node_no"])
        basin_v_set = basin_v_set.groupby(by="basin_node_id").sum().T
        basin_v_set = pd.concat([basin_v_set], keys=[set_name], names=["set"])
        basin_v = pd.concat([basin_v, basin_v_set])
    
    basin_a.index.names = ["set", "condition"]
    basin_v.index.names = ["set", "condition"]
    return basin_a, basin_v


def preprocessing_ribasim_model_tables(
    map_data, his_data, volume_data, nodes, weirs, pumps, basins, split_nodes, 
    basin_connections, boundary_connections, interpolation_lines
):
    # prepare all data
    basins_outflows = get_basins_outflows_including_settings(
        split_nodes=split_nodes, 
        basin_connections=basin_connections,
        boundary_connections=boundary_connections,
        weirs=weirs,
        pumps=pumps
    )
    node_targetlevel = get_target_levels_nodes_using_weirs_pumps(nodes, basins_outflows)
    node_surface_df, node_storage_df, node_bedlevel, orig_bedlevel = extract_bed_level_surface_storage(volume_data, nodes)
    node_h_df = get_waterlevels_table_from_simulations(map_data)
    node_h_basin = generate_node_waterlevels_table(node_h_df, node_bedlevel, node_targetlevel, interpolation_lines)
    node_h_node, node_a, node_v = generate_surface_storage_for_nodes(node_h_basin, node_surface_df, node_storage_df, orig_bedlevel)
    basin_h = generate_waterlevels_for_basins(basins, node_h_basin)
    basin_a, basin_v = generate_surface_storage_for_basins(node_a, node_v, nodes)
    edge_q_df = get_discharges_table_from_simulations(map_data)
    weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df = get_discharges_table_structures_from_simulations(his_data)

    return basins_outflows, node_h_basin, node_h_node, node_a, node_v, basin_h, basin_a, basin_v, \
        node_bedlevel, node_targetlevel, orig_bedlevel, edge_q_df, \
            weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df

