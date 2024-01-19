import pandas as pd
import geopandas as gpd
import numpy as np


def generate_basin_static_table(basin_h, basin_a, basins, decimals=3):
    if basin_h is None:
        return pd.DataFrame(
            data={
                "node_id": np.repeat(basins.basin_node_id.values, 2), 
                "level": [0.0, 1.0]*len(basins), 
                "area": [1000.0, 1000.0]*len(basins)
            }
        )
        
    basin_profile = pd.DataFrame(
        columns=['node_id', 'level', 'area', 'remarks']
    )
    for basin_node_id in basin_h.columns:
        basin_profile_col = basin_h[[basin_node_id]].reset_index(drop=True)
        basin_profile_col.columns = ['level']
        basin_profile_col["node_id"] = basin_node_id
        basin_profile_col['area'] = basin_a[[basin_node_id]].reset_index(drop=True)
        basin_profile = pd.concat([basin_profile, basin_profile_col])
    basin_profile["remarks"] = ""
    basin_profile = basin_profile[basin_profile['level'].notna()]
    basin_profile["area"] = basin_profile["area"].replace(0.0, 0.001)
    basin_profile = basin_profile.reset_index(drop=True)
    basin_profile = basin_profile[
        (basin_profile["level"].diff(1) > 0.0001) | 
        (basin_profile["level"].diff(1).isna()) | 
        (basin_profile["level"].diff(1) < 0.0)
    ].reset_index(drop=True)
    return basin_profile


def generate_basin_time_table_laterals(basins, basin_areas, laterals, laterals_data, saveat):
    laterals_basins = (laterals[["id", "geometry"]]
                       .sjoin(basin_areas[["basin_node_id", "geometry"]]).drop(columns=["index_right"])
                       [["id", "basin_node_id"]])
    if saveat is not None:
        laterals_data = laterals_data.resample(f"{saveat}S").interpolate()
    
    timeseries = pd.DataFrame()
    for basin_no in basins["basin_node_id"].to_numpy():
        if basin_no in basin_areas["basin_node_id"].to_numpy():
            laterals_basin = laterals_basins[laterals_basins["basin_node_id"]==basin_no]["id"].to_numpy()
            laterals_basin = [l for l in laterals_basin if l in laterals_data.columns]
            timeseries_basin = laterals_data[laterals_basin].sum(axis=1)
            timeseries_basin.name = "drainage"
            timeseries_basin = timeseries_basin.to_frame()
        else:
            timeseries_basin = pd.DataFrame(index=laterals_data.index, columns=["drainage"])
            timeseries_basin["drainage"] = 0.0
        timeseries_basin.index.name = "time"
        timeseries_basin = timeseries_basin.reset_index()

        timeseries_basin["potential_evaporation"] = 0.0
        timeseries_basin["precipitation"] = 0.0
        timeseries_basin["infiltration"] = 0.0
        timeseries_basin["urban_runoff"] = 0.0
        timeseries_basin["node_id"] = basin_no
        
        timeseries = pd.concat([
            timeseries,
            timeseries_basin
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time", "node_id", "precipitation", "potential_evaporation", "drainage", "infiltration", "urban_runoff"]]
    return timeseries

def generate_basin_time_table_areas_laterals_data(basins, areas, areas_laterals_data):
    timeseries = pd.DataFrame()
    for basin_no in basins["basin_node_id"].values:
        areas_basin = list(areas[areas['basin_node_id'] == basin_no]['area_code'].unique())
        timeseries_basin = areas_laterals_data[areas_basin].sum(axis=1).to_frame().rename(columns={0: 'Netto_flux'}).reset_index()
        
        timeseries_basin["drainage"] = timeseries_basin["Netto_flux"][timeseries_basin["Netto_flux"]>0]
        timeseries_basin["drainage"] = timeseries_basin["drainage"].fillna(0.0)

        timeseries_basin["infiltration"] = timeseries_basin["Netto_flux"][timeseries_basin["Netto_flux"]<0]
        timeseries_basin["infiltration"] = timeseries_basin["infiltration"].fillna(0.0)
        
        timeseries_basin["potential_evaporation"] = 0.0
        timeseries_basin["precipitation"] = 0.0
        timeseries_basin["urban_runoff"] = 0.0
        timeseries_basin["node_id"] = basin_no

        timeseries = pd.concat([
            timeseries,
            timeseries_basin
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time", "node_id", "precipitation", "potential_evaporation", "drainage", "infiltration", "urban_runoff"]]
    return timeseries


def generate_basin_time_table_drainage_per_ha(basins, basin_areas, drainage_per_ha):
    drainage_per_ha.name = "drainage"
    drainage_per_ha = drainage_per_ha.resample("H").interpolate()
    drainage_m3_s_ha = drainage_per_ha.to_frame() / 1000.0
    drainage_m3_s_ha.index.name = "time"

    basin_areas_ha = basin_areas.set_index('basin_node_id')

    timeseries = pd.DataFrame()
    for basin_no in basins["basin_node_id"].values:
        timeseries_basin = drainage_m3_s_ha.reset_index()
        timeseries_basin["potential_evaporation"] = 0.0
        timeseries_basin["precipitation"] = 0.0
        timeseries_basin["infiltration"] = 0.0
        timeseries_basin["urban_runoff"] = 0.0
        timeseries_basin["node_id"] = basin_no

        if basin_no in basin_areas_ha.index:
            area = basin_areas_ha.loc[basin_no, 'area_ha']
            timeseries_basin["drainage"] = timeseries_basin["drainage"] * area
        else:
            timeseries_basin["drainage"] = 0.0

        timeseries = pd.concat([
            timeseries,
            timeseries_basin
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time", "node_id", "precipitation", "potential_evaporation", "drainage", "infiltration", "urban_runoff"]]
    return timeseries


def generate_tabulate_rating_curve(basins_outflows, tabulated_rating_curves, basin_h, \
                                   edge_q_df, weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df):
    tabulated_rating_curves = tabulated_rating_curves.merge(
        basins_outflows.set_index("split_node")[["crestlevel", "crestwidth"]], 
        how="left", 
        left_on="split_node", 
        right_index=True
    )

    curves = pd.DataFrame()
    for i, trc in tabulated_rating_curves.iterrows():
        basin_node_id = trc["from_node_id"]
        water_levels_basin = basin_h[basin_node_id]

        split_type = trc["split_type"]
        split_node_name = trc["split_node_id"]
        if split_type == "weir":
            discharges = weir_q_df[split_node_name]
        if split_type == "universalWeir":
            discharges = uniweir_q_df[split_node_name]
        if split_type == "orifice":
            discharges = orifice_q_df[split_node_name]
        if split_type == "culvert":
            discharges = culvert_q_df[split_node_name]
        if split_type == "bridge":
            discharges = bridge_q_df[split_node_name]
        if split_type == "pump":
            discharges = pump_q_df[split_node_name]
        if split_type == "edge":
            discharges = edge_q_df[split_node_name]
        curve = pd.concat([water_levels_basin, discharges.replace(-999.0, np.nan)], axis=1)
        curve.columns = ["level", "discharge"]
        curve.iloc[0]["discharge"] = 0.0
        curve.iloc[1]["discharge"] = 0.0
        curve.iloc[2]["discharge"] = 0.0
        curve.iloc[3]["discharge"] = 0.0
        curve["node_id"] = trc["split_node_node_id"]
        curve = curve.interpolate().reset_index(drop=True)
        if curve.discharge.max() < 0.01:
            print(f" x basin_node_id {basin_node_id}: no discharge over split_node ({split_type}): {trc.split_node_id} in D-HYDRO simulation")

            def weir_formula(crestlevel, crestwidth, waterlevel):
                return 2.0/3.0 * max(0.0, waterlevel - crestlevel)**(3.0/2.0) * (2*9.81)**0.5 * crestwidth

            curve.loc[3:, "level"] = [i*0.05 + trc.crestlevel for i in range(0, len(curve)-3)]
            curve["discharge"] = curve.apply(lambda x: weir_formula(trc.crestlevel, trc.crestwidth, x["level"]), axis=1)

        curves = pd.concat([curves, curve])
    return curves.drop_duplicates().reset_index(drop=True)


def generate_manning_resistances(manningresistance):
    return pd.DataFrame(
        data={
            "node_id": manningresistance["split_node_node_id"],
            "length": [750.0]*len(manningresistance),
            "manning_n": [0.04]*len(manningresistance),
            "profile_width": [5.0]*len(manningresistance),
            "profile_slope": [3.0]*len(manningresistance),
        }
    )


def generate_ribasim_model_tables(dummy_model, basin_h, basin_a, basins, basin_areas, areas,
    laterals, laterals_data, boundaries, boundaries_data, 
    split_nodes, basins_outflows, set_name, basin_h_initial, 
    use_laterals_basis_network, use_laterals_areas, areas_laterals_data, saveat, drainage_per_ha,
    edge_q_df, weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df):

    # create tables for BASINS
    tables = dict()
    tables['basin_profile'] = generate_basin_static_table(basin_h, basin_a, basins, decimals=3)

    print('laterals')
    if use_laterals_basis_network and laterals is not None and laterals_data is not None:
        print(' - laterals based on lateral inflow according to dhydro network')
        tables['basin_time'] = generate_basin_time_table_laterals(
            basins, 
            basin_areas, 
            laterals,
            laterals_data,
            saveat
        )
    elif use_laterals_areas and areas_laterals_data is not None:
        print(' - laterals based on lateral inflow (timeseries) per area')
        tables['basin_time'] = generate_basin_time_table_areas_laterals_data(
            basins, 
            areas, 
            areas_laterals_data
        )
    else:
        print(' - laterals based on homogeneous lateral inflow timeseries')
        tables['basin_time'] = generate_basin_time_table_drainage_per_ha(
            basins, 
            basin_areas, 
            drainage_per_ha
        )
    if basin_h_initial is None:
        tables['basin_state'] = (basin_h.loc[(set_name, "targetlevel")]
                                 .rename("level")
                                 .reset_index()
                                 .rename(columns={"basin_node_id": "node_id"}))
    else:                            
        tables['basin_state'] = (basin_h_initial
                                 .rename("level")
                                 .reset_index()
                                 .rename(columns={"basin_node_id": "node_id"}))

    # create tables for BOUNDARIES
    level_boundaries = boundaries[boundaries['ribasim_type']=="LevelBoundary"]
    tables['level_boundary_static'] = pd.DataFrame(
        data={
            "node_id": level_boundaries["boundary_node_id"],
            "level": [7.15] * len(level_boundaries),
        }
    )
    flow_boundaries = boundaries[boundaries['ribasim_type']=="FlowBoundary"]
    tables['flow_boundary_static'] = pd.DataFrame(
        data={
            "node_id": flow_boundaries["boundary_node_id"],
            "flow_rate": [0.0] * len(flow_boundaries),
        }
    )
    
    # create tables for PUMPS
    pumps = split_nodes[split_nodes['ribasim_type'] == 'Pump']
    tables['pump_static'] = pd.DataFrame(
        data={
            "node_id": pumps["split_node_node_id"],
            "flow_rate": [0] * len(pumps),
        }
    )
    # create tables for OUTLETS
    outlets = split_nodes[split_nodes['ribasim_type'] == 'Outlet']
    tables['outlet_static'] = pd.DataFrame(
        data={
            "node_id": outlets["split_node_node_id"],
            "flow_rate": [0] * len(outlets),
        }
    )
    # create tables for TABULATED RATING CURVES
    tabulated_rating_curves = split_nodes[split_nodes['ribasim_type'] == 'TabulatedRatingCurve']
    tables['tabulated_rating_curve_static'] = generate_tabulate_rating_curve(
        basins_outflows, tabulated_rating_curves, 
        basin_h, edge_q_df, weir_q_df, uniweir_q_df, 
        orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df
    )
    
    # create tables for MANNINGRESISTANCE
    manningresistance = split_nodes[split_nodes['ribasim_type'] == 'ManningResistance']
    tables['manningresistance_static'] = generate_manning_resistances(manningresistance)

    return tables
