# -*- coding: utf-8 -*-
"""
Define functions related to network simulation with pandapipes.
"""
import logging
import matplotlib
import matplotlib.pyplot as plt
import momepy
import networkx as nx
import os
import pandas as pd

from .elevation import download_elevation_data
from .dhnx_addons import (plot_geometries,
                          custom_plot_save, save_gis_generic)

logger = logging.getLogger(__name__)  # Create a logger for this module

try:
    import pandapipes as pp
except ImportError as e:
    logger.exception(e)
    logger.warning("Optional dependency 'pandapipes' can be installed with "
                   "'pip install pandapipes'")


def pandapipes_run(network, gdf_pipes, df_DN=None, show_plot=False,
                   save_path='result_pandapipes',
                   save_gis_ext='.geojson',
                   save_plot=False,
                   P_th_kW=None,
                   f_length_loss=2,
                   pressure_net=12,  # [bar] (Pressure at the heat supply)
                   pressure_pn=20,  # [bar] The nominal pressure (used as initial value)
                   pressure_loss_coefficient_consumer=0,
                   elevation_col=None,
                   download_missing_elevation=False,
                   direction='forward',
                   pipe_section_length=50,
                   apply_flow_simultaneity=False,
                   **kwargs,
                   ):
    r"""Run pandapipes simulation with result network from dhnx.

    While DHNx uses a thermal transmittance (U-value) in unit W/(m*K) for
    heat loss calcultion, pandapipes requires a heat transfer coefficient
    input 'u_w_per_m2k' [W/(m^2*K)].

    U-value as given by pipe manufacturers typically describes the W of
    thermal loss per m pipe length and K temperature difference between mean
    operational temperature and external (ground) temperature:

    :math:`\frac{P_{loss}}{l} = (T_{op} - T_{ext}) \cdot U`

    As per documentation of pandapipes the losses are calculated as:

    :math:`P_{loss} = \alpha\cdot l \cdot\pi\cdot d \cdot (T_{op} - T_{ext})`

    It is also mentioned that :math:`d` describes the inner diameter.
    https://pandapipes.readthedocs.io/en/latest/components/pipe/pipe_component.html

    Therefore the area referenced by :math:`\alpha` is determined by the
    diameter :math:`d` and :math:`\alpha` can be calculated by:

    :math:`\alpha = \frac{U}{d\cdot \pi}`

    f_length_loss: It is assumed that the U-value in df_DN describes the total
    losses of forward and return flow, because this is required for DHNx
    to include the total losses in its calculation. In contrast, for the
    pandapipes calculation, we only want to evaluate pressure losses and
    temperature drop along the forward flow. Therefore the U-value in
    df_DN is divided by f_length_loss, i.e. halfed by default.

    Parameters
    ----------
    save_gis_ext : str, optional
        File type extension for saved GIS objects. Common options are
        '.gpkg', '.geojson' and '.shp'. The default is '.geojson'.

    pipe_section_length : int, optional
        Create 1 internal pipe section per e.g. 50m of pipe length, to improve
        accuracy of temperature profile (at the cost of computation time)


    Supports a closed loop network
    https://github.com/e2nIEE/pandapipes/pull/530
    https://hub.2i2c.mybinder.org/user/e2niee-pandapipes-dfbkk3gx/notebooks/tutorials/district_heating/circular_flow_in_a_district_heating_grid.ipynb

    TODO
    See if BadPointPressureLiftController gets merged
    https://github.com/e2nIEE/pandapipes/pull/711
    Then use it instead of defining pressure_net and have the user check
    the results.

    Check if network is split into multiple parts and solve each on its own

    """
    # Prepare the component tables of the DHNx network
    forks = network.components['forks'].copy()
    consumers = network.components['consumers'].copy()
    producers = network.components['producers'].copy()
    pipes = gdf_pipes.copy()
    # reset the index to later on merge the pandapipes results, that
    # do not know an 'id' or 'name' anymore
    idx_name = pipes.index.name
    if idx_name is None:
        idx_name = 'index'
    pipes = pipes.reset_index()


    pp_net, forks, consumers, producers, pipes, junctions_consumers = pp_setup_net(
            forks, consumers, producers, pipes, direction, pressure_pn,
            pressure_net, df_DN, elevation_col, P_th_kW,
            pipe_section_length, pressure_loss_coefficient_consumer,
            download_missing_elevation, f_length_loss,
            apply_flow_simultaneity=apply_flow_simultaneity)

    if show_plot:
        try:
            # Requires additional dependencies
            pp.plotting.simple_plot(pp_net, junction_size=0.01, pump_size=0.1,
                                    pipe_width=1, heat_exchanger_size=0.1,
                                    valve_size=0.1,
                                    flow_control_size=0.1, library='networkx')
        except Exception:
            pass

    # Set default options for pandapipes simulation
    # kwargs = dict()
    # kwargs.setdefault('iter', 10000)
    # kwargs.setdefault('tol_p', 1e-4)
    # kwargs.setdefault('tol_v', 1e-4)
    # kwargs.setdefault('stop_condition', "tol")
    # kwargs.setdefault('friction_model', "colebrook")
    # kwargs.setdefault('nonlinear_method', "automatic")
    # kwargs.setdefault('transient', False)

    if direction == 'circular':
        kwargs.setdefault('mode', "bidirectional")
    else:
        kwargs.setdefault('mode', "sequential")

    # Execute the pandapipes simulation
    try:
        pp.pipeflow(pp_net, **kwargs)
    except Exception as e:
        logger.exception(e)
        breakpoint()

    pipes, forks, consumers, producers = pp_merge_results(
        pp_net, pipes, forks, consumers, producers, direction,
        elevation_col, idx_name)

    if direction == 'circular':
        # For a closed-loop simulation, a recalculation is necessary after
        # determining the minimum pressure point in the forward flow. See
        # https://github.com/e2nIEE/pandapipes/discussions/786
        consumer_p_min_id = consumers.loc[[consumers['p_bar'].idxmin()],
                                          'id_full'].to_list()

        # Prepare the component tables of the DHNx network
        forks = network.components['forks'].copy()
        consumers = network.components['consumers'].copy()
        producers = network.components['producers'].copy()
        pipes = gdf_pipes.copy()
        # reset the index to later on merge the pandapipes results, that
        # do not know an 'id' or 'name' anymore
        idx_name = pipes.index.name
        if idx_name is None:
            idx_name = 'index'
        pipes = pipes.reset_index()

        pp_net, forks, consumers, producers, pipes, junctions_consumers = pp_setup_net(
                forks, consumers, producers, pipes, direction, pressure_pn,
                pressure_net, df_DN, elevation_col, P_th_kW,
                pipe_section_length, pressure_loss_coefficient_consumer,
                download_missing_elevation, f_length_loss,
                consumer_p_min_id=consumer_p_min_id,
                apply_flow_simultaneity=apply_flow_simultaneity)

        pp.pipeflow(pp_net, **kwargs)

        pipes, forks, consumers, producers = pp_merge_results(
            pp_net, pipes, forks, consumers, producers, direction,
            elevation_col, idx_name)

    # print(pp_net.res_junction.head(n=8))
    # print(pp_net.res_pipe.head(n=8))

    # Test for issues in pressure distribution
    df_junctions_consumers = pd.DataFrame.from_dict(junctions_consumers, 'index')
    if not df_junctions_consumers.empty:
        df_junctions_consumers = df_junctions_consumers.join(
            pp_net.res_junction[['p_bar']].rename(columns={'p_bar': 'p_to_bar'}),
            on='junction_to'
            )
        df_junctions_consumers = df_junctions_consumers.join(
            pp_net.res_junction[['p_bar']].rename(columns={'p_bar': 'p_from_bar'}),
            on='junction_from'
            )
        mask = df_junctions_consumers['p_from_bar'] > df_junctions_consumers['p_to_bar']
        df_junctions_consumers_mask = df_junctions_consumers.loc[mask]
        if not df_junctions_consumers_mask.empty:
            logger.debug('Pressure after these consumers is heigher '
                         'than in front\n%s', df_junctions_consumers_mask)

    # Export results to Excel
    if save_path is not None:
        filepath = os.path.join(save_path, 'pandapipes_result.xlsx')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with pd.ExcelWriter(filepath) as writer:
            pipes.reset_index().to_excel(
                writer, sheet_name='pipes',
                columns=[idx_name, 'type', 'from_node', 'to_node', 'length',
                         'capacity', 'Cost [€]', 'P_loss [kW]',
                         "Inner diameter [m]", "Roughness [mm]",
                         'U-value [W/mK]', "u [W/m2K]", 'DN']
            )
            pp_net.res_pipe.to_excel(writer, sheet_name='pandapipes_pipes')
            pp_net.res_junction.to_excel(writer,
                                         sheet_name='pandapipes_junctions')

    # Determine the low pressure and low temperature points ("Schlechtpunkt")
    forks_p_min = forks.loc[[forks['p_bar'].idxmin()]]
    forks_t_min = forks.loc[[forks['t_°C'].idxmin()]]

    pp_plot_results(pipes, consumers, producers, forks, forks_p_min,
                    forks_t_min, direction, elevation_col, show_plot,
                    save_plot, save_path, save_gis_ext)

    return pipes, forks, consumers, producers


def pp_setup_net(
        forks, consumers, producers, pipes, direction, pressure_pn,
        pressure_net, df_DN, elevation_col, P_th_kW,
        pipe_section_length, pressure_loss_coefficient_consumer,
        download_missing_elevation, f_length_loss,
        consumer_p_min_id=None, apply_flow_simultaneity=False):
    import math
    from CoolProp.CoolProp import PropsSI

    if consumer_p_min_id is None:
        consumer_p_min_id = consumers.loc[
            consumers.index[[-1]], 'id_full'].to_list()

    # Define the pandapipes parameters
    if df_DN is None:  # Use some default values
        dT = 30  # [K]
        feed_temp = 348  # 75 °C (Feed-in temperature at the heat supply)
        ext_temp = 283  # 10 °C (temperature of the ground)
    else:
        # Assume that the following values are the same for all DN types
        dT = (df_DN['T_forward [°C]'] - df_DN['T_return [°C]']).values[0]
        feed_temp = df_DN['T_forward [°C]'].values[0] + 273.15  # K
        ext_temp = df_DN['T_ground [°C]'].values[0] + 273.15  # K

    # Calculate heat transfer coefficient for pandapipes (see docstring above)
    df_DN["u [W/m2K]"] = df_DN['U-value [W/mK]'].div(
        df_DN['Inner diameter [m]'] * math.pi * f_length_loss)

    # elevation_col can be used to indicate which column contains elevation
    # data (height in meters).
    if not elevation_col is None:
        # Download elevation data and perform test for missing values
        for _df in [forks, consumers, producers]:
            if elevation_col not in _df:
                if download_missing_elevation:
                    df_elev = download_elevation_data(
                        _df, col_elevation=elevation_col,
                        show_plot=False)
                    _df[elevation_col] = df_elev[elevation_col]
            if _df[elevation_col].isna().any():
                logger.error("Height information from column '%s' has "
                             "missing values", elevation_col)

    # prepare the consumers dataframe
    # calculate massflow for each consumer and producer
    # Do not use the maximum power of each consumer, but a power that
    # considers the simultaneity. This must be provided by the user, either
    # as a name of a column in 'consumers', or as a series (or list, array)
    # with an entry for each consumer
    if P_th_kW is None:
        P_th_kW = 'P_heat_max'
        logger.warning("The pandapipes simulation requires a thermal power "
                       f"for each consumer. Per default, the column {P_th_kW} "
                       "in the consumers table is used. But this only makes "
                       "sense if the network was designed with a "
                       "simultaneity factor = 1, which is rarely the case. "
                       "Instead, please provide the name of a column that "
                       "contains the thermal power including the "
                       "simultaneity, or a list of those values with the "
                       "argument 'P_th_kW'.")
    if isinstance(P_th_kW, str):
        if P_th_kW in consumers.columns:
            P_th_kW = consumers[P_th_kW]
        else:
            raise ValueError(
                "Pandapipes requires a thermal power for each consumer. "
                f"The provided string {P_th_kW} is not found in the columns "
                "of the consumer table")
    elif isinstance(P_th_kW, pd.Series) or isinstance(P_th_kW, pd.DataFrame):
        # The dhnx consumer index is always of dtype string. Catch case where
        # the index of P_th_kW might be int or float and can be converted
        if (P_th_kW.index != consumers.index).any():
            P_th_kW.index = P_th_kW.index.astype(str)
            if (P_th_kW.index != consumers.index).any():
                raise ValueError(
                    "Cannot match the indices of the DHNx consumers and "
                    "the given thermal power for each consumer.")

    # Get required physical properties of water
    cp = PropsSI('C',
                 'T', feed_temp,
                 'P', pressure_net * 100000,   # pressure in [Pa],
                 'IF97::Water')  # [kJ/(kg K)]
    consumers['massflow'] = P_th_kW / (cp * dT * 0.001)  # [kg/s]

    # prepare the pipes dataframe

    # delete pipes with capacity of 0
    pipes = pipes.drop(pipes[pipes['capacity'] == 0].index)

    # At this point, 'forks' contains not only those connecting the chosen
    # pipes, but also the remaining possible forks. Pandapipes would set them
    # "out of service" anyway, but it is cleaner to just remove them here
    forks = forks.loc[forks.intersects(pipes.union_all())]
    producers = producers.loc[producers.intersects(pipes.union_all())]

    if apply_flow_simultaneity:
        # Add attributes from pipes to their 'to'-node (ending fork).
        # This requries that 'to_node' represents the actual flow direction,
        # which is not the case for raw dhnx results.
        if (pipes['direction'] != 1).any():
            logger.error("Flow simultaneity at forks NOT applied, "
                         "because flow direction is not forward in "
                         "all pipe segments. This should have been set "
                         "in apply_deterministic_simultaneity()")
            apply_flow_simultaneity = False
        else:
            if 'flow_diff_simultaneity' not in pipes.columns:
                logger.error("Flow simultaneity at forks NOT applied, "
                             "because column 'flow_diff_simultaneity' is "
                             "missing. This should have been set "
                             "in apply_deterministic_simultaneity()")
                apply_flow_simultaneity = False
            else:  # Merging is allowed
                # When merging, use groubpy.sum() to ensure forks with
                # multiple predecessors are considered.
                forks = pd.merge(
                    left=forks,
                    right=(pipes[['to_node', 'flow_diff_simultaneity']]
                           .groupby('to_node').sum()),
                    left_on='id_full', right_index=True)

    # Add data of technical data sheet with the DN numbers to the pipes table
    cols_select = ["Inner diameter [m]", "Roughness [mm]",
                   "U-value [W/mK]", "u [W/m2K]"]
    cols_select = [col for col in cols_select if col not in pipes.columns]
    if len(cols_select) > 0:
        cols_select.append("DN")
        pipes = pipes.join(df_DN[cols_select].set_index('DN'), on='DN')

    # Create the pandapipes model.
    # When direction is 'forward' or 'return', only a single direction
    # is modelled with pandapipes.
    # The setting 'circular' attempts to model the full system with supply
    # and return in a closed loop simulation.
    # This allows to calculate a minimum system pressure at the end of the
    # return. Depending on that, the user can derive a required
    # supply pressure.
    pp_net = pp.create_empty_network(fluid="water")
    junctions_consumers = dict()

    for df_junctions in [forks, consumers, producers]:
        j_list = pp.create_junctions(
            net=pp_net,
            nr_junctions=len(df_junctions),
            pn_bar=pressure_pn,
            tfluid_k=feed_temp,
            name=df_junctions['id_full'],
            height_m=(df_junctions[elevation_col]
                      if not elevation_col is None else 0),
            geodata=df_junctions.centroid.get_coordinates().values,
            )
        if 'flow_diff_simultaneity' in df_junctions.columns:
            pp.create_sources(
                net=pp_net,
                junctions=j_list,
                mdot_kg_per_s=(df_junctions['flow_diff_simultaneity']
                               *(-1) / (cp * dT * 0.001))  # kW --> kg/s
                )

        if direction == 'circular':
            # Each return flow fork needs its own junction
            j_list = pp.create_junctions(
                net=pp_net,
                nr_junctions=len(df_junctions),
                pn_bar=pressure_pn,
                tfluid_k=feed_temp,
                name=df_junctions['id_full']+'-return',
                height_m=(df_junctions[elevation_col]
                          if not elevation_col is None else 0),
                geodata=(df_junctions.centroid.translate(5, -5)
                         .get_coordinates().values),
                )
            if 'flow_diff_simultaneity' in df_junctions.columns:
                pp.create_sinks(
                    net=pp_net,
                    junctions=j_list,
                    mdot_kg_per_s=(df_junctions['flow_diff_simultaneity']
                                   *(-1) / (cp * dT * 0.001))  # kW --> kg/s
                    )

    if direction == 'forward':
        # Model and simulate only the forward flow in the network.
        # Mass flow enteres the system at a 'source', witch pressure and
        # temperature defined by an "external grid". Each consumer is a "sink"
        # where the mass flow leaves the system.

        # create sinks for consumers
        pp.create_sinks(
            net=pp_net,
            junctions=pd.merge(
                left=consumers, right=pp_net.junction.reset_index(),
                left_on='id_full', right_on='name')['index'],
            mdot_kg_per_s=consumers['massflow'],
            name=consumers['id_full']
        )

        # create source(s) for producers
        # It is assumed that there is only one producer that provides
        # the total massflow for all consumers.
        # Pandapipes supports multiple producers, but we would need to define
        # a mass flow for each of them.
        pp.create_sources(
            net=pp_net,
            junctions=pd.merge(
                left=producers, right=pp_net.junction.reset_index(),
                left_on='id_full', right_on='name')['index'],
            mdot_kg_per_s=consumers['massflow'].sum(),
            name=producers['id_full']
        )

        # EXTERNAL GRID as slip (Schlupf)
        pp.create_ext_grids(
            pp_net,
            junctions=pd.merge(
                left=producers, right=pp_net.junction.reset_index(),
                left_on='id_full', right_on='name')['index'],
            p_bar=pressure_net,
            t_k=feed_temp,
            name=producers['id_full'],
        )

    elif direction == 'return':
        # Opposite of the 'forward' mode.

        # create sources for consumers
        pp.create_sources(
            net=pp_net,
            junctions=pd.merge(
                left=consumers, right=pp_net.junction.reset_index(),
                left_on='id_full', right_on='name')['index'],
            mdot_kg_per_s=consumers['massflow'],
            name=consumers['id_full']
        )

        # create sink(s) for producers
        pp.create_sinks(
            net=pp_net,
            junctions=pd.merge(
                left=producers, right=pp_net.junction.reset_index(),
                left_on='id_full', right_on='name')['index'],
            mdot_kg_per_s=consumers['massflow'].sum(),
            name=producers['id_full']
        )

        # EXTERNAL GRID as slip (Schlupf)
        pp.create_ext_grids(
            pp_net,
            junctions=pd.merge(
                left=consumers, right=pp_net.junction.reset_index(),
                left_on='id_full', right_on='name')['index'],
            p_bar=pressure_net,
            t_k=feed_temp,
            name=consumers['id_full'],
        )

    elif direction == 'circular':
        # Model and simulate a closed-loop hyraulic network.
        # The pumps provide the total mass flow and the pressure
        # at the inlet.
        # All but one of the consumers also define their mass flow.
        # (If all consumers define it, the system is overdetermined)
        # This is achieved by combining heat exchangers and flow control
        # components.

        # pump_mode = "grid_plus_sink"
        # pump_mode = "circ_pump_pressure"
        pump_mode = "circ_pump_mass"

        # create pump for producers
        if pump_mode == 'circ_pump_mass':
            for index, producer in producers.iterrows():
                if len(producers)==1:
                    pipes_mask = pipes.index
                    consumers_mask = consumers.index
                else:
                    # For multiple producers, it is possible that each
                    # supplies a subnetwork. Find the pipes and consumers that
                    # are connected with the current producer
                    G = momepy.gdf_to_nx(pipes, approach="primal")
                    for nodes in nx.connected_components(G):
                        SG = G.subgraph(nodes)
                        # plot_networkx_graph(SG)
                        _, df_subnetwork = momepy.nx_to_gdf(SG)
                        df_sub = df_subnetwork.union_all()
                        if producer.geometry.intersects(df_sub):
                            pipes_mask = pipes.intersects(df_sub)
                            consumers_mask = consumers.intersects(df_sub)

                            # One consumer_p_min is required for each subnet
                            if not (consumers.loc[consumers_mask, 'id_full']
                                    .isin(consumer_p_min_id).any()):
                                consumer_p_min_id.append(
                                    consumers.loc[
                                        consumers_mask, 'id_full'].iloc[-1]
                                    )

                # When using circ_pump_const_mass_flow, the consumer
                # with id 'consumer_p_min_id' is left with an undefined
                # massflow. It will get the remaining mass flow, which
                # results in a solvable system
                if apply_flow_simultaneity:
                    # mdot_flow = producer['capacity']/(cp * dT * 0.001)
                    mdot_flow = (consumers.loc[consumers_mask, 'massflow'].sum()
                                 + (pipes.loc[pipes_mask,'flow_diff_simultaneity']
                                    .div(cp*dT*0.001).sum())
                                 )
                else:
                    # Here it is savest to use the sum of the consumers,
                    # for the 'perfect' mass balance
                    mdot_flow = consumers.loc[consumers_mask, 'massflow'].sum()
                    # print(mdot_flow)
                    # mdot_flow = producer['capacity'] / (cp*dT*0.001)
                    # print(mdot_flow)

                pp.create_circ_pump_const_mass_flow(
                    net=pp_net,
                    return_junction=pp_net.junction.index[
                        pp_net.junction['name'] == producer['id_full']+'-return'][0],
                    # flow_junction=j_pump,
                    flow_junction=pp_net.junction.index[
                        pp_net.junction['name'] == producer['id_full']][0],
                    p_flow_bar=pressure_net,
                    mdot_flow_kg_per_s=mdot_flow,
                    name=producer['id_full'],
                    t_flow_k=feed_temp,
                    )

                # pp.create_flow_control(
                #     pp_net,
                #     from_junction=j_pump,
                #     to_junction=pp_net.junction.index[
                #         pp_net.junction['name'] == producer['id_full']][0],
                #     controlled_mdot_kg_per_s=consumers['massflow'].sum())

        elif pump_mode == 'circ_pump_pressure':
            # Works, but a constant pressure lift at the pump makes no sense
            # to me. As a result there is a variable pressure loss at the
            # consumers
            for index, producer in producers.iterrows():
                pp.create_circ_pump_const_pressure(
                    net=pp_net,
                    return_junction=pp_net.junction.index[
                        pp_net.junction['name'] == producer['id_full']+'-return'][0],
                    flow_junction=pp_net.junction.index[
                        pp_net.junction['name'] == producer['id_full']][0],
                    p_flow_bar=pressure_net,
                    plift_bar=6,
                    name=producer['id_full'],
                    t_flow_k=feed_temp,
                    )

        elif pump_mode == 'grid_plus_sink':
            # create source(s) for producers
            pp.create_sources(
                net=pp_net,
                junctions=pd.merge(
                    left=producers, right=pp_net.junction.reset_index(),
                    left_on='id_full', right_on='name')['index'],
                mdot_kg_per_s=consumers['massflow'].sum(),
                name=producers['id_full']
            )

            # EXTERNAL GRID as slip (Schlupf)
            pp.create_ext_grids(
                pp_net,
                junctions=pd.merge(
                    left=producers, right=pp_net.junction.reset_index(),
                    left_on='id_full', right_on='name')['index'],
                p_bar=pressure_net,
                t_k=feed_temp,
                name=producers['id_full'],
            )

            for index, producer in producers.iterrows():
                pp.create_sink(
                    net=pp_net,
                    junction=pp_net.junction.index[
                        pp_net.junction['name'] == producer['id_full']+'-return'][0],
                    mdot_kg_per_s=consumers['massflow'].sum(),
                    name=producer['id_full']
                )

                pp.create_ext_grid(
                    pp_net,
                    junction=pp_net.junction.index[
                        pp_net.junction['name'] == producer['id_full']+'-return'][0],
                    p_bar=pressure_net-6,
                    t_k=feed_temp-50,
                    name=producer['id_full'],
                )

        # consumer_mode = 'heat_consumer'
        consumer_mode = 'heatexchanger_and_flowcontrol'

        if consumer_mode == 'heat_consumer':
            # create 'heat_consumer' for consumers
            consumers_return = consumers.copy()
            consumers_return['id_full'] = consumers_return['id_full']+'-return'
            pp.create_heat_consumers(
                net=pp_net,
                from_junctions=pd.merge(
                    left=consumers, right=pp_net.junction.reset_index(),
                    left_on='id_full', right_on='name')['index'],
                to_junctions=pd.merge(
                    left=consumers_return, right=pp_net.junction.reset_index(),
                    left_on='id_full', right_on='name')['index'],
                qext_w=P_th_kW.mul(1000),
                # deltat_k=dT,
                controlled_mdot_kg_per_s=consumers['massflow'],
                name=consumers['id_full'],
                )

        elif consumer_mode == 'heatexchanger_and_flowcontrol':
            # The component 'heat_consumer' does not allow to set a pressure
            # loss, so we use heat_exchanger and flow_control instead

            for index, consumer in consumers.iterrows():
                j_con_f = pp_net.junction.index[
                    pp_net.junction['name'] == consumer['id_full']][0]
                j_con_r = pp_net.junction.index[
                    pp_net.junction['name'] == consumer['id_full']+'-return'][0]

                junctions_consumers[consumer['id_full']] = dict(
                    junction_to=j_con_f, junction_from=j_con_r)

                if ((consumer['id_full'] in consumer_p_min_id)
                    and (pump_mode == "circ_pump_mass")):
                     pp.create_heat_exchanger(
                        net=pp_net,
                        from_junction=j_con_f,
                        to_junction=j_con_r,
                        qext_w=P_th_kW.loc[index] * (1000),  # kW --> W
                        loss_coefficient=pressure_loss_coefficient_consumer,
                        inner_diameter_mm=100,
                        name=consumer['id_full'],
                        )
                else:
                    j_con_i1 = pp.create_junction(
                        pp_net,
                        pn_bar=pressure_pn,
                        tfluid_k=feed_temp,
                        height_m=consumer[elevation_col] if not elevation_col is None else 0,
                        name=f"Junction Consumer {index}",
                        geodata=(consumer.geometry.centroid.x+10,
                                 consumer.geometry.centroid.y-20),
                        )
                    pp.create_heat_exchanger(
                        net=pp_net,
                        from_junction=j_con_f,
                        to_junction=j_con_i1,
                        qext_w=P_th_kW.loc[index] * (1000),  # kW --> W
                        loss_coefficient=pressure_loss_coefficient_consumer,
                        inner_diameter_mm=100,
                        name=consumer['id_full'],
                        )
                    pp.create_flow_control(
                        pp_net,
                        from_junction=j_con_i1,
                        to_junction=j_con_r,
                        controlled_mdot_kg_per_s=consumer['massflow'],
                        name=consumer['id_full']
                    )

    # create pipes
    pp.create_pipes_from_parameters(
        net=pp_net,
        from_junctions=pd.merge(
            left=pipes, right=pp_net.junction.reset_index(),
            left_on='from_node', right_on='name')['index'],
        to_junctions=pd.merge(
            left=pipes, right=pp_net.junction.reset_index(),
            left_on='to_node', right_on='name')['index'],
        length_km=pipes['length'].div(1000),  # convert to km
        # diameter_m=pipes["Inner diameter [m]"],
        inner_diameter_mm=pipes["Inner diameter [m]"]*1000,
        k_mm=pipes["Roughness [mm]"],
        u_w_per_m2k=pipes["u [W/m2K]"],
        text_k=ext_temp,
        # name=pipes[idx_name],
        sections=(pipes.length.div(pipe_section_length)
                  .round(0).clip(lower=1).astype(int)),
        flow_direction='forward',
    )

    if direction == 'circular':
        pipes_return = pipes.copy()
        pipes_return['from_node'] = pipes_return['from_node']+'-return'
        pipes_return['to_node'] = pipes_return['to_node']+'-return'
        pp.create_pipes_from_parameters(
            net=pp_net,
            from_junctions=pd.merge(
                left=pipes_return, right=pp_net.junction.reset_index(),
                left_on='from_node', right_on='name')['index'],
            to_junctions=pd.merge(
                left=pipes_return, right=pp_net.junction.reset_index(),
                left_on='to_node', right_on='name')['index'],
            length_km=pipes['length'].div(1000),  # convert to km
            # diameter_m=pipes["Inner diameter [m]"],
            inner_diameter_mm=pipes["Inner diameter [m]"]*1000,
            k_mm=pipes["Roughness [mm]"],
            u_w_per_m2k=pipes["u [W/m2K]"],
            text_k=ext_temp,
            # name=pipes[idx_name],
            sections=(pipes.length.div(pipe_section_length)
                      .round(0).clip(lower=1).astype(int)),
            flow_direction='return',
        )

    return pp_net, forks, consumers, producers, pipes, junctions_consumers


def pp_merge_results(
        pp_net, pipes, forks, consumers, producers, direction, elevation_col, idx_name):
    # Merge results of pipes to GeoDataFrame
    if direction == 'circular':
        pp_pipe = pd.merge(
            pp_net.pipe, pp_net.res_pipe, left_index=True,
            right_index=True, how='left'
            ).set_index(pp_net.pipe.index)

        pp_pipe_f = pp_pipe.loc[pp_pipe['flow_direction'] == 'forward',
                                pp_net.res_pipe.columns].reset_index(drop=True)
        pp_pipe_r = pp_pipe.loc[pp_pipe['flow_direction'] == 'return',
                                pp_net.res_pipe.columns].reset_index(drop=True)

        pipes_f = pd.merge(
            pipes, pp_pipe_f, left_index=True, right_index=True,
            how='left'
            )
        pipes_r = pd.merge(
            pipes, pp_pipe_r, left_index=True, right_index=True,
            how='left'
            )
        pipes_f.set_index(idx_name, inplace=True)  # restore the original index
        pipes_r.set_index(idx_name, inplace=True)  # restore the original index

        pipes = pd.concat(
            [pipes_f, pipes_r],
            keys=['forward', 'return'],
            names=['flow_direction'],
            # axis='columns',
            )

    else:
        pipes = pd.merge(
            pipes, pp_net.res_pipe, left_index=True, right_index=True,
            how='left'
            )
        pipes.set_index(idx_name, inplace=True)  # restore the original index
        pipes = pd.concat(
            [pipes],
            keys=[direction],
            names=['flow_direction'],
            # axis='columns',
            )

    junctions = pd.merge(
        pp_net.res_junction, pp_net.junction, left_index=True,
        right_index=True, how='left'
        ).set_index(pp_net.res_junction.index)
    junctions['t_°C'] = junctions['t_k'] - 273.15

    forks = pd.merge(
        forks, junctions.drop(columns=[elevation_col], errors='ignore'),
        left_on='id_full', right_on='name',
        how='left'
        ).set_index(forks.index)
    consumers = pd.merge(
        consumers, junctions.drop(columns=[elevation_col], errors='ignore'),
        left_on='id_full', right_on='name',
        how='left', suffixes=('', '_pp')
        ).set_index(consumers.index)
    producers = pd.merge(
        producers, junctions.drop(columns=[elevation_col], errors='ignore'),
        left_on='id_full', right_on='name',
        how='left', suffixes=('', '_pp')
        ).set_index(producers.index)

    # Convert Kelvin to degrees Celsius temperature columns
    pipes['t_from_°C'] = pipes['t_from_k'] - 273.15
    pipes['t_to_°C'] = pipes['t_to_k'] - 273.15
    # Calculate additional attributes
    pipes['vdot_m3_per_s_abs'] = pipes['vdot_m3_per_s'].abs()
    pipes['vdot_m3_per_h_abs'] = pipes['vdot_m3_per_s_abs']*3600
    pipes['v_mean_m_per_s_abs'] = pipes['v_mean_m_per_s'].abs()
    pipes['delta_p_Pa/m'] = (
        pipes['p_from_bar']-pipes['p_to_bar']).abs() / (pipes.length * 1e-5)
    # 'to' and 'from' may be confusing when the flow direction
    # of individual segments is reversed.
    pipes['p_mean_bar'] = pipes[['p_to_bar', 'p_from_bar']].mean(axis='columns')

    return pipes, forks, consumers, producers


def pp_plot_results(pipes, consumers, producers, forks, forks_p_min,
                    forks_t_min, direction, elevation_col, show_plot,
                    save_plot, save_path, save_gis_ext):

    # Plot the results of pandapipes simulation
    if show_plot or save_plot:
        if not elevation_col is None:
            # Plot elevation
            if save_plot:
                save_path_plots=os.path.join(save_path, 'plots', 'Elevation')
            else:
                save_path_plots=None

            plot_geometries(
                [pipes,
                 pd.concat([forks, consumers, producers]),
                 ],
                plt_kwargs=[
                    dict(label="Heating grid", color='red', zorder=0),
                    dict(
                        column=elevation_col,
                        legend=True,
                        legend_kwds=dict(label="Elevation [m]"),
                    ),
                ],
                title="Elevation data of heating grid junction points",
                set_axis_off=True,
                show_plot=show_plot,
                save_path=save_path_plots,
            )

        for direction_used in pipes.index.unique('flow_direction'):
            _pipes = pipes.xs(direction_used, level='flow_direction')

            # Plot pressure of pipes' ending nodes
            if save_plot:
                save_path_plots=os.path.join(
                    save_path, 'plots', f'Pressure ({direction_used})')
            else:
                save_path_plots=None

            plot_geometries(
                [consumers,
                 producers,
                 _pipes,
                 forks_p_min],
                plt_kwargs=[dict(label='Consumer', color='green'),
                            dict(label='Producer',
                                 color=matplotlib.colormaps['cividis'](1.0)),
                            dict(column='p_to_bar', linewidth=2, legend=True,
                                 label='Pipelines',
                                 cmap='cividis',
                                 legend_kwds={'label': 'Pressure [bar]'}),
                            dict(label='Minimum pressure',
                                 color=matplotlib.colormaps['cividis'](0.0)),
                            ],
                # plot_basemap=True,
                title=f'Pressure distribution ({direction_used})',
                set_axis_off=True,
                dpi=300,
                show_plot=show_plot,
                save_path=save_path_plots,
                )

            # Plot pressure loss per pipes segment
            if save_plot:
                save_path_plots=os.path.join(
                    save_path, 'plots', f'Pressure loss ({direction_used})')
            else:
                save_path_plots=None

            plot_geometries(
                [consumers, producers, _pipes],
                plt_kwargs=[dict(label='Consumer', color='green'),
                            dict(label='Producer',
                                 color=matplotlib.colormaps['cividis'](1.0)),
                            dict(column='delta_p_Pa/m', linewidth=2, legend=True,
                                 label='Pipelines',
                                 cmap='cividis',
                                 legend_kwds={'label': 'Pressure loss [Pa/m]'}),
                            ],
                # plot_basemap=True,
                title=f'Pressure loss ({direction_used})',
                set_axis_off=True,
                dpi=300,
                show_plot=show_plot,
                save_path=save_path_plots,
                )

            # Plot temperature of pipes' ending nodes
            if save_plot:
                save_path_plots=os.path.join(
                    save_path, 'plots', f'Temperature ({direction_used})')
            else:
                save_path_plots=None

            plot_geometries(
                [consumers,
                 producers,
                 _pipes,
                 forks_t_min],
                plt_kwargs=[dict(label='Consumer', color='black'),
                            dict(label='Producer',
                                 color=matplotlib.colormaps['Wistia'](1.0)),
                            dict(column='t_to_°C', linewidth=2, legend=True,
                                 label='Pipelines',
                                 cmap='Wistia',
                                 legend_kwds={'label': 'Temperature [°C]'}),
                            dict(label='Minimum temperature',
                                 color=matplotlib.colormaps['Wistia'](0.0)),
                            ],
                # plot_basemap=True,
                title=f'Temperature distribution ({direction_used})',
                set_axis_off=True,
                dpi=300,
                show_plot=show_plot,
                save_path=save_path_plots,
                )

            # Plot volume flow rate per pipe segment
            if save_plot:
                save_path_plots=os.path.join(
                    save_path, 'plots', f'Flow rate ({direction_used})')
            else:
                save_path_plots=None

            plot_geometries(
                [consumers,
                 producers,
                 _pipes],
                plt_kwargs=[dict(label='Consumer', color='black'),
                            dict(label='Producer',
                                 color=matplotlib.colormaps['Wistia'](1.0)),
                            dict(column='vdot_m3_per_h_abs', linewidth=2,
                                 legend=True, label='Pipelines', cmap='Wistia',
                                 legend_kwds={'label': 'Flow rate [m³/h]'})
                            ],
                # plot_basemap=True,
                title=f'Flow rate distribution ({direction_used})',
                set_axis_off=True,
                dpi=300,
                show_plot=show_plot,
                save_path=save_path_plots,
                )

            # Plot volume flow velocity per pipe segment
            if save_plot:
                save_path_plots=os.path.join(
                    save_path, 'plots', f'Velocity ({direction_used})')
            else:
                save_path_plots=None

            plot_geometries(
                [consumers,
                 producers,
                 _pipes],
                plt_kwargs=[dict(label='Consumer', color='black'),
                            dict(label='Producer',
                                 color=matplotlib.colormaps['Wistia'](1.0)),
                            dict(column='v_mean_m_per_s_abs', linewidth=2,
                                 legend=True, label='Pipelines', cmap='Wistia',
                                 legend_kwds={'label': 'Velocity [m/s]'})
                            ],
                # plot_basemap=True,
                title=f'Velocity distribution ({direction_used})',
                set_axis_off=True,
                dpi=300,
                show_plot=show_plot,
                save_path=save_path_plots,
                )

            # Find and plot pressure along shorest path from start to point with
            # lowest pressure
            if save_plot:
                save_path_plots=os.path.join(save_path, 'plots',
                                       f'Pressure (shortest path) ({direction_used})')
            else:
                save_path_plots=None

            gdf_shortest = find_shortest_path(_pipes, producers, forks_p_min)

            plot_geometries(
                    [_pipes, producers, forks_p_min, gdf_shortest],
                    plt_kwargs=[
                        dict(label='Pipelines', linewidth=0.5, color='red'),
                        dict(label='Producer',
                             color=matplotlib.colormaps['viridis'](1.0)),
                        dict(label='Minimum pressure',
                             color=matplotlib.colormaps['viridis'](0.0)),
                        dict(column='p_to_bar', legend=True, cmap='viridis',
                             legend_kwds=dict(label='Pressure [bar]')),],
                    title=f'Pressure distribution to point of minimum pressure ({direction_used})',
                    set_axis_off=True,
                    show_plot=show_plot,
                    save_path=save_path_plots,
                    )

    if direction == 'circular':
        gdf_shortest_f = find_shortest_path(
            pipes.xs('forward', level='flow_direction'),
            producers, forks_p_min)
        gdf_shortest_r = find_shortest_path(
            pipes.xs('return', level='flow_direction'),
            producers, forks_p_min)
        gdf_shortest = gdf_shortest_f

        if show_plot or save_plot:
            fig, ax = plt.subplots()
            ax.plot(gdf_shortest_f['distance'], gdf_shortest_f['p_mean_bar'], label='Forward')
            ax.plot(gdf_shortest_r['distance'], gdf_shortest_r['p_mean_bar'], label='Return')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Pressure [bar]')
            plt.legend()

            if save_plot:
                custom_plot_save(
                    os.path.join(save_path, 'plots', 'Pressure vs Distance'))
            if show_plot:
                plt.show()
            else:
                plt.close()

    else:
        gdf_shortest = find_shortest_path(
            pipes.xs(direction, level='flow_direction'),
            producers,
            forks_p_min)

    if save_path is not None:
        # export the GeoDataFrames with the simulation results to .geojson
        save_gis_generic(
            pipes, 'pandapipes_pipes', path=save_path, ext=save_gis_ext)
        save_gis_generic(
            forks, 'pandapipes_forks', path=save_path, ext=save_gis_ext)
        save_gis_generic(
            consumers, 'pandapipes_consumers', path=save_path,
            ext=save_gis_ext)
        save_gis_generic(
            producers, 'pandapipes_producers', path=save_path, ext=save_gis_ext)
        save_gis_generic(
            gdf_shortest, 'pandapipes_shortest', path=save_path,
            ext=save_gis_ext)
        if direction == 'circular':
            save_gis_generic(
                gdf_shortest_f, 'pandapipes_shortest_f',
                path=save_path, ext=save_gis_ext)
            save_gis_generic(
                gdf_shortest_r, 'pandapipes_shortest_r',
                path=save_path, ext=save_gis_ext)


def find_shortest_path(gdf, point_start, point_end, distance_col='distance',
                       show_plot=False):
    """Find shortest path from point_start to point_end in line network gdf.

    Use networkx to find the shortest path, then sort the edges along
    that path to ensure mixed directions in gdf do not cause issues.
    """
    gdf['length'] = gdf.length
    G = momepy.gdf_to_nx(gdf, approach="primal")
    # plot_networkx_graph(G)

    # Extract coordinates from point_start and point_end GeoDataFrames
    start = list(point_start.geometry.centroid.iloc[0].coords)[0]
    end = list(point_end.geometry.centroid.iloc[0].coords)[0]

    # Compute the shortest path
    shortest_path_nodes = nx.shortest_path(G, source=start, target=end,
                                           weight='length')

    G_s = nx.DiGraph(approach='primal')
    for i in range(len(shortest_path_nodes) - 1):
        u = shortest_path_nodes[i]
        v = shortest_path_nodes[i + 1]

        # Copy all edge attributes from G
        edge_attrs = G.get_edge_data(u, v)[0]
        G_s.add_edge(u, v, **edge_attrs)

    # plot_networkx_graph(G_s)

    # Compute the distance from the start to each node in the shortest path
    cumulative_distances = {node: nx.shortest_path_length(
        G, source=start, target=node, weight='length')
        for node in shortest_path_nodes}

    nx.set_node_attributes(G_s, values=cumulative_distances, name=distance_col)

    nodes_gdf, edges_gdf = momepy.nx_to_gdf(G_s)
    gdf_s = pd.merge(edges_gdf, nodes_gdf[['nodeID', distance_col]],
                     left_on='node_start', right_on='nodeID', how='left')
    gdf_s.crs = gdf.crs

    if show_plot:
        plot_geometries(
            [gdf, point_start, point_end, gdf_s],
            plt_kwargs=[
                dict(label='Network', color='red'),
                dict(label='Start'),
                dict(label='End'),
                dict(column=distance_col, legend=True,
                     legend_kwds=dict(label='Cumulative distance [m]')),],
            )

    # Test if the total length of the shortest path is all within gdf
    # equal_distances = np.isclose(
    #     gdf[gdf.within(gdf_s.geometry.union_all())].length.sum(),
    #     gdf_s['distance'].max(),
    #     atol=1)

    # if not equal_distances:
    #     logger.warning("There might be a problem with the shortest path")

    return gdf_s
