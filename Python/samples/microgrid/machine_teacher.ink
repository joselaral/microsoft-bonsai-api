# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Overview
# A microgrid is a local energy grid with various power sources and loads that typically operates 
# connected to a traditional grid but is also capable of disconnecting from the traditional grid
# and operate autonomously. Microgrids can improve the resiliency of power supply by providing 
# backup for the grid in case of emergencies. They can also integrate renewable energy resources 
# and reduce CO2 emissions.

# In this sample, a microgrid is connected to a traditional grid. The microgrid has a Solar 
# Photovoltaic (PV) system, a battery to store energy and a variable load.

# The microgrid is controlled to meet the load by taking actions such as charging or discharging 
# the battery, importing energy from or exporting energy to the traditional grid and consuming any 
# generated PV power.

# More details about the model are available at 
# https://github.com/microsoft/microsoft-bonsai-api/tree/main/Python/samples/microgrid

inkling "2.0"
using Goal
using Math #OPTIMISE USING MATH(S)

const data_length = 8760 # there are 8760 data points in the sim for a microgrid (1 year data captured at every hour)
const episode_length = 24 * 2 # episodes are two days long
const horizon = 24 # this is the number of hours we will forecast the data (pv, load) if we add forecasts to states (not yet implemented)

const upper_starting_index = data_length - episode_length - horizon - 1 # we will reset the sim to a random time index, this variable is the upper bound on the initial time index

const max_starting_charge = Math.Floor(58048.89)

# These are all the values the sim produces
type SimState {
    load: number<5000 .. 50000>,
    hour: number<0 .. 24>,
    pv: number<0 .. 75000>,
    battery_soc: number<0 .. 1>,
    capa_to_charge: number<0 .. 65305>,
    capa_to_discharge: number<0 .. 65305>,
    grid_status: number<0, 1, >,
    grid_co2: number<0 .. 0.4>,
    grid_price_import: number<0.075 .. 0.205>,
    grid_price_export: number<0 .. 1>,
    pv_load_diff: number<-45000 .. 65000>,
    cost: number,
    co2_emission: number,
    prev_load: number,
    prev_pv: number,
    prev_grid_price_import: number,
    prev_grid_co2: number,
    prev_action_grid_import: number,
    sum_load: number,
    cost_co2: number,
    co2_predict: number,
    load_predict: number,
    pv_predict: number,
}
# This is a subset of the SimState that we'll make available to the brain
# (these should all be values that will be available to a deployed brain)
type ObservedState {
    load: number<5000 .. 50000>,
    hour: number<0 .. 24>,
    pv: number<0 .. 75000>,
    capa_to_charge: number<0 .. 65305>,
    capa_to_discharge: number<0 .. 65305>,
    grid_status: number<0, 1, >,
    grid_co2: number<0 .. 0.4>,
    grid_price_import: number<0.075 .. 0.205>,
    co2_predict: number<0 .. 0.4>,
    load_predict: number<5000 .. 50000>,
    pv_predict: number<0 .. 75000>,
}
type Action {
    # if positive it means charge the battery, else means discharge
    battery_power: number<-16327 .. 16327>,
}
# Configuration variables for the simulator
type SimConfig {
    starting_time_step: number<0 .. upper_starting_index>,
    starting_charge: number,
    cost_overgeneration: number,
    cost_loss_load: number,
    cost_battery: number,
    cost_co2: number,
    episode_length: number
}
# The global simulator statement defines the simulator that: 
#  - can be configured for each episode using fields defined in SimConfig,
#  - accepts per-iteration actions defined in SimAction, and
#  - outputs states with the fields defined in SimState.
simulator Simulator(Action: Action, Config: SimConfig): SimState {
    # Automatically launch the simulator with this
    # registered package name.
    # package "SIM_NAME"
}
graph (input: ObservedState): Action {
    concept ProgrammedRules(input): Action {
        programmed function (State: ObservedState): Action {
            var minA = Math.Min(State.load - State.pv, max_starting_charge)
            var minB = Math.Min(State.pv - State.load, max_starting_charge)
            var p_disc: number<-16327 .. 16327> = -(Math.Max(0, Math.Min(minA, State.capa_to_discharge)))
            var p_char: number<-16327 .. 16327> = Math.Max(0, Math.Min(minB, State.capa_to_charge))
            if (State.load - State.pv >= 0) {
                return {
                    battery_power: p_disc
                }
            } else {
                return {
                    battery_power: p_char
                }
            }
        }
    }
    concept OptimizeGrid_DRL(input): Action {
        curriculum {
            algorithm {
                Algorithm: "SAC"
            }
            source Simulator
            training {
                EpisodeIterationLimit: episode_length,
                NoProgressIterationLimit: 3000000,
            }
            goal (State: SimState) {
                minimize Cost weight 1:
                    State.cost
                    in Goal.RangeBelow(1488)
                minimize CO2 weight 5:
                    State.co2_emission
                    in Goal.RangeBelow(2042)
            }
            lesson StartMGrid {
                scenario {
                    # we take a step size of 11, to make sure the two episodes brains sees are not very similar
                    starting_time_step: number<0 .. upper_starting_index step 11>,
                    starting_charge: number<0 .. max_starting_charge step 10>,
                    cost_battery: 0,
                    cost_loss_load: 10,
                    cost_co2: 0.1,
                    cost_overgeneration: 1,
                    episode_length: episode_length
                }
            }
        }
    }
    output concept ChooseTheRightApproach(OptimizeGrid_DRL, ProgrammedRules): Action {
        select ProgrammedRules
        select OptimizeGrid_DRL
        curriculum {
            source Simulator
            training {
                EpisodeIterationLimit: episode_length,
                NoProgressIterationLimit: 5000000,
            }
            goal (State: SimState) {
                minimize Cost weight 1:
                    State.cost
                    in Goal.RangeBelow(1488)
                minimize CO2 weight 4:
                    State.co2_emission
                    in Goal.RangeBelow(2042)
            }
            lesson StartMGrid {
                scenario {
                    # we take a step size of 11, to make sure the two episodes brains sees are not very similar
                    starting_time_step: number<0 .. upper_starting_index step 11>,
                    starting_charge: number<0 .. max_starting_charge step 10>,
                    cost_battery: 0,
                    cost_loss_load: 10,
                    cost_co2: 0.1,
                    cost_overgeneration: 1,
                    episode_length: episode_length
                }
            }
        }
    }
}
