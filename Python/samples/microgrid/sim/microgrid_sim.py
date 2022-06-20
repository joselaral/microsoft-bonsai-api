import csv
import math
import random
from collections import deque
from pymgrid import MicrogridGenerator
from .predict import train_and_save_model, load_model, predict_with_model


MAX_BATTERY_CHARGE = 58048.89

class MicrogridSim:
    """
    Model to simulate a microgrid.
    """
    def __init__(self, enable_ml_predicition):
        # we will use the 4th microgrid architecture in pymgrid25 benchmark set
        # with PV, battery, load and grid
        generator = MicrogridGenerator.MicrogridGenerator(nb_microgrid=25)
        pymgrid25 = generator.load('pymgrid25')
        self.enable_ml_predicition = enable_ml_predicition
        self.mg = pymgrid25.microgrids[4]
        self.mg._grid_price_import[self.mg._grid_price_import == 0.11] = 0.2 # increase the high grid price value
        self.prev_state = {}
        self.state = {}
        # Prediction attribute that holds the next four state predictions
        self.prediction_dict = {
            "co2_predict": deque([0, 0, 0, 0], maxlen=4),
            "co2_predict_average": 0,
            "load_predict": deque([0, 0, 0, 0], maxlen=4),
            "load_predict_average": 0,
            "pv_predict":deque([0, 0, 0, 0], maxlen=4),
            "pv_predict_average": 0
        }

        self.control_dict = {}
        self.cost_loss_load = 10 # penalty coefficient for not meeting the load
        self.cost_overgeneration = 1 # penalty coefficient for over-generating
        self.cost_battery = 0.02 # default cost of utilizing the battery
        self.cost_co2 = 0.1 # coefficient penalizing for CO2 usage
        self.episode_length = 24*2
        self.starting_charge = 0  # default starting charge
        self.load_ts = None
        self.pv_ts = None
        self.starting_time_step = None
    
    def get_state(self):
        """
        Returns a dictionary with the state variables of the sim along with the microgrid's cost.
        """
        self.prev_state = self.state
        state = self.mg.get_updated_values().copy()
        try:
            co2_emission = self.mg.get_co2()
            state["co2_emission"] = co2_emission
        except IndexError:
            state["co2_emission"] = 0
        state["pv_load_diff"] = state["pv"] - state["load"]
        costs = self.get_calculated_cost()
        state["cost"] = costs[0]
        state["normalized_cost"] = costs[1]
        # initialize the previous states/actions to calculate costs (initializes with 0 if first iteration). 
        state["prev_load"] = self.prev_state.get("load", 0)
        state["prev_pv"] = self.prev_state.get("pv", 0)
        state["prev_grid_price_import"] = self.prev_state.get("grid_price_import", 0)
        state["prev_grid_co2"] = self.prev_state.get("grid_co2", 0)
        state["prev_action_grid_import"] = self.control_dict.get("grid_import", 0)
        state["cost_co2"] = self.cost_co2
       
        if self.load_ts is not None:
            state["sum_load"] = self.load_ts.sum()
        else:
            state["sum_load"] = 1
        
        # Calculate average prediction
        self.state["prediction"] = self.prediction_dict


        if self.enable_ml_predicition:
            prediction = self.get_state_prediction(state["hour"], state["load"], state["grid_co2"], state["pv"])
            state["co2_predict"] = prediction["co2_predict"][-1]
            state["co2_predict_average"] = get_average(prediction["co2_predict"])
            state["load_predict"] = prediction["load_predict"][-1]
            state["load_predict_average"] = get_average(prediction["load_predict"])
            state["pv_predict"] = prediction["pv_predict"][-1]
            state["pv_predict_average"] = get_average(prediction["pv_predict"])
        else:
            state["co2_predict"] = self.prediction_dict["co2_predict"]
            state["co2_predict_average"] = 0
            state["load_predict"] = self.prediction_dict["load_predict"]
            state["load_predict_average"] = 5000
            state["pv_predict"] = self.prediction_dict["pv_predict"]
            state["pv_predict_average"] = 0
        
        self.state = state

        return state
    
    def episode_start(self, config):
        """
        Resets the sim state and re-initializes the sim with the config parameters.
        """

        # First we reset all the parameters
        self.prev_state = {}
        self.state = {}
        self.control_dict = {}
        self.starting_time_step = config["starting_time_step"]

        if "cost_loss_load" in config:
            self.cost_loss_load = config["cost_loss_load"]
        if "cost_overgeneration" in config:
            self.cost_overgeneration = config["cost_overgeneration"]
        if "cost_battery" in config:
            self.cost_battery = config["cost_battery"]
        if "cost_co2" in config:
            self.cost_co2 = config["cost_co2"]
        if "episode_length" in config:
            self.episode_length = config["episode_length"]    
        if "starting_charge" in config:
            self.starting_charge = config["starting_charge"]     
        
        # reset the sim
        self.mg.reset()
        # The sim originally resets all the data to the 0th index, we would like to reset the sim to any time index
        # TODO: Derive a new class and override the reset instead of accessing the private variables of the object
        self.mg._tracking_timestep = config["starting_time_step"]
        self.mg.update_variables()

        self.mg._df_record_state["load"] = [self.mg.load]
        self.mg._df_record_state["hour"] = [self.mg._tracking_timestep % 24]
        self.mg._df_record_state["pv"] = [self.mg.pv]
        self.mg._df_record_state["grid_status"] = [self.mg.grid.status]
        self.mg._df_record_state["grid_co2"] = [self.mg.grid.co2]
        self.mg._df_record_state["grid_price_import"] = [self.mg.grid.price_import]
        self.mg._df_record_state["grid_price_export"] = [self.mg.grid.price_export]
        
        # Initialize battery charge based on configuration
        self.initialize_battery_charge()

        self.load_ts = self.mg._load_ts.iloc[self.mg._tracking_timestep:self.mg._tracking_timestep + self.episode_length].values.flatten()
        self.pv_ts = self.mg._pv_ts.iloc[self.mg._tracking_timestep:self.mg._tracking_timestep + self.episode_length].values.flatten()   

    def initialize_battery_charge(self):
        """
        Changes the default zero charge from Pymgrid configuration to an initial charge. 

        """

        capa_to_charge = MAX_BATTERY_CHARGE - self.starting_charge 

        self.mg._df_record_state["capa_to_discharge"] = [self.starting_charge]
        self.mg._df_record_state["capa_to_charge"] = [capa_to_charge]

    def initialize_prediction(self):
        """
        Initialize ml predicition values to zero
        """

    def episode_step(self, action):
        control_dict = {"battery_charge": 0,
            "battery_discharge": 0,
            "grid_import": 0,
            "grid_export":0,
            "pv_consummed": 0,
            }
        
        # if battery_power > 0 it means charge the battery, else it means discharge the battery
        if action["battery_power"] > 0:
            control_dict["battery_charge"] = abs(action["battery_power"])
        else:
            control_dict["battery_discharge"] = abs(action["battery_power"])
        
        if "pv_to_consume" in action:
            control_dict["pv_consummed"] = action["pv_to_consume"]
        else:
            control_dict = self.get_pv_to_consume_power(control_dict)

        # if grid_power > 0 it means import from the grid, else it means export to the grid
        if "grid_power" in action:
            if action["grid_power"] > 0:
                control_dict["grid_import"] = abs(action["grid_power"])
            else:
                control_dict["grid_export"] = abs(action["grid_power"])
        else:
            control_dict = self.get_grid_power(control_dict)
        
        _ = self.mg.run(control_dict)
        self.control_dict = control_dict

    def get_grid_power(self, control_dict):
        """
        Calculates how much power to import/export from/to the grid to meet the load.
        """
        state = self.mg.get_updated_values().copy()
        load = state["load"]
        pv = state["pv"]
        capa_to_charge = state["capa_to_charge"]
        capa_to_discharge = state["capa_to_discharge"]
        grid_status = state["grid_status"]
        if grid_status == 0:
            # if there is a blackout we can't import from / export to grid
            control_dict["grid_import"] = 0
            control_dict["grid_export"] = 0
            return control_dict
        
        actual_pv_to_consume = min(pv, control_dict["pv_consummed"])
        actual_battery_charge = min(capa_to_charge, control_dict["battery_charge"])
        actual_battery_discharge = min(capa_to_discharge, control_dict["battery_discharge"]) 

        grid_power = load - (-actual_battery_charge + actual_battery_discharge + actual_pv_to_consume)

        if grid_power > 0:
            control_dict["grid_import"] = grid_power
            control_dict["grid_export"] = 0
        else:
            control_dict["grid_import"] = 0
            control_dict["grid_export"] = -grid_power
        return control_dict
    
    def get_pv_to_consume_power(self, control_dict):
        """
        Calculates how much power we should consume from the grid to meet the load
        """
        state = self.mg.get_updated_values().copy()
        load = state["load"]
        pv = state["pv"]
        capa_to_charge = state["capa_to_charge"]
        capa_to_discharge = state["capa_to_discharge"]
        grid_status = state["grid_status"]

        actual_battery_charge = min(capa_to_charge, control_dict["battery_charge"])
        actual_battery_discharge = min(capa_to_discharge, control_dict["battery_discharge"]) 

        total_load = load + actual_battery_charge - actual_battery_discharge
        # if there is any power deficiency, we cover as much of it as we can from the PV
        # TODO: This calculation doesn't account for the possibility to sell to the grid
        if total_load <= 0:
            control_dict["pv_consummed"] = 0
        else:
            control_dict["pv_consummed"] = min(total_load, pv)
        return control_dict
    
    def get_calculated_cost(self):
        """
        Calculates the cost of running the grid and normalizes it. Returns both the original cost and the normalized cost.
        """
        cost_loss_load = self.cost_loss_load
        cost_overgeneration = self.cost_overgeneration
        cost_battery = self.cost_battery
        cost_co2 = self.cost_co2

        # we haven't incurred any cost yet if we haven't taken an action
        if self.control_dict == {}:
            return 0, 0
        
        cost = cost_loss_load * self.control_dict["loss_load"] + cost_overgeneration * self.control_dict["overgeneration"] \
            + self.prev_state["grid_price_import"] * self.control_dict["grid_import"] \
            + self.prev_state["grid_price_export"] * self.control_dict["grid_export"] \
            + (self.control_dict["battery_charge"] + self.control_dict["battery_discharge"]) * cost_battery \
            + cost_co2 * self.control_dict["grid_import"] * self.prev_state["grid_co2"]
        
        normalized_cost = (cost - self.grid_cost_without_battery()) / self.load_ts.sum()
        return cost, normalized_cost

    def grid_cost_without_battery(self):
        """
        Calculates the cost of the grid in one iteration if we didn't have a battery.
        """
        load_pv = self.prev_state["load"] - self.prev_state["pv"]
        if load_pv <= 0:
            return 0
        grid_cost = self.prev_state["grid_price_import"] * load_pv \
                + self.cost_co2 * load_pv * self.prev_state["grid_co2"]
        return grid_cost
    
    def get_state_prediction(self, hour, load, grid_co2, pv, prediction_range=4, retrain_model=False):
        """
        Calcuates an average prediction of PV generation, CO2 cost and grid load based on historical data. 
        Args:
            hour(int):              Current hour of each day (0-24 hours)
            load(float):            Grid load
            grid_co2(flot):         CO2 generated by grid
            PV(float):              Current PV production from microgrid
            prediction_range(int):  Number of hour steps to predict
            retrain_model(bool):    Retrain ML model for prediction
        """

        learning_data_path = 'sim/supervised_learning_data.csv'
        trained_model_path = 'sim/predict_co2_load_pv.pkl'

        if retrain_model:
            train_and_save_model(learning_data_path, trained_model_path, False)

        model = load_model(trained_model_path)
        prediction = self.prediction_dict

        for i in range(0, prediction_range):
            state_params = [[hour, load, grid_co2, pv]]
            predict_iter = predict_with_model(model, state_params)
 
            for key in predict_iter.keys():         
                # print(f'item: {item}, key: {key}')   
                if key == "pv_predict":
                    # Predict zero pv generation at night time
                    if hour < 6 and hour > 20:
                        prediction[key].append(0)
                    # Else, set correct value
                    else:
                        prediction[key].append(max(predict_iter[key], 0)) 
                elif key=="load_predict":
                    # Set minimum load prediction to 5000 #TODO Units
                    prediction[key].append(max(predict_iter[key], 5000)) 
                else:
                    # print(f'Current CO2 Prediction: {predict_iter[key]}')
                    prediction[key].append(predict_iter[key])

            # Update values for next prediction                    
            hour = (hour+1) % 24
            load = predict_iter["load_predict"]
            grid_co2 = predict_iter["co2_predict"]
            pv = predict_iter["pv_predict"]
    
        return prediction

def get_average(data_list):
    
    avg = sum(data_list)/len(data_list)
    
    return avg 

