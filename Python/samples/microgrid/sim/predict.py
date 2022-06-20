import pickle
import pandas as pd
from sklearn import ensemble
#Flags, set as you see fit!
verbose=True
I_want_to_train = False
model_path = 'sim/predict_co2_load_pv.pkl'
learning_data_path = 'sim/supervised_learning_data.csv'

# if(verbose):
#     print("\n ---------------------------------\n| SUPER MEGA MICROGRID PREDICTION |\n ---------------------------------")

def train_and_save_model(data_file_path,save_model_path,verbose = False):
    if(verbose): 
            print("Training model from data in '",data_file_path,"'")
    
    training_data = pd.DataFrame(pd.read_csv(data_file_path))
    
    #You will notice I am using the PREV load values and CURRENT hour to create the prediciton
    #This is instead of manipulating the data to have a 'next' value in it. SMORT
    x = pd.DataFrame(training_data[['state_hour','state_prev_load','state_prev_grid_co2',"state_prev_pv"]].to_numpy())
    y_co2 = pd.DataFrame(training_data[['state_grid_co2']])#We are training to predict CO2 at state_hour based on prev. 
    y_load = pd.DataFrame(training_data[['state_load']])#Adding load
    y_pv = pd.DataFrame(training_data[['state_pv']])#Adding PV
    
    # Define parameters. Mostly defaults from the example
    params = {
        "n_estimators": 500,
        "max_depth": 5,     #Changing this from default (3) to 5 helped accuracy
        "min_samples_split": 5,
        "learning_rate": 0.01,#Changing this only made it worse
        "loss": "squared_error",#It's a regression, squared error suits me. Who needs a neural network?
    }

    r_co2 = ensemble.GradientBoostingRegressor(**params)
    r_co2.fit(x,y_co2.values.ravel())
    r_load = ensemble.GradientBoostingRegressor(**params)
    r_load.fit(x,y_load.values.ravel())
    r_pv = ensemble.GradientBoostingRegressor(**params)
    r_pv.fit(x,y_pv.values.ravel())
    if(verbose): print("Training complete!")
    model = {
        'co2_predict':r_co2,
        'load_predict':r_load,
        'pv_predict':r_pv
    }
    
    if(verbose): 
        print("Saving model to '",save_model_path,"'")
    with open(save_model_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    if(verbose): print("All work complete, sire.\n")

def load_model(load_model_path,verbose = False):
    """
    Load pckl model
    """
    if(verbose): 
        print("Loading model from '",model_path,"'\n")
    with open(load_model_path, 'rb') as f:
        return pickle.load(f)
    # TODO Does this need to be closed?

#This one uses the model to return a value
def predict_with_model(model,features,verbose = False):
    """
    Use trained model to return precition values
    """
    if(verbose): print("Predicting with '",features,"'")
    next_hour_co2_will_be = model['co2'].predict(features)[0] # note that it is giving back a list, so we want the VALUE of the first one for H+1
    next_hour_load_will_be = model['load'].predict(features)[0] # note that it is giving back a list, so we want the VALUE of the first one for H+1
    next_hour_pv_will_be = model['pv'].predict(features)[0] # note that it is giving back a list, so we want the VALUE of the first one for H+1
   
    # Build state structure to be added to the simulation state
    results = {
        'co2_predict':next_hour_co2_will_be,
        'load_predict':next_hour_load_will_be,
        'pv_predict':next_hour_pv_will_be
    }
    if(verbose): 
        print("I don't like musicals any more. And results will be", results, " at ",features[0][0],"\n")
    return results


# Only run this if you want a new pickle predict.

# if(I_want_to_train): 
#     train_and_save_model(learning_data_path, model_path,verbose)
# #Load the model for fun prediction times
# model = load_model(model_path,verbose)

# #You will need some features, here are some example ones
# features = [[#yes, double sqaure braces are needed.
#     15,#the hour of the data in 24 hour notation. Each iteraction has an hour value. So state_hour+1 is expected for next iteration prediction
#     10000,#this is the CURRENT state_hour state_load. You will have this from the simulation
#     0.20,#this is the CURRENT state_hour state_grid_co2. You will have this from the simulation
#     5000,#this is the CURRENT state_hour state_pv. You will have this from the simulation/
# ]]

# #It returns a single floating point number
# results = predict_with_model(model, features,verbose)
# print(results['co2_predict'])#Which I print regardless of verbosity
# print(results['load_predict'])#Which I print regardless of verbosity
# print(results['pv_predict'])#Which I print regardless of verbosity

# if(verbose):
#     print("\nDone!")