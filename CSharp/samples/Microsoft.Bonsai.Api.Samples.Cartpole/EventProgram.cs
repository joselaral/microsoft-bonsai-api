﻿using Microsoft.Bonsai.SimulatorApi.Client;
using Microsoft.Bonsai.SimulatorApi.Models;
using System;
using Newtonsoft.Json;
using System.Net.Http;
using System.Threading;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;

namespace Microsoft.Bonsai.Api.Samples.Cartpole
{
    class EventProgram
    {
        //the cartpole model
        private static Model model = new Model();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="args"></param>
        /// <remarks>
        /// You can run the example by itself, or pass the predict and http://localhost:5000/v1/prediction url 
        /// </remarks>
        public static async Task RunMain(string[] args)
        {
            if (args.Length == 0)
                await TrainAndAssess();
            else
            {
                if (args[0] == "predict")
                    await RunPrediction(args[1]);//assumes the second value is the URL
            }
        }

        /// <summary>
        /// Run the Train or Assessment loop
        /// </summary>
        private static async Task TrainAndAssess()
        {
            
            String workspaceName = GetWorkspace();
            String accessKey = GetAccessKey();

            BonsaiClientConfig bcConfig = new BonsaiClientConfig(workspaceName, accessKey);
            BonsaiClient client = new BonsaiClient(bcConfig);
            
            client.EpisodeStart += (o,e) => 
            { 
                Config config = new Config(); 
                model.Start(config); 
            };

            client.EpisodeStep += (o, e) => 
            {
                Action action = new Action();
                dynamic stepAction = e.Action;
                action.Command = stepAction.command.Value;

                // move the model forward
                model.Step(action);

            };

            client.EpisodeFinish += (o,e) => { Console.WriteLine(e.EpisodeFinishReason.ToString()); };

            await client.Connect(model);

            Console.ReadLine(); //hold open the console
        }

        private static async Task RunPrediction(string exportedBrainUrl)
        {
            BonsaiClientConfig bcConfig;

            bool useCredentials = true;

            if (useCredentials)
            {
                AzureADAppDetails aad = JsonConvert.DeserializeObject<AzureADAppDetails>(File.ReadAllText("aad.json"));
                bcConfig = new BonsaiClientConfig(aad.OAuthTokenUrl, aad.ApplicationId, aad.ClientSecret, exportedBrainUrl);
            }
            else
            {
                bcConfig = new BonsaiClientConfig(exportedBrainUrl);
            }
            
            BonsaiClient bonsaiClient = new BonsaiClient(bcConfig);

            bonsaiClient.EpisodeStep += (o, e) =>
            {
                Action action = new Action();
                dynamic stepAction = e.Action;
                action.Command = stepAction.command.Value;

                Console.WriteLine(action.Command);

                // move the model forward
                model.Step(action);
            };

            await bonsaiClient.Connect(model);
        }

        private static string GetWorkspace()
        {

            if (Environment.GetEnvironmentVariable("SIM_WORKSPACE") != null)
            {
                return Environment.GetEnvironmentVariable("SIM_WORKSPACE");
            }

            //fill in your Bonsai workspace ID
            return "<your_bonsai_workspace_id>";
        }

        private static String GetAccessKey()
        {

            if (Environment.GetEnvironmentVariable("SIM_ACCESS_KEY") != null)
            {
                return Environment.GetEnvironmentVariable("SIM_ACCESS_KEY");
            }

            //fill in your Bonsai access key
            return "<your_bonsai_access_key>";
        }
    }

    [JsonObject]
    public class AzureADAppDetails
    {
        [JsonProperty(PropertyName = "appId", Required = Required.Always)]
        public string ApplicationId { get; set; }

        [JsonProperty(PropertyName = "clientSecret", Required = Required.Always)]
        public string ClientSecret { get; set; }

        [JsonProperty(PropertyName = "tokenAddress", Required = Required.Always)]
        public string OAuthTokenUrl { get; set; }
    }
}
