
#
# Author : Felix Hulthén.
#

"""
A small (and crude) command interpreter for performing SMHI dataset loading,
automated testing and visualisation graph generation. You can type the command
'help' if you want instructions on how to use the command interpreter.
"""

import sys
import os
from src.utils.loadCSV import loadInterpolatedCSV, loadConfigCSV
from src.utils.params import NetowrkHyperparameterConfig, convConfigToHyperparams
from src.utils.plots import plotColumn, plot_comparison, plot_multi_comparison
from src.utils.simulation import SimulationData
import numpy

# Global parameters.

DATASET_CSV_FILE = "src/utils/example_data.csv"
CONFIG_LOCATION = "src/example_hyperparameters.csv"
LOADED_CONFIG_FILE = None

"""
OUTPUT_TARGET_HEADERS : You specify (exactly) the names of the output signals
  from the loaded SMHI CSV datasets that are supposed to be an output signal
  (e.g. part of the forecast values in the future). You can specify multiple;
  however, the order of them will depend on how they are listed in the CSV file.
"""
# OUTPUT_TARGET_HEADERS = ["gothenburg temperature", "gothenburg wind direction", "gothenburg wind speed", "gothenburg pressure"]
OUTPUT_TARGET_HEADERS=["gothenburg temperature"]

NUM_TARGET_COLUMNS=len(OUTPUT_TARGET_HEADERS)


def loadConfigFile(name):
    global LOADED_CONFIG_FILE
    LOADED_CONFIG_FILE=loadConfigCSV(CONFIG_LOCATION)


def genCompareGraphs(ids, names):
    if LOADED_CONFIG_FILE:
        cid=0
        conf=None
        nets=[]
        net_len=500
        while True:
            conf=LOADED_CONFIG_FILE.getConfig(cid)
            if conf:
                ocid=cid
                cid=cid + 1
                if not (str(ocid) in ids):
                    continue
                hyperparams=convConfigToHyperparams(conf)
                if not os.path.isdir(hyperparams.output_folder):
                    try:
                        os.mkdir(hyperparams.output_folder)
                    except OSError:
                        print(
                            "Unable to create output folder \"%s\" (you might have to remove the old folder)"%hyperparams.output_folder)
                        sys.exit(1)
                hyperparams.output_folder=hyperparams.output_folder + ("ID_%s/"%conf.id)
                if not os.path.isdir(hyperparams.output_folder):
                    try:
                        os.mkdir(hyperparams.output_folder)
                    except OSError:
                        print(
                            "Unable to create modified id output folder \"%s\" (you might have to remove the old files)"%hyperparams.output_folder)
                        sys.exit(1)
                net=SimulationData(hyperparams, OUTPUT_TARGET_HEADERS, data=data)
                net.splitAndNormalize()
                net.createValidationData()
                net.setupAndPerformSimulation(hyperparams)
                net_len=net.num_train
                net.id=names[str(ocid)]
                try:
                    net.model.load_weights(net.path_checkpoint)
                except Exception as error:
                    print("Error trying to load checkpoint: %s"%error)
                    return False
                nets.append(net)
            else:
                print("Ended config run at cid: %d"%cid)
                break
        plot_multi_comparison(start_idx=0, length=net_len, datas=nets, headers=OUTPUT_TARGET_HEADERS)
    else:
        print("No config CSV file has been loaded.")
        return False
    return True


def runConfigs():
    if LOADED_CONFIG_FILE:
        cid=0
        conf=None

        while True:
            conf=LOADED_CONFIG_FILE.getConfig(cid)
            if conf:
                hyperparams=convConfigToHyperparams(conf)
                if not os.path.isdir(hyperparams.output_folder):
                    try:
                        os.mkdir(hyperparams.output_folder)
                    except OSError:
                        print(
                            "Unable to create output folder \"%s\" (you might have to remove the old folder)"%hyperparams.output_folder)
                        sys.exit(1)
                hyperparams.output_folder=hyperparams.output_folder + ("ID_%s/"%conf.id)
                if not os.path.isdir(hyperparams.output_folder):
                    try:
                        os.mkdir(hyperparams.output_folder)
                    except OSError:
                        print(
                            "Unable to create modified id output folder \"%s\" (you might have to remove the old files)"%hyperparams.output_folder)
                        sys.exit(1)
                    hyperparams.show_output_after_sim=False
                    # Print config to file
                    conf_str=hyperparams.getConfigString()
                    file_object=open(hyperparams.output_folder + "config.txt", "w")
                    file_object.write(conf_str + "\n")
                    file_object.flush()
                    file_object.close()
                    performSimulation(cmd="simulate", hyperparams=hyperparams, data=data)
                else:
                    print(
                        "!! \n\nWARNING: SKIPPING ID %s (iteration %d) AS THE LAST SIMULATION FILES ARE STILL FOUND! PLEASE MOVE THEM AND KEEP THEM SAFE! \n\n!!"%(
                            conf.id, cid))
                cid=cid + 1

            else:
                print("Ended config run at cid: %d"%cid)
                break

    else:
        print("No config CSV file has been loaded.")
        return False
    return True


def printHelp(cmd="null"):
    print(
        "Dunno what \"{cmd}\" is. You can use 'view_input_plots', 'save_input_plots', 'run_config', 'agg' or 'simulate' at the "
        "moment.".format(cmd=cmd))


def performSimulation(cmd="", hyperparams=None, data=None):
    """Comments"""
    global OUTPUT_TARGET_HEADERS
    # OUTPUT_TARGET_HEADERS = OUTPUT_TARGET_HEADERS[:hyperparams.outputs]

    data=SimulationData(hyperparams, OUTPUT_TARGET_HEADERS, data=data)

    # Parse command.
    if cmd=="view_input_plots" or cmd=="save_input_plots":
        units=["Temperature [C]", "Wind direction [deg]", "Wind speed [m/s]", "Pressure [kilo Pascal]"]

        for i in range(2, len(data.smhi_in.header)):
            plotColumn(data.smhi_in.header[i], units[(i - 2)%len(units)], data.in_values, i + 4, cmd=cmd)

    elif cmd=="simulate":
        print("Reading %d rows of data from the dataset."%hyperparams.read_sequence_length)
        data.splitAndNormalize()
        data.createValidationData()
        data.setupAndPerformSimulation(hyperparams)
        data.performSimulation(hyperparams)

        try:
            data.model.load_weights(data.path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint: %s"%error)
            return False

        result=data.model.evaluate(
            x=numpy.expand_dims(data.in_test_scaled, axis=0),
            y=numpy.expand_dims(data.out_test_scaled, axis=0))

        print("loss (test-set):", result)

        # Render comparison graphs.
        plot_comparison(0, length=(24*60), data=data, headers=OUTPUT_TARGET_HEADERS, hyperparams=hyperparams)
    else:
        printHelp(cmd=cmd)
        return False

    return True


"""

Actual program loop

"""

# Load data from CSV files.
data=loadInterpolatedCSV(DATASET_CSV_FILE)

if not data:
    sys.exit(1)

while True:
    print("Input your command (or help): ")
    cmd=input()
    if cmd=="exit":
        break
    elif cmd=="help":
        printHelp(cmd=cmd)
    elif cmd[0:3]=="agg":
        args=cmd[4:]
        if args:
            loadConfigFile(CONFIG_LOCATION)
            pairs=args.split(" ")
            ids=[]
            names={}
            print(pairs)
            for pair in pairs:
                print(pair.split(","))
                [id, name]=pair.split(",")
                ids.append(id)
                names[id]=name
            genCompareGraphs(ids, names)
        else:
            print("Missing which pairs \"id,name\"")
    elif cmd=="run_config":
        loadConfigFile(CONFIG_LOCATION)
        runConfigs()
    else:
        conf=NetowrkHyperparameterConfig()
        conf.keep_city_as_input=True  # Remove (or set to 'False') ths line in order to not depend on the non-timeshifted output signals as input.
        performSimulation(cmd=cmd, hyperparams=conf, data=data)
