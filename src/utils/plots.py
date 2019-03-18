
#
# Authors: Rasmus Claesen and Felix Hulthén.
#

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def savePlotToFile(name, folder):
    plt.savefig("%s/%s.png" % (folder, name), bbox_inches="tight")
    plt.clf()


def plot_comparison_cpy(start_idx, length=100, data=None, headers=None, hyperparams=None):

    """Plot the predicted and true output-signals."""

    inp = data.in_val_scaled
    out_true = data.out_val

    inp2 = data.out_train

    end_idx = len(inp)
    inp = inp[(end_idx-length):end_idx]

    out_true = out_true[(end_idx-length):end_idx]
    out_actual = data.out_values[(end_idx-length):end_idx]
    inp = np.expand_dims(inp, axis=0)

    out_pred = data.model.predict(inp)
    out_pred_rescaled = data.out_scaler.inverse_transform(out_pred[0])

    for signal in range(0, len(headers)):
        signal_pred = out_pred_rescaled[:, signal]
        signal_true = out_true[:, signal]
        plt.figure(figsize=(15, 5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        plt.plot(out_actual, label='out_actual')
        plt.plot()
        header = headers[signal]
        plt.ylabel(header)
        plt.legend()
        if hyperparams.show_output_after_sim:
            plt.show()
        else:
            savePlotToFile("output_plot_%s%s" % (hyperparams.plot_output_sub_name, header),
                           hyperparams.output_folder)
        np.savetxt(hyperparams.output_folder + "out_pred_72.csv", signal_pred[:, signal], delimiter=",")
        np.savetxt(hyperparams.output_folder + "out_true_72.csv", signal_true[:, signal], delimiter=",")
        plt.clf()


def plot_comparison(start_idx, length=100, data=None, headers=None, hyperparams=None):

    """Plot the predicted and true output-signals."""

    inp=data.in_val_scaled
    out_true=data.out_val

    end_idx=len(inp)
    inp=inp[(end_idx - length):end_idx]

    out_true=out_true[(end_idx - length):end_idx]
    inp=np.expand_dims(inp, axis=0)

    out_pred=data.model.predict(inp)
    out_pred_rescaled=data.out_scaler.inverse_transform(out_pred[0])

    for signal in range(0, len(headers)):
        signal_pred=out_pred_rescaled[:, signal]
        signal_true=out_true[:, signal]
        plt.figure(figsize=(15, 5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        header=headers[signal]
        plt.ylabel(header)
        plt.legend()
        if hyperparams.show_output_after_sim:
            plt.show()
        else:
            savePlotToFile("output_plot_%s%s"%(hyperparams.plot_output_sub_name, header),
                           hyperparams.output_folder)
            np.savetxt(hyperparams.output_folder + "out_pred_72.csv", signal_pred, delimiter=",")
            np.savetxt(hyperparams.output_folder + "out_true_72.csv", signal_true, delimiter=",")
        plt.clf()


def plot_multi_comparison(start_idx, length=100, datas=[], headers=None):

    """Plot the predicted and true output-signals."""
    if len(datas) <= 0:
      return
    
    # They all should hold the same training data anyway, so
    # just pick the first one.
    end_idx = start_idx + length
    out_true = datas[0].out_train
    out_true = out_true[start_idx:end_idx]

    outs = []

    for data in datas:
      inp = data.in_train_scaled
      inp = inp[start_idx:end_idx]
      inp = np.expand_dims(inp, axis=0)
      out_pred = data.model.predict(inp)
      out_pred_rescaled = data.out_scaler.inverse_transform(out_pred[0])
      outs.append(out_pred_rescaled)

    for signal in range(0, len(headers)):
        signal_true = out_true[:, signal]
        plt.figure(figsize=(15, 5))
        plt.plot(signal_true, label='true')
        idxn = 0
        for out in outs:
          signal_pred = out[:, signal]
          plt.plot(signal_pred, label= "pred %s" % datas[idxn].id )
          idxn = idxn + 1
        header = headers[signal]
        plt.ylabel(header)
        plt.legend()
        savePlotToFile("multi graph - %s" % header, "./")
        plt.clf()


def plotColumn(header, y_axis_unit, rows, column_idx, cmd=None):
    column = []
    for row in rows:
        column.append(row[column_idx])
    plt.figure(figsize=(15, 5))
    plt.plot(column, label=header)
    plt.ylabel(y_axis_unit)
    plt.legend()
    if cmd == "view_input_plots":
        plt.show()
    else:
        savePlotToFile("input_plot_%s" % header, "./")
    plt.clf()


def plot_training(y_label=None, train_loss=None, val_loss=None, title="", sid=None, use_log=False):
    """
    :param y_label: y-axis label
    :param train_loss: Data of the training loss
    :param val_loss: Data of the validation loss
    :param title: Title of plot
    :param sid: Simulation ID to plot
    :param use_log: Use log scale for y-axis
    :return:
    """
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    if use_log:
        plt.yscale("log")
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.title(title)
    plt.legend()
    #  Storing plots in corresponding ID_{simulation id} folder
    if use_log:
        savePlotToFile("training_plot_%s_log" % sid, "./../simulation/ID_%s/" % sid)
    else:
        savePlotToFile("training_plot_%s" % sid, "./../simulation/ID_%s/" % sid)
    plt.clf()


def plot_double_csv(length=100):
    """
    Overall a bit fuzzy function to create some specific plots for the essay.
    Compare two predictions on a shared X-axis from saved log files.
    :param length: No. of hours for desired range the of X-axis.
    :return:
    """

    out_pred_lstm=np.genfromtxt("out_pred_lstm.csv", delimiter=",")
    out_true_lstm=np.genfromtxt("out_true_lstm.csv", delimiter=",")
    out_pred_gru=np.genfromtxt("out_pred_gru.csv", delimiter=",")
    out_true_gru=np.genfromtxt("out_true_gru.csv", delimiter=",")

    matplotlib.rc('font', size=20)
    #matplotlib.rc('axes', titlesize=18)
    plt.figure(figsize=(18, 6))
    #fig, ax = plt.subplots(1, 1, sharex = True)


    #ax[0].plot(true_24[100:length], label="true")
    #ax[0].plot(pred_24[100:length], label="predicted")
    #ax[0].legend()

    plt.plot(out_true_lstm[100:length], label="true")
    plt.plot(out_pred_lstm[100:length], label="predicted")
    plt.legend()
    #ax[1].plot(true_72[100:length])
    #ax[1].plot(pred_72[100:length])


    plt.ylabel("Temperature (°C)")
    #ax[1].set_ylabel("GRU")
    plt.xlabel("Hours")
    # plt.subplots_adjust(hspace=0)
    savePlotToFile("pred_plot", ".")
    plt.clf()


def train_comparison(y_label=None, train_loss_a=None, val_loss_a=None, train_loss_b=None, val_loss_b=None,
                     title=None):
    """
    Compares training from to given networks.
    Like previous function the ambition was dynamic but using hardcoded values to create plots for the essay.
    See ~plot_training. 
    """

    matplotlib.rc('font', size=14)
    plt.figure(figsize=(15, 5))
    fig, ax=plt.subplots(2, 1, sharex=True, figsize=(10,5))

    ax[0].plot(train_loss_a, label="LSTM")
    ax[0].plot(train_loss_b, label="GRU")
    ax[0].legend()
    ax[1].plot(val_loss_a)
    ax[1].plot(val_loss_b)

    ax[0].set_ylabel("Training " + y_label)
    ax[1].set_ylabel("Validation " + y_label)
    ax[1].set_xlabel("Epoch")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_title(title)
    plt.subplots_adjust(hspace=0)
    savePlotToFile("Comparison plot", ".")
    plt.clf()
