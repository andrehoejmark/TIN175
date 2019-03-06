import matplotlib.pyplot as plt
import numpy as np


def savePlotToFile(name, folder):
    plt.savefig("%s/%s.png" % (folder, name), bbox_inches="tight")
    plt.clf()


def plot_comparison(start_idx, length=100, data=None, headers=None, hyperparams=None):

    """Plot the predicted and true output-signals."""

    inp = data.in_train_scaled
    out_true = data.out_train

    end_idx = len(inp)
    inp = inp[(end_idx-length):end_idx]

    out_true = out_true[(end_idx-length):end_idx]
    inp = np.expand_dims(inp, axis=0)

    out_pred = data.model.predict(inp)
    out_pred_rescaled = data.out_scaler.inverse_transform(out_pred[0])

    for signal in range(0, len(headers)):
        signal_pred = out_pred_rescaled[:, signal]
        signal_true = out_true[:, signal]
        plt.figure(figsize=(15, 5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        header = headers[signal]
        plt.ylabel(header)
        plt.legend()
        if hyperparams.show_output_after_sim:
            plt.show()
        else:
            savePlotToFile("output_plot_%s%s" % (hyperparams.plot_output_sub_name, header),
                           hyperparams.output_folder)
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
        for out in outs:
          signal_pred = out[:, signal]
          plt.plot(signal_pred, label= "pred %s" % data.id )
        header = headers[signal]
        plt.ylabel(header)
        plt.legend()
        savePlotToFile("multi_graph.png", "./")
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


def plot_training(y_label=None, train_loss=None, val_loss=None, title=None, sid=None, use_log=False):
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
