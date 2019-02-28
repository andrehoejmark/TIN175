import matplotlib.pyplot as plt
import numpy as np


def savePlotToFile(name, folder):
    plt.savefig("%s/%s.png" % (folder, name), bbox_inches="tight")


def plot_comparison(start_idx, length=100, data=None, headers=None, hyperparams=None):

    """Plot the predicted and true output-signals."""

    inp = data.in_test
    out_true = data.out_test

    end_idx = start_idx + length
    inp = inp[start_idx:end_idx]

    out_true = out_true[start_idx:end_idx]
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
