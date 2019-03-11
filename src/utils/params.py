class NetowrkHyperparameterConfig:
    def __init__(self):
        self.read_sequence_length = 24 * 7 * 3  # How much data to read from the dataset.
        self.start_learning_rate = 1e-3  # The initial learning rate.
        self.loss_factor = 0.1  # The change of learning rate when unsuccessful in improving the result.
        self.min_learning_rate = 1e-4  # The absolute minimum learning rate.
        self.learning_patience = 0  # The patience to have when not able to improve the learning rate (0 is pretty good to avoid overfitting).
        self.num_epochs = 10
        self.steps_per_epoch = 5
        self.early_stop_patience = 0
        self.warmup_steps = 10  # Number to steps before exiting the warm up phase.
        self.num_gated_reoccurring_units = 512
        self.activation_function = 'sigmoid'
        self.time_shift_in_hours = 12
        self.training_splitting = 0.9
        self.show_output_after_sim = True
        self.read_batch_size = 16
        self.network_type = "GRU"
        self.output_folder = "simulation/"
        self.plot_output_sub_name = ""
        self.outputs = 4
        self.use_full_timestamp = True
        self.keep_city_as_input = False

    def getConfigString(self):
        return ("""
# Current config
Read sequence length = %d
Start learning rate = %f
Loss factor = %f
Min learning rate = %f
Learning patience = %d
Num epochs = %d
Steps per epoch = %d
Early stop patience = %d
Warpup steps = %d
Num GRU = %d
Activation function = %s
Time shift in hours = %d
Training splitting = %f
Show output after sim = %d
Read batch size = %d
Network type = %s
Output folder = %s
Plot output sub name = %s
Num outputs = %d (deprecated)
Use full timestamp : %s
Use city as input : %s
""" % (self.read_sequence_length, self.start_learning_rate, self.loss_factor, self.min_learning_rate,
       self.learning_patience,
       self.num_epochs, self.steps_per_epoch, self.early_stop_patience, self.warmup_steps,
       self.num_gated_reoccurring_units,
       self.activation_function, self.time_shift_in_hours, self.training_splitting, self.show_output_after_sim,
       self.read_batch_size,
       self.network_type, self.output_folder, self.plot_output_sub_name, self.outputs,
       str(self.use_full_timestamp), str(self.keep_city_as_input)))


def convConfigToHyperparams(conf):
    hyperparams = NetowrkHyperparameterConfig()
    hyperparams.time_shift_in_hours = conf.time_shift
    hyperparams.training_splitting = conf.training_split
    hyperparams.start_learning_rate = conf.start_learning_rate
    hyperparams.loss_factor = conf.loss_factor
    hyperparams.min_learning_rate = conf.min_learning_rate
    hyperparams.learning_patience = conf.learning_patience
    hyperparams.num_epochs = conf.num_epochs
    hyperparams.steps_per_epoch = conf.steps_per_epoch
    hyperparams.early_stop_patience = conf.early_stop_patience
    hyperparams.warmup_steps = conf.warmup_steps
    hyperparams.num_gated_reoccurring_units = conf.num_gru
    hyperparams.activation_function = conf.activation_function
    hyperparams.network_type = conf.network_type
    hyperparams.plot_output_sub_name = "%s_" % str(conf.id)
    hyperparams.read_batch_size = conf.read_batch_size
    hyperparams.outputs = conf.outputs
    return hyperparams

