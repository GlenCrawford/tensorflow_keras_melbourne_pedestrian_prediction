import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column

import numpy as np
import pandas as pd
import functools

### Load the input data ###

# See README on how to obtain this file.
INPUT_DATA_PATH = 'Pedestrian_Counting_System___2009_to_Present__counts_per_hour_.csv'
INPUT_DATA_COLUMN_NAMES = ['ID', 'Date_Time', 'Year', 'Month', 'Mdate', 'Day', 'Time', 'Sensor_ID', 'Sensor_Name', 'Hourly_Counts']
INPUT_DATA_COLUMNS_TO_USE = ['Day', 'Time', 'Sensor_ID', 'Hourly_Counts']

BATCH_SIZE = 5000

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Load the input CSV file into a tf.data Dataset.
# Each item in the returned dataset is a batch of the specified size, each batch being a tuple of (many input features, many labels).
# The features are a dictionary mapping CSV column names to Tensors containing the batch's data data in the corresponding column.
# The labels are a Tensor containing the labels for the features in the batch.
def load_dataset():
  return tf.data.experimental.make_csv_dataset(
    INPUT_DATA_PATH,
    batch_size = BATCH_SIZE,
    header = True, # Exclude the header row.
    shuffle=True, # Randomize the rows.
    label_name = 'Hourly_Counts',
    na_value = '?',
    num_epochs = 1,
    ignore_errors = True,
    column_names = INPUT_DATA_COLUMN_NAMES,
    select_columns = INPUT_DATA_COLUMNS_TO_USE
  )

# Print what a single batch looks like.
def show_dataset(dataset):
  for features, labels in dataset.take(1):
    for feature, values in features.items():
      print("{:9s}: {}".format(feature, values.numpy()))
    print('Labels:   ' + str(labels))

# Helper function to build a tf.data Dataset. Do this so we can pipe data through the feature columns in the preprocessing layer of the model.
def build_dataset_from_dictionary(data):
  data_frame = pd.DataFrame(data)
  labels = data_frame.pop('labels')
  dataset = tf.data.Dataset.from_tensor_slices((dict(data_frame), labels))
  dataset = dataset.batch(BATCH_SIZE)
  return dataset

train_dataset = load_dataset()

print('Example train dataset batch:')
show_dataset(train_dataset)

### Data preprocessing ###
# Instead of preprocessing the data before training the model, we're going to take advantage of the tf.feature_column API to preprocess *inside* the model. This means that we can pass raw data directly into the model.

# Helper function to take a feature column and pass the batch through it, in order to test preprocessing done in the feature column.
def pass_example_batch_through_feature_column(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

# Function to normalize the numeric data based on the mean and standard deviation of the relevant column (calculated below).
def normalize_numeric_data(data, mean, std):
  return (data-mean) / std

# Get a batch so we can look at some example values.
# The below values, means, standard deviations, normalization, etc comments all come from one of these batches, where the batch size was 5.
example_batch = next(iter(train_dataset))[0]

# Take a look at the example batch.
print('\n\nExample batch:')
print(str(example_batch))
# OrderedDict([
#   ('Day', <tf.Tensor: id=74, shape=(5,), dtype=string, numpy=array([b'Tuesday', b'Sunday', b'Friday', b'Tuesday', b'Thursday'], dtype=object)>),
#   ('Time', <tf.Tensor: id=76, shape=(5,), dtype=int32, numpy=array([ 8,  5,  5, 16, 16], dtype=int32)>),
#   ('Sensor_ID', <tf.Tensor: id=75, shape=(5,), dtype=int32, numpy=array([15, 12, 15, 11, 18], dtype=int32)>)
# ])

# Use Pandas to summarize the mean, standard deviation, etc of the numeric columns.
# Uncomment bit at the end to make it more readable (ie, remove exponent notation).
# Note that this isn't *good*, since we're loading the input data twice, but it'll do for learning.
input_data_statistical_summary = pd.read_csv(INPUT_DATA_PATH)[['Time', 'Sensor_ID']].describe()#.apply(lambda s: s.apply(lambda x: format(x, 'g')))

# Output of input_data_statistical_summary:
print('\n\nNumeric data summary/statistics:')
print(input_data_statistical_summary)
#               Time    Sensor_ID
# count  2.88666e+06  2.88666e+06
# mean       11.4523        20.97
# std         6.9475      14.7097
# min              0            1
# 25%              5            9
# 50%             11           18
# 75%             17           31
# max             23           62

# Store the mean and standard deviation of both the Time and Sensor_ID columns.
TIME_MEAN, SENSOR_ID_MEAN = np.array(input_data_statistical_summary.T['mean'])
TIME_STD, SENSOR_ID_STD = np.array(input_data_statistical_summary.T['std'])

print('\n\nTime mean: ' + str(TIME_MEAN))
print('Time std: ' + str(TIME_STD))

print('Sensor_ID mean: ' + str(SENSOR_ID_MEAN))
print('Sensor_ID std: ' + str(SENSOR_ID_STD))

# Build normalizer functions that the numeric feature columns will use to normalize their values.
time_feature_normalizer = functools.partial(normalize_numeric_data, mean=TIME_MEAN, std=TIME_STD)
sensor_id_feature_normalizer = functools.partial(normalize_numeric_data, mean=SENSOR_ID_MEAN, std=SENSOR_ID_STD)

# Now let's go through a batch (where batch size is 5) and look at each of the features and their values, and test feature columns to preprocess them.

# Day feature (raw):
# [b'Tuesday', b'Sunday', b'Friday', b'Tuesday', b'Thursday']
# This is a "categorical" column, where the values are from a finite (small) set, but we don't want to be passing in the days of the week as strings.

# Day feature (after preprocessing):
print('\n\nDay feature (preprocessed):')
pass_example_batch_through_feature_column(feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])))
# [
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1.]
#   [0. 0. 0. 0. 1. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0.]
# ]

# Time feature (raw):
# [ 8,  5,  5, 16, 16]

# Time feature (before normalization/preprocessing):
print('\n\nTime feature (before normalization/preprocessing):')
pass_example_batch_through_feature_column(feature_column.numeric_column('Time'))
# [
#   [ 8.]
#   [ 5.]
#   [ 5.]
#   [16.]
#   [16.]
# ]

# Time feature (after normalization/preprocessing):
print('\n\nTime feature (after normalization/preprocessing):')
pass_example_batch_through_feature_column(feature_column.numeric_column('Time', normalizer_fn = time_feature_normalizer))
# [
#   [-0.5  ]
#   [-1.   ]
#   [-1.   ]
#   [ 0.833]
#   [ 0.833]
# ]

# Sensor_ID feature (raw):
# [15, 12, 15, 11, 18]

# Sensor_ID feature (before normalization/preprocessing):
print('\n\nSensor_ID feature (before normalization/preprocessing):')
pass_example_batch_through_feature_column(feature_column.numeric_column('Sensor_ID'))
# [
#   [15.]
#   [12.]
#   [15.]
#   [11.]
#   [18.]
# ]

# Sensor_ID feature (after normalization/preprocessing):
print('\n\nSensor_ID feature (after normalization/preprocessing):')
pass_example_batch_through_feature_column(feature_column.numeric_column('Sensor_ID', normalizer_fn = sensor_id_feature_normalizer))
# [
#   [-0.357]
#   [-0.571]
#   [-0.357]
#   [-0.643]
#   [-0.143]
# ]

print('\n\n')

### Define the model ###

# Now that we've decided how to send each feature through a feature column and preprocess it (map values, normalize numeric data, etc), create a DenseFeatures input layer to preprocess our inputs.
preprocessing_layer = tf.keras.layers.DenseFeatures([
  #feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])),
  feature_column.numeric_column('Time', normalizer_fn = time_feature_normalizer),
  feature_column.numeric_column('Sensor_ID', normalizer_fn = sensor_id_feature_normalizer)
])

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(50, activation = 'relu'),
  tf.keras.layers.Dense(50, activation = 'relu'),
  tf.keras.layers.Dense(50, activation = 'relu'),
  tf.keras.layers.Dense(1, activation = 'relu'),
])

# Configure the model parameters.
model.compile(
  optimizer = 'adam',
  loss = 'mean_absolute_error',
  metrics = ['mean_absolute_error']
)

### Train the model ###

model.fit(train_dataset, epochs=5)

print('\n\nModel architecture:')
model.summary()

### Test the model ###

test_data = {
  'Day': ['Wednesday', 'Saturday', 'Sunday', 'Sunday', 'Tuesday', 'Saturday', 'Monday', 'Wednesday', 'Monday', 'Tuesday'],
  'Time': [23, 0, 4, 15, 8, 10, 18, 4, 14, 9],
  'Sensor_ID': [18, 18, 14, 13, 13, 17, 12, 2, 13, 1],
  'labels': [61, 48, 76, 55, 5176, 211, 182, 14, 976, 1025]
}

test_dataset = build_dataset_from_dictionary(test_data)

print('\n\nTest dataset:')
show_dataset(test_dataset)
print('\n\n')

test_loss, test_metrics = model.evaluate(test_dataset)
print('\n\nTest loss: ' + str(test_loss))
print('Test metrics: ' + str(test_metrics))

predictions = model.predict(test_dataset)

print('\n\nPredictions for test dataset:')
print(str(predictions))
