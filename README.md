# Predicting pedestrian traffic around Melbourne, Australia with a Tensorflow 2 and Keras neural network

This is a Tensorflow 2 and Keras neural network to predict the number of pedestrians that will be walking around key areas in the Melbourne (Australia) city on a given day, at a given hour, at a given location.

The network is trained on a large dataset that contains hourly metrics of pedestrians as detected by sensor devices at key areas across the city since 2009 to present. Once trained, the model can be used to predict, for example, how many pedestrians will be walking through the Flinders Street Station Underpass at 5pm on Friday. This could then be used to understand pedestrian movements around the CBD, plan infrastructure upgrades, prepare for public events, etc.

The dataset is updated monthly (the actual data frequency is hourly) and is very large; at the time of writing it contains 2.89 million records, and is 233MB.

Here is the structure of the dataset, showing the header, first and last rows:

| ID      | Date_Time              | Year | Month   | Mdate | Day      | Time | Sensor_ID | Sensor_Name                | Hourly_Counts |
|:-------:|:----------------------:|:----:|:-------:|:-----:|:--------:|:----:|:---------:|:--------------------------:|:-------------:|
| 1       | 05/01/2009 12:00:00 AM | 2009 | May     | 1     | Friday   | 0    | 1         | Bourke Street Mall (North) | 53            |
| 2886663 | 10/31/2019 11:00:00 PM | 2019 | October | 31    | Thursday | 23   | 62        | La Trobe St (North)        | 100           |

The data is provided by the City of Melbourne, Australia, under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode). I do not include the data in this repository (it's in the `.gitignore` file), you can download it [here](https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-2009-to-Present-counts-/b2ak-trbp).

The current mean absolute error of the model after training is in the 50s. The goal was to get it below 100, so mission accomplished!

By the way, I'm aware that a neural network isn't really the best tool for this job; one might be better off just doing it with a simple SQL query like the following. It's just that this is a big dataset that I have had my eye on for a while and have been looking for an excuse to play with ;)

```
melbourne_pedestrian_sensor_counts=# SELECT AVG(Hourly_Counts) AS prediction FROM melbourne_pedestrian_sensor_counts WHERE Day = 'Tuesday' AND Time = 8 AND Sensor_ID = 13;
      prediction       
-----------------------
 4828.8401937046004843
(1 row)
```

## Requirements

Python version: 3.7.4

See dependencies.txt for packages and versions (and below to install).

## Architecture of the neural network

### Data preprocessing

This program uses Tensorflow's `tf.feature_column` API to preprocess data. This means that instead of preprocessing data _before_ passing it into the model (for example, to map day of the week strings such as Monday, Tuesday, Wednesday, etc to something more usable), the input data is mapped _inside_ the model, using a "DenseFeatures" layer made up of "feature columns" that map and normalize data. As a result of the preprocessing being done inside the model, the feature mapping layer is included when exporting the trained model, and subsequent raw data can be passed directly into the model, rather than preprocessed beforehand.

__Input layer:__ The above-mentioned DenseFeatures layer that handles the preprocessing of three features (`Day`, `Time`, `Sensor_ID`), all input through indicator colunns with one-hot encoding.

__One hidden layer:__ 60 neurons.

__Output layer:__ 1 neuron, with the value being the prediction of the model.

## Setup

Clone the Git repo.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

Download the [input data file](https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-2009-to-Present-counts-/b2ak-trbp) into the root directory.

## Run

```bash
python main.py
```

## Monitoring/logging

After training, run:

```
$ tensorboard --logdir logs/fit
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.0.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then open the above URL in your browser to view the model in TensorBoard.
