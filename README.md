# Predicting pedestrian traffic around Melbourne, Australia with a Tensorflow 2 and Keras neural network

This is a Tensorflow 2 and Keras neural network to predict the number of pedestrians that will be walking around key areas in the Melbourne (Australia) city on a given day, at a given hour.

The network is trained on a large dataset that contains hourly metrics of pedestrians as detected by sensor devices at key areas across the city. The data is recorded at hourly intervals, from 2009 to present. Once trained, the model can be used to predict, for example, how many pedestrians will be walking through the Flinders Street Station Underpass at 5pm on Friday. This could then be used to understand pedestrian movements around the CBD, plan infrastructure upgrades, or prepare for public events, concerts, protests, etc.

The dataset is updated monthly (the actual data frequency is hourly) and is very large, at the time of writing it contains 2.89 million records, and is 233MB.

Here is the structure of the dataset, showing the header, first and last rows:

| ID      | Date_Time              | Year | Month   | Mdate | Day      | Time | Sensor_ID | Sensor_Name                | Hourly_Counts |
| ------- | ---------------------- | ---- | ------- | ----- | -------- | ---- | --------- | -------------------------- | ------------- |
| 1       | 05/01/2009 12:00:00 AM | 2009 | May     | 1     | Friday   | 0    | 1         | Bourke Street Mall (North) | 53            |
| 2886663 | 10/31/2019 11:00:00 PM | 2019 | October | 31    | Thursday | 23   | 62        | La Trobe St (North)        | 100           |

The data is provided by the City of Melbourne, Australia, under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode). I do not include the data in this repository (it's in the `.gitignore` file), you can download it [here](https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-2009-to-Present-counts-/b2ak-trbp).

## Requirements

Python version: 3.7.4
See dependencies.txt for packages and versions (and below to install).

## Architecture of the neural network

### Data preprocessing

This program uses Tensorflow's `tf.feature_column` API to preprocess data. This means that instead of preprocessing data _before_ passing it into the model (for example, to map days of the week strings such as Monday, Tuesday, Wednesday, etc to something more usable), the input data is mapped _inside_ the model, using a "DenseFeatures" layer made up of "feature columns" that map and normalize data. As a result of the preprocessing being done inside the model, it is included when exporting the trained model, and subsequent raw data can be passed directly into the model, rather than preprocessed beforehand.

TODO: More details about the architecture.

## Setup

Clone the Git repo.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

Download the [input data file](https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-2009-to-Present-counts-/b2ak-trbp) into the root directoy.

## Run

```bash
python main.py
```
