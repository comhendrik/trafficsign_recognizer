# Traffic Sign Recognizer

This project involves developing a Traffic Sign Detection and Recognition system using a deep learning model built with Keras. Its goal is interpreting traffic signs in real-time.

## Table of Contents

- [Introduction](#introduction)
- [Disclaimer](#disclaimer)
- [Features](#features)
- [Installation](#installation)
- [Downloading GTSRB Dataset](#downloading-gtsrb-dataset)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

Traffic Sign Recognizer is a project aimed at detecting and recognizing traffic signs using deep learning techniques. This system interprets traffic signs in real-time, enhancing the safety and efficiency of autonomous vehicles and driver assistance systems.

## Disclaimer

At this time detection for signs like unlimited is not as precise, but all other signs could be detected.

## Features

- Real-time traffic sign detection
- Traffic sign recognition using a deep learning model
- Built with Keras and Python

## Installation

To install and run this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/comhendrik/trafficsign_recognizer.git
    ```
2. Navigate to the project directory:
    ```sh
    cd trafficsign_recognizer
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```
4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Downloading GTSRB Dataset

This dataset is needed to train the model, feel free to modify the model to train it on a different dataset

To download the GTSRB (German Traffic Sign Recognition Benchmark) dataset, follow these steps:

Download the dataset from the official website:
GTSRB Dataset Download
Extract the downloaded zip file to a directory of your choice.
Place the extracted dataset in the appropriate directory within the project.

## Usage

To train the traffic sign recognizer, run the following command:
```sh
python3 load.py
```

To use the traffic sign recognizer, run the following command:
```sh
python3 predict.py
```

This will start the system and begin processing video input for traffic sign detection and recognition.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

