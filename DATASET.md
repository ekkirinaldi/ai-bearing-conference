# About Dataset

## Context

There are few datasets on mechanical engineering, in particular devoted to apply Machine Learning in industrial environment. The present data bundle is a community contribution to make available in Kaggle motor performance data provided by Case Western Bearing Data Center.

## Content

### Test Bench Setup

The test bench for motor performance assessment consists of:

- Motor with 2 HP power
- Torque transducer
- Dynamometer
- Control electronics

### Bearing Defects

The test bearings support the motor shaft. Defects were introduced at a single point by EDM machining. The diameters of defects in inches (millimeters):

- 0.007 inches (0.178 millimeters)
- 0.014 inches (0.356 millimeters)
- 0.021 inches (0.533 millimeters)

There is a time series for each defect located in 1 of 3 parts of the bearing: ball, inner race, outer race.

### Accelerometer Positions

Telemetry measurements come from 3 accelerometers installed on 3 positions in the system:

- **Drive end (DE)**
- **Fan end (FE)**
- **Base (BA)**

### Experimental Conditions

This dataset corresponds to the following conditions:

- 1 HP load applied to the motor
- Shaft rotating speed of 1772 rpm
- 48 kHz sampling frequency of the accelerometers

## Features

The following 9 features are calculated in order to run the fault identification prediction:

- Maximum
- Minimum
- Mean
- Standard deviation
- RMS
- Skewness
- Kurtosis
- Crest factor
- Form factor

Each feature is computed for time segments of 2048 points (0.04 seconds at the 48kHz accelerometer sampling frequency).

## Acknowledgements

This dataset is publicly available thanks to [Case Western Reserve University](https://case.edu/).

The experiments were initiated in order to characterize the performance of IQ PreAlert, a motor bearing condition assessment system developed at Rockwell. From that point on, the experimental program has expanded to provide a motor performance database which can be used to validate and/or improve a host of motor condition assessment techniques.

## Inspiration

This dataset serves the purpose of applying Machine Learning to Predictive Maintenance of industrial machinery. Its scope is the data-driven fault diagnosis. A common task in this field is the fault detection and classification.
