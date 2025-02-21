# param_estimation
This repository contains Python codes used to generate trajectories and create figures for the scientific paper:

Parameter Optimization for a Neurotransmission Recovery Model
Ariane Ernst, Meida Jusyte, Toluwani Okunola, Tino Petrov, Alexander Walter, and Stefanie Winkelmann

Contents
The repository includes the following files:

custom_optimization_old_rates.py
Performs a multi-start optimization to minimize the mean-squared error between the measured and simulated currents over the model parameters.

recovery_model_60Hz_no_F_F.yml
Defines the system of ordinary differential equations (ODEs) for the recovery model presented in the paper, using SBML format.

recovery_model_60Hz_no_F_F.xml
Another representation of the ODE system for the recovery model in SBML format.

ts.npy
Contains the measurement time points.
