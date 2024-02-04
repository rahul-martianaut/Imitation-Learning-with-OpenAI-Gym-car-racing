# Imitation Learning with OpenAI Gym Car Racing

## Overview

Welcome to the Imitation Learning with OpenAI Gym Car Racing project! This repository contains code and resources for training a car racing agent using imitation learning.

## Project Structure

- **`Car_Racing_Simulation.py`**: Script for generating training and testing data by manually controlling the car using the keyboard in the OpenAI Gym Car Racing environment.

- **`model.py`**: Module containing the architecture of the neural network model used for imitation learning.

- **`Data_loader.py`**: Module responsible for loading and preprocessing the dataset. The dataset is generated by the `Car_Racing_Simulation.py` script.

- **`model_training.py`**: Script for training the imitation learning model using the processed dataset.

- **`Simulator.py`**: Script for playing the car racing game using the trained model. This demonstrates the effectiveness of the imitation learning approach.

## Dependencies

Ensure you have the following dependencies installed before running the scripts:

- Python 3.6 or later
- [OpenAI Gym](https://gym.openai.com/) (==0.26.2)
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)

You have two options for installing Gym:

1. **Install Gym with Box2D:**
    ```bash
    pip install gym[box2d]
    ```

   **Conda installation:**
    ```bash
    conda install -c conda-forge gym-box2d
    ```

2. **Install Gym and Box2D separately:**
    ```bash
    pip install gym
    pip install box2d-py
    ```

Choose the installation method that suits your preferences or project requirements.

### Pygame

If your project involves Pygame, you can install it using the following command:

```bash
pip install pygame
