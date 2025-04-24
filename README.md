# Mamamia Gen Project

This repository contains the implementation of the **Mamamia Gen** project, which leverages advanced deep learning techniques for medical image generation and analysis. The project utilizes frameworks such as PyTorch, MONAI, and WandB for training, evaluation, and visualization.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Mamamia Gen project focuses on generating and analyzing medical images using state-of-the-art generative models like SPADE and Diffusion Models. It includes training pipelines, evaluation scripts, and utilities for data preprocessing and augmentation.

## Features
- **SPADE Trainer**: Implements a training pipeline for SPADE-based models.
- **Diffusion Models**: Includes utilities for training and loading diffusion models.
- **Data Augmentation**: Provides a set of transformations for medical image preprocessing.
- **Distributed Training**: Supports Distributed Data Parallel (DDP) for scalable training.
- **WandB Integration**: Logs training metrics and visualizations to WandB.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mamamia_gen.git
   cd mamamia_gen