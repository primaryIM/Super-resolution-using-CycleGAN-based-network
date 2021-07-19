# Super-resolution using CycleGAN-based network

The repository contains Tensorflow implementation of a Cycle-Consistent Adversarial Network-based super-resolution neural network. 

## Purpose of usage/implementation

Internal structures of concrete specimens can be assessed by reconstructing their 3D volumes from the micro-CT images. However, if the evaluated specimen is large, micro-CT device will yield blurry, degraded images. In order to provide a possibility to analyse large specimens, super-resolution neural network was applied. It was implemented to increase the resolution and recover lost details in images.

## Overview of attempted approaches to solving the problem

The given micro-CT image dataset was unpaired, i.e. there is no direct correspondence between low-resolution (LR) image and its high-resolution (HR) counterpart.


## Examples of super-resolved micro-CT images generated by the network

<img width="927" alt="main" src="https://user-images.githubusercontent.com/66497140/126190319-34fdeab8-277a-49ee-8552-c8aa04e3b418.png">
