# Super-resolution using CycleGAN-based network

The repository contains Tensorflow implementation of a Cycle-Consistent Adversarial Network-based super-resolution neural network (SRNN).

## Purpose of usage/implementation

Internal structures of concrete specimens can be assessed by reconstructing their 3D volumes from the micro-CT images. However, if the evaluated specimen is large, the micro-CT device will yield blurry, degraded images. In order to provide a possibility to analyze large specimens, SRNN was applied. Its aim is to increase quality and recover lost details in micro-CT images.

## Overview of attempted approaches to solving the problem

The given micro-CT image dataset was unpaired, i.e. there is no direct correspondence between low-resolution (LR) image and its high-resolution (HR) counterpart. After the paper study, I decided to implement and improve either cycle-consistency image translation, reference-based, or artificial dataset generation-based SRNNs. At the moment I saw more potential for improving specifically cycle-consistency image translation SRNN. Therefore, I used vanilla CycleGAN to modify it in a number of ways:
1) Changed generators (Enhanced deep super-resolution network (EDSR), Enhanced Super-Resolution Generative Adversarial Network)
2) Changed discriminators from Markovian to Relativistic average
3) Added multi-modality super-resolution loss function apart from adversarial loss, cycle consistency loss, and identity loss
4) Minor changes like hyperparameter tuning/random sampling/added inference framework/subpixel convolution layers

The presented build of the network uses EDSR as generators and Markovian discriminators.

## Examples of super-resolved micro-CT images generated by the network

<img width="927" alt="main" src="https://user-images.githubusercontent.com/66497140/126190319-34fdeab8-277a-49ee-8552-c8aa04e3b418.png">
