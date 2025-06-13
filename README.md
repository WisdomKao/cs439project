# CS439 Project

## Lion Optimizer Analysis

Analysis of Lion optimizer vs SGD/AdamW on CIFAR-10.

## Experiments
1. Loss landscape analysis
2. Training dynamics
3. Weight decay scaling
4. Architecture comparison

## Results
Plots saved to results/:
1. loss_landscape_analysis.png
2. training_dynamics.png
3. weight_decay_analysis.png
4. architecture_generalization.png

## Key Findings
1. Lion competitive performance (~89.9% vs SGD 90.4%)
2. Requires 5× higher weight decay than AdamW
3. Different training dynamics patterns
4. Smooth loss landscapes for all optimizers
