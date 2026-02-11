---
language: en
license: mit
tags:
- swarm-intelligence
- quantum-inspired
- multi-agent-systems
- reinforcement-learning
- distance-prediction
- iraq
---

# 🐜 Q-MAS Distance Predictor

## Model Description

This is the trained MLP predictor for **Layer 7** of the Q-MAS framework. It estimates the Euclidean distance from an agent's current position to the target.

| Attribute | Value |
|-----------|-------|
| Architecture | 10-32-16-8-1 |
| Training samples | 50 |
| MAE | 0.055 |
| Input features | 10 |

## Performance

| Metric | 6-Layer | 7-Layer | Improvement |
|--------|---------|---------|-------------|
| Survivors/10 | 2.6 | **3.6** | **+38%** |
| Steps to target | 55.0 | 52.2 | -2.8 |

## Contact

**Abdullah Hawas**  
Independent Researcher, Dhi Qar, Iraq  
abdullahhawas93@gmail.com
