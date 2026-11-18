# DNN Programming Assignment – RNN, LSTM, GRU and Transformer

**Student:** Samriddhi Mehta  
**Student ID:** 2026aj05286

## 1. Project Overview

This submission covers two parts of the Deep Neural Networks programming assignment, focusing on recurrent neural networks and Transformer-based architectures.

The implementation includes custom neural network components, model training and evaluation, comparative experiments, ablation studies and result visualizations.

The submission contains two notebooks:

- **Problem 1 – `problem1/problem1.ipynb`:** Implementation of custom RNN, LSTM and GRU cells; Human Activity Recognition (HAR) classification; sequence-addition experiments; gradient-flow analysis; hidden-size experiments; and gradient-clipping ablation.
- **Problem 2 – `problem2/problem2.ipynb`:** Implementation of scaled dot-product attention, multi-head attention and Transformer encoder blocks; HAR classification using a Transformer; Tiny Shakespeare character-level language modelling; and attention, context-length and positional-encoding experiments.

## 2. Environment and Dependencies

The notebooks were developed using Python and PyTorch in Kaggle notebook environments.

**Environment details:**
- Python: 3.13.18
- PyTorch: 2.11.0
- NumPy: 2.1.3
- pandas: 2.3.3
- Matplotlib: 3.10.0
- scikit-learn: 1.6.1

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

GPU execution is recommended, particularly for the Transformer language-model experiments.

## 3. How to Run

1. Set up a Python environment with the required dependencies.
2. Install the packages listed in `requirements.txt`.
3. Open `problem1/problem1.ipynb` and execute the cells in sequential order.
4. Open `problem2/problem2.ipynb` and execute the cells in sequential order.
5. Ensure the required UCI HAR and Tiny Shakespeare datasets are accessible to the respective data-loading cells.
6. Allow the notebooks to generate the experiment results, figures and text samples.

The notebooks contain the implementation, training, evaluation and visualization steps for their respective problems.

## 4. Experiments and Evaluation

**Problem 1:**
- Compared custom recurrent cells against their corresponding PyTorch implementations.
- Evaluated RNN, LSTM and GRU models on HAR classification.
- Studied sequence-length effects on the sequence-addition task.
- Analysed gradient propagation and the impact of hidden size.
- Compared vanilla RNN performance with and without gradient clipping.

**Problem 2:**
- Implemented and tested scaled dot-product attention and multi-head attention.
- Evaluated Transformer positional-encoding strategies on HAR classification.
- Trained a character-level language model using the Tiny Shakespeare dataset.
- Analysed attention-head configurations and the effect of removing causal masking.
- Measured computational time and memory usage across different context lengths.
- Generated text samples using different temperature settings.

## 5. Runtime

**Total execution time on an NVIDIA T4 GPU:** **7 minutes 32 seconds (14m 32s)**.

Execution time may vary depending on GPU availability, environment configuration and the number of experiments executed.

## 6. Individual Variant

The final digit of my student ID is even. Therefore, the assigned Problem 1 ablation compares vanilla RNN training at sequence length T = 200 with gradient clipping (maximum gradient norm 1.0) against training without gradient clipping.

## 7. Implementation and Reproducibility Notes

- Custom recurrent cells and Transformer attention components are implemented and evaluated through dedicated test cases.
- Random seeds are used in the experiments to support reproducibility.
- Classification performance is evaluated using test accuracy and macro-F1, while regression performance is evaluated using mean squared error.
- Model parameter counts and training time are recorded to compare model complexity and computational cost.
- Training and validation metrics are used to observe model convergence and generalization.
- Attention visualizations and experiment plots are included to support the analysis of model behaviour.
- Generated text samples at different temperatures are saved as a separate output artifact.
- Relative paths