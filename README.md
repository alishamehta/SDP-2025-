# SDP-2025- Adaptive Profiling and Optimization of Transformer Models for Fake News Detection

## Overview
This project investigates the performance characteristics of Transformer-based models for fake news detection on the **LIAR dataset**, with a focus on **operator-level profiling** and **GPU-accelerated inference**. While many misinformation detection systems rely on CPU-based or lightly optimized models, such approaches struggle to meet real-time latency requirements.

We evaluate how different Transformer architectures behave under GPU execution, analyze their operator composition (GEMM vs. non-GEMM), and demonstrate how optimization frameworks such as **TensorRT** can significantly reduce inference latency without sacrificing accuracy.

---

## Motivation
Fake news detection systems are increasingly deployed in real-world settings where **latency, scalability, and efficiency** matter. However, many existing approaches:
- Rely on CPU-based inference
- Use minimally optimized Transformer models
- Do not account for operator-level bottlenecks

At the same time, prior work shows that **non-GEMM operations can account for a substantial fraction of inference time** in modern ML workloads. This motivates a deeper look at how Transformer operators execute on GPUs and where optimization efforts should be focused.

---

## Dataset

### LIAR Dataset
The **LIAR dataset** consists of short political statements labeled across multiple levels of truthfulness (e.g., true, false, half-true).

It supports experiments in:
- Fake news detection  
- Political fact-checking automation  
- Multi-class and binary text classification  
- Explainable AI for trustworthiness assessment  

For this project, the dataset is preprocessed and used for supervised Transformer-based classification.

---

## Models Evaluated
We evaluate a set of Transformer models chosen to reflect different trade-offs between **accuracy, model size, and inference efficiency**:

- **BERT (Baseline Transformer)**  
  Serves as a reference point for comparing distilled and optimized variants.

- **DistilBERT**  
  Evaluates the efficiency–accuracy trade-off of distilled models on short, low-resource text inputs.

- **DistilBERT + TensorRT**  
  Assesses real-time inference optimization using kernel fusion and GPU-specific execution optimizations.

- **GPT-2**  
  Explores whether generative language models capture stylistic or semantic cues relevant to misinformation detection and how their operator profiles differ from encoder-only Transformers.

---

## Methodology
1. **Model Training & Inference**
   - Train and evaluate Transformer models on the LIAR dataset.
   - Run inference on both CPU and GPU backends where applicable.

2. **Operator-Level Profiling**
   - Profile inference execution to categorize operators into:
     - GEMM (matrix multiplication)
     - Non-GEMM (softmax, layer normalization, element-wise operations)
   - Measure execution time contributions of each operator class.

3. **Performance Benchmarking**
   - Compare inference latency across models.
   - Analyze the impact of GPU acceleration and TensorRT optimization.

---

## Metrics
- **Accuracy** (used as a sanity check for model correctness)
- **Inference execution time**
- **Percentage of GEMM vs. non-GEMM operations**
- **CPU vs. GPU execution breakdown**

---

## Key Findings
- Transformer inference is dominated by a small set of operators, including:
  - Matrix multiplications (GEMM)
  - Softmax attention
  - Layer normalization
  - Element-wise activations
- **Non-GEMM operations remain a significant contributor to inference latency**, even on GPUs.
- **DistilBERT + TensorRT achieves substantial inference speedups** compared to vanilla BERT, demonstrating the feasibility of real-time fake news detection using optimized Transformers.
- Operator fusion and GPU-aware execution are critical for reducing overhead and improving throughput.

---

## Artifacts
- 📓 **Colab Notebook** — End-to-end profiling and benchmarking workflow  
- 📊 **Poster (PDF)** — *Adaptive Profiling and Optimization at the Operator Level for Machine Learning Models*  

(Links provided in the repository.)

---

## Limitations
- Accuracy is not the primary optimization target and is used mainly for validation.
- Results are specific to the LIAR dataset and short-text classification.
- Profiling reflects inference-time behavior rather than training dynamics.

---

## Future Work
- Explore operator fusion strategies in greater depth.
- Extend profiling to additional Transformer architectures and datasets.
- Investigate adaptive optimization strategies for non-GEMM-heavy workloads.
- Apply the pipeline to real-time social media data streams.

---

## References
- LIAR Dataset: https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset  
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine  
- DistilBERT baseline on LIAR: https://www.kaggle.com/code/adamhsueh/distilbertmodel  

---

## Author
**Alisha Mehta** 
**Jack T, Keefer** 
**Dr. Beverly Abadines Quon** 
Department of Electrical and Computer Engineering
