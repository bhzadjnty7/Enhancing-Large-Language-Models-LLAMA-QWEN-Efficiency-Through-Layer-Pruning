# Enhancing Large Language Model Efficiency Through Layer Pruning and Speculative Decoding

**Authors:** Sobhan Abedi, Behzad Jannati, Mahsa Abarghani Aghdam  
**Course:** Large Language Models - Final Project  
**Instructors:** Prof. Mohammad Javad Dousti & Prof. Yadoulah Yaghoubzadeh

---

## 1. Abstract

Large Language Models (LLMs) have demonstrated exceptional capabilities but are notoriously resource-intensive and expensive to deploy. Their autoregressive nature, which requires a full forward pass for every generated token, creates a significant bottleneck due to memory bandwidth limitations. This project combines two state-of-the-art efficiency techniques—**progressive layer pruning** and **speculative decoding**—into a unified, lossless acceleration framework. Our central thesis is that a pruned version of a target LLM serves as an ideal "draft" model for speculative decoding, eliminating the need for a separate, smaller model and ensuring distributional alignment. This repository contains the first phase of this project: a comprehensive analysis tool to identify redundant layers in modern LLMs for optimal pruning.

## 2. Project Overview & Pipeline

The core goal is to accelerate LLM inference without sacrificing output quality. We achieve this through a three-stage pipeline:

1.  **Stage 1: Recursive Pruning:** We use a similarity-informed schedule to identify and remove the most redundant groups of transformer layers from models like Llama 3.1 and Qwen.
2.  **Stage 2: Targeted Healing:** To counteract any performance degradation from pruning, we apply a lightweight QLoRA fine-tuning pass on a small, targeted dataset.
3.  **Stage 3: Speculative Integration:** The healed, pruned model is then used as an efficient in-model "drafter" within a speculative decoding loop, sharing the KV cache with the original full model to accelerate token generation.

The entire pipeline is visualized below, as proposed in our initial plan:![Project Pipeline]

<img width="755" height="431" alt="image" src="https://github.com/user-attachments/assets/838c8ac0-522f-4bba-8c7e-ab93dc14266d" />

*> **Figure 1:** Our end-to-end acceleration pipeline. (1) A similarity scan identifies redundant layers for removal. (2) The pruned model is healed via QLoRA. (3) The healed model acts as a drafter for the full LLM in a speculative decoding loop.*

### Current StatusThis repository currently implements **Part 1: The Layer Pruning Analysis Framework**. The code provides the tools to load popular LLMs, analyze the similarity between their layers, and generate detailed reports recommending which layers are the best candidates for pruning. The implementation of the actual pruning, healing, and speculative decoding stages is planned as future work.

## 3. Features of the Analysis Framework

*   **Layer Redundancy Analysis:** Utilizes angular distance (derived from cosine similarity) to quantify the functional redundancy between transformer layers.
*   **Optimal Block Identification:** Identifies the most redundant contiguous blocks of layers for efficient, structured pruning.
*   **Pruning Recommendations:** Generates concrete suggestions for which layers to remove to achieve specific compression ratios (e.g., 10%, 20%, 30%).
*   **Rich Visualizations:** Creates detailed plots, including layer similarity heatmaps and optimal pruning distance curves, to provide insight into model architecture.
*   **Resource-Efficient:** Optimized to run on a single T4 GPU by leveraging 8-bit quantization via `bitsandbytes`.
*   **Broad Model Support:** Successfully tested on **Qwen 2.5 7B**, **Qwen 3 8B**, and **Llama 3.1 8B**.

## 4. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```2.  **Install the required libraries:**
    ```bash
    pip install -q transformers bitsandbytes accelerate peft datasets huggingface-hub fsspec
    ```

3.  **Log in to Hugging Face:**
    Accessing gated models like Llama 3 requires a Hugging Face access token.
    ```bash
    huggingface-cli login --token YOUR_HF_TOKEN --add-to-git-credential
    ```

## 5. How to Use the Analyzer

The core logic is encapsulated in the `LayerPruningAnalyzer` class. You can run the analysis for a supported model by following these steps:

```python
# Main function to run a complete analysis
def main():
    """Run a complete layer pruning_sweep analysis on a model."""
    print("Starting Llama-3.1-8B layer pruning_sweep analysis...")

    # 1. Create the analyzer instance
    analyzer = LayerPruningAnalyzer(        model_name="meta-llama/Llama-3.1-8B",
        use_8bit=True  # Use 8-bit quantization for efficiency
    )

    try:
        # 2. Load the model and tokenizer
        analyzer.load_model()

        # 3. Prepare the dataset for analysis        tokenized_texts = analyzer.prepare_dataset()

        # 4. Extract hidden state representations from each layer
        analyzer.extract_layer_representations(tokenized_texts, batch_size=1)

        # 5. Analyze layer similarities to find optimal pruning_sweep candidates
        analyzer.analyze_layer_similarities(max_block_size=12)

        # 6. Visualize the results and save plots
        analyzer.visualize_results()

        # 7. Generate a final report with statistics and recommendations
        report = analyzer.generate_report()

        print("\n============================================================")
        print("Analysis completed successfully!")    except Exception as e:
        print(f"An error occurred during analysis: {e}")

# Run the analysisif __name__ == "__main__":
    main()
```

This script will produce two output files:
*   `Llama-3.1-8B_analysis_results.png`: A visualization of the analysis.
*   `Llama-3.1-8B_pruning_report.json`: A detailed report with pruning recommendations.

## 6. Results from Analysis

Our analysis framework has yielded insightful results across three different models. The key finding is that middle-to-late layers consistently show higher functional redundancy, making them prime candidates for pruning.

| Model | Total Layers | Avg. Angular Distance | Recommendation for 20% Pruning (Remove 6 Layers) |
| :--- | :---: | :---: | :--- |
| **Qwen/Qwen2.5-7B** | 28 | 0.2118 | Remove 5 layers starting from **layer 14** (Angular Distance: 0.1899) |
| **Qwen/Qwen3-8B** | 36 | 0.1865 | Remove 7 layers starting from **layer 15** (Angular Distance: 0.1981) |
| **meta-llama/Llama-3.1-8B**| 32 | 0.2227 | Remove 6 layers starting from **layer 22** (Angular Distance: 0.2129) |

*Note: The number of layers to remove for a target percentage is rounded.*

These results confirm that a significant portion of layers can be identified for removal with minimal expected impact on model representations, paving the way for the next stages of the project.

## 7. Future Work

This repository currently focuses on the analysis phase. The next steps will build upon these findings to create the full end-to-end acceleration pipeline:

-   [ ] **Implement Pruning Functionality:** Add methods to physically remove the identified layers from the model.
-   [ ] **Develop the "Healing" Stage:** Integrate a QLoRA fine-tuning process to restore any performance loss post-pruning.
-   [ ] **Integrate Speculative Decoding:** Use the pruned, healed model as an in-model drafter for the original LLM.
-   [ ] **End-to-End Benchmarking:** Rigorously evaluate the final accelerated model on throughput, latency, and task-specific accuracy metrics (SQuAD, GSM8K, etc.).
-   [ ] **Release Toolkit:** Package the code into an open-source toolkit for the community.

## 8. License

This project is licensed under the MIT License. See the `LICENSE` file for details.
