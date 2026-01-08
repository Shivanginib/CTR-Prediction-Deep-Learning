# Click-Through Rate (CTR) Prediction: Deep Learning for Digital Advertising

## Project Overview
This project explores high-performance deep learning architectures for predicting user clicks in digital advertising. Using the **Criteo 1TB Click Logs Dataset**, I developed a scalable pipeline that overcomes memory constraints through chunk-wise processing and compares the performance of a self-attentive TabNet model against a standard 3-layer Neural Network.

## The Challenge: Big Data Constraints
* **Dataset Size:** 1TB+ (Criteo Click Prediction).
* **Memory Management:** Implemented **chunk-wise processing** (1,000,000 rows per chunk) to train models iteratively without crashing system memory.
* **Feature Complexity:** Managed 13 numerical features and 26 categorical features with missing values.

## Technical Workflow
### **Data Preprocessing**
* **Handling Missing Values:** Numerical features were imputed with the median; categorical features were marked as "Unknown."
* **Scaling & Encoding:** Applied **StandardScaler** to numerical data and **Label Encoding** to categorical features.
* **Chunking Strategy:** Data was read and trained in 1M-row increments with iterative model updates.

### **Model Architectures Compared**
1. **TabNet Classifier:** A modern, self-attentive deep learning model designed specifically for structured/tabular data.
2. **3-Layer Fully Connected NN:** A standard dense architecture with ReLU activation and Adam optimization.

## Performance Comparison
| Metric | TabNet Classifier | 3-Layer Neural Network |
| :--- | :--- | :--- |
| **Final Accuracy** | **81.2% (Winner)** | 79.5% |
| **Feature Selection** | Dynamic & Interpretable | None (Standard) |
| **Activation** | Entmax | Softmax |
| **Strategy** | Learns feature importance | Global pattern recognition |

## Key Insights
* **Superiority of TabNet:** TabNet outperformed the standard NN by nearly 2%, proving that dynamic feature selection is highly effective for mixed (numerical & categorical) structured data.
* **Scalability:** Successfully demonstrated that iterative training on 1,000,000-row chunks allows for high-accuracy modeling on datasets that exceed RAM capacity.
* **Interpretability:** TabNet provided better insights into which features (out of the 39) were the primary drivers for a "Click" prediction.

## Future Improvements
* **Hyperparameter Tuning:** Further optimize TabNet’s decision steps and relaxation factors.
* **Embeddings:** Implement Entity Embedding layers for categorical features in the 3-Layer NN to bridge the performance gap.
* **Distributed Training:** Transition to a distributed computing environment (e.g., Spark or multi-GPU) for faster processing of the full 1TB dataset.

