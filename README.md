# Word2Vec from Scratch: Contrastive Learning for Text Representation

This project implements the Word2Vec (Skip-Gram with Negative Sampling) model from scratch using PyTorch, with the aim of examining it through the modern lens of contrastive self-supervised learning.

The objective is to learn meaningful word representations from unlabeled data (WikiText-2) and to evaluate the transferability of these representations on a supervised downstream task (AG News classification).

## Project Structure

```
word2vec-as-contrastive-learning/
├── assets/                           # Result plots and figures
├── notebooks/
│   ├── word2vec.ipynb                # Word2Vec model training on WikiText-2
│   └── classification.ipynb          # Downstream classification experiments (AG News)
├── checkpoints/                      # Saved embedding weights (.ckpt)
│   └── *.ckpt
└── requirements.txt                  # Python dependencies
```

## Results Summary

The experiments confirm the effectiveness of pre-training:

1.  **Pre-training vs. Vanilla**: Initializing the classifier with pre-trained Word2Vec embeddings **outperforms** a randomly initialized model, leading to lower validation loss and higher accuracy.

![Pre-trained vs Randomly Initialized Classifier (25k labeled samples)](assets/comparison_25k.png)  
*Validation accuracy and loss comparison between the classifier initialized with pre-trained Word2Vec embeddings and Vanilla baseline (**25,000** labeled samples).*


2.  **Effect of Data Scarcity:**: The advantage of pre-training increases when fewer labeled samples are available (e.g., 5k vs. 25k), confirming that self-supervised pre-training is most valuable in low-resource regimes.

![Pre-trained vs Randomly Initialized Classifier (5k labeled samples)](assets/comparison_5k.png)  
*Validation accuracy and loss comparison between the classifier initialized with pre-trained Word2Vec embeddings and Vanilla baseline (**5,000** labeled samples).*


3.  **Ablation on `R` (Context Radius)**: For this specific classification task, smaller context windows (e.g., **`R=5`**) yielded the best performance. This suggests that local semantic information was more valuable than a wider, more general context.

![Ablation study on context radius R](assets/ablation_R.png)  
*Impact of context radius R on validation accuracy and loss.*

4.  **Ablation on `K` (Negative Sample Ratio)**: A smaller ratio of negative samples (e.g., **`K=1`**) produced the best classification results. Larger values of `K` seemed to focus the optimization too much on distinguishing random pairs, resulting in less informative embeddings for this task.

![Ablation study on negative samples K](assets/ablation_K.png)  
*Impact of negative samples K on validation accuracy and loss.*

## How to Run

First, clone the repository and install the required dependencies with `pip install -r requirements.txt`.

### Step 1: Train Word2Vec Embeddings

Run the `word2vec.ipynb` notebook cells in order. It will :
- Train the Word2Vec model on the WikiText-2 dataset.
- Train all the model configurations required for the next step (e.g., varying `R` and `K`).
- Create a `checkpoints/` directory and populate it with the trained embedding weights (`.ckpt` files).

### Step 2: Run Classification Experiments

Run the `classification.ipynb` notebook cells in order. It will :
- Load the pre-trained embeddings from the `checkpoints/` folder.
- Train the `ClassAttentionModel` on the AG News dataset.
- Run the experiment comparing Word2Vec initialization vs. Vanilla (random) initialization.
- Run the ablation studies on the `R` and `K` hyperparameters.
- Display the final plots in the notebook itself.

## References

[1] Chopra, S., Hadsell, R., & LeCun, Y. (2005). Learning a Similarity Metric Discriminatively, with Application to Face Verification.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Vol. 1, pp. 539–546.
https://doi.org/10.1109/CVPR.2005.202