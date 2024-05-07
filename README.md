# Insight-SNE: Explore t-SNE through interactive explanation

## Introduction

Welcome to Insight-SNE! This interactive tool allows you to explore t-Distributed Stochastic Neighbor Embedding (t-SNE), a popular technique for dimensionality reduction and visualization of high-dimensional data. By using this tool, you can gain a deeper understanding of how t-SNE works and how it affects the representation of your data.

## Setup

To run Insight-SNE, follow these simple steps:

1. **Clone the Repository**: Clone the Insight-SNE repository to your local machine.

2. **Install Dependencies**: Navigate to the project directory and install the required dependencies using `pip` and the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Dash App**: Once the dependencies are installed, you can run the Dash app using the following command:
    ```bash
    python app.py
    ```

4. **Explore t-SNE**: Open your web browser and navigate to the address where the Dash app is running (typically http://127.0.0.1:8050/). You can now interact with the t-SNE visualization and explore its behavior with different datasets and parameters.

## Usage

Insight-SNE provides the following features:

- **Data Selection**: Choose from a variety of built-in datasets or upload your own data to visualize with t-SNE.
- **Parameter Adjustment**: Adjust parameters such as the perplexity and number of iterations to see how they affect the t-SNE embedding.
- **Interactive Visualization**: Interact with the t-SNE plot to inspect individual data points and clusters.

## License

Insight-SNE is licensed under the MIT License. See the `LICENSE` file for more details.
