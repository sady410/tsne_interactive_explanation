# Insight-SNE — Original Interactive Prototype

This repository contains the first, basic prototype of Insight-SNE: a Python/Dash tool for interactively exploring t-SNE embeddings and their gradient-based explanations. It was developed as part of the work presented at **ESANN 2024** in *Insight-SNE: Understanding t-SNE Embeddings through Interactive Explanation*.

**Looking for the current application? Visit [insight-sne.netlify.app](https://insight-sne.netlify.app/)** for the desktop version, screenshots, downloads, and getting-started instructions.

## From prototype to desktop application

This prototype explored how an interactive interface could help users inspect an embedding together with explanations of local feature influence. It is an early research implementation and provides context for the original publication.

Insight-SNE has since evolved into a C++/Qt desktop application integrating the C++ gradient-based explanation implementation. The current version supports point and selection inspection, feature importance and directions, target comparisons, gradient vectors, point scope, saved sessions, and data and figure exports.

| Project | Purpose |
| --- | --- |
| **This repository** | First Python/Dash interactive prototype associated with the ESANN 2024 publication. |
| **[Insight-SNE website](https://insight-sne.netlify.app/)** | Current product, downloads, screenshots, and first-session instructions. |
| **[Insight-SNE desktop repository](https://github.com/sady410/Insight-SNE-desktop)** | Source repository for the current C++/Qt desktop application. |
| **[Original Qt application](https://github.com/sady410/Insight-SNE)** | Earlier Qt version using the Python explanation implementation. |
| **[Python research implementation](https://github.com/sady410/tsne_gradients_explanation)** | Basic numerical implementation and research examples for gradient-based t-SNE explanations. |
| **[C++ implementation](https://github.com/sady410/tsne-explanations-cpp)** | Standalone C++17 implementation of exact t-SNE and the explanation method. |

## Run the original prototype

These instructions apply to the Python/Dash prototype in this repository. To use the current desktop application, see the [website's download section](https://insight-sne.netlify.app/#download).

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/sady410/tsne_interactive_explanation.git
   cd tsne_interactive_explanation
   ```

2. Install the Python dependencies, preferably in a virtual environment:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the Dash application:

   ```bash
   python app.py
   ```

4. Open the local address printed in the terminal, typically `http://127.0.0.1:8050/`.

The prototype provides built-in datasets and data upload, controls for parameters such as perplexity and iteration count, and an interactive embedding view for exploring samples and their explanations.

This is experimental research software. Its dependencies and behavior may require adjustments on newer environments. Check the implementation, assumptions, and outputs for your intended use; explanations describe local feature influence and should be interpreted alongside the original data and embedding settings.

## Publication

Sacha Corbugy, Thibaut Septon, Bruno Dumas, and Benoît Frénay. **Insight-SNE: Understanding t-SNE Embeddings through Interactive Explanation.** European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2024.

**[Read the paper](https://doi.org/10.14428/esann/2024.es2024-190)** · **[University of Namur publication record and manuscript](https://researchportal.unamur.be/en/publications/insight-sne-understanding-t-sne-embeddings-through-interactive-ex/)**

```bibtex
@inproceedings{corbugy2024insightsne,
  title = {Insight-SNE: Understanding t-SNE Embeddings through Interactive Explanation},
  author = {Corbugy, Sacha and Septon, Thibaut and Dumas, Bruno and Fr{\'e}nay, Beno{\^i}t},
  booktitle = {European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
  year = {2024},
  publisher = {ESANN (i6doc.com)},
  doi = {10.14428/esann/2024.es2024-190}
}
```

For the underlying gradient-based explanation method, see also [Gradient-based explanation for non-linear non-parametric dimensionality reduction](https://doi.org/10.1007/s10618-024-01055-6).

If you use this prototype in research, please cite the ESANN paper and identify the version of the code used.

## License

This prototype is distributed under the [MIT License](LICENSE). Related repositories and application packages provide their own licenses and third-party notices.
