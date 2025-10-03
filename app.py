# ct-pr

## Overview

**ct-pr** is an end-to-end pipeline for tracking, analyzing, and visualizing clinical trials using advanced NLP and machine learning techniques. The project is designed to help researchers and clinicians easily search for topics, track research progress on specific diseases or conditions, and understand advancements, drugs, and final results achieved in clinical trials. It also enables users to view the authors, universities, and collaborators involved in each trial.

This project processes clinical trials data downloaded from [ClinicalTrials.gov](https://clinicaltrials.gov/), making it easier to extract insights from large-scale medical research datasets.

---

## Key Features

- **Topic Search & Tracking:** Find and track research on specific diseases or conditions.
- **Advancement & Results Analysis:** See what drugs, interventions, and results have been achieved for each trial.
- **Collaboration Insights:** Identify authors, universities, and collaborators involved in research.
- **NLP-Powered Ranking:** Uses BERT re-ranking, BM25 ranking, and cosine similarity fusion to surface the most relevant trials for a given query.
- **Summarization:** Leverages Pegasus-PubMed LLM for summarizing medical trial data.
- **Visualization:** Generates a variety of plots and dashboards for data exploration.
- **Audio Summaries:** Converts summaries to audio for accessibility.

---

## Tech Stack

- **Python 3.9+**: Main programming language.
- **Flask**: Web application framework.
- **PyTorch**: Deep learning framework for model development and tensor computations.
- **Transformers (HuggingFace)**: For BERT and Pegasus-PubMed models.
- **Sentence Transformers**: For semantic search and embeddings.
- **BM25 (rank_bm25)**: Classic IR ranking algorithm.
- **NumPy / Pandas / Dask**: Data manipulation and analysis.
- **Matplotlib / Seaborn / Plotly / Cartopy**: Data visualization.
- **NetworkX**: Network graph visualization.
- **WordCloud / squarify**: Word clouds and treemaps.
- **gTTS**: Text-to-speech for audio summaries.
- **XNNPACK**: High-efficiency neural network inference library (integrated as a third-party dependency).
- **VulkanMemoryAllocator**: GPU memory management for Vulkan (integrated as a third-party dependency).
- **Other dependencies**: See [`requirements.txt`](requirements.txt) for a full list.

---

## Project Structure

```
.
├── app.py                  # Main application script (Flask web app)
├── requirements.txt        # Python dependencies
├── data/                   # Input CSV data files (downloaded from ClinicalTrials.gov)
├── pytorch/                # PyTorch source and third-party libraries
├── static/                 # Static assets (images, audio)
├── templates/              # HTML templates for Flask
└── README.md               # Project documentation
```

---

## How to Download and Use Locally

### 1. Download Clinical Trials Data

- Go to [ClinicalTrials.gov](https://clinicaltrials.gov/).
- Download the relevant clinical trials data as CSV files.
- Place your CSV files in the `data/` directory (e.g., `data/Book3.csv`).

### 2. Clone the Repository

```sh
git clone <repository-url>
cd ct-pr
```

### 3. Set Up Python Environment

It is recommended to use a virtual environment:

```sh
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

### 4. Install Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download Additional Resources

- Ensure your data files (CSVs) are in the `data/` directory.
- The `pytorch/third_party` directory already contains required native libraries (XNNPACK, VulkanMemoryAllocator, etc.).

---

## Usage

### Run the Main Application

```sh
python app.py
```

- Open your browser and go to `http://127.0.0.1:5000/`.
- Enter a disease or condition in the search box.
- The dashboard will display:
  - Top relevant trials (using BERT, BM25, and cosine similarity fusion)
  - Summaries (using Pegasus-PubMed LLM)
  - Visualizations (clusters, timelines, status, network graphs, word clouds, etc.)
  - Audio summary for accessibility
  - Tables of trials, collaborators, interventions, and more

---

## NLP & ML Pipeline

- **Preprocessing:** Cleans and standardizes clinical trial text data.
- **Ranking:** Uses BM25 and BERT-based re-ranking, then fuses results with cosine similarity for best relevance.
- **Summarization:** Uses Pegasus-PubMed LLM to generate concise summaries of trial results and advancements.
- **Clustering & Visualization:** Clusters trials by topic and visualizes research trends, collaborations, and outcomes.
- **Audio:** Converts summaries to speech for accessibility.

---

## Development & Contribution

- All main logic is in [`app.py`](app.py).
- For advanced development, explore the `pytorch/` directory for source and third-party integrations.
- To run tests, use the appropriate test scripts or frameworks as described in the codebase.

---

## Troubleshooting

- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- For issues with native libraries (XNNPACK, VulkanMemoryAllocator), refer to their respective documentation in `pytorch/third_party/`.
- If you encounter issues with missing data, verify that the required CSV files are present in the `data/` directory.

---

## License

See the LICENSE file for details.

---

## Acknowledgements

- [ClinicalTrials.gov](https://clinicaltrials.gov/)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/)
- [XNNPACK](https://github.com/google/XNNPACK)
- [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)