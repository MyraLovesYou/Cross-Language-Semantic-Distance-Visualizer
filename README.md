# Cross-Language-Semantic-Distance-Visualizer
A modular data science pipeline and interactive dashboard that mathematically quantifies and visualizes "Semantic Drift" between English and Japanese parallel sentences. By mapping high-dimensional vector embeddings into a shared 2D space, this tool exposes where contextual nuances, idioms, and cultural pragmatics diverge.
Longer distance between two corresponding points means their meanings are less similar, while closer distances signal higher similarity in meaning.
![Shows preview of project homepage](assets/preview.jpg)
### [**Try it out here!**](https://cross-language-semantice-distance-visualizer.streamlit.app/)

---

##  Features

- **Automated SQLite Database:**  backend pipeline that automatically caches sentence vectors and UMAP coordinates upon first launch, eliminating redundant deep-learning computation.
- **Multilingual Vector Alignment:** Leverages Hugging Face's `paraphrase-multilingual-MiniLM-L12-v2` transformer model to map diverse language characters into a unified mathematical space.
- **Dimensionality Reduction:** Compresses 384-dimensional semantic properties down to a human-viewable 2D topography using UMAP while preserving relative linguistic proximity.
- **Interactive UI Dashboard:** Built with Streamlit and Plotly to allow dynamic category filtering, structural gap analysis, and hover-triggered translation trajectory tracking.

## Running on local machine

1. Clone this repository.

2. Navigate to project directory

3. Set up virtual environment

`python -m venv .venv`

`# Windows command prompt
.venv\Scripts\activate.bat`

`# Windows PowerShell
.venv\Scripts\Activate.ps1`

`# macOS and Linux
source .venv/bin/activate`

4. Run the command `pip install -r requirements.txt` in the terminal

5. To start app run `streamlit run main.py` in the terminal

alternatively try `python -m streamlit run main.py`

**If you want to edit the provided database you will need a Hugging Face token to generate vectors**

Create a `.streamlit` directory and inside that directory make `secrets.toml`. In you secrets file enter this line `HF_TOKEN = "your-token-here"` and replace with your token.

---

## Limitations
The app works only for English-Japanese paired phrases, but theoretically could work for any two language supported in [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) the structure of the project would need to be slightly adapted to deal with more languages.

The current dataset was quickly put together and I think if a more carefully dataset of sentences was created then the results could be more useful to look at.

## Background
Developed as an independent undergraduate research track project exploring the intersection of Computational Linguistics and Artificial Intelligence at Purdue University.

Developer: Myra Bagga
