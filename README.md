# MNIST: 3 Models, 1 Digit ✏️

A side-by-side comparison of three classic architectures on handwritten digit
recognition — a single-layer **Perceptron**, a fully-connected **ANN**, and a
**CNN** — with a live Streamlit demo where you draw a digit and watch all
three vote on what it is.

**[▶ Try the live demo](#)** *(add your deployed Streamlit link here once deployed)*

![demo screenshot placeholder](docs/demo.png)

## Why this project

Most MNIST tutorials stop at "here's 99% accuracy." This project instead asks
*where and why* the simpler models fail — the Perceptron and ANN routinely
misread digits the CNN gets right, and watching that happen live on a digit
you drew yourself makes the difference between architectures concrete rather
than abstract.

## Models

| Model | Architecture | Test accuracy* |
|---|---|---|
| Perceptron | Single dense layer, softmax | ~92% |
| ANN | 2 hidden layers (128 → 64), dropout | ~97% |
| CNN | 2 conv + pool blocks, dense head | ~99% |

\* Typical results after 5 epochs on standard MNIST; re-run `src/train.py` to
reproduce exact numbers, which vary slightly by seed/hardware.

## Project structure

```
mnist-digit-classifier/
├── app.py                 # Streamlit app (draw + predict)
├── src/
│   └── train.py            # Trains and saves all three models
├── saved_models/            # Trained .keras models (committed for the demo)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git
cd mnist-digit-classifier
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 1. Get the data

Download `mnist_train.csv` and `mnist_test.csv` (e.g. from the
[Kaggle "Digit Recognizer"](https://www.kaggle.com/competitions/digit-recognizer)
or [pjreddie's CSV mirror](https://pjreddie.com/projects/mnist-in-csv/)) and
place them in a `data/` folder:

```
data/
├── mnist_train.csv
└── mnist_test.csv
```

### 2. Train the models

```bash
python src/train.py --train_csv data/mnist_train.csv --test_csv data/mnist_test.csv --epochs 5
```

This saves `perceptron.keras`, `ann.keras`, and `cnn.keras` into
`saved_models/`.

### 3. Run the app locally

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, draw a digit, and see all three
predictions update live.

## Deploying so you can share a link

The easiest free option is **Streamlit Community Cloud**:

1. Push this repo to GitHub, **including the `saved_models/*.keras` files**
   (the app loads them at runtime — without them deployed the app has nothing
   to predict with).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app."
3. Point it at your repo, branch `main`, and file path `app.py`.
4. Deploy. You'll get a public `https://<something>.streamlit.app` URL —
   that's the link to put in your LinkedIn post and GitHub README.

Alternative: [Hugging Face Spaces](https://huggingface.co/spaces) also
supports Streamlit apps for free and is popular for ML demos specifically.

## Tech stack

TensorFlow / Keras · Streamlit · streamlit-drawable-canvas · scikit-learn ·
NumPy / Pandas

## License

MIT — feel free to fork and build on this.
