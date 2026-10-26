# Singapore Food Classifier

An image classifier that recognises 10 Singapore hawker dishes, built with **Python + TensorFlow** using transfer learning (EfficientNetV2-B0 pre-trained on ImageNet), and deployed as a free **Gradio** app on **Hugging Face Spaces**.

Dishes: chicken rice, laksa, char kway teow, nasi lemak, roti prata, satay, chilli crab, kaya toast, bak chor mee, hokkien mee.

## Project layout

```
sg-food-classifier/
├── scripts/download_images.py   # builds the dataset into data/<dish>/
├── train.py                     # trains the model, saves to space/
├── space/                       # everything that gets deployed to Hugging Face
│   ├── app.py                   # Gradio web app
│   ├── requirements.txt
│   ├── README.md                # Space config (title, emoji, SDK)
│   ├── model.keras              # created by train.py
│   └── labels.json              # created by train.py
├── outputs/                     # training charts (history.png, confusion.png)
└── data/                        # images (not committed)
```

## Steps

All commands run from this folder, in Git Bash or PowerShell.

### 1. Activate the virtual environment

```bash
source .venv/Scripts/activate      # Git Bash
# .venv\Scripts\Activate.ps1       # PowerShell
```

(First time on a new machine: `python -m venv .venv` then `pip install -r requirements.txt`.)

### 2. Download images

```bash
python scripts/download_images.py
```

About 150 images per dish. Takes ~10–15 minutes. If DuckDuckGo rate-limits you, just re-run; finished dishes are skipped.

### 3. Clean the data (important)

Open each `data/<dish>/` folder in File Explorer (large icons view) and delete anything that's wrong: other dishes, menus, people, logos, collages. 10–15 minutes of cleaning typically adds more accuracy than any model tweak.

### 4. Train

```bash
python train.py
```

~10–20 minutes on CPU. Prints validation accuracy and saves the model to `space/` plus charts to `outputs/`. Check `outputs/confusion.png` to see which dishes get mixed up (e.g. char kway teow vs hokkien mee) and add/clean images for those.

### 5. Test the app locally

```bash
cd space
python app.py
```

Open http://127.0.0.1:7860 and upload a food photo. Optionally put a few test photos in `space/examples/` (as `.jpg`) to show clickable examples.

### 6. Deploy to Hugging Face Spaces (free)

1. Create a free account at https://huggingface.co.
2. Create an access token with **write** permission: https://huggingface.co/settings/tokens
3. Create the Space at https://huggingface.co/new-space: name `sg-food-classifier`, SDK **Gradio**, template **Blank**, hardware **CPU basic (free)**.
4. Log in and upload the `space/` folder:

```bash
hf auth login
hf upload <your-username>/sg-food-classifier ./space . --repo-type space
```

The Space builds automatically (a few minutes) and is live at `https://huggingface.co/spaces/<your-username>/sg-food-classifier`. Free CPU Spaces sleep after inactivity and wake on the next visit.

### 7. Add it to your portfolio

Embed or link the Space on your personal website, and include `outputs/history.png` and `outputs/confusion.png` as screenshots.

## Notes

- Images are scraped from the web for personal/educational use; don't redistribute the dataset.
- To add a dish: add it to `CLASSES` in `scripts/download_images.py`, download, retrain, redeploy.
