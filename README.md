# segmentation_gradio

An interactive **image segmentation** demo built with **Gradio**, wired to a medical/biomedical segmentation backend (via the `MedSAM2` submodule) and a lightweight API layer.

> **Note:** This README was prepared from the repository structure. If any details differ from your actual code, tell me and I'll update it immediately.

---

## ✨ Features

- 🔌 **Point‑and‑click / box prompts** for interactive segmentation (Gradio UI)
- 🧠 **Model backend via `MedSAM2`** submodule
- 🧰 **Utilities & configs** for reproducible runs (`utils/`, `configs/`)
- 🌐 Optional **local API service** (`api_service/`) to decouple UI vs. inference
- 🧪 **Quick tests** / examples in `test.py`

---

## 📦 Repository structure

```
segmentation_gradio/
├─ MedSAM2/           # model submodule (pulled via git submodules)
├─ api_service/       # optional REST API server for inference
├─ configs/           # model / app config files
├─ utils/             # helpers and common utilities
├─ app.py             # Gradio UI app (entry point)
├─ main.py            # alternate entry / orchestration script
├─ segmentation.py    # segmentation logic / pipeline wrapper
├─ test.py            # quick tests / example usage
├─ requirements.txt   # Python dependencies
├─ .env-sample        # sample environment variables
└─ .gitmodules        # submodule definitions
```

---

## 🚀 Quickstart

### 1) Clone with submodules

```bash
git clone --recurse-submodules https://github.com/mehrn79/segmentation_gradio.git
cd segmentation_gradio
git submodule update --init --recursive
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configure environment variables

```bash
cp .env-sample .env
```
Common variables (adjust to your setup):

```
DEVICE=cuda            # or "cpu"
MODEL_NAME=medsam2     # model id / variant
API_URL=http://127.0.0.1:8000  # if using the api_service
# HF_TOKEN=...         # if the backend pulls weights from Hugging Face
```

### 5) Obtain model weights

`MedSAM2` is included as a submodule. Follow its own README to download or place the required weights. Then point this repo to the weights via your config or `.env`.

### 6) Run the app

```bash
# Option A: launch the Gradio UI directly
python app.py

# Option B: use the main orchestrator (if that’s how you run it)
python main.py
```
By default Gradio serves on `http://127.0.0.1:7860/` — the console will print the exact URL.

> If you’re using the API service, run it first (see below), then launch the UI.

---

## 🌐 API service

> I attempted to open `api_service/` via GitHub in this environment, but the page failed to load here. So this section is a precise **template** ready to be filled with your actual route names and payloads. If you share the files (e.g., `api_service/main.py`), I'll finalize it 1:1 with your code.

### Run
```bash
cd api_service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment variables (suggested)
```
DEVICE=cuda            # or cpu
MODEL_NAME=medsam2
WEIGHTS_PATH=/path/to/weights
API_PORT=8000
API_HOST=0.0.0.0
```

### Endpoints (template)
- `GET /healthz` — health check
  ```json
  {"status":"ok","device":"cuda","model":"medsam2"}
  ```
- `POST /segment` — run segmentation
  - **Input** (choose one):
    - `multipart/form-data` with `image` file + optional JSON fields:
      ```json
      {
        "points": [[x, y], [x, y]],
        "box": [x0, y0, x1, y1],
        "return_mask": "png|rle|array"
      }
      ```
    - or `application/json` with `image_b64` instead of file.
  - **Output** example (for `png`):
    ```json
    {
      "mask_png": "<base64>",
      "shape": [H, W],
      "latency_ms": 123
    }
    ```

#### Call examples
**cURL (file upload):**
```bash
curl -X POST "http://127.0.0.1:8000/segment" \
  -F "image=@/path/to/image.png" \
  -F 'payload={"points":[[120,220],[160,240]],"return_mask":"png"}'
```

**Python client:**
```python
import requests, json
url = "http://127.0.0.1:8000/segment"
files = {"image": open("/path/to/image.png", "rb")}
payload = {"points": [[120,220],[160,240]], "return_mask": "png"}
resp = requests.post(url, files=files, data={"payload": json.dumps(payload)})
print(resp.json())
```

---

## 🖼️ Using the UI

1. Upload an image (medical or natural images, depending on model).
2. (Optional) Provide prompts: click foreground points or draw a box.
3. Hit **Segment**.
4. Review the mask overlay; download the resulting mask if needed.

---

## ⚙️ Configuration

- **Configs:** YAML/JSON files in `configs/`, referenced by CLI args or `.env`.
- **Runtime:** switch devices (`cpu`/`cuda`), tweak model variant, post‑processing, etc.

---

## ✅ Testing

```bash
python test.py
```

Add CI later with GitHub Actions to lint and run smoke tests.

---

## 🧪 Example: headless inference

```python
from segmentation import segment_image
mask = segment_image(
    image_path="/path/to/image.png",
    points=[(x1, y1), (x2, y2)],   # or box=(x0, y0, x1, y1)
    device="cuda",
    model_name="medsam2",
)
```

---

## 🧩 Troubleshooting

- **CUDA not found** → Set `DEVICE=cpu` or install a CUDA‑enabled PyTorch.
- **Model weights missing** → Ensure `MedSAM2` weights are in place and paths are correct.
- **Gradio not opening** → Check the terminal URL/port and firewall on `7860`.
- **API 404s** → Verify the API server is running and `API_URL` is set.

---

## 🗺️ Roadmap

- [ ] Precise API docs from `api_service` code (routes & schemas)
- [ ] Demo notebooks (batch inference, evaluation)
- [ ] Dockerfile + Compose (UI + API)
- [ ] Basic unit tests and CI
- [ ] Example datasets and masks for quick trials

---

## 🤝 Contributing

PRs welcome! Please open an issue with context (data type, model variant, reproduction steps) before large changes.

---

## 📜 License

Add a license file (e.g., MIT) at the project root if you intend others to use or extend the code.

---

## 🙏 Acknowledgments

- `MedSAM2` authors and maintainers
- The Gradio team for the UI toolkit
