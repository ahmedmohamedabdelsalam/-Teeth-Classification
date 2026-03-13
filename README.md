# Dental Imaging System

Clinical software for intraoral scan classification. This system analyzes dental images to assist in identifying common clinical conditions.

## Clinical Scope
The system provides differential analysis for the following conditions:
- **CaS**: Calculus
- **CoS**: Caries
- **Gum**: Gingivitis
- **MC**: Mouth Cancer
- **OC**: Orthodontics
- **OLP**: Oral Lichen Planus
- **OT**: Other

## System Architecture
- **Inference Core**: TensorFlow-based neural net optimized for dental feature extraction.
- **Backend**: FastAPI high-concurrency server.
- **Interface**: Clinical console developed with HTML5/Tailwind/JS.

## Local Deployment
1. **Initialize Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch System**:
   ```bash
   python -m app.main
   ```
Navigate to `http://localhost:8000` via a secure browser.

## Model Retraining
To train a new model from scratch with improved accuracy (using Transfer Learning):
1. Place your dataset in `data/raw/Training` and `data/raw/Validation`.
2. Ensure sub-folders are named according to the 7 clinical classes.
3. Run the training script:
   ```bash
   python scripts/train.py
   ```
The system will automatically use the newly trained `teeth_model.keras`.

## Cloud Integration

### Standard Web Hosting (Render / Railway)
1. Deploy via GitHub integration.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Containerized Hosting (AWS / Docker)
```bash
docker build -t dental-diagnostic-system .
docker run -p 8000:8000 dental-diagnostic-system
```

## Disclaimer
This software is intended for diagnostic decision support only. Clinical findings should be verified by a licensed dental professional.
