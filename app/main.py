from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.inference import get_prediction
from app.model_loader import model_loader
from app import config
import logging
import uvicorn
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dental Imaging System")

# Ensure static directory exists and mount it
if not os.path.exists(config.STATIC_DIR):
    os.makedirs(config.STATIC_DIR)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Configure templates
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content="", media_type="image/x-icon")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to handle image uploads and return predictions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        logging.info(f"Received prediction request for file: {file.filename}")
        image_bytes = await file.read()
        if not image_bytes:
            raise ValueError("Empty image bytes received")
            
        results = get_prediction(image_bytes)
        return results
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error in prediction route:\n{error_details}")
        raise HTTPException(
            status_code=500, 
            detail={
                "message": str(e),
                "traceback": error_details.splitlines()[-5:] # Send more context
            }
        )


@app.get("/health")
async def health_check():
    try:
        model = model_loader.load_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_input": str(model.input_shape)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
