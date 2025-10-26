from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import clip
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import base64

app = FastAPI()

#mounting with the other pages
app.mount("/static", StaticFiles(directory="static"), name="static")

#template addition to connect backend with frontend
templates = Jinja2Templates(directory="template")

#load the clip vit-b/32 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# functions

def compute_histogram_similarity(img1, img2):
    img1_cv = np.array(img1)
    img2_cv = np.array(img2)
    hist1 = cv2.calcHist([img1_cv], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2_cv], [0], None, [256], [0,256])
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return round(correlation * 100, 2)

def compute_orb_similarity(img1, img2):
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_cv, None)
    kp2, des2 = orb.detectAndCompute(img2_cv, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    similarity = len(matches) / max(len(kp1), len(kp2)) * 100
    return round(similarity, 2)



def draw_orb_keypoints(img: Image.Image):
    """Return ORB keypoints image as base64"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    img_kp = cv2.drawKeypoints(img_cv, kp, None, color=(0,255,0), flags=0)
    _, buffer = cv2.imencode('.png', img_kp)
    return base64.b64encode(buffer).decode()

# Similarity Function

def compute_similarity_metrics(image1: Image.Image, image2: Image.Image):
    # CLIP embedding
    img1_input = preprocess(image1).unsqueeze(0).to(device)
    img2_input = preprocess(image2).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = model.encode_image(img1_input)
        emb2 = model.encode_image(img2_input)
    emb1 /= emb1.norm(dim=-1, keepdim=True)
    emb2 /= emb2.norm(dim=-1, keepdim=True)
    clip_score = (emb1 @ emb2.T).item() * 100

    # Grayscale for pixel metrics
    img1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    img2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
    h, w = img1_cv.shape
    img2_cv = cv2.resize(img2_cv, (w, h))

    mse_score = np.mean((img1_cv - img2_cv) ** 2)
    ssim_score = ssim(img1_cv, img2_cv) * 100

    hist_corr = compute_histogram_similarity(image1, image2)
    orb_feat = compute_orb_similarity(image1, image2)

    explanation = (
        "CLIP shows semantic similarity, "
        "MSE shows pixel-level difference, "
        "SSIM shows structural similarity, "
        "Histogram correlation compares color distribution, "
        "ORB matches key visual features."
    )

    # Generate ORB visuals only
    orb_img1 = draw_orb_keypoints(image1)
    orb_img2 = draw_orb_keypoints(image2)

    return {
        "clip": round(clip_score, 2),
        "mse": round(mse_score, 2),
        "ssim": round(ssim_score, 2),
        "histogram": hist_corr,
        "orb": orb_feat,
        "explanation": explanation,
        "orb_img1": orb_img1,
        "orb_img2": orb_img2
    }

# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = Image.open(image1.file).convert("RGB")
    img2 = Image.open(image2.file).convert("RGB")

    similarity = compute_similarity_metrics(img1, img2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "similarity": similarity
        }
    )
