# Logo Detection Inference API

Serves the YOLOv8 model trained in `../colab/train_logo_detector.ipynb` as a
simple HTTP API. Deployed separately from the Next.js app since Vercel's
serverless functions can't run PyTorch/YOLO.

## Deploying to Hugging Face Spaces (recommended, free)

1. Go to https://huggingface.co/new-space
2. Name it (e.g. `logo-detector-api`), choose **Docker** as the Space SDK,
   visibility **Private** if you don't want it publicly browsable (it'll
   still be reachable at its API URL either way -- Private just hides it
   from HF's public directory).
3. Once created, you'll get a git remote URL for the Space. Clone it, copy
   in `main.py`, `requirements.txt`, and `Dockerfile` from this folder, plus
   your trained `best.pt` file (downloaded from the Colab notebook) into
   the same folder.
4. `git add . && git commit -m "Add logo detection API" && git push`
5. The Space will build and start automatically (check the "Logs" tab if it
   fails). Once running, your API is live at:
   `https://YOUR-USERNAME-logo-detector-api.hf.space`

Test it works:

```bash
curl -X POST https://YOUR-USERNAME-logo-detector-api.hf.space/detect \
  -F "file=@/path/to/test-photo.jpg"
```

You should get back JSON like:

```json
{
  "detected": true,
  "best_match": { "brand": "nike", "confidence": 0.91 },
  "all_detections": [{ "brand": "nike", "confidence": 0.91 }]
}
```

## Alternative: Render.com

Also works if you'd rather not use Hugging Face. Create a new "Web Service",
point it at a repo containing these files, Render auto-detects the
Dockerfile. Free tier spins down when idle (first request after a while
takes ~30-60s to wake up) -- fine for a class project, worth upgrading if
this goes to real customers.

## Updating the model later

If you retrain with more data (recommended once you test on real customer
photos), just replace `best.pt` in the Space's repo and push again -- no
other changes needed.
