# Deployment Guide: Hugging Face Spaces 🚀

Your project is now 100% prepared and configured for a seamless deployment to **Hugging Face Spaces**. We have optimized the configurations (like changing the default port to `7860`) so that it works perfectly out-of-the-box.

## Steps to Deploy on Hugging Face Spaces

1. **Create a Hugging Face Account**
   If you don't have one, sign up for free at [huggingface.co](https://huggingface.co/join).

2. **Create a New Space**
   - Go to your profile and click **New Space** (or visit [huggingface.co/new-space](https://huggingface.co/new-space)).
   - **Space name**: `Teeth-Classification` (or whatever you prefer)
   - **License**: Choose your preferred license (e.g., MIT).
   - **Space SDK**: Select **Docker**.
   - **Docker Template**: Select **Blank** (since we already have a custom Dockerfile).
   - **Space Hardware**: Free tier (CPU basic - 16GB RAM) is sufficient.
   - Click **Create Space**.

3. **Deploy the Code**
   Once the space is created, you have two options to deploy the code from your GitHub repository:

   **Option A: Import from GitHub (Easiest)**
   Hugging Face allows you to link a GitHub repository directly to a Space. Go to the settings in your Space and link your repo:
   `https://github.com/ahmedmohamedabdelsalam/-Teeth-Classification`

   **Option B: Clone and Push**
   Hugging Face will provide you with Git commands to push your code directly to their platform. 
   Since your code is already perfectly structured locally, you can simply add the Hugging Face remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```

4. **Wait for Build**
   Hugging Face Spaces will automatically detect the `Dockerfile`, install the dependencies from `requirements.txt`, and launch the app.
   You will see the building logs, and once it says **"Running"**, your premium diagnostic UI will be live globally!

---
*Note: Due to the high-accuracy model size (`teeth_model.keras`), the initial Docker build process on Hugging Face may take a few minutes.*
