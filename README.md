# Qwen-Image Full-Stack Generator 🎨

A professional, locally-run text-to-image generation system using the Qwen-Image model, featuring a modern full-stack architecture with a FastAPI backend and a React/Tailwind CSS frontend.

## Features

- **Modern UI**: A sleek and responsive user interface built with React and Tailwind CSS.
- **Advanced Controls**: Professional options for image generation, including multiple modes (Text-to-Image, Image-to-Image, Inpainting, Super-Resolution).
- **Style Picker**: Choose from a variety of professional styles to enhance your creations.
- **FastAPI Backend**: A robust and fast backend server that exposes the power of the Qwen-Image model through a REST API.
- **Local & Private**: All processing happens on your local machine, ensuring privacy and control.
- **Hardware Optimized**: Tuned for high-end hardware for optimal performance.

## Architecture

This project consists of two main components:

-   **`backend/`**: A Python application powered by **FastAPI**. It handles the core image generation logic using the `diffusers` library and the Qwen-Image models. It exposes several API endpoints for the frontend to consume.
-   **`frontend/`**: A modern single-page application (SPA) built with **React** and styled with **Tailwind CSS**. It provides a user-friendly interface for interacting with the image generation capabilities of the backend.

The two components run concurrently and communicate via a REST API.

## System Requirements

The system requirements are the same as the original project, with a focus on high-end hardware for the best experience.

### Recommended Hardware

-   **GPU**: NVIDIA RTX 4080 (16GB VRAM) or better
-   **CPU**: High-core-count processor (e.g., AMD Threadripper)
-   **RAM**: 32GB minimum, 128GB recommended
-   **Storage**: 60-70GB for models and generated images

## Quick Start

### 1. Setup

**Backend:**

First, set up the Python environment and install the required dependencies.

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

**Frontend:**

Next, set up the Node.js environment for the React application.

```bash
# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### 2. Launch the Application

A convenient launch script is provided to start both the backend and frontend servers concurrently.

```bash
# From the backend directory
python launch.py
```

This will:
-   Start the FastAPI backend server on `http://localhost:8000`.
-   Start the React frontend development server on `http://localhost:3000`.

Open your browser and navigate to **`http://localhost:3000`** to use the application.

## Development

If you want to run the servers independently for development purposes:

**To run the backend:**

```bash
cd backend/src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**To run the frontend:**

```bash
cd frontend
npm run dev -- --port 3000
```

## Project Structure

```
/
├── backend/                # FastAPI backend application
│   ├── src/
│   │   ├── main.py         # FastAPI app definition
│   │   └── qwen_generator.py # Core generation logic
│   ├── launch.py           # Script to launch the full-stack app
│   └── requirements.txt
│
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── components/     # UI components
│   │   └── api.js          # API client
│   └── package.json
│
└── README.md               # This file
```
