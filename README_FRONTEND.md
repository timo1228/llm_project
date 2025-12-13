# NutriVision Frontend

**NutriVision: A Parameter-Efficient LMM for Food Image-to-Nutrition Analysis**

## Tech Stack

- **Frontend Framework**: React 18
- **Build Tool**: Vite
- **Styling Framework**: Tailwind CSS
- **Backend API**: FastAPI

## Features

1. **Image Upload**: Support for selecting image files from local storage
2. **Image Preview**: Preview selected images before submission
3. **Task Submission**: Submit images to backend API (`localhost:8000/generate`)
4. **Loading Animation**: Display animated loading indicators while processing
5. **Text Display**: Properly display returned text content with line breaks and long text support

## Installation & Setup

### 1. Create Conda Environment

```bash
bash setup_conda_env.sh
```

Or manually:

```bash
conda create -n image-upload python=3.10 -y
conda activate image-upload
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

Requires Node.js (v18 or higher recommended), then:

```bash
cd frontend
npm install
```

### 4. Start Services

**Start Backend Server** (Terminal 1):
```bash
conda activate image-upload
cd backend
python main.py
```

Backend will run at `http://localhost:8000`.

**Start Frontend Development Server** (Terminal 2):
```bash
cd frontend
npm run dev
```

Frontend will run at `http://localhost:3000`.

### 5. Access Application

Open `http://localhost:3000` in your browser.

## API Endpoint

### POST /generate

Receive an image file and return generated nutrition report.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response**:
```json
{
  "text": "Generated nutrition report text...",
  "report": "Generated nutrition report text...",
  "filename": "image.jpg",
  "status": "success"
}
```

## Project Structure

```
Project/
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt
└── setup_conda_env.sh       # Environment setup script
```

## Notes

- Ensure backend service is running at `localhost:8000` before starting frontend
- Frontend connects to backend API via proxy configuration
- Recommended image file size: under 10MB
- Supports common image formats (jpg, png, gif, webp, etc.)
