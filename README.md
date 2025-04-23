# MNIST Foundation Project

A containerized Streamlit + PyTorch digit recognizer, backed by PostgreSQL feedback logging.

---
## 🔧 Prerequisites

Before you begin, make sure you have installed:

- **Git** (to clone this repository)  
  - Windows: [Git for Windows](https://git-scm.com/download/win)  
  - macOS: `brew install git` or `xcode-select --install`  
  - Linux: `sudo apt install git` (Debian/Ubuntu)  
- **Docker & Docker Compose**  
  - Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)  
  - Linux: follow your distro’s [Docker install docs](https://docs.docker.com/engine/install/)

Once those are in place, you can run:

## 🚀 Quickstart With Commands

1. **Clone** the repo:  
   ```bash
   git clone https://github.com/EsperanzaShi/MLinstitute_Foundation.git
   cd MLinstitute_Foundation
   ```

2. **Launch** the app and database:  
   ```bash
   docker compose up -d --build
   ```

3. **Browse** to:  
   - **Local:**  `http://localhost:8501`  
   - **Server:** `http://188.245.86.47:8501`
  
---

## 📁 Repository Structure
```text
MLinstitute_Foundation/
├── db-init/                       # SQL to initialize PostgreSQL schema
│   └── 01-create-predictions.sql
├── mnist_classifier/              # Python app code & Dockerfile
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── trainmodel_CNN.py
│   ├── interactive_front_end.py
│   ├── db.py
│   └── …
├── docker-compose.yml             # Defines web (Streamlit) & db (Postgres) services
└── README.md                      # Project overview and instructions
```
---

## 🎓 How to Use the App

1. **Draw** a handwritten digit (0–9) on the left canvas.  
2. **Click** **Classify** to see:  
   - **Prediction:** model’s guess (0–9)  
   - **Confidence:** probability of its guess  
3. **Click** **Give Feedback**, select the **true label**, then **Submit true label**.  
4. Your feedback is logged to the history table (and to Postgres).

---

## ⚙️ Deployment

- **Server:** Hetzner CX11 (1 vCPU, 2 GB RAM)  
- **Launch:**  
  ```bash
  docker compose up -d --build
  ```
- Access: http://188.245.86.47:8501
  
