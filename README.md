# MNIST Foundation Project

A containerised Streamlit + PyTorch digit recogniser, backed by PostgreSQL feedback logging.

---
## 🚀 To run this project, there are 2 options..
**Option 1** (Recommanded): Run on a public server
- **Browse** to: `http://188.245.86.47:8501`
---

**Option 2**: Run locally

1. 🔧 To prepare the **Local environment**, make sure you have installed:

- **Git** (to clone this repository)  
  - Windows: [Git for Windows](https://git-scm.com/download/win)  
  - macOS: `brew install git` or `xcode-select --install`  
  - Linux: `sudo apt install git` (Debian/Ubuntu)  
- **Docker & Docker Compose**  
  - Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)  
  - Linux: follow your distro’s [Docker install docs](https://docs.docker.com/engine/install/)

Once those are in place, you can run these in the Git Bash:

2. **Clone** the repo:  
   ```bash
   git clone https://github.com/EsperanzaShi/MLinstitute_Foundation.git
   ```
   ```bash
   cd MLinstitute_Foundation
   ```

3. **Launch** the app and database:  
   ```bash
   docker compose up -d --build
   ```

4. **Browse** to: `http://localhost:8501`  
---

## 📁 Repository Structure
```text
MLinstitute_Foundation/
├── db-init/                       # SQL to initialise PostgreSQL schema
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
2. **Click** **'Classify'** to see:  
   - **Prediction:** model’s guess (0–9)  
   - **Confidence:** probability of its guess  
3. **Click** **'Give Feedback'**, select the **true label**, then **Submit true label**.  
4. Your feedback is logged to the history table (and to Postgres).
5. To continue drawing and predicting more numbers, **click** **'New'** twice to reset the canvas.

---

## ⚙️ Deployment

- **Server:** Hetzner CX11 (1 vCPU, 2 GB RAM)  
- **Launch:**  
  ```bash
  docker compose up -d --build
  ```
- Access: http://188.245.86.47:8501
  
