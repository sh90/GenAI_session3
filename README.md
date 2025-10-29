# 🧠 GenAI Module — Session 3  
### *Generative AI and Agentic AI Basics*

---

## ⚙️ Setup Instructions

Follow these steps to set up and run the project locally.

---

### 🪣 Step 1: Clone or Download the Repository

```bash
git clone https://github.com/sh90/GenAI_session3.git
```

Alternatively, you can [download the ZIP file](https://github.com/sh90/GenAI_session3/archive/refs/heads/master.zip) and extract it.

---

### 💻 Step 2: Download PyCharm IDE

You can install **PyCharm** (Community or Professional edition) from the official JetBrains website:

👉 [Download PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)

---

### 📂 Step 3: Open the Project in PyCharm

**File → Open → Select the cloned or downloaded folder**

---

### 🧩 Step 4: Create and Activate Virtual Environment

Run the following command in your project directory:

```bash
uv venv
source .venv/bin/activate    # for macOS/Linux
# OR
.venv\Scripts\activate       # for Windows
```

---

### 📦 Step 5: Install Dependencies

```bash
uv sync
```

---

### 🔐 Step 6: Create the `.env` File

Copy the example environment file to a new `.env` file:

```bash
cp .env.example .env
```

Add your keys or credentials inside the `.env` file as required.

---

### 🚀 Step 7: Run the Application

```bash
streamlit run .\vanilla_rag\streamlit_app_langchain.py
streamlit run .\hybrid_rag\streamlit_app_hybrid_rag.py
```

---

## ✅ Notes

- Ensure you have **Python ≥ 3.12** installed (as specified in `pyproject.toml`).
- The `uv` package manager should be installed globally:

- If `uv` is not recognized, check its path using:

  ```bash
  which uv      # macOS/Linux
  where uv      # Windows
  ```

