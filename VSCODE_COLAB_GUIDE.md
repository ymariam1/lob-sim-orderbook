# Running Colab Notebook in VSCode

## Setup (One-time)

### 1. Install VSCode Colab Extension
- Open VSCode
- Press `Cmd+Shift+X` (Extensions)
- Search: "Colaboratory"
- Install the official Google Colab extension by Google

### 2. Verify Installation
- Press `Cmd+Shift+P` (Command Palette)
- Type: "Colab"
- You should see commands like:
  - "Colab: Open in Google Colab"
  - "Colab: Sign in"

---

## Using Your Notebook

### 1. Open the Notebook in VSCode
```bash
# Already done - you have colab_training.ipynb open
```

### 2. Connect to Colab
**Option A: Through Command Palette**
- Press `Cmd+Shift+P`
- Type: `Colab: Open in Google Colab`
- Select `colab_training.ipynb`

**Option B: Through Kernel Selector**
- Click the kernel selector (top right of notebook)
- Select "Select Another Kernel"
- Choose "Google Colab"
- Sign in with your Google account

**Option C: Status Bar**
- Look for Colab icon in the status bar (bottom)
- Click to connect

### 3. Enable GPU
Once connected to Colab:
- In VSCode, you'll see a "Runtime" option
- Or go to the Colab web interface
- Runtime → Change runtime type → GPU (T4)

### 4. Run Cells
Now you can run cells directly in VSCode:
- Click the "Run" button on each cell
- Or press `Shift+Enter` to run current cell
- Or use "Run All" from the toolbar

**The notebook will:**
- ✅ Mount your Google Drive (you'll need to authorize)
- ✅ Access your 28 CSV files from `/csv` folder
- ✅ Auto-split 80/20 (22 train, 6 test)
- ✅ Train on Colab GPU
- ✅ Save models to Drive

---

## Workflow Summary

```
VSCode (Local IDE)
    ↓
Colab Extension
    ↓
Google Colab (Remote GPU)
    ↓
Google Drive (Your CSV data)
```

### Benefits:
- 🎨 Edit in VSCode (better IDE)
- 🚀 Run on Colab GPU (free)
- 💾 Data stays in Drive (accessible)
- 📊 See output in VSCode

---

## Step-by-Step First Run

### Cell 1: Check GPU
```python
!nvidia-smi  # Should show Tesla T4
```

### Cell 2: Clone Repo
```python
# Replace YOUR_USERNAME with your GitHub username
!git clone https://github.com/YOUR_USERNAME/lob-sim-orderbook.git
%cd lob-sim-orderbook
```

### Cell 3: Install Dependencies
```python
!pip install -e .
!pip install stable-baselines3[extra] wandb tensorboard
# Takes ~2 minutes
```

### Cell 4: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')  # Click auth link, allow access
```
**Important:** A popup will appear asking for Google Drive authorization. Click the link and authorize.

### Cell 5: Auto Split Data
```python
# Automatically creates:
# - data/train/ (22 files - 80%)
# - data/test/ (6 files - 20%)
```

### Cell 7: Train!
```python
# Runs training on Colab GPU
# Models save to: /content/drive/MyDrive/lob_models/
# Takes ~20 mins for 100k timesteps
```

---

## Tips for VSCode Colab

### Keep Session Alive
Colab will disconnect after ~90 mins of inactivity. To prevent this:
- Keep the VSCode window active
- Or add a keep-alive cell (Cell 6 has TensorBoard which keeps it active)

### View Outputs
- Outputs appear directly in VSCode
- TensorBoard will open in a new browser tab
- Print statements show in the cell output

### Edit Code
You can edit cells in VSCode then run them on Colab:
- Change hyperparameters
- Adjust training config
- Modify code
- Run on GPU immediately

### Debugging
- Add print statements
- Use `!ls`, `!pwd` to navigate
- Check files with `!cat filename`

---

## Troubleshooting

### "Cannot connect to Colab"
1. Sign out and sign back in:
   - `Cmd+Shift+P` → "Colab: Sign out"
   - `Cmd+Shift+P` → "Colab: Sign in"
2. Restart VSCode
3. Make sure you're logged into Google in your browser

### "No GPU detected"
1. In Colab web interface: Runtime → Change runtime type → GPU
2. Or add this cell at the top:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

### "CSV files not found"
1. Check Drive is mounted:
   ```python
   !ls /content/drive/MyDrive/
   ```
2. Verify csv folder exists:
   ```python
   !ls /content/drive/MyDrive/csv/
   ```
3. Check symlink:
   ```python
   !ls -la data/csv
   ```

### "Session disconnected"
Your Drive data is safe! Just:
1. Reconnect to Colab
2. Re-run cells 1-5 (setup cells)
3. Cell 7 will automatically resume from checkpoint

---

## Quick Start Checklist

- [ ] VSCode Colab extension installed
- [ ] `colab_training.ipynb` open in VSCode
- [ ] Connected to Colab (kernel shows "Colab")
- [ ] GPU enabled (T4 visible in nvidia-smi)
- [ ] CSV files uploaded to Drive `/csv` folder
- [ ] Ready to run!

**Run cells in order (1 → 2 → 3 → 4 → 5 → 7) and you're training!**

---

## Advanced: Custom Configuration

Edit Cell 7 to customize training:

```python
# Quick test (5 mins)
--timesteps 25000 --net-arch 32 32

# Small (20 mins) - Current default
--timesteps 100000 --net-arch 64 64

# Medium (1 hour)
--timesteps 500000 --net-arch 128 64

# Large (3 hours)
--timesteps 1000000 --net-arch 128 128 64
```

Just edit the cell in VSCode and run!
