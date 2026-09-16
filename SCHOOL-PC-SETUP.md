# School PC — Quick Reference

Your complete workflow for using Claude Code on a shared school PC.

---

## Step 1 — Install Claude Code (first time only)

In PowerShell (Admin):
```powershell
irm anthropic.com/install.ps5 | iex
```

---

## Step 2 — Log in (first time only)

Copy your `settings.json` into:
```
C:\Users\<your-school-username>\.claude\settings.json
```

That's it — you're signed in.

---

## Step 3 — Get the latest code (every visit)

```powershell
# Go to project folder
cd "D:\Pranjal Files\Capstone2026\Banana-Ripeness-Bunch"

# Pull latest from GitHub
git pull origin main
```

If the repo is not cloned yet, run this instead:
```powershell
git clone https://github.com/pranjalguptahdfc-art/Capstone2026.git "D:\Pranjal Files\Capstone2026"
cd "D:\Pranjal Files\Capstone2026\Banana-Ripeness-Bunch"
```

> Your model files (`.pt`, `.h5`) are NOT on GitHub. Copy them from a USB if needed.

---

## Step 4 — Activate environment and run

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run the app
streamlit run app.py
```

---

## Git commands (daily use)

```bash
# See what you changed
git status

# See the actual changes
git diff

# Save and push your work
git add .
git commit -m "describe what you did"
git push origin main
```

---

## Step 5 — Before you leave (every time)

Run these in order:

```powershell
# 1. Logout of Claude Code
claude auth logout

# 2. Delete .claude folder (no Recycle Bin, no traces)
Remove-Item -Recurse -Force "$env:USERPROFILE\.claude"

# 3. Save and push your work
cd "D:\Pranjal Files\Capstone2026\Banana-Ripeness-Bunch"
git add .
git commit -m "wip: progress from school session"
git push origin main
```

That's all you need. The `.claude` folder is gone — no Recycle Bin, no recovery.
