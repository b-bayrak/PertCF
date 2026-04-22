# GitHub & PyPI Deployment Guide

Follow these steps in order to go from this folder to a live PyPI package
and GitHub Pages documentation site.

---

## Step 1 — Prepare your GitHub repository

```bash
# In this folder:
git init
git add .
git commit -m "feat: PertCF 1.0.0 — dependency-light rewrite (removes myCBR)"

# Push to your existing repo (replace the remote if needed)
git remote add origin https://github.com/b-bayrak/PertCF-Explainer.git
git push -u origin main --force
```

> **Note:** This rewrites the repo history. If you want to preserve the
> old myCBR-based commits on a separate branch first:
> ```bash
> git push origin main:main-legacy   # save old history
> git push origin main --force       # push new clean version
> ```

---

## Step 2 — Reserve the PyPI package name

1. Go to https://pypi.org and sign in (or create an account).
2. The name `pertcf` must not already be taken — check at
   https://pypi.org/project/pertcf/
3. If available, it will be claimed automatically on your first upload.

---

## Step 3 — Set up Trusted Publisher (no API tokens needed)

This is the modern OIDC approach — no secrets to manage.

1. On PyPI: go to **Account settings → Publishing → Add a new pending publisher**
2. Fill in:
   - PyPI project name: `pertcf`
   - Owner: `b-bayrak`
   - Repository name: `PertCF-Explainer`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. On GitHub: go to **Settings → Environments → New environment**
   - Name: `pypi`
   - No protection rules needed for a personal repo

---

## Step 4 — Create a GitHub Release to trigger PyPI publish

```bash
# Tag the release
git tag v1.0.0
git push origin v1.0.0

# Then on GitHub:
# Releases → Draft a new release → choose tag v1.0.0
# → "Generate release notes" → Publish release
```

The `publish.yml` workflow triggers automatically and uploads
`pertcf-1.0.0-py3-none-any.whl` and `pertcf-1.0.0.tar.gz` to PyPI.

---

## Step 5 — Enable GitHub Pages for documentation

```bash
pip install mkdocs-material mkdocstrings[python]

# Preview locally:
mkdocs serve
# → open http://127.0.0.1:8000

# Deploy to GitHub Pages:
mkdocs gh-deploy
```

This pushes the built site to the `gh-pages` branch.
Then on GitHub: **Settings → Pages → Source: Deploy from branch → gh-pages**.

Your docs will be live at: https://b-bayrak.github.io/PertCF-Explainer

---

## Step 6 — Register on Papers with Code

1. Go to https://paperswithcode.com/paper/pertcf-a-perturbation-based-counterfactual
   (search if the URL differs)
2. Click **"Add implementation"** and link to:
   - Repo: `https://github.com/b-bayrak/PertCF-Explainer`
   - PyPI: `https://pypi.org/project/pertcf/`
3. This adds your paper to the XAI / counterfactual explanations leaderboard
   and makes it discoverable by researchers browsing the site.

---

## Step 7 — Get a Zenodo DOI (software citation)

1. Go to https://zenodo.org and log in with your GitHub account.
2. Under **GitHub → Flip the switch** next to `PertCF-Explainer`.
3. Create the v1.0.0 GitHub Release (Step 4 above).
4. Zenodo automatically archives it and mints a DOI like `10.5281/zenodo.XXXXXXX`.
5. Update `CITATION.cff` with the DOI and add the badge to README.md:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

---

## Step 8 — Add the Colab launch badge (already in notebooks)

The `.ipynb` files already contain Colab badges. Once pushed to GitHub,
the links will be live:

```
https://colab.research.google.com/github/b-bayrak/PertCF-Explainer/blob/main/examples/quickstart_german_credit.ipynb
```

---

## Checklist

| Task | Est. time | Status |
|---|---|---|
| Push repo to GitHub | 5 min | ☐ |
| Set up Trusted Publisher on PyPI | 10 min | ☐ |
| Create GitHub Release v1.0.0 | 5 min | ☐ |
| Verify PyPI page looks correct | 5 min | ☐ |
| Run `mkdocs gh-deploy` | 5 min | ☐ |
| Register on Papers with Code | 10 min | ☐ |
| Flip Zenodo switch + get DOI | 10 min | ☐ |
| Update CITATION.cff with Zenodo DOI | 5 min | ☐ |
| Open ORCID account and add paper | 15 min | ☐ |

**Total: ~70 minutes of one-time setup.**  
After that, future releases are one GitHub tag away.

---

## Maintenance

- New release: bump `version` in `pyproject.toml` → update `CHANGELOG.md`
  → `git tag vX.Y.Z` → push → create GitHub Release.
- Docs update: edit `docs/` → `mkdocs gh-deploy`.
- No server to maintain. No external dependencies to update.
