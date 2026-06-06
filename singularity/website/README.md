# Singularity download site (Vercel)

A one-page download site. It detects your friend's OS and hands him the right
file. The actual app is built by GitHub Actions and hosted on **GitHub
Releases** — this site just links to it. (Vercel hosts the page, not the 100MB+
binary, and can't compile the app.)

## The whole pipeline (one-time)

```
You push the code to GitHub
        │
        ├─ tag a version  →  GitHub Actions builds Windows + Mac apps
        │                    and attaches them to a GitHub Release
        │
        └─ deploy this /website folder to Vercel
                             │
                  Friend visits your Vercel URL → clicks Download
                  → gets the right file from the GitHub Release
```

## Step 1 — Publish the app to GitHub Releases

1. Put the whole `singularity` project in a GitHub repo.
2. Create a version tag so the build runs and publishes a Release:
   - Easiest in the browser: repo → **Releases** → *Draft a new release* →
     *Choose a tag* → type `v1.0.0` → **Publish release**.
   - The `Build Singularity` Action runs and attaches
     `Singularity-Windows.zip` and `Singularity-macOS.zip` to that release.
3. Your stable download links are now:
   - `https://github.com/<you>/<repo>/releases/latest/download/Singularity-Windows.zip`
   - `https://github.com/<you>/<repo>/releases/latest/download/Singularity-macOS.zip`

## Step 2 — Point the site at your repo

Open `website/index.html` and edit the two lines near the bottom:

```js
const GH_USER = "your-github-username";
const GH_REPO = "singularity";
```

## Step 3 — Deploy to Vercel (pick one)

- **GitHub import (recommended):** vercel.com → *Add New → Project* → import your
  repo → set **Root Directory** to `website` → Deploy.
- **Drag & drop:** vercel.com → *Add New → Project → deploy a folder* → drop the
  `website` folder.
- **CLI:** `npm i -g vercel`, then from the `website` folder run `vercel`.

Vercel gives you a URL like `https://singularity.vercel.app`. Send that to your
friend — he clicks Download, gets the right file, opens it, and follows the
in-app setup. Same flow, nicer front door.

## Updating later

Push a new tag (e.g. `v1.0.1`). The Action rebuilds and the `latest/download`
links automatically point to the new files — the Vercel page needs no change.
