# abc-pilot-annotation rendered episodes — static site

Browsable gallery of the 31 rendered videos from the
`abc_pilot_sample_annotation_20260416` delivery. Videos themselves are hosted
as assets on a GitHub **Release** (not in git), and the static HTML below
references them by URL.

## What this directory contains

```
pilot_site/
  index.html                 # the site (vanilla HTML + JS, no build step)
  episodes.json              # metadata for all 31 episodes
  build_episodes_json.py     # regenerate episodes.json from pilot_data/ + pilot_renders/
  README.md                  # this file
```

The 31 mp4s live in a sibling directory (`pilot_renders/`) and are **uploaded
to a GitHub Release**, not committed to the repo.

## Deployment (once)

Copy `pilot_site/` onto your laptop. Also copy or download `pilot_renders/`
(~689 MB) — you'll upload those to the release below.

### 1. Create the repo (private)

```bash
# from the laptop, inside where you copied pilot_site/
cd pilot_site
git init
git add index.html episodes.json build_episodes_json.py README.md
git commit -m "Initial pilot-annotation gallery"

# Create a private repo and push. Requires GitHub Pro/Team/Enterprise
# to enable Pages on a private repo.
gh repo create abc-pilot-annotation-site --private --source=. --push
```

Save the final repo slug as `<owner>/<repo>` — you'll need it twice more.

### 2. Create a release and upload the 31 mp4s

```bash
# From anywhere on your laptop that has the mp4s (adjust path):
cd path/to/pilot_renders

# Create the release tag "v1" in the private repo
gh release create v1 \
    --repo <owner>/abc-pilot-annotation-site \
    --title "Pilot rendered episodes v1" \
    --notes "31 rendered mp4s from abc_pilot_sample_annotation_20260416."

# Upload all 31 mp4s (in parallel-ish; gh uploads serially)
gh release upload v1 *.mp4 \
    --repo <owner>/abc-pilot-annotation-site
```

If the upload is interrupted, rerun with `--clobber` to overwrite existing
assets.

### 3. Point `index.html` at your release

Open `index.html` and edit the `CONFIG` block near the top:

```js
const CONFIG = {
    repoOwner: "<owner>",
    repoName:  "abc-pilot-annotation-site",
    releaseTag: "v1",
};
```

Commit and push:

```bash
git commit -am "Wire CONFIG to <owner>/abc-pilot-annotation-site/v1"
git push
```

### 4. Enable GitHub Pages

In the repo on github.com: **Settings → Pages**.
- Source: *Deploy from a branch*, branch `main`, folder `/ (root)`
- Save. First deploy takes 1–3 minutes.

Your site will be at `https://<owner>.github.io/abc-pilot-annotation-site/`.

## Important: private-repo authentication

GitHub Release assets on a **private** repo require GitHub authentication to
download. A `<video src="https://github.com/<owner>/<repo>/releases/download/v1/file.mp4">`
tag works in the browser **if and only if** the viewer is currently logged
into github.com in the same browser and has access to the repo.

Anonymous visitors will see broken/404 videos. Options if that's a problem:
- Make the repo public (videos become world-readable).
- Host the mp4s on a separate, *public* GitHub repo's release (keep the
  metadata site private).
- Host the mp4s in S3 / Drive with signed URLs and rewrite `videoUrl()` in
  `index.html`.

## Regenerating `episodes.json`

If `pilot_data/` or `pilot_renders/` changes upstream, re-run:

```bash
uv run python pilot_site/build_episodes_json.py \
    --bundle-root pilot_data \
    --render-root pilot_renders \
    --output pilot_site/episodes.json
```

Then `git add` + commit + push the updated JSON.

## Local preview (before pushing)

Serve the directory locally and open in a browser:

```bash
cd pilot_site
python3 -m http.server 8000
# open http://localhost:8000/
```

(You won't see videos until `CONFIG` points at a real release and you're
authenticated, but the card grid, filter, and metadata are all browsable
locally.)
