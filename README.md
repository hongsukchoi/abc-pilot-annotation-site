# abc-pilot-annotation rendered episodes — static site

Browsable gallery of rendered episodes from two annotation deliveries:

- **Finetuned tab** — 31 episodes from `abc_pilot_sample_annotation_20260416`
  (release `v1`).
- **Pretrained tab** — episodes from `abc_pilot_pretraining_20260424`
  (release `v2`).

Videos themselves are hosted as assets on GitHub **Releases** (not in git);
the static HTML references them by URL. Each tab shows
`<n> tasks · <m> episodes` next to its name so you can see the size at a
glance before clicking.

## What this directory contains

```
pilot_site/
  index.html                  # the site (vanilla HTML + JS, no build step)
  episodes.json               # metadata for the finetuned tab (31 episodes)
  episodes_pretrained.json    # metadata for the pretrained tab
  build_episodes_json.py      # regenerate <episodes_*>.json from a bundle + renders pair
  scripts/                    # data pipeline (manifest → bundles → renders)
    build_pilot_manifest.py
    materialize_pilot_bundles.py
    extract_subtask_annotations.py
    dataset_episode_to_video.py
  README.md                   # this file
```

The mp4s live in sibling directories (`pilot_renders/` for finetuned,
`pilot_renders_pretrained/` for pretrained) and are **uploaded to GitHub
Releases**, not committed to the repo.

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
# finetuned tab
uv run python pilot_site/build_episodes_json.py \
    --bundle-root pilot_data \
    --render-root pilot_renders \
    --output pilot_site/episodes.json

# pretrained tab
uv run python pilot_site/build_episodes_json.py \
    --bundle-root pilot_data_pretrained \
    --render-root pilot_renders_pretrained \
    --output pilot_site/episodes_pretrained.json
```

Then `git add` + commit + push the updated JSON.

## Adding the pretrained tab from scratch

The static site's pretrained tab pulls from `episodes_pretrained.json` and
videos from release `v2`. To regenerate end-to-end:

```bash
# 1. Cross-reference annotation mcaps with raw YAM datasets
AWS_PROFILE=far-compute python scripts/build_pilot_manifest.py \
    --preset pretrained \
    --annotation-profile xdof-bair-abc \
    --output pilot_manifest_pretrained.json

# 2. Materialize per-episode bundles (downloads cameras + extracts mcap labels)
AWS_PROFILE=far-compute python scripts/materialize_pilot_bundles.py \
    --manifest pilot_manifest_pretrained.json \
    --bundle-root pilot_data_pretrained \
    --annotation-profile xdof-bair-abc \
    --skip-existing

# 3. Render mp4s with the same banner layout as finetuned
python scripts/dataset_episode_to_video.py \
    --bundle-root pilot_data_pretrained \
    --output-root pilot_renders_pretrained \
    --layout simple

# 4. Generate episodes_pretrained.json
python build_episodes_json.py \
    --bundle-root pilot_data_pretrained \
    --render-root pilot_renders_pretrained \
    --output episodes_pretrained.json

# 5. Upload to release v2
gh release create v2 \
    --repo <owner>/abc-pilot-annotation-site \
    --title "Pilot rendered episodes v2 — pretraining" \
    --notes "Rendered mp4s from abc_pilot_pretraining_20260424."
gh release upload v2 pilot_renders_pretrained/*.mp4 \
    --repo <owner>/abc-pilot-annotation-site
```

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
