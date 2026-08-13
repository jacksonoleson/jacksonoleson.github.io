# jacksonoleson.github.io

Personal site + blog, built with [Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages.

## Structure

- `index.html` — profile home page, lists all posts
- `blog.html` — full post archive at `/blog/`
- `_posts/` — blog posts, named `YYYY-MM-DD-title.md`
- `_layouts/` — page templates
- `assets/css/style.scss` — styles
- `_config.yml` — site title, bio, and social links

## Run it locally

### Option A — Docker (no system Ruby setup needed)

```bash
./serve.sh
```

Open http://localhost:4000. Edits to posts, layouts, and CSS reload automatically
(changes to `_config.yml` require restarting the server).

### Option B — native Ruby

Requires Ruby dev headers, which need a one-time root install:

```bash
sudo apt install -y ruby-dev build-essential zlib1g-dev
gem install --user-install bundler
export PATH="$(ruby -e 'puts Gem.user_dir')/bin:$PATH"   # add to ~/.bashrc
bundle install
bundle exec jekyll serve --livereload
```

To preview drafts (files in `_drafts/` with no date in the filename), add `--drafts`.

## Adding a post

Create `_posts/YYYY-MM-DD-my-title.md`:

```markdown
---
title: "My Title"
description: "One-line summary shown on the home page."
tags: [ai, math]
---

Content here. Math works with `$inline$` and `$$display$$`.
```

Leave a blank line above and below `$$...$$` blocks, otherwise they render
inline in the middle of the paragraph instead of centered on their own line.

### Images

Each post gets its own folder under `assets/images/`, named after the post slug
(the filename minus the date):

```
_posts/2026-01-15-linear_regression_backprop.md
assets/images/linear_regression_backprop/two_pts.png
```

Reference them with an absolute path so they resolve from any post URL:

```markdown
![Two points](/assets/images/linear_regression_backprop/two_pts.png)
```

Images can't live inside `_posts/` itself — Jekyll reads subfolders there as
categories and doesn't publish non-post files.
