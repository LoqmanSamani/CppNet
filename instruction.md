# CppNet — Web Page Creation Instructions

> Instructions for an agent tasked with building and deploying a static GitHub Pages website for the **CppNet** C++ networking library.

---

## Phase 0 — Reconnaissance (Do This First)

Before writing a single line of code:

1. Read `README.md` in full. Extract: package purpose, feature list, logo path, any badges or links, and the repository URL.
2. Scan the `examples/` directory. For each file, note its filename and read its content to infer a short description and the most illustrative code snippet.
3. Scan the `benchmark/` directory. Read `benchmarks.md` and note all benchmark categories, metrics (throughput, latency, etc.), and result tables or charts. List any supporting files (scripts, raw data, graphs) present in the directory.
4. Identify the package's core value proposition in one sentence — this will anchor all copy on the site.
5. Do **not** invent or assume any content — all copy must be grounded in what you find in the repository.

---

## Phase 1 — Project Setup

6. Create a `docs/` directory at the repo root. GitHub Pages is configured to serve from `docs/` on the `main` branch — this avoids extra configuration.
7. Add an empty `docs/.nojekyll` file to prevent GitHub's Jekyll processor from interfering with the static site.
8. Build a **fully static site** (pure HTML + CSS + JS, no build step, no framework). The site must work by simply opening `docs/index.html` locally.
9. Use a **single-page layout** with smooth-scroll anchor navigation. Do **not** create separate HTML files per section. Required anchor IDs:
   - `#home`
   - `#quickstart`
   - `#examples`
   - `#benchmarks`
   - `#author`
10. All asset paths (logo, fonts, etc.) must be **relative** so the site works both locally and when served from GitHub Pages.

---

## Phase 2 — Design System

11. Use a **dark theme** appropriate for a C++ systems library:
    - Background: deep navy (`#0d1117`) or dark charcoal (`#161b22`)
    - Accent: electric blue or cyan (e.g., `#58a6ff`)
    - Body font: clean sans-serif (Inter or system-ui)
    - Code font: monospace (JetBrains Mono or Fira Code via Google Fonts CDN)
12. Embed **[highlight.js](https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js)** via CDN for C++ syntax highlighting on all code blocks. Initialize with `hljs.highlightAll()`.
13. The site must be **fully responsive** — test layouts at mobile (≤768px) and desktop (≥1024px).
14. Include a **sticky top navigation bar** containing:
    - Package logo + name (left)
    - Nav links: Home · Quick Start · Examples · Benchmarks · Author (center/right)
    - "View on GitHub" button (rightmost, styled as an outlined button)

---

## Phase 3 — Section Content

### 3.1 — Home (`#home`)

15. Display the package logo (use the path found in `README.md`), the package name as an `<h1>`, and the one-liner description.
16. Add two CTA buttons:
    - **"Get Started"** — scrolls to `#quickstart`
    - **"View on GitHub"** — links to the repository URL
17. List 3–5 key features (extracted from `README.md`) as icon cards or short bullet rows beneath the hero.

### 3.2 — Quick Start (`#quickstart`)

18. Show all installation steps as copy-able code blocks. Add a **clipboard icon button** to every code block — clicking it copies the content and briefly changes the icon to a checkmark.
19. Show a minimal "Hello World" equivalent — the simplest possible working usage of the library.
20. Clearly label any prerequisites (compiler version, CMake version, OS requirements, dependencies) in a callout box or highlighted note.

### 3.3 — Examples (`#examples`)

21. For **each file** found in `examples/`, generate a card containing:
    - Example name (derived from the filename, humanized)
    - A one-sentence description (inferred from the file content — do not invent)
    - The most illustrative code snippet from that file, displayed in a syntax-highlighted block
    - A "View on GitHub" link pointing to the raw file in the repository
22. If there are more than 4 examples, display them in a 2-column grid on desktop, 1-column on mobile.

### 3.4 — Benchmarks (`#benchmarks`)

23. Read `benchmark/benchmarks.md` fully before writing this section.
24. Display a short introductory paragraph explaining what was benchmarked and the test environment (extract from `benchmarks.md`).
25. Render all benchmark result **tables** from `benchmarks.md` as styled HTML tables (not raw markdown). Apply alternating row shading and highlight the best-performing row per table.
26. If `benchmark/benchmarks.md` contains or references any **charts or graphs**, reproduce them — either as inline SVG, a Canvas-based chart using [Chart.js](https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js) (CDN), or as `<img>` tags if image files exist in `benchmark/`.
27. Include a callout noting the benchmark methodology (hardware specs, compiler flags, number of runs) if that information is present in `benchmarks.md`.
28. Add a link to the raw `benchmark/benchmarks.md` file on GitHub for users who want the full detail.

### 3.5 — Author (`#author`)

29. Fetch information from **https://loqmansamani.github.io/** — use the name, one-line bio, and available contact links.
30. Display:
    - Name as `<h2>`
    - One-line bio
    - Contact icon links: GitHub, LinkedIn, X/Twitter — use SVG icons from [Simple Icons CDN](https://cdn.simpleicons.org/) or inline SVGs. Icons should be white/light colored and enlarge slightly on hover.
31. Keep this section **minimal**: no walls of text. Avatar image optional — only include if one is clearly available from the personal site.

---

## Phase 4 — Quality Checks

Before committing:

32. Open `docs/index.html` in a browser and verify:
    - [ ] All nav links scroll to the correct section
    - [ ] Logo image loads correctly
    - [ ] All code blocks are syntax-highlighted
    - [ ] Clipboard buttons work
    - [ ] All example cards render with correct file-linked URLs
    - [ ] Benchmark tables are fully rendered with no raw markdown visible
    - [ ] Author contact icons link to correct URLs
    - [ ] Site is readable on mobile (≤768px)
33. Verify all internal asset paths are **relative** (no absolute `/` paths that would break on the GitHub Pages subdirectory URL).
34. Run a quick HTML validation pass and fix any broken tags or missing `alt` attributes on images.

---

## Phase 5 — GitHub Pages Deployment

35. In the repository settings, set GitHub Pages source to: **Branch: `main`, Folder: `/docs`**.
36. Commit and push the `docs/` directory to the `main` branch.
37. The site will be available at:
    ```
    https://<username>.github.io/<repository-name>/
    ```
38. After deployment, do a final smoke test on the live URL — confirm the logo, code blocks, benchmark tables, and author links all render correctly in the deployed environment.

---

## Constraints & Non-Negotiables

- **No frameworks** (no React, Vue, Tailwind CLI, etc.) — CDN-only external dependencies.
- **No build step** — the site must be openable as plain HTML.
- **No invented content** — every claim about the library must come from `README.md`, `examples/`, or `benchmark/`.
- **No separate HTML files** — everything in one `index.html` with anchor navigation.
- All **external CDN links** must use pinned versions (not `latest`) for reproducibility.
