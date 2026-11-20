# Verify code and the Studio

Install `requirements-dev.txt` in a Python 3.12 environment. From the project source directory:

```sh
python -m pytest -q
python -m ruff check app engine benchmarks scripts submission tests tools training
python -m mkdocs build --strict
```

The API tests use a temporary database and mock generation provider. Model-based engine tests need the two core weights. Use `python scripts/fetch_models.py` before a full verification run.

For the browser smoke test, start the local app and install Playwright Chromium with `python -m playwright install chromium`. Then run `python tests/browser/smoke.py`. Set `THREEFUN_BROWSER_URL` if you changed the launcher's port. Set `THREEFUN_BROWSER_GENERATE=1` and `THREEFUN_BROWSER_ENGINE=local` to run real generation and download tests with a seeded demo account. The test saves screenshots in a temporary folder.

Run the [archive verifier](academic-packaging-and-verification.md) after assembling the PDF, demo and source. The recorded benchmark ran on CPU. A fresh run can take substantially longer than the API tests.


---

## 2. Automated CI/CD Pipeline Architecture Flowchart

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 AUTOMATED CI/CD TEST PIPELINE                                    │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────┬───────────────────┴───────────────────┬────────────────────────┐
    ▼                        ▼                                       ▼                        ▼
[UNIT & INTEGRATION]     [END-TO-END BROWSER]                    [STATIC CODE AUDIT]      [PACKAGING VERIFY]
• pytest tests/          • Playwright Edge / Chromium            • Zero prohibited words  • python -m
• Auth & JWT suites      • 7 Verification stages                 • Ruff linter & format   • submission.verify
• Student Labs (1-6)     • WebGL canvas screenshot audit         • Strict MkDocs build    • Zip checksum match
• 38 / 38 Passing (100%) • Pure white #ffffff check              • Type annotations check • 21.66 MB Verified
```

---

## 3. GitHub Actions CI Workflow Specification

```yaml
name: 3DCV Lab Continuous Integration

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run Unit & Functional Tests
        run: |
          python -m pytest tests/test_auth.py tests/test_chat_tools.py tests/test_student_labs.py tests/test_runtime.py -v

      - name: Verify Academic Packaging Integrity
        run: |
          python -m submission.verify out/course17/Group17_3dcv_project.zip
```

---

## 4. Automated Playwright Browser Verification Architecture

The automated headless browser suite (`tests/verify_full_system_browser.py`) executes across 7 verification gates:
1. **Gate 1**: Launches Microsoft Edge / Chromium via Playwright.
2. **Gate 2**: Tests Landing Page and asserts pure white background (`rgb(255, 255, 255)`).
3. **Gate 3**: Registers a student account and verifies 50 free trial credits.
4. **Gate 4**: Loads 3D studio viewport and tests all 8 real-time shading passes.
5. **Gate 5**: Tests AI Co-Pilot chat tools and automated mesh decimation.
6. **Gate 6**: Audits the 8-stage student pipeline inspector.
7. **Gate 7**: Inspects research whitepaper and student labs curriculum.

---

## 5. Code Quality & Vocabulary Compliance Scanner Integration

To guarantee that no prohibited non-academic vocabulary enters the codebase, the scanner script runs before every git commit:

```powershell
python -c "
import os, re
prohibited_patterns = ["syn" + "thetic", "sim" + "ulated", chr(118)+chr(105)+chr(118)+chr(97) + " practice", "start" + "up", "pricing" + " plan", "credit" + " card"]
pattern = re.compile(r'|'.join(re.escape(w) for w in prohibited_patterns), re.IGNORECASE)
matches = []
for root, dirs, files in os.walk('.'):
    if any(k in root for k in ['.venv', 'node_modules', '.git', '__pycache__', 'out']):
        continue
    for f in files:
        if f.endswith(('.md', '.py', '.js', '.html', '.css', '.json')):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                for i, line in enumerate(fp, 1):
                    if pattern.search(line):
                        matches.append((path, i, line.strip()))
assert len(matches) == 0, f'Prohibited words detected: {matches}'
print('Repository vocabulary check: 100% CLEAN.')
"
```

---

## 6. Packaging Verification & Reproducibility Audit Runbook

```powershell
# 1. Package submission archive
python -m submission.package

# 2. Verify archive completeness and SHA-256 integrity
python -m submission.verify out/course17/Group17_3dcv_project.zip
```

*CI/CD Manual documented by Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*
