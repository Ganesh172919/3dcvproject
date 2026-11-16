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
