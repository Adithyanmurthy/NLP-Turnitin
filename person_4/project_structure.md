# Project Structure

```
nlp-content-integrity/
├── data/
│   ├── raw/                    # Person 1's raw datasets
│   ├── processed/              # Person 1's cleaned datasets
│   └── reference_index/        # Person 2's plagiarism index
├── models/
│   ├── ai_detection/           # Person 1's trained models
│   ├── plagiarism/             # Person 2's trained models
│   └── humanization/           # Person 3's trained models
├── src/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── ai_detector.py      # Person 1's module
│   │   ├── plagiarism_detector.py  # Person 2's module
│   │   └── humanizer.py        # Person 3's module
│   ├── pipeline.py             # Person 4: Integration
│   ├── config.py               # Person 4: Configuration
│   └── utils.py                # Person 4: Utilities
├── api/
│   ├── __init__.py
│   ├── app.py                  # Person 4: FastAPI backend
│   ├── models.py               # Person 4: API models
│   └── routes.py               # Person 4: API routes
├── frontend/
│   ├── index.html              # Person 4: Web UI
│   ├── styles.css              # Person 4: Styling
│   └── script.js               # Person 4: Frontend logic
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py        # Person 4: Integration tests
│   ├── test_api.py             # Person 4: API tests
│   └── test_edge_cases.py      # Person 4: Edge case tests
├── benchmarks/
│   ├── benchmark_ai_detection.py
│   ├── benchmark_plagiarism.py
│   └── benchmark_humanization.py
├── main.py                     # Person 4: CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```
