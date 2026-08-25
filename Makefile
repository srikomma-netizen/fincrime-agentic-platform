.PHONY: install data train index setup demo serve test docker

install:
	pip install -r requirements.txt

data:
	python scripts/generate_synthetic_data.py

train:
	python scripts/train_models.py

index:
	python scripts/build_vector_index.py

setup: data train index   ## one-shot: generate data, train models, build index

demo:
	python scripts/demo.py

serve:
	uvicorn app.main:app --app-dir src --reload --port 8000

test:
	PYTHONPATH=src pytest -q

docker:
	docker compose up --build
