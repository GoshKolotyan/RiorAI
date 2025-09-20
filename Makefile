run_floor_plane_test:
	python -m src.floor_plane.main

syntax:
	mypy .

format:
	ruff format .