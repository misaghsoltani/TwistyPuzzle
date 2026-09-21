SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
# Every recipe drives cargo against the same target directory, and `fix`
# rewrites sources that other targets read.
.NOTPARALLEL:

CARGO ?= cargo
PYTHON ?= python
UV ?= uv
MATURIN ?= maturin
PYTEST ?= pytest
RUFF ?= ruff
PYREFLY ?= pyrefly
DOCKER ?= docker
BAKE ?= $(DOCKER) buildx bake

WHEEL_OUT ?= dist
CATALOG_OUT ?= docs/catalog.png
GUI_SCREENSHOT_OUT ?= docs/gui.png

.PHONY: \
	help all ci check build develop \
	wheel wheel-abi3 wheel-free-threaded sdist clean \
	test test-rust test-bigint test-python \
	fmt format fmt-rust fmt-python fmt-check \
	lint lint-rust lint-python typecheck clippy fix \
	bench bench-rust bench-batch bench-threads \
	catalog-sheet gen-font \
	gui gui-build gui-wheel gui-screenshot \
	check-gil test-free-threaded bench-free-threaded \
	docker-ci docker-ci-amd64 docker-ci-arm64 docker-native docker-emulated docker-all \
	docker-rust-test docker-msrv docker-gui docker-gui-screenshot docker-wheels

help: ## Show all available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

all: fmt-check lint test ## Run formatting checks, linters, and all tests

ci: fmt-check lint test test-bigint gui-build gui-screenshot ## Run the local CI-equivalent checks

check: ## Typecheck all Rust targets and features
	$(CARGO) check --all-targets --all-features

build: ## Build release Rust binaries and libraries
	$(CARGO) build --release

develop: ## Build and install the extension into the active Python environment
	$(MATURIN) develop --release

wheel: ## Build a release wheel for the active Python interpreter into dist/
	$(MATURIN) build --release --out $(WHEEL_OUT) --interpreter $(PYTHON)

wheel-abi3: ## Build the abi3 release wheel into dist/
	$(MATURIN) build --release --out $(WHEEL_OUT)

wheel-free-threaded: ## Build a free-threaded release wheel for the active interpreter
	$(MATURIN) build --release --out $(WHEEL_OUT) --interpreter $(PYTHON)

sdist: ## Build source distribution into dist/
	$(MATURIN) sdist --out $(WHEEL_OUT)

test: test-rust test-python ## Run all Rust and Python tests

test-rust: ## Run Rust tests
	$(CARGO) test --release

test-bigint: ## Run Rust tests with bigint-only feature
	$(CARGO) test --release --features bigint-only

test-python: develop ## Run Python tests against the installed extension
	$(PYTEST) tests/python

fmt: fmt-rust fmt-python ## Format all Rust and Python code

format: fmt ## Alias for fmt

fmt-rust: ## Format Rust code
	$(CARGO) fmt --all

fmt-python: ## Format Python code
	$(RUFF) format

fmt-check: ## Check Rust and Python formatting
	$(CARGO) fmt --all -- --check
	$(RUFF) format --check

lint: lint-rust lint-python typecheck ## Run all linters and type checks

lint-rust: clippy ## Lint Rust code with clippy

clippy: ## Run cargo clippy on all targets and features
	$(CARGO) clippy --all-targets --all-features

lint-python: ## Lint Python code with ruff
	$(RUFF) check

typecheck: ## Typecheck Python code with pyrefly
	$(PYREFLY) check

fix: ## Auto-fix Rust and Python lint/format issues
	$(CARGO) clippy --fix --allow-dirty --allow-staged --all-targets --all-features
	$(CARGO) fmt --all
	$(RUFF) check --fix --unsafe-fixes
	$(RUFF) format

bench: bench-rust bench-batch bench-threads ## Run every benchmark path

bench-rust: ## Run Rust benchmark example
	$(CARGO) run --release --example bench

bench-batch: ## Benchmark a batch against one puzzle at a time
	$(CARGO) run --release --example bench_batch

bench-threads: develop ## Benchmark Python thread scaling
	$(PYTHON) scripts/bench_threads.py

catalog-sheet: develop ## Generate the puzzle catalog contact sheet
	$(PYTHON) scripts/catalog_sheet.py -o $(CATALOG_OUT)

# The generator wraps at its own width, so the regenerated file fails
# fmt-check until rustfmt reflows it.
gen-font: ## Regenerate src/render/font_data.rs
	$(PYTHON) tools/gen_font.py
	$(CARGO) fmt --all

gui: ## Run the desktop interface
	$(CARGO) run --release -p twistypuzzle-gui

gui-build: ## Build the desktop interface
	$(CARGO) build --release -p twistypuzzle-gui

gui-wheel: ## Build the twistypuzzle-gui wheel into dist/
	$(MATURIN) build --release --manifest-path crates/gui/Cargo.toml --out $(WHEEL_OUT)

gui-screenshot: ## Render the GUI screenshot to docs/gui.png
	$(CARGO) run --release -p twistypuzzle-gui --example screenshot -- $(GUI_SCREENSHOT_OUT)

check-gil: develop ## Verify GIL is disabled in free-threaded Python
	$(PYTHON) -X gil=0 -c "import sys, twistypuzzle; assert not sys._is_gil_enabled(), 'GIL is still enabled!'; assert twistypuzzle.free_threaded, 'twistypuzzle not compiled with free-threading support'; print('✓ Free-threaded Python verified: GIL disabled, twistypuzzle.free_threaded is True')"

test-free-threaded: develop ## Run Python tests under free-threaded Python
	$(PYTHON) -X gil=0 -m pytest tests/python

bench-free-threaded: develop ## Benchmark thread scaling under free-threaded Python
	$(PYTHON) -X gil=0 scripts/bench_threads.py

docker-ci: ## Run docker buildx bake ci
	$(BAKE) ci

docker-ci-amd64: ## Run docker buildx bake ci-amd64
	$(BAKE) ci-amd64

docker-ci-arm64: ## Run docker buildx bake ci-arm64
	$(BAKE) ci-arm64

docker-native: ## Run docker buildx bake native
	$(BAKE) native

docker-emulated: ## Run docker buildx bake emulated
	$(BAKE) emulated

docker-all: ## Run docker buildx bake all
	$(BAKE) all

docker-rust-test: ## Run Rust test target inside docker bake
	$(BAKE) rust-test

docker-msrv: ## Run MSRV target inside docker bake
	$(BAKE) msrv

docker-gui: ## Build GUI target inside docker bake
	$(BAKE) gui

docker-gui-screenshot: ## Render GUI screenshot via docker bake
	$(BAKE) gui-screenshot

docker-wheels: ## Build wheel artifacts via docker bake
	$(BAKE) wheels

clean: ## Remove build artifacts created by common targets
	$(CARGO) clean
	rm -rf $(WHEEL_OUT) docker/out
