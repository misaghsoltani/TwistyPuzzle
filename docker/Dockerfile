# syntax=docker/dockerfile:1
#
# Wheel build and verification inside the same containers the release ships
# from, so what CI proves and what PyPI receives are built the same way.
#
# The stages are targets, not layers of one image: each ends in a command that
# fails the build if the thing it checks is wrong. `docker buildx bake` picks
# between them. See docker/README.md.

ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:latest
# Pinned rather than `latest`: the point of a container matrix is that a rerun
# a month from now builds what it built today.
ARG UV_VERSION=0.9.9
ARG RUST_VERSION=stable

# --------------------------------------------------------------------------
# Toolchain: the base image plus Rust. The manylinux and musllinux images
# already carry uv, but a pinned version is installed anyway so the builds do
# not drift as the base images are rebuilt. It comes from the installer and not
# from ghcr.io/astral-sh/uv, because that image is published for amd64 and
# arm64 only while uv itself ships a binary for every architecture here.
# --------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS toolchain
ARG RUST_VERSION
ARG UV_VERSION
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:/root/.local/bin:/usr/local/bin:$PATH CARGO_TERM_COLOR=always RUST_BACKTRACE=1 CARGO_INCREMENTAL=0 UV_NO_PROGRESS=1 UV_PYTHON_DOWNLOADS=never
RUN curl --proto '=https' --tlsv1.2 -sSfL --retry 3 "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh \
    && curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" --no-modify-path \
    && rustc --version && cargo --version && uv --version

# --------------------------------------------------------------------------
# The source tree, with its dependencies fetched.
#
# Deliberately not the usual manifests-only prefetch layer: this crate's
# `[lib]` names a path that has to exist before Cargo will parse the manifest
# at all, so a manifest-only stage needs stub sources that then have to be kept
# in step with the real layout. The registry cache mount below survives between
# builds, which is where nearly all of that layer's value was.
#
# `--locked` because Cargo.lock is committed, and a container build has no
# business resolving something different from what CI resolved.
# --------------------------------------------------------------------------
FROM toolchain AS source
# `TARGETPLATFORM` is supplied by buildx, but only to stages that declare it.
# It keys the cache mounts below so an amd64 build and an arm64 build never
# share one `target/` directory.
ARG TARGETPLATFORM
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETPLATFORM},sharing=locked cargo fetch --locked

# --------------------------------------------------------------------------
# Wheels.
#
# One abi3 wheel covers CPython 3.10 and every later version. The
# free-threaded build cannot load a stable-ABI wheel, so it gets its own, one
# per version, matching the release split.
# --------------------------------------------------------------------------
FROM source AS wheels
ARG TARGETPLATFORM
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/src/target,id=cargo-target-${TARGETPLATFORM},sharing=locked <<'EOF'
set -eux
uv tool install 'maturin>=1.9,<2.0'

# The oldest interpreter the abi3 floor names. Building against it is what
# makes the resulting wheel loadable on every later version.
maturin build --release --locked --out /wheels --interpreter /opt/python/cp310-cp310/bin/python

# Free-threading is a separate ABI, not a later version of the stable one, so
# there is one wheel per version instead of one wheel for all of them.
#
# Every free-threaded interpreter in the image, not a hardcoded one: these
# images carry 3.15t already, which no GitHub runner offers. Covering it is the
# whole reason to have a container matrix, and it is what the release workflow
# is otherwise waiting on a runner image for.
#
# Best-effort per interpreter. PyO3 gains support for each new free-threaded
# version some time after CPython ships it, and a version it cannot build for
# yet is news instead of a failure. The test stage reports any interpreter
# that ends up without a wheel.
for ft in /opt/python/*t/bin/python; do
    [ -x "$ft" ] || continue
    if maturin build --release --locked --out /wheels --interpreter "$ft"; then
        echo "built a free-threaded wheel for $ft"
    else
        echo "WARNING: no free-threaded wheel for $ft (PyO3 may not support it yet)" >&2
    fi
done

ls -l /wheels
EOF

# --------------------------------------------------------------------------
# Verification: install the built wheels on every interpreter the image
# carries and run the suite against each.
#
# Nothing here reads the source tree as a package. The tests run from a
# directory that is not the repository root, exactly as CI does, so an import
# that resolves to the checkout instead of the installed wheel is a failure
# rather than a silent pass.
# --------------------------------------------------------------------------
FROM wheels AS test
ARG TARGETPLATFORM
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-${TARGETPLATFORM},sharing=locked <<'EOF'
set -eu

status=0
tested=0

for py in /opt/python/cp31*/bin/python; do
    [ -x "$py" ] || continue
    tag=$("$py" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}" + ("t" if not getattr(sys, "_is_gil_enabled", lambda: True)() else ""))')

    # Newer interpreters land in these images ahead of the wheels for
    # everything else. Skip one that cannot even get pytest, rather than
    # reporting a failure that is not this project's.
    if ! uv pip install --python "$py" --quiet pytest >/dev/null 2>&1; then
        echo "=== $tag: no pytest for this interpreter, skipping ==="
        continue
    fi

    echo "=== $tag ($py) ==="

    # `--no-index` proves the wheel under test is the one just built, and
    # never something resolved from an index.
    if ! uv pip install --python "$py" --quiet --no-index --find-links /wheels twistypuzzle; then
        case "$tag" in
          *t)
            # One wheel per version here, and PyO3 gains each new
            # free-threaded version some time after CPython ships it. An
            # interpreter it cannot build for yet is reported, not failed.
            echo "--- $tag: no free-threaded wheel for this interpreter, skipping ---"
            ;;
          *)
            # The abi3 wheel claims to serve 3.10 and every later version. If
            # it will not install on one of them, that claim is false.
            echo "!!! $tag: the abi3 wheel does not cover this interpreter" >&2
            status=1
            ;;
        esac
        continue
    fi

    # The base install has to stand on its own: the package declares no
    # dependencies, and this is where a stray top-level import of an optional
    # one would show up. Checked before the extra is installed, because
    # afterward it is unfalsifiable.
    "$py" - <<'PROBE' || status=1
import importlib.util, sys
for forbidden in ("gymnasium", "numpy"):
    if importlib.util.find_spec(forbidden) is not None:
        sys.exit(f"a bare install pulled in {forbidden}")
import twistypuzzle
print("  base install is self-contained:", twistypuzzle.__version__)
PROBE

    # Now the optional half. Where wheels exist for this interpreter the whole
    # suite runs. Where they do not, conftest.py drops the tests that need them
    # and the rest still runs. Compiling numpy from source inside every
    # container in the matrix would cost more than the coverage is worth.
    uv pip install --python "$py" --quiet numpy gymnasium || echo "  [note] numpy/gymnasium unavailable here, so their tests will skip"

    if ( cd /tmp && "$py" -m pytest -q /src/tests/python --import-mode=importlib ); then
        tested=$((tested + 1))
    else
        echo "!!! $tag: suite failed" >&2
        status=1
    fi

    # A free-threaded build that re-enables the GIL on import is the one
    # regression this project cannot detect from behavior alone.
    case "$tag" in
      *t)
        "$py" -c "
import sys, twistypuzzle
assert not sys._is_gil_enabled(), 'importing twistypuzzle re-enabled the GIL'
assert twistypuzzle.free_threaded
print('  free-threaded, GIL stays off')
" || status=1
        ;;
    esac

    uv pip uninstall --python "$py" --quiet twistypuzzle gymnasium numpy >/dev/null 2>&1 || true
done

[ "$tested" -gt 0 ] || { echo "no interpreter was tested at all" >&2; exit 1; }
echo "=== $tested interpreter(s) passed ==="
exit "$status"
EOF

# --------------------------------------------------------------------------
# The Rust suite, including the run that widens every integer. That one proves
# the inline small-integer path is observationally equal to arbitrary
# precision, and it is the reason `bigint-only` exists.
# --------------------------------------------------------------------------
FROM source AS rust-test
ARG TARGETPLATFORM
ARG BIGINT_ONLY=1
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/src/target,id=cargo-target-${TARGETPLATFORM},sharing=locked <<'EOF'
set -eux
# `--no-default-features` drops the pyo3 dependency, so the suite links against
# no interpreter and tests the library on its own terms.
cargo test --release --locked --no-default-features
if [ "${BIGINT_ONLY}" = "1" ]; then
    cargo test --release --locked --no-default-features --features bigint-only
fi
EOF

# --------------------------------------------------------------------------
# The source distribution, built and then installed from source with the
# wheels deliberately out of reach. This is the path a platform with no
# published wheel takes.
# --------------------------------------------------------------------------
FROM source AS sdist
ARG TARGETPLATFORM
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/root/.cache/uv,id=uv-cache-${TARGETPLATFORM},sharing=locked <<'EOF'
set -eux
uv tool install 'maturin>=1.9,<2.0'
maturin sdist --out /sdist
ls -l /sdist

py=/opt/python/cp312-cp312/bin/python
uv pip install --python "$py" --quiet pytest numpy gymnasium
# `--no-binary` is the whole point: forbid the wheel and make it compile.
uv pip install --python "$py" --no-binary twistypuzzle --find-links /sdist twistypuzzle
cd /tmp && "$py" -m pytest -q /src/tests/python --import-mode=importlib
EOF

# --------------------------------------------------------------------------
# The floor in `rust-version`. A claimed MSRV that nobody compiles is a guess.
# --------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS msrv
ARG TARGETPLATFORM
ARG MSRV=1.85.0
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH CARGO_TERM_COLOR=always CARGO_INCREMENTAL=0
RUN curl --proto '=https' --tlsv1.2 -sSf --retry 3 https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${MSRV}" --no-modify-path
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-msrv-${TARGETPLATFORM},sharing=locked --mount=type=cache,target=/src/target,id=cargo-target-msrv-${TARGETPLATFORM},sharing=locked cargo check --locked --all-targets --no-default-features

# --------------------------------------------------------------------------
# Collectible output: the built wheels, for a run that wants them on the host.
# --------------------------------------------------------------------------
FROM scratch AS artifacts
COPY --from=wheels /wheels /wheels
