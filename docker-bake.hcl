# Container build matrix.
#
# `bake` rather than a shell loop for one reason that matters: it builds the
# targets in parallel and shares the layers they have in common, so the
# toolchain stage is built once and not once per platform.
#
#   docker buildx bake -f docker/docker-bake.hcl ci          # what CI runs
#   docker buildx bake -f docker/docker-bake.hcl native      # this machine only
#   docker buildx bake -f docker/docker-bake.hcl emulated    # the slow ones
#   docker buildx bake -f docker/docker-bake.hcl all
#
# Every target ends in a command that fails the build when the thing it checks
# is wrong, so a target that builds is a target that passed. Nothing is pushed
# and nothing is tagged. These are tests that happen to be Dockerfiles.

variable "RUST_VERSION" { default = "stable" }
# The Debian `rust` images are tagged by version instead of by channel, so
# the desktop build cannot reuse RUST_VERSION's "stable".
variable "RUST_IMAGE_VERSION" { default = "1" }
variable "MSRV" { default = "1.85.0" }
# Pinned, so a rerun next month builds what it builds today.
variable "UV_VERSION" { default = "0.9.9" }

variable "MANYLINUX" { default = "quay.io/pypa/manylinux_2_28" }
variable "MUSLLINUX" { default = "quay.io/pypa/musllinux_1_2" }

# Located at repository root to keep the build context rooted at `.`,
# avoiding the need for extended filesystem permissions (`--allow=fs.read=..`).

target "_common" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  args = {
    RUST_VERSION = RUST_VERSION
    UV_VERSION   = UV_VERSION
  }
  # These are checks. Producing an image would only fill the local store.
  output = ["type=cacheonly"]
}

# --------------------------------------------------------------------------
# Wheels, built and then tested on every interpreter in the image.
#
# `platforms` is what drives the architecture: buildx runs the whole stage
# under the matching base image, natively where the builder can and under QEMU
# where it cannot.
# --------------------------------------------------------------------------

target "manylinux-x86_64" {
  inherits  = ["_common"]
  target    = "test"
  args      = { BASE_IMAGE = "${MANYLINUX}_x86_64:latest" }
  platforms = ["linux/amd64"]
}

target "manylinux-aarch64" {
  inherits  = ["_common"]
  target    = "test"
  args      = { BASE_IMAGE = "${MANYLINUX}_aarch64:latest" }
  platforms = ["linux/arm64"]
}

target "musllinux-x86_64" {
  inherits  = ["_common"]
  target    = "test"
  args      = { BASE_IMAGE = "${MUSLLINUX}_x86_64:latest" }
  platforms = ["linux/amd64"]
}

target "musllinux-aarch64" {
  inherits  = ["_common"]
  target    = "test"
  args      = { BASE_IMAGE = "${MUSLLINUX}_aarch64:latest" }
  platforms = ["linux/arm64"]
}

# --------------------------------------------------------------------------
# The architectures release.yml publishes for but no runner and no developer
# laptop can execute. They are correct here and ruinously slow: QEMU plus
# `lto = "fat"` and `codegen-units = 1` is hours, not minutes. Deliberately
# outside the `ci` group: run them before a release, not on every push.
#
# `BIGINT_ONLY=0` on the emulated wheel targets does not apply (that argument
# belongs to `rust-test`). The cost here is the release profile itself.
# --------------------------------------------------------------------------

target "manylinux-armv7" {
  inherits = ["_common"]
  target   = "test"
  # Not `${MANYLINUX}`: armv7 has no manylinux_2_28 image. The architecture was
  # only ever given a glibc 2.31 floor, so this is the oldest tag that exists
  # for it instead of an inconsistency.
  args      = { BASE_IMAGE = "quay.io/pypa/manylinux_2_31_armv7l:latest" }
  platforms = ["linux/arm/v7"]
}

target "manylinux-ppc64le" {
  inherits  = ["_common"]
  target    = "test"
  args      = { BASE_IMAGE = "${MANYLINUX}_ppc64le:latest" }
  platforms = ["linux/ppc64le"]
}

target "manylinux-s390x" {
  inherits  = ["_common"]
  target    = "test"
  # Big-endian. Everything else in this matrix is little-endian, so this is the
  # only target that would catch an endianness assumption in the bit patterns
  # the renderer and the exact-arithmetic tables are full of.
  args      = { BASE_IMAGE = "${MANYLINUX}_s390x:latest" }
  platforms = ["linux/s390x"]
}

# --------------------------------------------------------------------------
# The Rust suite, the source distribution, the MSRV floor and the interface.
# --------------------------------------------------------------------------

target "rust-test" {
  inherits  = ["_common"]
  target    = "rust-test"
  args      = { BASE_IMAGE = "${MANYLINUX}_x86_64:latest", BIGINT_ONLY = "1" }
  platforms = ["linux/amd64"]
}

target "sdist" {
  inherits  = ["_common"]
  target    = "sdist"
  args      = { BASE_IMAGE = "${MANYLINUX}_x86_64:latest" }
  platforms = ["linux/amd64"]
}

target "msrv" {
  inherits  = ["_common"]
  target    = "msrv"
  args      = { BASE_IMAGE = "${MANYLINUX}_x86_64:latest", MSRV = MSRV }
  platforms = ["linux/amd64"]
}

target "gui" {
  inherits   = ["_common"]
  dockerfile = "docker/Dockerfile.gui"
  target     = "build"
  args       = { RUST_VERSION = RUST_IMAGE_VERSION }
  platforms  = ["linux/amd64"]
}

# Writes the rendered frame to `docker/out/gui.png` instead of discarding it.
target "gui-screenshot" {
  inherits = ["gui"]
  target   = "screenshot"
  output   = ["type=local,dest=./docker/out"]
}

# Writes the built wheels to `docker/out/wheels/`.
target "wheels" {
  inherits  = ["_common"]
  target    = "artifacts"
  args      = { BASE_IMAGE = "${MANYLINUX}_x86_64:latest" }
  platforms = ["linux/amd64"]
  output    = ["type=local,dest=./docker/out"]
}

# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------

# What a change is worth checking against on every push: both libc flavors on
# both architectures, plus the paths nothing else covers.
#
# Split by architecture because CI runs each half on a runner of that
# architecture. Building arm64 on an amd64 runner would mean QEMU, and QEMU
# against `lto = "fat"` with `codegen-units = 1` is the same arithmetic that
# makes the `emulated` group take hours. Locally, `ci` builds both.
group "ci" {
  targets = ["ci-amd64", "ci-arm64"]
}

group "ci-amd64" {
  targets = [
    "manylinux-x86_64",
    "musllinux-x86_64",
    "rust-test",
    "sdist",
    "gui",
  ]
}

group "ci-arm64" {
  targets = [
    "manylinux-aarch64",
    "musllinux-aarch64",
  ]
}

# Host-native targets that execute without QEMU emulation.
group "native" {
  targets = ["manylinux-aarch64", "musllinux-aarch64"]
}

group "emulated" {
  targets = ["manylinux-armv7", "manylinux-ppc64le", "manylinux-s390x"]
}

group "default" {
  targets = ["ci"]
}

group "all" {
  targets = [
    "manylinux-x86_64",
    "manylinux-aarch64",
    "musllinux-x86_64",
    "musllinux-aarch64",
    "manylinux-armv7",
    "manylinux-ppc64le",
    "manylinux-s390x",
    "rust-test",
    "sdist",
    "msrv",
    "gui",
  ]
}
