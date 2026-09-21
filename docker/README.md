# Container Builds

Reproducible build environments for release targets across diverse system architectures.

While standard GitHub Actions runners cover typical host environments, container builds extend coverage to the complete Linux wheel matrix: musl libc, 32-bit ARM, and big-endian systems.

```bash
docker buildx bake  # default group: standard CI targets
docker buildx bake native  # host-native targets without emulation
docker buildx bake emulated  # emulated architectures (ARMv7, PPC64LE, s390x)
docker buildx bake all
docker buildx bake manylinux-x86_64  # single target
```

Execute from the repository root so that the build context defaults to `.` without requiring extended filesystem permissions.

## Test Targets vs Images

Build targets terminate in test commands that fail the build if assertions fail, producing no output image (`type=cacheonly`). A successful target build indicates that test suites and assertions passed.

Two targets write artifacts to `docker/out/`:

```bash
docker buildx bake wheels  # built wheels
docker buildx bake gui-screenshot  # headless render test output
```

## Target Matrix Verification

| Target | Validation Objective |
| --- | --- |
| `manylinux-x86_64`, `manylinux-aarch64` | Verifies glibc wheel builds and test suites across all supported Python interpreters |
| `musllinux-x86_64`, `musllinux-aarch64` | Validates musl libc compatibility (Alpine Linux and minimal containers) |
| `manylinux-armv7` | 32-bit architecture verification (pointer width and `usize` assumptions) |
| `manylinux-ppc64le` | Secondary 64-bit RISC architecture validation |
| `manylinux-s390x` | **Big-endian.** Verifies endian-neutrality in IEEE-754 conversions and rasterizer pixel formats |
| `rust-test` | `cargo test`, including the `bigint-only` feature verifying equivalence of the inline integer path |
| `sdist` | Verifies source distribution build and source installation without pre-built wheels |
| `msrv` | Validates that the Minimum Supported Rust Version (`rust-version = "1.85"`) compiles without error |
| `gui` | Verifies desktop GUI compilation against X11/Wayland headers and validates headless rendering |

Each wheel target installs the wheel across every Python interpreter in the image, asserts that a **minimal installation introduces neither gymnasium nor numpy**, and subsequently installs optional extras and executes test suites.

Free-threaded interpreters also verify that importing the native extension does not re-enable the Python GIL.

## Performance Considerations

- **`.dockerignore` optimization.** The local working tree can exceed 3 GB (primarily `target/` and `.pixi/`). Excluding these directories avoids multi-gigabyte build context transfers to the Docker daemon.
- **Cache mounts**, keyed by `TARGETPLATFORM`, preserve the Cargo registry, git checkouts, `target/` build cache, and uv cache across build invocations.
- **Concurrent `bake` execution.** Docker Buildx builds targets in parallel and deduplicates shared base stages.

`crates/` is included in the build context because `crates/gui` is a member of the root Cargo workspace.

## Emulated Architectures

`armv7`, `ppc64le`, and `s390x` execute under QEMU emulation with full Link-Time Optimization (`lto = "fat"`, `codegen-units = 1`). Due to emulation overhead, these targets are designated for pre-release validation instead of per-commit CI runs.

Automated CI targets are partitioned into `ci-amd64` and `ci-arm64` on native runners. To execute emulated targets locally, ensure QEMU binfmt handlers are registered (e.g., via `docker run --privileged --rm tonistiigi/binfmt --install all`).

## Scope

Non-Linux platforms (macOS and Windows) are validated directly via native GitHub Actions runners. Docker build targets validate the Linux matrix, where libc, endianness, and architecture variations require explicit verification.
