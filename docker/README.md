# Container builds

Every build this project ships, run inside the containers the release actually ships from, on architectures no CI runner offers.

The GitHub runners cover three operating systems on one architecture each. What they cannot cover is the rest of the wheel matrix: musl libc, 32-bit ARM, and a big-endian machine.

```bash
docker buildx bake  # the default group: what CI runs
docker buildx bake native  # only what this machine runs without emulation
docker buildx bake emulated  # the slow architectures
docker buildx bake all
docker buildx bake manylinux-x86_64  # one target
```

Run from the repository root, because `docker-bake.hcl` lives there so the build context is plainly `.` and needs no `--allow=fs.read=..`.

## These are tests, not images

Every target ends in a command that fails the build when the thing it checks is wrong, and produces no image (`type=cacheonly`). A target that builds is a target that passed. Nothing is tagged and nothing is pushed.

Two targets do write something out, to `docker/out/`:

```bash
docker buildx bake wheels  # the built wheels
docker buildx bake gui-screenshot  # the interface, rendered with no display
```

## What each target checks

| Target | What it proves |
| --- | --- |
| `manylinux-x86_64`, `manylinux-aarch64` | The glibc wheels build and the suite passes on every interpreter in the image |
| `musllinux-x86_64`, `musllinux-aarch64` | The same on musl (Alpine and most containers) |
| `manylinux-armv7` | 32-bit. Pointer width and `usize` assumptions surface here and nowhere else |
| `manylinux-ppc64le` | A second 64-bit architecture |
| `manylinux-s390x` | **Big-endian.** The only target that can catch an endianness assumption in the baked IEEE-754 tables or the renderer's pixel packing |
| `rust-test` | `cargo test`, and the `bigint-only` run that proves the inline small-integer path equals arbitrary precision |
| `sdist` | The sdist builds *and installs from source* with wheels forbidden, which is the path a platform with no published wheel takes |
| `msrv` | The `rust-version = "1.85"` floor actually compiles. A claimed MSRV nobody compiles is a guess |
| `gui` | The desktop crate builds against real X11/Wayland headers and renders a PNG with no display attached |

Each wheel target installs the wheel on every CPython in the image, asserts that a **bare install pulls in neither gymnasium nor numpy**, and only then installs the optional half and runs the suite. That ordering matters: once the extras are installed the question cannot be asked anymore.

Free-threaded interpreters additionally have to prove that importing the module leaves the GIL off.

## Speed

- **`.dockerignore` is load-bearing.** The working tree is over 3 GB, nearly all of it `target/` and `.pixi/`. Docker sends the context to the daemon before the first instruction runs, so without that file every target in the matrix would pay a multi-gigabyte transfer before compiling anything.
- **Cache mounts**, keyed by `TARGETPLATFORM`, hold the Cargo registry, the git checkouts, `target/` and uv's cache across builds. The key matters: without it an amd64 and an arm64 build would share one `target/` and corrupt it.
- **`bake` over a shell loop**, because it builds targets in parallel and shares the layers they have in common, so the toolchain stage is built once rather than once per platform.

`crates/` is deliberately *not* ignored even though it never reaches the wheel: `crates/gui` is a workspace member the root manifest names, so Cargo will not parse the workspace without it.

## The emulated targets are genuinely slow

`armv7`, `ppc64le` and `s390x` run under QEMU, against a release profile that is `lto = "fat"` with `codegen-units = 1`. Expect hours, not minutes. They are deliberately outside the `ci` group: run them before a release, not on every push.

CI does not run them, and so installs no QEMU: the `ci` group is split into `ci-amd64` and `ci-arm64`, each built on a runner of that architecture. To run the emulated group anywhere, register the binfmt handlers first, using `docker/setup-qemu-action` on a runner, or `docker run --privileged --rm tonistiigi/binfmt --install all` locally.

## What this does not cover

Docker cannot test macOS or Windows. Those stay on GitHub runners, where the `rust`, `python` and `gui` jobs cover them across the full interpreter matrix. "Every platform" here means every Linux platform, which is where the architectural variety actually is.
