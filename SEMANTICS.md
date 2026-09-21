# Semantics

Most internal operations of this library represent implementation details, including data structure selection for key lookups, intermediate numerical refinement levels, and thread scheduling across rasterization bands. Behaviors that directly influence observable outputs are formally specified in this document.

The criterion for inclusion is strict: a design decision is documented here if altering it would modify an externally observable result instead of execution latency. The implementation references these sections by number where behavioral contracts are assumed. Modifications to these specifications alter the semantic contract of the library.

## 1. Refinement is shared and mutable

An element within an algebraic number field $\mathbb{Q}(\theta)$ is represented as a polynomial in $\theta$ over $\mathbb{Q}$. To evaluate signs or ordering comparisons, the field's isolating interval around $\theta$ is iteratively contracted until the evaluation is unambiguous. The isolating interval is a property of the field instead of individual elements, and invoking `refine()` contracts the interval in place for all numbers associated with that field.

Consequently, the internal state of a field depends on computational history, specifically which comparisons were evaluated and in what order. Executions that evaluate different queries may leave the field refined to different interval widths.

Comparison operations evaluate deterministically regardless of the initial interval width, and conversions to `f64` refine the interval until yielding the correctly rounded 64-bit float representation. Computations arriving at the same logical state produce identical coordinates, rasterized pixels, and sticker configurations, irrespective of intermediate query sequences.

## 2. Iteration order is normative

Several key-value collections are iterated in specified orders that determine computational outputs instead of merely internal layout:

- In `movement.rs`, `planes` and `stops` maintain insertion order via `IndexMap`. The stops associated with a grip are retrieved from this collection and subsequently sorted by angle. When two stops have identical angles, insertion order resolves the tie, deterministically selecting the rotation applied by a turn.
- In `piece.rs`, `edge_index` is traversed in insertion order (`IndexMap`), and `cutgraph` is traversed in ascending key order (`BTreeMap`). The traversal order of `cutgraph` determines the initial vertex during cut-face polygon traversal, establishing polygon winding and outward normal orientation.
- Sticker slots are indexed in canonical color order (`symbolic.rs`), ensuring that the solved sticker array of a monochromatic-face puzzle yields contiguous sequences `0,0,...,1,1,...` (for a 3x3x3 cube, `arange(54) // 9`).

Collections accessed exclusively via key lookups utilize standard hash tables.

Canonical string representations of exact numbers serve as dictionary keys during piece enumeration. Consequently, textual serialization is observable, necessitating the explicit `reduce` flag in in-place `Fraction` arithmetic (§5) and strict formatting invariants in `Polynomial`'s `Display` implementation: `"2/4"` and `"1/2"` represent identical algebraic quantities but distinct dictionary keys.

## 3. Field identity is pointer identity

The functions `check_same_field`, `extend`, and `promote` evaluate field equivalence via pointer equality (`Arc::ptr_eq`) instead of structural comparison. Two field instances possessing identical minimal polynomials remain distinct entities because each maintains an independent, mutable isolating interval (§1). Combining elements between unlinked field instances would conflate their mutable intervals.

Consequently, comparing or combining numbers from distinct instances requires explicit embedding into a shared field via `promote`, instead of assuming equivalence based on identical construction parameters.

## 4. Field extension and element re-embedding

Arithmetic involving numbers from distinct fields is supported: `extend` constructs an extension field containing both subfields, and `promote` maps a slice of elements into this extension. This produces a *new* field instance, and mapped elements constitute newly instantiated values. References to prior elements remain unchanged.

Contractual implications:

- An element's field reference may change across computations. A field reference obtained prior to an operation is not guaranteed to equal the field of the resulting element.
- Because `extend` computes a primitive element, the minimal polynomial of the composite field is not guaranteed to follow a specific algebraic presentation. `Real.field()` returns the current containing field.
- Arithmetic and relational operations are defined strictly within a shared field. Rationals (fields of degree 1) coerce with all fields. Other cross-field evaluations raise an exception, whereas `Elem::equals` evaluates to `false`. The Python bindings automatically promote operands before evaluation. In Rust, callers must invoke `promote` explicitly or supply elements sharing a field instance.

## 5. Explicit reduction control in rational arithmetic

In-place operations on `Fraction` accept a `reduce` boolean flag instead of enforcing automatic reduction. An unreduced fraction represents the same rational value with a distinct string representation, and string serializations serve as dictionary keys (§2). Reduction is therefore an explicit caller decision instead of an automatic type invariant.

Value-returning methods reduce by default. The accessor methods `numerator` and `denominator` return lowest terms irrespective of internal reduction state. Consequently, the reduction flag affects only string serialization and the arithmetic overhead of subsequent operations.

## 6. Distinction between in-place and value-returning operations

Ring operations are implemented separately for in-place and value-returning variants instead of delegating one to the other. In specific algebraic structures, their semantics intentionally diverge: `AlgebraicNumber::iadd` does **not** reduce modulo the minimal polynomial after each addition, whereas `add` performs immediate reduction. This distinction optimizes multi-term summations by amortizing modular reduction to a single operation at the conclusion of the summation.

An implementation of `Elem` defining `iadd` via `*self = self.add(b)` violates this performance contract despite mathematical equivalence.

## 7. Constant ambient shading and sRGB encoding

The rendering model utilizes uniform ambient illumination (intensity 1.5) with Lambertian materials, without directional light sources or specular reflection. Consequently, per-fragment shading simplifies to scaling the facet's linear color by ambient intensity followed by sRGB transfer function encoding. All fragments of a given polygon face evaluate to identical RGBA byte quadruplets.

This model eliminates barycentric attribute interpolation across triangles, allows supersampling anti-aliasing to resolve via unweighted box averaging over sub-pixel samples, and enables frame reconstruction via precomputed pixel-to-slot lookup tables (§10).

## 8. Deterministic parallel rasterization

The software rasterizer bins primitives into horizontal tiles of 32 rows and rasterizes them across threads. Because tiles span disjoint scanlines, pixel writes and depth tests remain mutually independent across worker threads. Rendered output is invariant to thread count and scheduling order.

Within individual scanline bands, primitives are rasterized in submission order with standard depth testing. Coplanar geometry at identical depth values resolves according to submission order. Outline edges utilize a polygon depth offset bias toward the camera to prevent z-fighting instead of relying on pass ordering.

Thread pool configuration via `RAYON_NUM_THREADS` impacts rendering throughput but guarantees identical pixel values.

## 9. Single-threaded simulator encapsulation

Operations such as rendering, state extraction, or coordinate conversion to `f64` trigger isolating interval refinement in underlying algebraic numbers (§1). Because these operations mutate shared numerical state, the Python API enforces exclusive simulator access for `render`, `piece`, `grips`, and related methods. Concurrent access to a single simulator instance raises a `RuntimeError` instead of permitting race conditions.

Callers must maintain an affinity of at most one thread per simulator instance. Independent simulator instances share no mutable state and execute concurrently without contention, as implemented in `build_many`, `render_many`, and `PuzzleBatch`.

## 10. Precomputed pixel lookup tables

Non-jumbling puzzles preserve their geometric envelope across all reachable configurations: puzzle pieces occupy identical spatial domains regardless of permutation, with only facet colors varying. Consequently, under invariant camera projection parameters and frame dimensions, the mapping from pixel coordinates to visible sticker slots is invariant to puzzle state, allowing frames to be synthesized via direct table lookups (`render/lut.rs`).

A precomputed lookup table is valid only under the following conditions:

- **Stickers remain on the canonical solved lattice.** Jumbling puzzles admit moves that place pieces outside solved lattice sites, precluding a static assignment of pixel coordinates to discrete slots.
- **The puzzle is at rest.** During turn animation, geometry occupies intermediate non-lattice configurations.
- **Turn arrows are disabled.** Arrows are alpha-blended over the scene instead of written opaquely. Their pixel colors depend on background composition instead of mapping to a discrete slot.
- **Viewpoint and frame dimensions remain unchanged.** Camera parameters, render resolution, and scene options must match the configuration under which the table was compiled. Altering any parameter invalidates the cache and triggers table reconstruction.

Fidelity guarantees: on a solved puzzle, table lookup rendering reproduces software rasterization bit-for-bit. On scrambled configurations, minor variations occur exclusively along piece boundary seams (<0.5% of pixels), where piece rotation translates boundary edge geometry relative to fixed screen-space pixel centers. Interior sticker pixels remain unaffected. Because all frames within a batch evaluate against the same lookup table, rendered batches remain internally consistent and deterministic.

## 11. Move rotation reference frame

`make_move` applies rotation exclusively to pieces on the positive half-space of the cut plane. The opposing half-space remains stationary, and no compensatory global rotation is applied to the puzzle.

An alternative convention (rotating the negative half-space in the inverse direction alongside a compensatory global rotation) would yield identical projected imagery under a fixed camera, but introduces artificial frame transformations. Under that alternative, slot coordinates would become coupled to global rotation state, requiring inverse frame transformations prior to state observation extraction.

Restricting rotation to the positive half-space establishes three invariants:

- The sticker array directly corresponds to the reference frame of the rendered image.
- Actions correspond directly to standard face rotations, and `action_permutations` represent the canonical permutation group generators.
- Rotations are applied symmetrically without designating privileged reference pieces.

`turnable_cuts` canonicalizes off-center cutting planes such that the positive half-space forms the smaller cap, minimizing the number of transformed piece geometries per move.

## 12. Face-major layout indexing

`Layout` provides a face-major indexing scheme for puzzle stickers. Because facelet ordering represents an arbitrary convention instead of an intrinsic geometric invariant, the specifications are formalized as follows:

**Standard Cube Conventions.** For cube puzzles, faces follow the standard ordering `U, D, L, R, B, F`, aligned such that `U` lies at `+y`, `R` at `+x`, and `F` at `+z`. Within each face, facelets are ordered in row-major sequence, with rows spanning `ROW`, columns spanning `COL`, and `ROW × COL` defining the outward surface normal. The initial 12 moves represent face quarter turns ordered face-major: index `2k` represents counterclockwise rotation as viewed from the exterior, and `2k + 1` represents clockwise rotation. The reference permutations and 128 move sequences recorded in `tests/data/facelet_cube.txt` define this specification, and the implementation is verified against this reference fixture.

**Canonical Order for Non-Cube Puzzles.** For general polyhedral puzzles lacking established external standards, indexing follows the intrinsic geometric order: faces correspond to puzzle colors, facelets to sticker slots, and moves to primitive actions in their native order (§2). Because slots are numbered in color order, solved state observations equal `0,0,...,1,1,...` across all puzzle families, making translation the identity mapping. No artificial intra-face ordering is imposed on non-grid facets (e.g., dodecahedral pentagons), avoiding unstable tie-breaking between symmetric coordinates.

Contractual properties:

- Cube layouts retain all 12 canonical face turns regardless of mechanical reachability, and the `actions` array indicates which entries map to valid simulator actions. Moves outside the primary 12 face turns (such as inner slice rotations) append using native grip identifiers prefixed with a colon (e.g., `:E`), avoiding collision with outer face grip names `A` through `F`.
- Move inverses are derived from permutation cycle structure instead of naming conventions. Self-inverse operations return their own index. If multiple valid inverses exist (such as commuting moves where $(UD)^{-1} = D^{-1}U^{-1} = U^{-1}D^{-1}$), the minimal index is selected.
- A layout reflects native puzzle actions without filtering: identity actions or duplicate permutations are preserved verbatim to maintain parity between search policies and physical simulation.

External state injection: `set_states` accepts explicit slot permutations, while `set_colors` accepts color assignments (resolving sticker identities in slot order). Input vectors are validated for structural cardinality (valid permutation of slots or matching color histogram). Reachability from the solved state is not evaluated (reachability determination is computationally intractable in general). Unreachable states are simulated and rendered faithfully without achieving the solved state.

## 13. Minimal dynamic integer width representation

Arrays returned by this library utilize the narrowest unsigned integer datatype capable of representing the maximum theoretical bounds of the puzzle recipe, instead of static 32-bit or 64-bit containers. For a 3x3x3 cube (54 slots, 6 colors), color indices, slot identifiers, and permutation tables are stored in 8-bit integers (`uint8`). For a 9x9x9 cube (486 slots, 6 colors), slot indices are allocated as 16-bit integers (`uint16`), while color indices remain 8-bit (`uint8`).

Datatype bounds depend strictly on puzzle parameters instead of instantiated sample values, ensuring that array `dtype` is an invariant property of the puzzle definition. `Array.typestr` reports this format, and Gymnasium observation and action spaces conform to this schema.

Integer width minimization does not impose arbitrary upper limits: geometries with more than 256 facets widen storage types automatically. Values exceeding target array widths trigger explicit range errors instead of arithmetic overflow or truncation.

In the native backend, batch buffers store states and permutations at the minimal integer width required by the slot count, reducing memory traffic by 75% relative to 32-bit representations.

## 14. Kinematic stops for single-piece grips

A turn operation applies rotation to the positive half-space of a cutting plane (§11). Permissible angular stops are determined by rotational symmetries that align cutting plane families of the rotating subset with cutting planes of the stationary puzzle. Cutting planes that do not intersect a piece subset are algebraically coplanar with the exterior and are included in rotational matching. `find_stops` matches planes between subsets and extracts rotations about the normal axis that preserve cutting plane alignment.

Restricting plane matching strictly to planes intersecting piece interiors succeeds for all multi-piece components but fails for terminal vertex pieces ("tips"). Because a single piece contains no interior dividing planes, an interior-only formulation yields zero matching planes, failing to identify rotational stops. For piece subsets with no interior dividing planes, the algorithm includes all non-intersecting planes in the matching set. For multi-piece subsets, this formulation is strictly equivalent to interior plane matching. Across all 85 cataloged puzzles, the grips requiring non-intersecting plane inclusion correspond precisely to single-piece grips.

Four cataloged puzzles feature single-piece tip grips: Pyraminx and Master Pyraminx (3-fold rotational symmetry, $120^\circ$ stops), Magic Octahedron (4-fold symmetry, $90^\circ$ stops), and Tutt's Icosaminx (5-fold symmetry, $72^\circ$ stops). A tip piece carries only its own facet stickers, though deeper face rotations encompass tip pieces within their rotating subsets.
