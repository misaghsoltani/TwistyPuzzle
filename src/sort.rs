//! A stable sort whose comparator is allowed to fail.
//!
//! `slice::sort_by` cannot be used directly here: comparing two algebraic
//! numbers can fail, and the standard comparator has nowhere to put the error.
//! Swallowing it and returning `Equal` is worse than it looks, because the sort may
//! then see an inconsistent order and panic, and this crate does not panic on
//! a degenerate value (see `SEMANTICS.md`).
//!
//! # Why a binary insertion sort
//!
//! Two properties of the callers decide the algorithm, and both point the same
//! way.
//!
//! *The inputs are tiny.* Building and turning all eighty-five catalog
//! puzzles enters the sort about fifty thousand times at a mean length of four.
//! Sorting the four corner values of an interval product in [`crate::exact`] is
//! ninety-five percent of those calls, and the longest slice measured anywhere
//! in that run held thirty-one elements. At those lengths a single call to the
//! allocator costs more than the comparisons do.
//!
//! *Comparisons are the expensive operation, not moves.* Comparing two
//! algebraic numbers may refine the field they share, which is unbounded work,
//! while moving one is a `memmove` of a few machine words. So the thing worth
//! minimizing is the comparison count, and data movement can be spent freely.
//!
//! A binary insertion sort is what those two facts ask for. It performs
//! `O(n log n)` comparisons, which is the same order as a merge sort and within
//! a constant of the information-theoretic bound. It moves elements with
//! `rotate_right`, which lowers to `memmove` and never clones, and it needs no
//! memory at all. A merge sort has to hold a second copy of the array, and the
//! obvious implementation clones an element on every merge step: `O(n log n)`
//! clones of a value that owns a `Vec` and an `Arc`, to avoid `O(n^2)` byte
//! moves that at these lengths are faster than one call to the allocator.
//!
//! Quadratic movement does eventually lose. The crossover is far away: with
//! comparisons this expensive it sits in the thousands of elements, two orders
//! of magnitude past the longest slice any caller passes. [`MERGE_THRESHOLD`]
//! guards it anyway, so the cost stays `O(n log n)` in both comparisons and
//! moves no matter what a future caller does.

/// Above this length [`sort_by`] switches from insertion to merge sort.
///
/// Chosen to sit above the longest slice any caller passes, so the measured
/// workload never takes the merge path and never allocates.
const MERGE_THRESHOLD: usize = 32;

/// Sort `v` in place, stably, by `cmp`.
///
/// `cmp` returns a negative number, zero, or a positive number, which is the
/// convention the callers already speak. On failure the first error is
/// returned and the contents of `v` are unspecified, as every caller discards
/// the value in that case.
pub fn sort_by<T, E, F: FnMut(&T, &T) -> Result<i32, E>>(v: &mut [T], mut cmp: F) -> Result<(), E> {
    if v.len() < 2 {
        return Ok(());
    }
    if v.len() <= MERGE_THRESHOLD {
        insertion_sort(v, &mut cmp)
    } else {
        merge_sort(v, &mut cmp)
    }
}

/// Binary insertion sort: `O(n log n)` comparisons, no allocation.
fn insertion_sort<T, E, F: FnMut(&T, &T) -> Result<i32, E>>(v: &mut [T], cmp: &mut F) -> Result<(), E> {
    for i in 1..v.len() {
        // Already in order against its predecessor, so it is already in place.
        // One comparison buys the whole already-sorted case, which is the
        // common one: it takes the sort to `n - 1` comparisons where the
        // binary search alone would always spend `n log n`.
        if cmp(&v[i - 1], &v[i])? <= 0 {
            continue;
        }
        // Otherwise find where it belongs: the leftmost position holding
        // something strictly greater. Searching for *strictly* greater is the
        // stability: an element equal to this one keeps its place ahead of
        // it.
        let (mut lo, mut hi) = (0usize, i - 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if cmp(&v[mid], &v[i])? > 0 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        // `memmove` the run up by one and drop the element into the gap.
        v[lo..=i].rotate_right(1);
    }
    Ok(())
}

/// Bottom-up merge sort over a permutation, for the lengths where quadratic
/// movement would cost more than an allocation.
///
/// The elements themselves are never cloned and never copied more than once:
/// the merge orders `u32` indices, and the permutation is applied at the end
/// with at most `n - 1` swaps.
fn merge_sort<T, E, F: FnMut(&T, &T) -> Result<i32, E>>(v: &mut [T], cmp: &mut F) -> Result<(), E> {
    let n = v.len();
    let mut src: Vec<u32> = (0..n as u32).collect();
    let mut dst: Vec<u32> = vec![0; n];

    let mut width = 1;
    while width < n {
        let mut lo = 0;
        while lo < n {
            let mid = (lo + width).min(n);
            let hi = (lo + 2 * width).min(n);
            let (mut i, mut j) = (lo, mid);
            for slot in &mut dst[lo..hi] {
                // `<= 0` instead of `< 0` is the stability: on a tie the left
                // run, which came first in the input, goes first in the output.
                let take_left = if i >= mid {
                    false
                } else if j >= hi {
                    true
                } else {
                    cmp(&v[src[i] as usize], &v[src[j] as usize])? <= 0
                };
                if take_left {
                    *slot = src[i];
                    i += 1;
                } else {
                    *slot = src[j];
                    j += 1;
                }
            }
            lo += 2 * width;
        }
        core::mem::swap(&mut src, &mut dst);
        width *= 2;
    }

    // `src[k]` is where the element for position `k` currently lives. Invert
    // it, because a cycle of swaps needs to be told where each element goes
    // instead of where it comes from.
    for (dest, &from) in src.iter().enumerate() {
        dst[from as usize] = dest as u32;
    }
    for i in 0..n {
        while dst[i] as usize != i {
            let j = dst[i] as usize;
            v.swap(i, j);
            dst.swap(i, j);
        }
    }
    Ok(())
}

/// [`sort_by`] for a comparator that cannot fail.
pub fn sort_by_infallible<T, F: FnMut(&T, &T) -> i32>(v: &mut [T], mut cmp: F) {
    let _: Result<(), core::convert::Infallible> = sort_by(v, |a, b| Ok(cmp(a, b)));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg() -> impl FnMut() -> u32 {
        let mut rng = 0x1234_5678u32;
        move || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            rng
        }
    }

    /// Both paths, either side of the threshold, against the standard sort.
    #[test]
    fn sorts_correctly() {
        let mut next = lcg();
        for len in [0usize, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 4097] {
            let input: Vec<i32> = (0..len).map(|_| (next() % 50) as i32).collect();
            let mut got = input.clone();
            sort_by_infallible(&mut got, |a, b| a - b);
            let mut want = input.clone();
            want.sort_unstable();
            assert_eq!(got, want, "len {len}");
        }
    }

    /// Inputs that are already ordered, reversed, or all equal: the shapes
    /// where an insertion sort either shines or degenerates.
    #[test]
    fn handles_degenerate_orders() {
        for len in [2usize, 5, 31, 32, 33, 200] {
            let asc: Vec<i32> = (0..len as i32).collect();
            let mut v = asc.clone();
            sort_by_infallible(&mut v, |a, b| a - b);
            assert_eq!(v, asc, "ascending, len {len}");

            let mut v: Vec<i32> = asc.iter().rev().copied().collect();
            sort_by_infallible(&mut v, |a, b| a - b);
            assert_eq!(v, asc, "descending, len {len}");

            let mut v = vec![7i32; len];
            sort_by_infallible(&mut v, |a, b| a - b);
            assert_eq!(v, vec![7i32; len], "all equal, len {len}");
        }
    }

    /// Equal keys, distinguishable payloads: a stable sort leaves them in the
    /// order they arrived. Checked on both paths.
    #[test]
    fn is_stable() {
        let mut v: Vec<(i32, usize)> = vec![(1, 0), (0, 1), (1, 2), (0, 3), (1, 4)];
        sort_by_infallible(&mut v, |a, b| a.0 - b.0);
        assert_eq!(v, vec![(0, 1), (0, 3), (1, 0), (1, 2), (1, 4)]);

        // Long enough to take the merge path, with only four distinct keys so
        // every key has a long run of ties to keep in order.
        let mut next = lcg();
        let long: Vec<(i32, usize)> = (0..500).map(|i| ((next() % 4) as i32, i)).collect();
        let mut got = long.clone();
        sort_by_infallible(&mut got, |a, b| a.0 - b.0);
        let mut want = long.clone();
        want.sort_by_key(|x| x.0);
        assert_eq!(got, want);
    }

    /// Comparison counts, at the size that dominates the workload.
    ///
    /// Exact numbers instead of a bound: they are what distinguishes a binary
    /// insertion sort from a linear one, and a regression to linear scanning
    /// would still satisfy any loose inequality at these lengths.
    #[test]
    fn spends_few_comparisons() {
        let count = |mut v: Vec<i32>| {
            let mut n = 0usize;
            sort_by_infallible(&mut v, |a, b| {
                n += 1;
                a - b
            });
            n
        };

        // n = 4 is ninety-five percent of the real calls. Ordered input costs
        // one comparison per element and nothing else, whereas reversed input is the
        // worst case and stays within `sum(1 + ceil(log2 i))`.
        assert_eq!(count((0..4).collect()), 3, "ordered n=4");
        assert_eq!(count((0..4).rev().collect()), 6, "reversed n=4");

        // The early exit is what makes an ordered input linear. Without it the
        // binary search alone would spend about `n log n` here.
        assert_eq!(count((0..32).collect()), 31, "ordered n=32");
    }

    /// Which algorithm runs on each side of [`MERGE_THRESHOLD`].
    ///
    /// The two paths have opposite signatures on ordered input (insertion
    /// spends one comparison per element, merge cannot), so the comparison
    /// count identifies the path. Without this, moving the boundary by one
    /// would silently start allocating on the hot size and every other test
    /// would still pass.
    #[test]
    fn dispatches_on_the_threshold() {
        let count = |mut v: Vec<i32>| {
            let mut n = 0usize;
            sort_by_infallible(&mut v, |a, b| {
                n += 1;
                a - b
            });
            n
        };

        let at = count((0..MERGE_THRESHOLD as i32).collect());
        let over = count((0..=(MERGE_THRESHOLD as i32)).collect());
        assert_eq!(
            at,
            MERGE_THRESHOLD - 1,
            "at the threshold the insertion path should cost one comparison per element"
        );
        assert!(
            over > MERGE_THRESHOLD * 2,
            "one past the threshold the merge path should run, but {over} comparisons \
             looks like the insertion path"
        );
    }

    #[test]
    fn reports_the_first_failure() {
        let mut v = vec![5, 4, 3, 2, 1];
        let err = sort_by(&mut v, |a, b| if *a == 3 { Err("no") } else { Ok(a - b) });
        assert_eq!(err, Err("no"));

        // And on the merge path.
        let mut v: Vec<i32> = (0..100).rev().collect();
        let err = sort_by(&mut v, |a, b| if *a == 42 { Err("no") } else { Ok(a - b) });
        assert_eq!(err, Err("no"));
    }

    /// The sort must not require `Clone`: the callers hold values that own a
    /// `Vec` and an `Arc`, and copying them is the cost this module exists to
    /// avoid. A non-`Clone` element type makes that a compile-time guarantee.
    #[test]
    fn does_not_clone_its_elements() {
        struct NoClone(i32);
        let mut v: Vec<NoClone> = (0..64).rev().map(NoClone).collect();
        sort_by_infallible(&mut v, |a, b| a.0 - b.0);
        assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
    }
}
