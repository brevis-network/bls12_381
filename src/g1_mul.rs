//! Variable-time $\mathbb{G}_1$ arithmetic for the zkVM, where the prover knows every input anyway:
//! a multiscalar multiplication that splits each scalar into two 128-bit halves with the GLV
//! endomorphism before Straus or Pippenger.
//!
//! The algorithms run over [`G1Ops`]: projective points off the zkVM, and in it [`RawG1`], affine
//! points in canonical coordinates whose additions and doublings are the G1 precompiles.

// Off the zkVM the canonical-coordinate path only runs in the tests.
#![cfg_attr(not(target_os = "zkvm"), allow(dead_code))]

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

#[cfg(feature = "alloc")]
use crate::g1::endomorphism;
use crate::g1::{G1Affine, G1Projective};
use crate::g2_mul::{fp_from_canonical, fp_to_canonical, neg_canonical};
#[cfg(feature = "alloc")]
use crate::Scalar;
use subtle::Choice;

#[cfg(target_os = "zkvm")]
use pico_patch_libs::{syscall_bls12381_add, syscall_bls12381_double};

/// The group operations the algorithms need.
pub(crate) trait G1Ops: Copy {
    fn identity() -> Self;
    fn from_affine(p: &G1Affine) -> Self;
    fn plus(&self, other: &Self) -> Self;
    fn twice(&self) -> Self;
    fn negated(&self) -> Self;
    fn into_projective(self) -> G1Projective;
}

impl G1Ops for G1Projective {
    fn identity() -> Self {
        G1Projective::identity()
    }
    fn from_affine(p: &G1Affine) -> Self {
        G1Projective::from(p)
    }
    fn plus(&self, other: &Self) -> Self {
        self + other
    }
    fn twice(&self) -> Self {
        self.double()
    }
    fn negated(&self) -> Self {
        -self
    }
    fn into_projective(self) -> G1Projective {
        self
    }
}

/// An affine point as the G1 precompiles take it: `x, y`, six canonical little-endian limbs each,
/// or the point at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawG1 {
    limbs: [u64; 12],
    infinity: bool,
}

impl RawG1 {
    const IDENTITY: RawG1 = RawG1 {
        limbs: [0; 12],
        infinity: true,
    };

    fn x(&self) -> &[u64] {
        &self.limbs[..6]
    }

    fn y(&self) -> &[u64] {
        &self.limbs[6..]
    }

    fn to_affine(self) -> G1Affine {
        if self.infinity {
            return G1Affine::identity();
        }
        G1Affine {
            x: fp_from_canonical(self.x()),
            y: fp_from_canonical(self.y()),
            infinity: Choice::from(0),
        }
    }

    /// `self + other` for `self.x != other.x`.
    fn add_distinct(&self, other: &Self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            let mut limbs = self.limbs;
            unsafe { syscall_bls12381_add(&mut limbs, &other.limbs) };
            RawG1 {
                limbs,
                infinity: false,
            }
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            debug_assert!(self.x() != other.x(), "the add syscall refuses equal x");
            let sum = G1Projective::from(self.to_affine()) + G1Projective::from(other.to_affine());
            RawG1::from_affine(&G1Affine::from(sum))
        }
    }

    /// `2 self` for `self.y != 0`.
    fn double_nonzero_y(&self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            let mut limbs = self.limbs;
            unsafe { syscall_bls12381_double(&mut limbs) };
            RawG1 {
                limbs,
                infinity: false,
            }
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            debug_assert!(
                self.y().iter().any(|&l| l != 0),
                "the double syscall refuses y = 0"
            );
            RawG1::from_affine(&G1Affine::from(
                G1Projective::from(self.to_affine()).double(),
            ))
        }
    }
}

impl G1Ops for RawG1 {
    fn identity() -> Self {
        RawG1::IDENTITY
    }
    fn from_affine(p: &G1Affine) -> Self {
        if bool::from(p.infinity) {
            return RawG1::IDENTITY;
        }
        let mut limbs = [0u64; 12];
        limbs[..6].copy_from_slice(&fp_to_canonical(&p.x));
        limbs[6..].copy_from_slice(&fp_to_canonical(&p.y));
        RawG1 {
            limbs,
            infinity: false,
        }
    }

    fn plus(&self, other: &Self) -> Self {
        if self.infinity {
            *other
        } else if other.infinity {
            *self
        } else if self.x() != other.x() {
            self.add_distinct(other)
        } else if self.y() == other.y() {
            self.twice()
        } else {
            // the same x and another y: `other = -self`
            RawG1::IDENTITY
        }
    }

    fn twice(&self) -> Self {
        if self.infinity || self.y().iter().all(|&l| l == 0) {
            RawG1::IDENTITY
        } else {
            self.double_nonzero_y()
        }
    }

    fn negated(&self) -> Self {
        if self.infinity || self.y().iter().all(|&l| l == 0) {
            return *self;
        }
        let mut out = *self;
        out.limbs[6..].copy_from_slice(&neg_canonical(self.y()));
        out
    }

    fn into_projective(self) -> G1Projective {
        G1Projective::from(self.to_affine())
    }
}

/// $|x|$, the BLS parameter.
const X: u64 = crate::BLS_X;

/// The accumulator the zkVM proves with, and projective points elsewhere.
#[cfg(target_os = "zkvm")]
type Native = RawG1;
#[cfg(not(target_os = "zkvm"))]
type Native = G1Projective;

/// [`G1Projective::msm_variable_base`] in the zkVM.
#[cfg(feature = "alloc")]
pub(crate) fn msm_native(points: &[G1Projective], scalars: &[Scalar]) -> G1Projective {
    let mut affine = vec![G1Affine::identity(); points.len()];
    G1Projective::batch_normalize(points, &mut affine);
    msm::<Native>(&affine, scalars).into_projective()
}

/// `acc + t`, `t` when there is no `acc` yet.
fn add_to<A: G1Ops>(acc: Option<A>, t: &A) -> A {
    match acc {
        Some(a) => a.plus(t),
        None => *t,
    }
}

/// $\sum_i \[s_i\] P_i$ over `A`, for points in $\mathbb{G}_1$: the GLV split relies on
/// $\phi = \[-x^2\]$, which holds only there.
#[cfg(feature = "alloc")]
pub(crate) fn msm<A: G1Ops>(points: &[G1Affine], scalars: &[Scalar]) -> A {
    let (bases, halves) = glv_split(points, scalars);
    let bases: Vec<A> = bases.iter().map(A::from_affine).collect();
    if bases.is_empty() {
        return A::identity();
    }
    match best_method(bases.len()) {
        Method::Straus(w) => straus(&bases, &halves, w),
        Method::Pippenger(c) => pippenger(&bases, &halves, c),
    }
}

/// `k = lo + hi |x|^2` with both halves below $|x|^2 < 2^{128}$, as $q < |x|^4$.
#[cfg(feature = "alloc")]
fn glv_halves(k: &Scalar) -> [u128; 2] {
    let bytes = k.to_bytes();
    let mut limbs = [0u64; 4];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        *limb = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    // base-|x| digits, then pairs of them
    let mut digits = [0u64; 4];
    for digit in digits.iter_mut() {
        let mut rem = 0u128;
        for limb in limbs.iter_mut().rev() {
            let cur = (rem << 64) | u128::from(*limb);
            *limb = (cur / u128::from(X)) as u64;
            rem = cur % u128::from(X);
        }
        *digit = rem as u64;
    }
    debug_assert_eq!(limbs, [0; 4]);
    let x = u128::from(X);
    [
        u128::from(digits[0]) + u128::from(digits[1]) * x,
        u128::from(digits[2]) + u128::from(digits[3]) * x,
    ]
}

/// `(P, k)` pairs as `(P, lo)` and `([|x|^2] P, hi) = (-phi(P), hi)`. Pairs whose point is the
/// identity, and halves that are zero, are dropped; pairs past the shorter input are ignored.
#[cfg(feature = "alloc")]
fn glv_split(points: &[G1Affine], scalars: &[Scalar]) -> (Vec<G1Affine>, Vec<u128>) {
    let mut bases = Vec::with_capacity(2 * points.len());
    let mut halves = Vec::with_capacity(2 * points.len());
    for (p, k) in points.iter().zip(scalars) {
        if bool::from(p.infinity) {
            continue;
        }
        let [lo, hi] = glv_halves(k);
        if lo != 0 {
            bases.push(*p);
            halves.push(lo);
        }
        if hi != 0 {
            bases.push(-endomorphism(p));
            halves.push(hi);
        }
    }
    (bases, halves)
}

/// Bits in a half: $|x|^2 < 2^{128}$.
#[cfg(feature = "alloc")]
const BITS: u32 = 128;

#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    /// Interleaved width-`w` NAF.
    Straus(u32),
    /// Buckets over `c`-bit windows.
    Pippenger(u32),
}

/// The cheaper method for `m` 128-bit halves, by a cost model in tenths of an addition.
#[cfg(feature = "alloc")]
fn best_method(m: usize) -> Method {
    const ADD: u64 = 10;
    const DOUBLE: u64 = 6;
    let m = m as u64;
    let bits = u64::from(BITS);
    let straus = (2..=7).map(|w| {
        let table = 1u64 << (w - 2);
        let cost = m * (DOUBLE + ADD * (table - 1))
            + m * (bits + 1) * ADD / (u64::from(w) + 1)
            + bits * DOUBLE;
        (cost, Method::Straus(w))
    });
    // capped: a wider window only grows the 2^c buckets
    let pippenger = (2..=12).map(|c| {
        let windows = (bits + u64::from(c) - 1) / u64::from(c);
        let cost = ADD * windows * (m + 2 * (1 << c)) + bits * DOUBLE;
        (cost, Method::Pippenger(c))
    });
    straus
        .chain(pippenger)
        .min_by_key(|(cost, _)| *cost)
        .map(|(_, method)| method)
        .unwrap()
}

/// Width-`w` NAF of `k < 2^127.5`, least significant digit first.
#[cfg(feature = "alloc")]
fn wnaf(k: u128, w: u32) -> ([i8; 130], usize) {
    let mut k = k;
    let mut out = [0i8; 130];
    let mut len = 0;
    while k != 0 {
        if k & 1 == 1 {
            let mut d = (k & ((1 << w) - 1)) as i32;
            if d >= 1 << (w - 1) {
                d -= 1 << w;
            }
            out[len] = d as i8;
            if d > 0 {
                k -= d as u128;
            } else {
                k += (-d) as u128;
            }
        }
        len += 1;
        k >>= 1;
    }
    (out, len)
}

/// Interleaved width-`w` NAF: one shared chain of doublings.
#[cfg(feature = "alloc")]
fn straus<A: G1Ops>(bases: &[A], halves: &[u128], w: u32) -> A {
    let half = 1usize << (w - 2);
    // table[j * half + i] = [2i + 1] bases[j]
    let mut table = Vec::with_capacity(bases.len() * half);
    for p in bases {
        let twice = p.twice();
        table.push(*p);
        for _ in 1..half {
            let next = table[table.len() - 1].plus(&twice);
            table.push(next);
        }
    }
    let nafs: Vec<_> = halves.iter().map(|&d| wnaf(d, w)).collect();
    let len = nafs.iter().map(|(_, len)| *len).max().unwrap_or(0);
    let mut acc: Option<A> = None;
    for i in (0..len).rev() {
        if let Some(a) = acc.as_mut() {
            *a = a.twice();
        }
        for (j, (naf, _)) in nafs.iter().enumerate() {
            let d = naf[i];
            if d == 0 {
                continue;
            }
            let t = &table[j * half + (d.unsigned_abs() as usize >> 1)];
            let t = if d < 0 { t.negated() } else { *t };
            acc = Some(add_to(acc, &t));
        }
    }
    acc.unwrap_or_else(A::identity)
}

/// Pippenger's bucket method over `c`-bit windows of the halves.
#[cfg(feature = "alloc")]
fn pippenger<A: G1Ops>(bases: &[A], halves: &[u128], c: u32) -> A {
    let windows = (BITS + c - 1) / c;
    let mask = (1u128 << c) - 1;
    let mut total: Option<A> = None;
    let mut buckets: Vec<Option<A>> = vec![None; mask as usize];
    for window in (0..windows).rev() {
        if let Some(t) = total.as_mut() {
            for _ in 0..c {
                *t = t.twice();
            }
        }
        buckets.iter_mut().for_each(|b| *b = None);
        for (p, &d) in bases.iter().zip(halves) {
            let b = (d >> (window * c)) & mask;
            if b != 0 {
                let bucket = &mut buckets[b as usize - 1];
                *bucket = Some(add_to(*bucket, p));
            }
        }
        // sum_b [b] bucket_b, as a running sum from the top bucket down
        let mut running: Option<A> = None;
        let mut sum: Option<A> = None;
        for bucket in buckets.iter().rev() {
            if let Some(b) = bucket {
                running = Some(add_to(running, b));
            }
            if let Some(r) = running {
                sum = Some(add_to(sum, &r));
            }
        }
        if let Some(s) = sum {
            total = Some(add_to(total, &s));
        }
    }
    total.unwrap_or_else(A::identity)
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use ff::Field;
    use group::Group;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    fn rng() -> XorShiftRng {
        XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ])
    }

    fn edge_scalars() -> Vec<Scalar> {
        let x = Scalar::from(X);
        vec![
            Scalar::zero(),
            Scalar::one(),
            -Scalar::one(),
            x,
            x * x,
            x * x - Scalar::one(),
            x * x * x,
            -(x * x),
            Scalar::from(u64::MAX),
        ]
    }

    #[test]
    fn phi_is_minus_x_squared() {
        let mut rng = rng();
        let x = Scalar::from(X);
        for _ in 0..4 {
            let p = G1Affine::from(G1Projective::random(&mut rng));
            assert_eq!(
                G1Projective::from(endomorphism(&p)),
                -(G1Projective::from(p) * (x * x))
            );
        }
    }

    #[test]
    fn glv_halves_recompose() {
        let mut rng = rng();
        let x2 = Scalar::from(X) * Scalar::from(X);
        let scalars = edge_scalars()
            .into_iter()
            .chain((0..64).map(|_| Scalar::random(&mut rng)));
        for k in scalars {
            let [lo, hi] = glv_halves(&k);
            assert!(lo < u128::from(X) * u128::from(X) && hi < u128::from(X) * u128::from(X));
            let s = |v: u128| {
                Scalar::from((v >> 64) as u64) * Scalar::from(1u64 << 32) * Scalar::from(1u64 << 32)
                    + Scalar::from(v as u64)
            };
            assert_eq!(s(lo) + s(hi) * x2, k);
        }
    }

    #[test]
    fn wnaf_recomposes() {
        let x2 = u128::from(X) * u128::from(X);
        for w in 2..=7 {
            for k in [
                0,
                1,
                2,
                3,
                7,
                1u128 << 64,
                u128::from(u64::MAX),
                x2 - 1,
                x2 / 3,
            ] {
                let (naf, len) = wnaf(k, w);
                let back = naf[..len]
                    .iter()
                    .rev()
                    // `k` can exceed `i128::MAX`; mod 2^128 the fold is exact
                    .fold(0u128, |acc, &d| {
                        acc.wrapping_mul(2).wrapping_add(i128::from(d) as u128)
                    });
                assert_eq!(back, k, "k {k:#x} w {w}");
            }
        }
    }

    #[test]
    fn raw_points_round_trip_and_negate() {
        let mut rng = rng();
        for _ in 0..4 {
            let p = G1Affine::from(G1Projective::random(&mut rng));
            let raw = RawG1::from_affine(&p);
            assert_eq!(raw.to_affine(), p);
            assert_eq!(raw.negated().to_affine(), -p);
        }
        let id = RawG1::from_affine(&G1Affine::identity());
        assert_eq!(id, RawG1::IDENTITY);
        assert_eq!(id.negated(), id);
    }

    #[test]
    fn raw_arithmetic_covers_every_case() {
        let mut rng = rng();
        let p = RawG1::from_affine(&G1Affine::from(G1Projective::random(&mut rng)));
        let q = RawG1::from_affine(&G1Affine::from(G1Projective::random(&mut rng)));
        let id = RawG1::IDENTITY;
        let proj = |r: RawG1| r.into_projective();
        assert_eq!(proj(p.plus(&q)), proj(p) + proj(q));
        assert_eq!(p.plus(&p), p.twice());
        assert_eq!(proj(p.twice()), proj(p).double());
        assert_eq!(p.plus(&p.negated()), id);
        assert_eq!(p.plus(&id), p);
        assert_eq!(id.plus(&p), p);
        assert_eq!(id.twice(), id);
    }

    fn naive(points: &[G1Affine], scalars: &[Scalar]) -> G1Projective {
        points
            .iter()
            .zip(scalars)
            .map(|(p, s)| G1Projective::from(p) * s)
            .sum()
    }

    fn inputs(n: usize, rng: &mut XorShiftRng) -> (Vec<G1Affine>, Vec<Scalar>) {
        let mut points: Vec<G1Affine> = (0..n)
            .map(|_| G1Affine::from(G1Projective::random(&mut *rng)))
            .collect();
        let mut scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut *rng)).collect();
        // repeats, opposites, the identity and edge scalars
        if n >= 4 {
            points[1] = points[0];
            points[2] = -points[0];
            points[3] = G1Affine::identity();
            for (s, e) in scalars.iter_mut().zip(edge_scalars()) {
                *s = e;
            }
        }
        (points, scalars)
    }

    #[test]
    fn msm_matches_naive() {
        let mut rng = rng();
        for n in [0, 1, 2, 3, 5, 16, 40, 128, 300] {
            let (points, scalars) = inputs(n, &mut rng);
            let expected = naive(&points, &scalars);
            assert_eq!(msm::<G1Projective>(&points, &scalars), expected, "n {n}");
            let proj: Vec<G1Projective> = points.iter().map(G1Projective::from).collect();
            assert_eq!(msm_native(&proj, &scalars), expected, "native, n {n}");
            assert_eq!(
                G1Projective::msm_variable_base(&proj, &scalars),
                expected,
                "msm_variable_base, n {n}"
            );
            if n <= 16 {
                assert_eq!(
                    msm::<RawG1>(&points, &scalars).into_projective(),
                    expected,
                    "raw, n {n}"
                );
            }
        }
    }

    #[test]
    fn every_method_matches_naive() {
        let mut rng = rng();
        let (points, scalars) = inputs(12, &mut rng);
        let expected = naive(&points, &scalars);
        let (bases, halves) = glv_split(&points, &scalars);
        let proj: Vec<G1Projective> = bases.iter().map(G1Projective::from).collect();
        let raw: Vec<RawG1> = bases.iter().map(RawG1::from_affine).collect();
        for w in 2..=7 {
            assert_eq!(straus(&proj, &halves, w), expected, "straus {w}");
        }
        for c in 1..=16 {
            assert_eq!(pippenger(&proj, &halves, c), expected, "pippenger {c}");
        }
        for (w, c) in [(2, 1), (4, 3), (6, 8)] {
            assert_eq!(
                straus(&raw, &halves, w).into_projective(),
                expected,
                "raw straus {w}"
            );
            assert_eq!(
                pippenger(&raw, &halves, c).into_projective(),
                expected,
                "raw pippenger {c}"
            );
        }
        // the same scalar against P and -P cancels
        let p = points[0];
        assert_eq!(
            msm::<G1Projective>(&[p, -p], &[scalars[4], scalars[4]]),
            G1Projective::identity()
        );
    }

    #[test]
    fn best_method_grows_with_size() {
        assert!(matches!(best_method(2), Method::Straus(_)));
        assert!(matches!(best_method(4096), Method::Pippenger(_)));
    }
}
