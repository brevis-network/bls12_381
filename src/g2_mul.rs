//! Variable-time $\mathbb{G}_2$ scalar multiplication, for the zkVM, where the prover knows every
//! input anyway: a windowed single multiplication, a multiscalar multiplication that splits each
//! scalar into four 64-bit digits with the $\psi$ endomorphism before Straus or Pippenger, and
//! the subgroup check.
//!
//! The algorithms run over [`G2Ops`]: projective points off the zkVM, and in it [`RawG2`], affine
//! points in canonical coordinates whose additions and doublings are the G2 precompiles.

// Off the zkVM the canonical-coordinate path only runs in the tests.
#![cfg_attr(not(target_os = "zkvm"), allow(dead_code))]

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::fp::Fp;
use crate::fp2::Fp2;
use crate::g2::{G2Affine, G2Projective};
#[cfg(feature = "alloc")]
use crate::Scalar;
use subtle::Choice;

// Declared here rather than imported from `pico_patch_libs`: the patch-libs revision this crate
// pins predates the G2 syscalls, which pico-sdk defines.
#[cfg(target_os = "zkvm")]
extern "C" {
    fn syscall_bls12381_g2_add(p: *mut [u64; 24], q: *const [u64; 24]);
    fn syscall_bls12381_g2_double(p: *mut [u64; 24]);
}

/// The group operations the algorithms need.
pub(crate) trait G2Ops: Copy {
    fn identity() -> Self;
    fn from_affine(p: &G2Affine) -> Self;
    fn plus(&self, other: &Self) -> Self;
    fn twice(&self) -> Self;
    fn negated(&self) -> Self;
    fn into_projective(self) -> G2Projective;
}

impl G2Ops for G2Projective {
    fn identity() -> Self {
        G2Projective::identity()
    }
    fn from_affine(p: &G2Affine) -> Self {
        G2Projective::from(p)
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
    fn into_projective(self) -> G2Projective {
        self
    }
}

/// An affine point as the G2 precompiles take it: `x.c0, x.c1, y.c0, y.c1`, six canonical
/// little-endian limbs each, or the point at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawG2 {
    limbs: [u64; 24],
    infinity: bool,
}

/// `f`'s canonical limbs.
fn fp_to_canonical(f: &Fp) -> [u64; 6] {
    #[cfg(target_os = "zkvm")]
    {
        let mut f = *f;
        f.mul_r_inv_internal();
        f.0
    }
    #[cfg(not(target_os = "zkvm"))]
    {
        let be = f.to_bytes();
        let mut limbs = [0u64; 6];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb = u64::from_be_bytes(be[40 - 8 * i..48 - 8 * i].try_into().unwrap());
        }
        limbs
    }
}

/// The field element of canonical limbs.
fn fp_from_canonical(limbs: &[u64]) -> Fp {
    let limbs: [u64; 6] = limbs.try_into().unwrap();
    #[cfg(target_os = "zkvm")]
    {
        let mut f = Fp(limbs);
        f.mul_r_internal();
        f
    }
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut be = [0u8; 48];
        for (i, limb) in limbs.iter().enumerate() {
            be[40 - 8 * i..48 - 8 * i].copy_from_slice(&limb.to_be_bytes());
        }
        Fp::from_bytes(&be).unwrap()
    }
}

/// `p - a` for a canonical nonzero `a`.
fn neg_canonical(a: &[u64]) -> [u64; 6] {
    let mut out = [0u64; 6];
    let mut borrow = 0u64;
    for i in 0..6 {
        let (d, b1) = crate::fp::MODULUS[i].overflowing_sub(a[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[i] = d;
        borrow = u64::from(b1 | b2);
    }
    out
}

impl RawG2 {
    const IDENTITY: RawG2 = RawG2 {
        limbs: [0; 24],
        infinity: true,
    };

    /// Coordinate `i` of `x.c0, x.c1, y.c0, y.c1`.
    fn coord(&self, i: usize) -> &[u64] {
        &self.limbs[6 * i..6 * i + 6]
    }

    fn x(&self) -> &[u64] {
        &self.limbs[..12]
    }

    fn y(&self) -> &[u64] {
        &self.limbs[12..]
    }

    fn to_affine(self) -> G2Affine {
        if self.infinity {
            return G2Affine::identity();
        }
        let c = |i: usize| fp_from_canonical(self.coord(i));
        G2Affine {
            x: Fp2 { c0: c(0), c1: c(1) },
            y: Fp2 { c0: c(2), c1: c(3) },
            infinity: Choice::from(0),
        }
    }

    /// `self + other` for `self.x != other.x`.
    fn add_distinct(&self, other: &Self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            let mut limbs = self.limbs;
            unsafe { syscall_bls12381_g2_add(&mut limbs, &other.limbs) };
            RawG2 {
                limbs,
                infinity: false,
            }
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            debug_assert!(self.x() != other.x(), "the add syscall refuses equal x");
            let sum = G2Projective::from(self.to_affine()) + G2Projective::from(other.to_affine());
            RawG2::from_affine(&G2Affine::from(sum))
        }
    }

    /// `2 self` for `self.y != 0`.
    fn double_nonzero_y(&self) -> Self {
        #[cfg(target_os = "zkvm")]
        {
            let mut limbs = self.limbs;
            unsafe { syscall_bls12381_g2_double(&mut limbs) };
            RawG2 {
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
            RawG2::from_affine(&G2Affine::from(
                G2Projective::from(self.to_affine()).double(),
            ))
        }
    }
}

impl G2Ops for RawG2 {
    fn identity() -> Self {
        RawG2::IDENTITY
    }
    fn from_affine(p: &G2Affine) -> Self {
        if bool::from(p.infinity) {
            return RawG2::IDENTITY;
        }
        let mut limbs = [0u64; 24];
        for (i, f) in [p.x.c0, p.x.c1, p.y.c0, p.y.c1].iter().enumerate() {
            limbs[6 * i..6 * i + 6].copy_from_slice(&fp_to_canonical(f));
        }
        RawG2 {
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
            RawG2::IDENTITY
        }
    }

    fn twice(&self) -> Self {
        if self.infinity || self.y().iter().all(|&l| l == 0) {
            RawG2::IDENTITY
        } else {
            self.double_nonzero_y()
        }
    }

    fn negated(&self) -> Self {
        if self.infinity {
            return *self;
        }
        let mut out = *self;
        for c in 2..4 {
            let y = self.coord(c);
            if y.iter().any(|&l| l != 0) {
                out.limbs[6 * c..6 * c + 6].copy_from_slice(&neg_canonical(y));
            }
        }
        out
    }

    fn into_projective(self) -> G2Projective {
        G2Projective::from(self.to_affine())
    }
}

/// The accumulator the zkVM proves with, and projective points elsewhere.
#[cfg(target_os = "zkvm")]
type Native = RawG2;
#[cfg(not(target_os = "zkvm"))]
type Native = G2Projective;

/// `acc + t`, `t` when there is no `acc` yet.
fn add_to<A: G2Ops>(acc: Option<A>, t: &A) -> A {
    match acc {
        Some(a) => a.plus(t),
        None => *t,
    }
}

/// `[k] p` for `k` as little-endian bytes of any length, with 4-bit windows.
pub(crate) fn multiply_windowed<A: G2Ops>(p: A, by: &[u8]) -> A {
    let mut table = [p; 15];
    for i in 1..15 {
        table[i] = table[i - 1].plus(&p);
    }
    let mut acc: Option<A> = None;
    for byte in by.iter().rev() {
        for nibble in [byte >> 4, byte & 0xf] {
            if let Some(a) = acc.as_mut() {
                for _ in 0..4 {
                    *a = a.twice();
                }
            }
            if nibble != 0 {
                let t = &table[nibble as usize - 1];
                acc = Some(add_to(acc, t));
            }
        }
    }
    acc.unwrap_or_else(A::identity)
}

/// Whether `[x] p = psi(p)`, as [`G2Affine::is_torsion_free`] checks it: `[|x|] p = -psi(p)`.
pub(crate) fn is_torsion_free_with<A: G2Ops + PartialEq>(p: &G2Affine) -> bool {
    let point = A::from_affine(p);
    let mut acc = point;
    for bit in (0..63 - X.leading_zeros()).rev() {
        acc = acc.twice();
        if (X >> bit) & 1 == 1 {
            acc = acc.plus(&point);
        }
    }
    acc == A::from_affine(&psi_affine(p)).negated()
}

impl G2Projective {
    /// `[k]self` for `k` as little-endian bytes of any length, in variable time. Unlike the
    /// multiexponentiation, `self` need not be in $\mathbb{G}_2$.
    pub(crate) fn multiply_vartime(&self, by: &[u8]) -> G2Projective {
        #[cfg(target_os = "zkvm")]
        let p = RawG2::from_affine(&G2Affine::from(self));
        #[cfg(not(target_os = "zkvm"))]
        let p = *self;
        multiply_windowed::<Native>(p, by).into_projective()
    }

    /// Variable-base multiscalar multiplication $\sum_i \[s_i\] P_i$, in variable time.
    ///
    /// Every point must be in $\mathbb{G}_2$ (as [`G2Affine::is_torsion_free`] checks): the
    /// scalars are split with $\psi$, which acts as $\[x\]$ only there. `points` and `scalars`
    /// pair up to the shorter of the two.
    #[cfg(feature = "alloc")]
    pub fn msm_variable_base(points: &[G2Affine], scalars: &[Scalar]) -> G2Projective {
        msm::<Native>(points, scalars).into_projective()
    }
}

/// [`G2Projective::msm_variable_base`] over `A`.
#[cfg(feature = "alloc")]
pub(crate) fn msm<A: G2Ops>(points: &[G2Affine], scalars: &[Scalar]) -> A {
    let (bases, digits) = gls_split(points, scalars);
    let bases: Vec<A> = bases.iter().map(A::from_affine).collect();
    if bases.is_empty() {
        return A::identity();
    }
    match best_method(bases.len()) {
        Method::Straus(w) => straus(&bases, &digits, w),
        Method::Pippenger(c) => pippenger(&bases, &digits, c),
    }
}

/// $|x|$, the BLS parameter: on $\mathbb{G}_2$, $\psi = \[x\] = \[-|x|\]$.
const X: u64 = crate::BLS_X;

/// `k` in base $|x|$: four digits below $2^{64}$, as $q = x^4 - x^2 + 1 < |x|^4$.
#[cfg(feature = "alloc")]
fn gls_digits(k: &Scalar) -> [u64; 4] {
    let bytes = k.to_bytes();
    let mut limbs = [0u64; 4];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        *limb = u64::from_le_bytes(chunk.try_into().unwrap());
    }
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
    digits
}

/// $\psi$ of an affine point, which stays affine: $\psi$ maps $z = 1$ to $z = 1$.
fn psi_affine(p: &G2Affine) -> G2Affine {
    let q = G2Projective::from(p).psi();
    G2Affine {
        x: q.x,
        y: q.y,
        infinity: p.infinity,
    }
}

/// `(P, k)` pairs as `([|x|^i] P, digit_i)`: $[|x|] P = -\psi(P)$. Pairs whose point is the
/// identity, and digits that are zero, are dropped.
#[cfg(feature = "alloc")]
fn gls_split(points: &[G2Affine], scalars: &[Scalar]) -> (Vec<G2Affine>, Vec<u64>) {
    let mut bases = Vec::with_capacity(4 * points.len());
    let mut digits = Vec::with_capacity(4 * points.len());
    for (p, k) in points.iter().zip(scalars) {
        if bool::from(p.infinity) {
            continue;
        }
        let mut base = *p;
        for (i, d) in gls_digits(k).into_iter().enumerate() {
            if i > 0 {
                base = -psi_affine(&base);
            }
            if d != 0 {
                bases.push(base);
                digits.push(d);
            }
        }
    }
    (bases, digits)
}

#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    /// Interleaved width-`w` NAF.
    Straus(u32),
    /// Buckets over `c`-bit windows.
    Pippenger(u32),
}

/// The cheaper method for `m` 64-bit digits, by a cost model in tenths of an addition.
#[cfg(feature = "alloc")]
fn best_method(m: usize) -> Method {
    const ADD: u64 = 10;
    const DOUBLE: u64 = 6;
    let m = m as u64;
    let straus = (2..=7).map(|w| {
        let table = 1u64 << (w - 2);
        // per digit: a table of one doubling and `table - 1` additions, then about 65 / (w + 1)
        // additions; 64 doublings shared
        let cost =
            m * (DOUBLE + ADD * (table - 1)) + m * 65 * ADD / (u64::from(w) + 1) + 64 * DOUBLE;
        (cost, Method::Straus(w))
    });
    let pippenger = (2..=16).map(|c| {
        let windows = (64 + u64::from(c) - 1) / u64::from(c);
        // per window: one addition per digit and two per bucket; 64 doublings in all
        let cost = ADD * windows * (m + 2 * (1 << c)) + 64 * DOUBLE;
        (cost, Method::Pippenger(c))
    });
    straus
        .chain(pippenger)
        .min_by_key(|(cost, _)| *cost)
        .map(|(_, method)| method)
        .unwrap()
}

/// Width-`w` NAF of `k`, least significant digit first: odd digits below $2^{w-1}$ in size.
#[cfg(feature = "alloc")]
fn wnaf(k: u64, w: u32) -> ([i8; 66], usize) {
    let mut k = u128::from(k);
    let mut out = [0i8; 66];
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

/// Interleaved width-`w` NAF: one shared chain of 64 doublings.
#[cfg(feature = "alloc")]
fn straus<A: G2Ops>(bases: &[A], digits: &[u64], w: u32) -> A {
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
    let nafs: Vec<_> = digits.iter().map(|&d| wnaf(d, w)).collect();
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

/// Pippenger's bucket method over `c`-bit windows of the 64-bit digits.
#[cfg(feature = "alloc")]
fn pippenger<A: G2Ops>(bases: &[A], digits: &[u64], c: u32) -> A {
    let windows = (64 + c - 1) / c;
    let mask = (1u64 << c) - 1;
    let mut total: Option<A> = None;
    let mut buckets: Vec<Option<A>> = vec![None; mask as usize];
    for window in (0..windows).rev() {
        if let Some(t) = total.as_mut() {
            for _ in 0..c {
                *t = t.twice();
            }
        }
        buckets.iter_mut().for_each(|b| *b = None);
        for (p, &d) in bases.iter().zip(digits) {
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
    use crate::Scalar;
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
            x - Scalar::one(),
            x * x,
            x * x * x,
            x * x * x - Scalar::one(),
            Scalar::from(u64::MAX),
        ]
    }

    #[test]
    fn psi_squared_is_psi2_and_psi_is_minus_x() {
        let mut rng = rng();
        for _ in 0..4 {
            let p = G2Projective::random(&mut rng);
            assert_eq!(p.psi().psi(), p.psi2());
            assert_eq!(-p.psi(), p * Scalar::from(X));
        }
    }

    #[test]
    fn gls_digits_recompose() {
        let mut rng = rng();
        let x = Scalar::from(X);
        let scalars = edge_scalars()
            .into_iter()
            .chain((0..64).map(|_| Scalar::random(&mut rng)));
        for k in scalars {
            let d = gls_digits(&k);
            let back = d
                .iter()
                .rev()
                .fold(Scalar::zero(), |acc, &d| acc * x + Scalar::from(d));
            assert_eq!(back, k);
        }
    }

    #[test]
    fn wnaf_recomposes() {
        for w in 2..=7 {
            for k in [0, 1, 2, 3, 7, 0x8000_0000_0000_0000, u64::MAX, X, X - 1] {
                let (naf, len) = wnaf(k, w);
                let back = naf[..len]
                    .iter()
                    .rev()
                    .fold(0i128, |acc, &d| 2 * acc + i128::from(d));
                assert_eq!(back, i128::from(k), "k {k:#x} w {w}");
                assert!(naf[..len]
                    .iter()
                    .all(|&d| d == 0 || (d % 2 != 0 && i32::from(d).abs() < 1 << (w - 1))));
            }
        }
    }

    #[test]
    fn raw_points_round_trip_and_negate() {
        let mut rng = rng();
        for _ in 0..4 {
            let p = G2Affine::from(G2Projective::random(&mut rng));
            let raw = RawG2::from_affine(&p);
            assert_eq!(raw.to_affine(), p);
            assert_eq!(raw.negated().to_affine(), -p);
        }
        let id = RawG2::from_affine(&G2Affine::identity());
        assert_eq!(id, RawG2::IDENTITY);
        assert_eq!(id.negated(), id);
    }

    #[test]
    fn raw_arithmetic_covers_every_case() {
        let mut rng = rng();
        let p = RawG2::from_affine(&G2Affine::from(G2Projective::random(&mut rng)));
        let q = RawG2::from_affine(&G2Affine::from(G2Projective::random(&mut rng)));
        let id = RawG2::IDENTITY;
        let proj = |r: RawG2| r.into_projective();
        assert_eq!(proj(p.plus(&q)), proj(p) + proj(q));
        assert_eq!(p.plus(&p), p.twice());
        assert_eq!(proj(p.twice()), proj(p).double());
        assert_eq!(p.plus(&p.negated()), id);
        assert_eq!(p.plus(&id), p);
        assert_eq!(id.plus(&p), p);
        assert_eq!(id.twice(), id);
    }

    #[test]
    fn multiply_vartime_matches_constant_time() {
        let mut rng = rng();
        let points = [
            G2Projective::identity(),
            G2Projective::generator(),
            G2Projective::random(&mut rng),
        ];
        let scalars = edge_scalars()
            .into_iter()
            .chain((0..8).map(|_| Scalar::random(&mut rng)));
        for k in scalars {
            let by = k.to_bytes();
            for p in &points {
                let expected = p.multiply(&by);
                assert_eq!(p.multiply_vartime(&by), expected);
                let raw = RawG2::from_affine(&G2Affine::from(p));
                assert_eq!(multiply_windowed(raw, &by).into_projective(), expected);
            }
        }
        // a point outside G2 is in `test_clear_cofactor`
    }

    #[test]
    fn the_raw_subgroup_check_agrees() {
        let mut rng = rng();
        for _ in 0..4 {
            let p = G2Affine::from(G2Projective::random(&mut rng));
            assert!(is_torsion_free_with::<RawG2>(&p));
            assert!(is_torsion_free_with::<G2Projective>(&p));
        }
        // off the subgroup in `test_is_torsion_free` and `test_clear_cofactor`
    }

    fn naive(points: &[G2Affine], scalars: &[Scalar]) -> G2Projective {
        points
            .iter()
            .zip(scalars)
            .map(|(p, s)| G2Projective::from(p).multiply(&s.to_bytes()))
            .sum()
    }

    fn inputs(n: usize, rng: &mut XorShiftRng) -> (Vec<G2Affine>, Vec<Scalar>) {
        let mut points: Vec<G2Affine> = (0..n)
            .map(|_| G2Affine::from(G2Projective::random(&mut *rng)))
            .collect();
        let mut scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut *rng)).collect();
        // repeats, opposites, the identity and edge scalars
        if n >= 4 {
            points[1] = points[0];
            points[2] = -points[0];
            points[3] = G2Affine::identity();
            for (s, e) in scalars.iter_mut().zip(edge_scalars()) {
                *s = e;
            }
        }
        (points, scalars)
    }

    #[test]
    fn msm_matches_naive() {
        let mut rng = rng();
        for n in [0, 1, 2, 3, 5, 16, 40, 130] {
            let (points, scalars) = inputs(n, &mut rng);
            let expected = naive(&points, &scalars);
            assert_eq!(
                G2Projective::msm_variable_base(&points, &scalars),
                expected,
                "n {n}"
            );
            if n <= 16 {
                assert_eq!(
                    msm::<RawG2>(&points, &scalars).into_projective(),
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
        let (bases, digits) = gls_split(&points, &scalars);
        let proj: Vec<G2Projective> = bases.iter().map(G2Projective::from).collect();
        let raw: Vec<RawG2> = bases.iter().map(RawG2::from_affine).collect();
        for w in 2..=7 {
            assert_eq!(straus(&proj, &digits, w), expected, "straus {w}");
        }
        for c in 1..=16 {
            assert_eq!(pippenger(&proj, &digits, c), expected, "pippenger {c}");
        }
        for (w, c) in [(2, 1), (4, 3), (6, 8)] {
            assert_eq!(
                straus(&raw, &digits, w).into_projective(),
                expected,
                "raw straus {w}"
            );
            assert_eq!(
                pippenger(&raw, &digits, c).into_projective(),
                expected,
                "raw pippenger {c}"
            );
        }
        // the same scalar against P and -P cancels
        let p = points[0];
        assert_eq!(
            G2Projective::msm_variable_base(&[p, -p], &[scalars[4], scalars[4]]),
            G2Projective::identity()
        );
    }

    #[test]
    fn best_method_grows_with_size() {
        assert!(matches!(best_method(4), Method::Straus(_)));
        assert!(matches!(best_method(4096), Method::Pippenger(_)));
    }
}
