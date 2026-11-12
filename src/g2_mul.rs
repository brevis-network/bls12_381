//! Variable-time $\mathbb{G}_2$ scalar multiplication, for the zkVM, where the prover knows every
//! input anyway: a windowed single multiplication, and a multiscalar multiplication that splits
//! each scalar into four 64-bit digits with the $\psi$ endomorphism before Straus or Pippenger.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::g2::{G2Affine, G2Projective};
#[cfg(feature = "alloc")]
use crate::Scalar;

impl G2Projective {
    /// `[k]self` for `k` as little-endian bytes of any length, in variable time, with 4-bit
    /// windows. Unlike the multiexponentiation, `self` need not be in $\mathbb{G}_2$.
    #[cfg_attr(not(target_os = "zkvm"), allow(dead_code))]
    pub(crate) fn multiply_vartime(&self, by: &[u8]) -> G2Projective {
        let mut table = [G2Projective::identity(); 16];
        for i in 1..16 {
            table[i] = table[i - 1] + self;
        }
        let mut acc: Option<G2Projective> = None;
        for byte in by.iter().rev() {
            for nibble in [byte >> 4, byte & 0xf] {
                if let Some(a) = acc.as_mut() {
                    for _ in 0..4 {
                        *a = a.double();
                    }
                }
                if nibble != 0 {
                    let t = &table[nibble as usize];
                    acc = Some(match acc {
                        Some(a) => a + t,
                        None => *t,
                    });
                }
            }
        }
        acc.unwrap_or_else(G2Projective::identity)
    }

    /// Variable-base multiscalar multiplication $\sum_i \[s_i\] P_i$, in variable time.
    ///
    /// Every point must be in $\mathbb{G}_2$ (as [`G2Affine::is_torsion_free`] checks): the
    /// scalars are split with $\psi$, which acts as $\[x\]$ only there. `points` and `scalars`
    /// pair up to the shorter of the two.
    #[cfg(feature = "alloc")]
    pub fn msm_variable_base(points: &[G2Affine], scalars: &[Scalar]) -> G2Projective {
        let (bases, digits) = gls_split(points, scalars);
        if bases.is_empty() {
            return G2Projective::identity();
        }
        match best_method(bases.len()) {
            Method::Straus(w) => straus(&bases, &digits, w),
            Method::Pippenger(c) => pippenger(&bases, &digits, c),
        }
    }
}

/// $|x|$, the BLS parameter: on $\mathbb{G}_2$, $\psi = \[x\] = \[-|x|\]$.
#[cfg(feature = "alloc")]
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
#[cfg(feature = "alloc")]
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

/// The cheaper method for `m` 64-bit digits, counting additions, a doubling as 0.6 of one.
#[cfg(feature = "alloc")]
fn best_method(m: usize) -> Method {
    let m = m as u64;
    let doublings = 64 * 6;
    let straus = (2..=7).map(|w| {
        let table = 1u64 << (w - 2);
        // per point: one doubling and `table - 1` additions for the table, 65/(w+1) additions
        let cost = m * (6 + 10 * (table - 1)) + m * 650 / (u64::from(w) + 1) + doublings;
        (cost, Method::Straus(w))
    });
    let pippenger = (2..=16).map(|c| {
        let windows = (64 + u64::from(c) - 1) / u64::from(c);
        let cost = 10 * windows * (m + 2 * (1 << c)) + 6 * 64;
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
fn straus(bases: &[G2Affine], digits: &[u64], w: u32) -> G2Projective {
    let half = 1usize << (w - 2);
    // table[j * half + i] = [2i + 1] bases[j]
    let mut table = Vec::with_capacity(bases.len() * half);
    for p in bases {
        let p = G2Projective::from(p);
        let twice = p.double();
        table.push(p);
        for i in 1..half {
            let next = table[table.len() - 1] + twice;
            debug_assert_eq!(table.len() % half, i);
            table.push(next);
        }
    }
    let nafs: Vec<_> = digits.iter().map(|&d| wnaf(d, w)).collect();
    let len = nafs.iter().map(|(_, len)| *len).max().unwrap_or(0);
    let mut acc: Option<G2Projective> = None;
    for i in (0..len).rev() {
        if let Some(a) = acc.as_mut() {
            *a = a.double();
        }
        for (j, (naf, _)) in nafs.iter().enumerate() {
            let d = naf[i];
            if d == 0 {
                continue;
            }
            let t = &table[j * half + (d.unsigned_abs() as usize >> 1)];
            let t = if d < 0 { -t } else { *t };
            acc = Some(match acc {
                Some(a) => a + t,
                None => t,
            });
        }
    }
    acc.unwrap_or_else(G2Projective::identity)
}

/// Pippenger's bucket method over `c`-bit windows of the 64-bit digits.
#[cfg(feature = "alloc")]
fn pippenger(bases: &[G2Affine], digits: &[u64], c: u32) -> G2Projective {
    let windows = (64 + c - 1) / c;
    let mask = (1u64 << c) - 1;
    let mut total: Option<G2Projective> = None;
    let mut buckets: Vec<Option<G2Projective>> = vec![None; mask as usize];
    for window in (0..windows).rev() {
        if let Some(t) = total.as_mut() {
            for _ in 0..c {
                *t = t.double();
            }
        }
        buckets.iter_mut().for_each(|b| *b = None);
        for (p, &d) in bases.iter().zip(digits) {
            let b = (d >> (window * c)) & mask;
            if b != 0 {
                let bucket = &mut buckets[b as usize - 1];
                *bucket = Some(match bucket {
                    Some(sum) => sum.add_mixed(p),
                    None => G2Projective::from(p),
                });
            }
        }
        // sum_b [b] bucket_b, as a running sum from the top bucket down
        let mut running: Option<G2Projective> = None;
        let mut sum: Option<G2Projective> = None;
        for bucket in buckets.iter().rev() {
            if let Some(b) = bucket {
                running = Some(match running {
                    Some(r) => r + b,
                    None => *b,
                });
            }
            if let Some(r) = running {
                sum = Some(match sum {
                    Some(s) => s + r,
                    None => r,
                });
            }
        }
        if let Some(s) = sum {
            total = Some(match total {
                Some(t) => t + s,
                None => s,
            });
        }
    }
    total.unwrap_or_else(G2Projective::identity)
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
            for p in &points {
                assert_eq!(p.multiply_vartime(&k.to_bytes()), p.multiply(&k.to_bytes()));
            }
        }
        // a point outside G2 is in `test_clear_cofactor`
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
            assert_eq!(
                G2Projective::msm_variable_base(&points, &scalars),
                naive(&points, &scalars),
                "n {n}"
            );
        }
    }

    #[test]
    fn every_method_matches_naive() {
        let mut rng = rng();
        let (points, scalars) = inputs(12, &mut rng);
        let expected = naive(&points, &scalars);
        let (bases, digits) = gls_split(&points, &scalars);
        for w in 2..=7 {
            assert_eq!(straus(&bases, &digits, w), expected, "straus {w}");
        }
        for c in 1..=16 {
            assert_eq!(pippenger(&bases, &digits, c), expected, "pippenger {c}");
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
