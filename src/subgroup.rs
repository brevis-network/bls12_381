//! Subgroup checks on the zkVM's G1 and G2 precompiles: the point stays in the canonical limbs the
//! precompiles take, and the double-and-add loop runs in place.

#[cfg(any(target_os = "zkvm", test))]
use crate::g1::{G1Affine, BETA};
#[cfg(any(target_os = "zkvm", test))]
use crate::g1_mul::{G1Ops, RawG1};

#[cfg(target_os = "zkvm")]
use crate::g2::G2Affine;
#[cfg(target_os = "zkvm")]
use crate::g2_mul::{fp_to_canonical, neg_canonical};
#[cfg(target_os = "zkvm")]
use pico_patch_libs::{syscall_bls12381_add, syscall_bls12381_double};

#[cfg(target_os = "zkvm")]
extern "C" {
    fn syscall_bls12381_g2_add(p: *mut [u64; 24], q: *const [u64; 24]);
    fn syscall_bls12381_g2_double(p: *mut [u64; 24]);
}

/// Whether `endomorphism(p) = -[x^2] p`, as [`G1Affine::is_torsion_free`] checks it: with `x^2 =
/// |x|^2`, that is `[|x|]([|x|] p) = (beta * p.x, -p.y)`. Canonical limbs compare as field
/// elements: the precompiles' outputs are range-checked below `p`.
#[cfg(any(target_os = "zkvm", test))]
pub(crate) fn is_torsion_free_raw(p: &G1Affine) -> bool {
    if bool::from(p.infinity) {
        return true;
    }
    let target = RawG1::from_affine(&G1Affine {
        x: p.x * BETA,
        y: p.y,
        infinity: p.infinity,
    })
    .negated();
    #[cfg(target_os = "zkvm")]
    {
        // an off-curve point would reach precompile calls the prover cannot prove
        if !bool::from(p.is_on_curve()) {
            return false;
        }
        let base = *RawG1::from_affine(p).limbs();
        times_abs_x(&base)
            .and_then(|q| times_abs_x(&q))
            .map_or(false, |q| q == *target.limbs())
    }
    #[cfg(not(target_os = "zkvm"))]
    {
        let p = RawG1::from_affine(p);
        mul_by_abs_x(&mul_by_abs_x(&p)) == target
    }
}

/// `[|x|] p` over [`G1Ops`], for the host-side test of the check.
#[cfg(all(test, not(target_os = "zkvm")))]
fn mul_by_abs_x<A: G1Ops>(p: &A) -> A {
    let mut acc = *p;
    for bit in (0..63 - crate::BLS_X.leading_zeros()).rev() {
        acc = acc.twice();
        if (crate::BLS_X >> bit) & 1 == 1 {
            acc = acc.plus(p);
        }
    }
    acc
}

/// `[|x|] p` on the precompiles in place, `None` for the identity. `p` is a curve point other than
/// the identity; doubling a point with `y = 0` gives the identity (unreachable on the curve).
#[cfg(target_os = "zkvm")]
fn times_abs_x(p: &[u64; 12]) -> Option<[u64; 12]> {
    let mut acc = *p;
    let mut identity = false;
    for bit in (0..63 - crate::BLS_X.leading_zeros()).rev() {
        if !identity {
            if acc[6..].iter().all(|&l| l == 0) {
                identity = true;
            } else {
                unsafe { syscall_bls12381_double(&mut acc) };
            }
        }
        if (crate::BLS_X >> bit) & 1 == 1 {
            if identity {
                acc = *p;
                identity = false;
                continue;
            }
            // the add precompile refuses `acc = ±p`; a point of small order can get there
            if acc[..6] != p[..6] {
                unsafe { syscall_bls12381_add(&mut acc, p) };
            } else if acc[6..] == p[6..] {
                unsafe { syscall_bls12381_double(&mut acc) };
            } else {
                identity = true;
            }
        }
    }
    (!identity).then_some(acc)
}

/// Whether `[x] p = psi(p)`, as [`G2Affine::is_torsion_free`] checks it: `[|x|] p = -psi(p)`.
#[cfg(target_os = "zkvm")]
pub(crate) fn g2_is_torsion_free_raw(p: &G2Affine) -> bool {
    if bool::from(p.infinity) {
        return true;
    }
    // an off-curve point would reach precompile calls the prover cannot prove
    if !bool::from(p.is_on_curve()) {
        return false;
    }
    let limbs = |q: &G2Affine| {
        let mut out = [0u64; 24];
        for (i, f) in [q.x.c0, q.x.c1, q.y.c0, q.y.c1].iter().enumerate() {
            out[6 * i..6 * i + 6].copy_from_slice(&fp_to_canonical(f));
        }
        out
    };
    let mut target = limbs(&crate::g2_mul::psi_affine(p));
    for c in 2..4 {
        let y = &mut target[6 * c..6 * c + 6];
        if y.iter().any(|&l| l != 0) {
            let n = neg_canonical(y);
            y.copy_from_slice(&n);
        }
    }
    g2_times_abs_x(&limbs(p)).map_or(false, |q| q == target)
}

/// `[|x|] p` on the G2 precompiles in place, `None` for the identity. `p` is a curve point other
/// than the identity; doubling a point with `y = 0` gives the identity (unreachable on the curve).
#[cfg(target_os = "zkvm")]
fn g2_times_abs_x(p: &[u64; 24]) -> Option<[u64; 24]> {
    let mut acc = *p;
    let mut identity = false;
    for bit in (0..63 - crate::BLS_X.leading_zeros()).rev() {
        if !identity {
            if acc[12..].iter().all(|&l| l == 0) {
                identity = true;
            } else {
                unsafe { syscall_bls12381_g2_double(&mut acc) };
            }
        }
        if (crate::BLS_X >> bit) & 1 == 1 {
            if identity {
                acc = *p;
                identity = false;
                continue;
            }
            // the add precompile refuses `acc = ±p`; a point of small order can get there
            if acc[..12] != p[..12] {
                unsafe { syscall_bls12381_g2_add(&mut acc, p) };
            } else if acc[12..] == p[12..] {
                unsafe { syscall_bls12381_g2_double(&mut acc) };
            } else {
                identity = true;
            }
        }
    }
    (!identity).then_some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::Fp;
    use crate::g1::G1Projective;
    use group::Group;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use subtle::Choice;

    fn rng() -> XorShiftRng {
        XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ])
    }

    /// A point on the curve `y^2 = x^3 + 4`, almost surely outside the subgroup.
    fn curve_point(rng: &mut XorShiftRng) -> G1Affine {
        loop {
            let x = Fp::random(&mut *rng);
            let y2 = x.square() * x + crate::g1::B;
            if let Some(y) = Option::<Fp>::from(y2.sqrt()) {
                let p = G1Affine {
                    x,
                    y,
                    infinity: Choice::from(0),
                };
                assert!(bool::from(p.is_on_curve()));
                return p;
            }
        }
    }

    #[test]
    fn raw_check_agrees_with_the_projective_one() {
        let mut rng = rng();
        let mut points = vec![G1Affine::identity(), G1Affine::generator()];
        for _ in 0..24 {
            points.push(G1Affine::from(G1Projective::random(&mut rng)));
            let off = curve_point(&mut rng);
            points.push(off);
            points.push(-off);
            // a subgroup point and the same point plus a cofactor-torsion component
            points.push(G1Affine::from(G1Projective::from(off).clear_cofactor()));
        }
        let (mut inside, mut outside) = (0, 0);
        for p in &points {
            let expected = bool::from(p.is_torsion_free());
            assert_eq!(is_torsion_free_raw(p), expected, "{p:?}");
            if expected {
                inside += 1
            } else {
                outside += 1
            }
        }
        assert!(inside > 20 && outside > 20, "{inside} in, {outside} out");
    }

    #[test]
    fn mul_by_abs_x_matches_scalar_multiplication() {
        let mut rng = rng();
        for _ in 0..8 {
            let a = G1Affine::from(G1Projective::random(&mut rng));
            let by_x = G1Projective::from(a) * crate::Scalar::from(crate::BLS_X);
            assert_eq!(
                mul_by_abs_x(&RawG1::from_affine(&a)),
                RawG1::from_affine(&G1Affine::from(by_x))
            );
        }
    }
}
