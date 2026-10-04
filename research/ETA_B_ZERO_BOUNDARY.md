# The `eta_B=0` response boundary

## Status

The complete-response theorem uses both the router-logit block
`eta_Z>0` and the shared-basis block `eta_B>0`.  This note audits whether the
second rate is logically necessary.  It does **not** change the theorem in the
paper: the paper states a sufficient assumption set and already says that the
general `eta_B=0` stabilizer is unresolved.

Four proper subfamilies are now closed.

1. For any number of bases, an entrywise nonnegative gauge is already forced
   to be a permutation by one strictly positive router row.
2. For two bases, the same conclusion holds without a sign restriction on the
   gauge, provided the router has full column rank.
3. For any number of bases, the signed family whose induced regular-simplex
   isometry has a coordinate-permutation orthogonal part cannot support a
   full-rank router unless the gauge is itself a permutation.
4. For three bases, the signed family whose induced isometry has zero affine
   translation also reduces to permutations.

For `K>=3`, the only unresolved case is a signed gauge `M` with at least one
negative entry even though both observed routers `A` and `A M` are strictly
positive.  The searches recorded below did not find such a response
stabilizer, but numerical failure is not a proof.  This route is therefore
`blocked`, not `complete`.

## Algebraic reduction

Let `A,B` and `A',B'` be full-rank factorizations of the same effective
parameter matrix, with strictly positive row-stochastic routers.  Standard
factorization algebra gives a unique invertible gauge

```text
A' = A M,       B' = M^{-1} B,       M 1 = 1.                 (E1)
```

Write a router row as a column `p`, put `q=M^T p`, and define

```text
J(p) = diag(p) - p p^T.
```

When `eta_B=0`, equality of the local response block and full row rank of `B`
reduce exactly to

```text
J(q)^2 = M^T J(p)^2 M                                      (E2)
```

for every router row.  This is the only equation studied below.  It is not a
claim about Adam, natural gradient, a finite optimizer step, or a historical
sharing graph.

### Exact diagonal-scaling compatibility formulation

Fix one positive solution row `p,q` and define

```text
T = D_p M D_q^{-1},       D_p=diag(p), D_q=diag(q).          (C1)
```

Then

```text
1^T T=1^T,       Tq=p,       J(p)M=TJ(q).                   (C2)
```

Consequently (E2) is equivalent to

```text
J(q) T^T T J(q)=J(q)^2.                                    (C3)
```

The range of `J(q)` is `1^perp`.  Since `1^T T=1^T`, `T` maps that
subspace to itself, and (C3), including polarization, says exactly that
`T|_(1^perp)` is orthogonal.  Equivalently,

```text
T=O+t 1^T,       O^T O=I,       O1=1,       1^Tt=0.         (C4)
```

This is an exact equivalence, not a numerical ansatz.

For a second positive router row, form its componentwise ratio to the fixed
row and put

```text
r=D_p^{-1}p',       h=T^T r,
T_r=D_r T D_h^{-1}.                                         (C5)
```

The transformed row is exactly `q'=D_q h`.  Thus it satisfies (E2) if and
only if `h>0` and `T_r|_(1^perp)` is orthogonal.  Multiplying `r` by a
positive scalar changes neither `T_r` nor positivity.  If `r_i` are the
ratios of all router rows to the fixed row, then

```text
rank(A)=rank([r_1^T; ...; r_L^T]).                           (C6)
```

Indeed the router on the left is the ratio matrix on the right times the
invertible diagonal `D_p`, up to positive row normalizations.  Equations
(C4)--(C6) turn the remaining problem into compatibility of diagonal scalings
of one affine regular-simplex isometry.

## Theorem 1: a nonnegative gauge is a permutation

**Claim.**  Let `p,q` be strictly positive probability vectors, let
`M 1=1`, let `q=M^T p`, and suppose (E2) holds.  If `M` is entrywise
nonnegative, then `M` is a permutation matrix.  Thus one positive row is
enough; router full rank is not needed for this subcase.

**Proof.**  Set `D_p=diag(p)`, `D_q=diag(q)`, and

```text
K = D_p M D_q^{-1}.                                         (E3)
```

The identities `p^T M=q^T` and `M 1=1` give

```text
1^T K=1^T,       K q=p.                                     (E4)
```

Moreover,

```text
J(p)M = (K-p 1^T)D_q,
J(q)  = (I-q 1^T)D_q.
```

Since `D_q` is invertible, (E2) is equivalent to the exact column-Gram
identity

```text
(K-p 1^T)^T(K-p 1^T)
  = (I-q 1^T)^T(I-q 1^T).                                  (E5)
```

Subtracting two columns eliminates the centers.  Hence every two distinct
columns of `K` are at Euclidean distance `sqrt(2)`, exactly as two distinct
standard basis vectors.

By nonnegativity of `M` and positivity of the diagonal scalings, `K>=0`.
Together with `1^T K=1^T`, every column of `K` lies in the probability
simplex.  Its Euclidean diameter is `sqrt(2)`, and equality is attained only
by two distinct vertices: for simplex points `u,v`,
`||u-v||_2^2<=||u||_2^2+||v||_2^2<=2`, with equality forcing each point to be
a different standard basis vector.  There are `K` columns, all pairwise at
that distance, so `K=P` for a permutation matrix `P`.

Finally (E4) gives `Pq=p`.  From (E3),
`M=D_p^{-1} P D_q`; at the unique nonzero entry in row `r`, the ratio is
`q_j/p_r=1`.  Therefore `M=P`.  QED.

The sign assumption is material to this proof and is not implied by valid
router charts.  Strictly positive `A` and `AM` can coexist with negative
entries in `M`; the rejected cyclic construction below is an explicit
example.  Its failure is response inequivalence, not chart infeasibility.

## Theorem 2: two bases need no sign assumption

**Claim.**  Let `K=2`, let `A` and `AM` be strictly positive row-stochastic
routers, and suppose `rank(A)=2`.  If (E2) holds for every row, then `M` is
either the identity or the two-column swap.

**Proof.**  Every invertible row-stochastic two-by-two matrix has the form

```text
M = [[x,1-x], [y,1-y]],       d=x-y != 0.                   (E6)
```

Composing the candidate chart with the two-column swap changes the sign of
`d` and preserves the question, so assume `d>0`.  Write a router row as
`(p,1-p)` and its transformed first coordinate as

```text
q = y+d p.                                                   (E7)
```

For `L=[[1,-1],[-1,1]]`,

```text
J(p)=p(1-p)L,       J(p)^2=2[p(1-p)]^2 L,
M^T L M=d^2 L.
```

Strict positivity therefore turns (E2) into

```text
q(1-q)=d p(1-p).                                             (E8)
```

Full column rank of `A` supplies two distinct rows
`0<p_1<p_2<1`.  Put `q_j=y+d p_j`,
`S=p_1+p_2`, and `Delta=p_2-p_1>0`.  Subtracting the two copies of (E8) and
dividing by `d Delta` yields

```text
q_1+q_2=S.                                                   (E9)
```

Also `q_2-q_1=d Delta`.  Summing the two copies of (E8), and using the sum and
difference formula for two squares, gives

```text
0 = (1-d)/2 * [S(2-S)+d Delta^2].                            (E10)
```

The bracket is strictly positive because `0<S<2`, `d>0`, and `Delta>0`.
Thus `d=1`.  Equations (E9) and `q_2-q_1=p_2-p_1` give `q_j=p_j`, whence
`y=0` and `M=I`.  Undoing the optional column swap leaves exactly the two
permutations.  QED.

Full router rank is used precisely to obtain two distinct values of `p`.
With one repeated row, (E8) is only one scalar equation and admits signed
non-permutation solutions, so the proof cannot survive removal of that
assumption.

## Theorem 3: coordinate-permutation affine direction, any `K`

**Claim.**  In (C4)--(C6), suppose

```text
T=P+t 1^T
```

for a coordinate permutation `P`.  If `t!=0`, every positive compatible
ratio `r` is constant.  Hence probability-valued router rows are identical
and cannot have full column rank.  If `t=0`, the original gauge `M` is the
same coordinate permutation.

**Proof.**  First take `P=I`.  Put `c=t^T r`.  Then

```text
h=T^T r=r+c 1.                                               (C7)
```

Both `T` and a compatible `T_r` preserve column sums and are orthogonal on
`1^perp`, so their determinants have absolute value one.  Here `det(T)=1`;
because the diagonal determinant factors in (C5) are positive,

```text
1=det(T_r)=prod_j r_j / prod_j h_j.                          (C8)
```

If `c>0`, every positive entry of `h` is strictly larger than the
corresponding entry of `r`, contradicting (C8).  If `c<0`, strict positivity
of `h` makes every entry strictly smaller, with the same contradiction.
Thus `c=0`, `h=r`, and

```text
T_r=I+(D_r t)(1/r)^T.                                       (C9)
```

On `1^perp`, `T_r-I` has rank at most one.  The restriction is orthogonal
and has determinant `+1` by (C8).  A nonidentity real orthogonal matrix with
rank-one displacement from the identity is a reflection and has determinant
`-1`; hence the restriction in (C9) is the identity.  As `t!=0`, this forces
`(1/r)^T` to annihilate `1^perp`, so `1/r` and therefore `r` are constant.

For general `P`, set `S=P^T T=I+(P^Tt)1^T` and
`tilde r=P^T r`.  Left multiplication by `P^T` preserves the tangent-space
isometry condition, and direct substitution gives

```text
P^T T_r=D_tilde_r S D_(S^T tilde_r)^{-1}.
```

The preceding argument applies.  Finally, if `t=0`, then `T=P`, `Pq=p`, and
`M=D_p^{-1}PD_q=P`.  QED.

Strict positivity is used in the product monotonicity step.  Full rank is
used only to rule out the one-dimensional family of constant ratios.  The
result is not a general affine-isometry theorem: its orthogonal part is
assumed to be a coordinate permutation.

## Theorem 4: zero affine translation for `K=3`

**Claim.**  Let `K=3` and suppose the fixed-row map in (C4) has `t=0`, so
`T=O` is orthogonal and fixes `1`.  If a nonconstant positive ratio `r` with
`h=O^T r>0` is compatible, then `O` is a coordinate permutation.  Therefore
a full-rank router in this subfamily forces the original gauge `M` to be a
permutation.

**Proof.**  Compatibility and (C5) give

```text
prod(r)=prod(h).                                             (C10)
```

Orthogonality and `O1=1` also give equal sums and equal squared norms for
`r` and `h`.  In three dimensions these are equality of the elementary
symmetric polynomials `e_1,e_2,e_3`, so `h=P^T r` for some coordinate
permutation `P`.

Put `S=OP^T`.  Then `S` is orthogonal, fixes both `1` and `r`, and

```text
R=T_r P^T=D_r S D_r^{-1}                                   (C11)
```

is orthogonal on `1^perp`.  If `S=I`, then `O=P`.  Otherwise, because `r`
is nonconstant, `span{1,r}` is a plane and `S` is reflection across it.  With
`r=(x,y,z)>0`, a normal is

```text
n=(y-z,z-x,x-y),       S=I-2 n n^T/(n^T n).                 (C12)
```

The `-1` tangent eigenvector of `R` is `a=D_r n`, while its fixed tangent
line has normal `Pi D_r^{-1}n`, where `Pi=I-11^T/3`.  Orthogonality of `R`
therefore requires these two vectors to be parallel.  Direct expansion gives
all three components of their cross product as

```text
-2 (x-y)(x-z)(y-z)(xy+xz+yz)/(3xyz).                        (C13)
```

Because `x,y,z>0`, (C13) can vanish only when two coordinates of `r` are
equal.  In that case (C12) is exactly the transposition of those two
coordinates.  Thus `S` is a coordinate permutation, and so is `O=SP`.

A full-rank router supplies a nonconstant ratio by (C6).  Once `O` is a
permutation, Theorem 3 with `t=0` gives `M=O`.  QED.

The use of three dimensions is material: equal `e_1,e_2,e_3` determines an
unordered triple, and the complement of `span{1,r}` is one-dimensional.
Neither step directly proves the result for `K>3`.

## Why the tempting cyclic construction fails

An order-three gauge can be generated from a positive invertible stochastic
matrix `A` and the row-cycle `R` by

```text
M=A^{-1} R A,       AM=RA,       M^3=I.                     (E11)
```

It is possible to tune `A` so that (E2) holds to numerical precision for its
first row while all router entries remain well inside the simplex.  One such
rounded diagnostic is

```text
A = [[1/6,       1/5,        19/30],
     [0.63314916,0.20079942, 0.16605142],
     [0.12947217,0.74011599, 0.13041184]].
```

It has `sigma_min(A)=0.46688265` and minimum entry `0.12947217`.  The induced
gauge is approximately

```text
M = [[-0.07770671, 1.14850805,-0.07080134],
     [ 0.06250496, 0.07063394, 0.86686110],
     [ 1.00041993,-0.00749270, 0.00707277]].
```

Although `M^3=I` and `AM=RA`, the three Frobenius residuals in (E2) are

```text
[4.39e-9, 6.7977e-2, 5.6124e-2].                            (E12)
```

Thus order three does **not** propagate a congruence involving `M^T` from one
orbit point to the next.  This construction is a useful regression test, not
a counterexample.

## Exhausted mechanisms and the exact blocked gap

- **Two-basis affine gauges:** closed analytically by Theorem 2; independent
  large random scans found no exception.
- **Nonnegative gauges for any `K`:** closed analytically by Theorem 1.  A
  positive `A` and positive `AM` do not make `M` nonnegative, so this does not
  settle the model's full feasible gauge group.
- **Simple signed involutions:** the examined fixed-row equations factor into
  branches that either lie on the simplex boundary or lose strict positivity.
- **Coordinate-permutation affine directions:** Theorem 3 closes this family
  for every `K`, including the coordinate-reflection involutions.  The product
  invariant forces every compatible diagonal scaling onto one ratio ray.
- **Zero-translation three-basis maps:** Theorem 4 closes every orthogonal
  direction, not only coordinate reflections, by exact elimination (C13).
- **Order-three/circulant gauges:** the one-row construction above fails the
  other two blocks.  `research/search_eta_b_circulant.py` explicitly evaluates
  all rows rather than relying on the invalid orbit shortcut.
- **General signed gauges:** `research/search_eta_b_zero.py` pins
  non-permutation gauge coordinates; `research/search_eta_b_two_positive_charts.py`
  parameterizes both routers as interior and permits a signed `M=A^{-1}A'`.
  Low-residual solutions observed in these searches collapse router rank or
  approach a boundary/permutation.  This is negative numerical evidence only.

The exact diagonal-scaling formulation makes the unresolved lemma more
specific:

> Let `K>=3`, `T=O+t1^T` as in (C4), and suppose there are `K` positive,
> linearly independent ratio vectors `r_i`, including `r_1=1`, such that
> `h_i=T^T r_i>0` and every
> `D_(r_i) T D_(h_i)^{-1}` is orthogonal on `1^perp`.  Must `t=0` and `O`
> be a coordinate permutation?

Theorems 3 and 4 prove this only when the orthogonal part is already a
coordinate permutation (arbitrary `K,t`) or when `K=3,t=0` (arbitrary `O`).
The remaining intersection--a non-coordinate orthogonal part together with a
nonzero translation--survived square-system root searches in both rotation
and reflection components, including fixed nonzero translations, but that is
negative numerical evidence only.  No proof or valid counterexample is
known.  Route R14 is therefore **BLOCKED**.  Reopen it only with a new exact
compatibility invariant or an exact full-row positive construction; another
least-squares/root run or a one-row witness is insufficient.

For reproducibility, the final square-system reconnaissance used the exact
residual equations but floating-point root finding.  It returned no accepted
interior, full-rank candidate in the following batches:

| parameterization | component / constraint | starts | seed | accepted |
|---|---:|---:|---:|---:|
| original `A,M` square system | fixed signed first gauge row `(-0.2,0.6)` | 1,000 | 20260823 | 0 |
| diagonal scaling (C5) | rotations, free translation | 2,000 | 20260823 | 0 |
| diagonal scaling (C5) | reflections, free translation | 2,000 | 20260823 | 0 |
| diagonal scaling (C5) | rotation, `t_1=0.2` | 2,000 | 20260823 | 0 |
| diagonal scaling (C5) | rotation, `t_1=-0.5` | 2,000 | 20260824 | 0 |
| diagonal scaling (C5) | reflection, `t_1=0.2` | 2,000 | 20260825 | 0 |
| diagonal scaling (C5) | reflection, `t_1=-0.5` | 2,000 | 20260826 | 0 |

The diagonal-scaling filter first solves a linear program for a positive base
chart `q>0,Tq>0`; it does not assume the affine image of the simplex centroid
is positive.  These counts document search coverage only and do not increase
the theorem's scope.

## Deterministic verification

The regression checks do not prove the theorems; they verify their displayed
algebra and prevent the rejected cyclic witness from being silently promoted:

```bash
pytest -q tests/test_eta_b_zero_boundary.py
```

The exploratory searches are runnable diagnostics, not certificates:

```bash
python research/search_eta_b_circulant.py
python research/search_eta_b_zero.py --seed 20260823 --targets 40 --starts 20
python research/search_eta_b_zero_involution.py --seed 20260823 --starts 500
python research/search_eta_b_two_positive_charts.py 20260823 500
python research/eta_b_zero_algebraic_attack.py --starts 2000
python research/eta_b_zero_diagonal_scaling_attack.py --starts 2000
```

No novelty claim is attached to these boundary lemmas without a separate
prior-art audit.  They refine an assumption ledger; they do not replace the
paper's jointly trained complete-response theorem.
