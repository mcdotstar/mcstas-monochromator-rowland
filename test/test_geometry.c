/*******************************************************************************
 * test_geometry.c  –  Unity unit tests for mono-rowland-lib
 *
 * Tests cover:
 *   mono_rowland_gauss              – peak value, symmetry
 *   mono_rowland_circle_xz          – right-angle, collinear, abs-bug
 *                                     regression, equilateral triangle
 *   mono_rowland_coverage_limit_pt  – zero-angle identity, mirror symmetry
 *   mono_rowland_exact_focus_angle  – symmetric geometry, no OOB access
 ******************************************************************************/
#include "unity.h"
#include "mono-rowland-lib.h"

#include <math.h>

#define TOL 1e-9   /* tight tolerance for exact algebra */
#define RTOL 1e-6  /* relaxed tolerance for trig / sqrt */

void setUp(void)    {}
void tearDown(void) {}

/* ─── mono_rowland_gauss ─────────────────────────────────────────────────── */

void test_gauss_peak_value(void) {
    /* At the mean the pdf equals 1/(√(2π)·σ). */
    double sigma = 0.05;
    double expected = 1.0 / (sqrt(2.0 * M_PI) * sigma);
    TEST_ASSERT_DOUBLE_WITHIN(expected * RTOL, expected,
                              mono_rowland_gauss(0.3, 0.3, sigma));
}

void test_gauss_symmetry(void) {
    double mean = 1.0, sigma = 0.2;
    double left  = mono_rowland_gauss(mean - 0.1, mean, sigma);
    double right = mono_rowland_gauss(mean + 0.1, mean, sigma);
    TEST_ASSERT_DOUBLE_WITHIN(left * RTOL, left, right);
}

/* ─── mono_rowland_circle_xz ─────────────────────────────────────────────── */

/* Helper: verify that the computed circle actually passes through all three
   points (origin, (x0,z0), (x1,z1)) within tolerance. */
static void assert_circle_passes_through(double x0, double z0,
                                         double x1, double z1,
                                         double *c, double r) {
    double d_origin = sqrt(c[0]*c[0] + c[1]*c[1]);
    double d0 = sqrt((c[0]-x0)*(c[0]-x0) + (c[1]-z0)*(c[1]-z0));
    double d1 = sqrt((c[0]-x1)*(c[0]-x1) + (c[1]-z1)*(c[1]-z1));
    TEST_ASSERT_DOUBLE_WITHIN(r * RTOL, r, d_origin);
    TEST_ASSERT_DOUBLE_WITHIN(r * RTOL, r, d0);
    TEST_ASSERT_DOUBLE_WITHIN(r * RTOL, r, d1);
}

void test_circle_right_angle(void) {
    /* Circle through (1,0), origin, (0,1):
       center = (0.5, 0.5), radius = √2/2. */
    double c[2];
    double r = mono_rowland_circle_xz(1.0, 0.0, 0.0, 1.0, c);
    double expected_r = sqrt(2.0) / 2.0;
    TEST_ASSERT_DOUBLE_WITHIN(expected_r * RTOL, expected_r, r);
    TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.5, c[0]);
    TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.5, c[1]);
    assert_circle_passes_through(1.0, 0.0, 0.0, 1.0, c, r);
}

void test_circle_collinear_x_axis(void) {
    /* All three points on x-axis → collinear → -1. */
    double c[2] = {0, 0};
    double r = mono_rowland_circle_xz(1.0, 0.0, 2.0, 0.0, c);
    TEST_ASSERT_EQUAL_DOUBLE(-1.0, r);
}

void test_circle_collinear_z_axis(void) {
    /* All three points on z-axis → collinear → -1. */
    double c[2] = {0, 0};
    double r = mono_rowland_circle_xz(0.0, 1.0, 0.0, 3.0, c);
    TEST_ASSERT_EQUAL_DOUBLE(-1.0, r);
}

void test_circle_abs_bug_regression(void) {
    /* This case has fractional slopes m0=0.7/0.9≈0.78 and m1=0.3/0.9≈0.33,
       both < 1.  The old code used integer abs(), which truncated both to 0
       and therefore always picked the else-branch regardless of which slope
       is larger.  With fabs() the correct branch is chosen and the circle
       passes through all three points. */
    double x0 =  0.9, z0 = -0.174;   /* source in mono frame (80° geometry) */
    double x1 =  0.866, z1 = -0.500; /* sink in mono frame */
    double c[2];
    double r = mono_rowland_circle_xz(x0, z0, x1, z1, c);
    TEST_ASSERT_TRUE(r > 0.0);
    assert_circle_passes_through(x0, z0, x1, z1, c, r);
}

void test_circle_equilateral_triangle(void) {
    /* Equilateral triangle with side s = 1.
       Vertices: (0,0), (1,0), (0.5, √3/2).
       Circumradius = s/√3 = 1/√3. */
    double s = 1.0;
    double x1 = s, z1 = 0.0;
    double x2 = s / 2.0, z2 = sqrt(3.0) / 2.0 * s;
    double c[2];
    double r = mono_rowland_circle_xz(x1, z1, x2, z2, c);
    double expected_r = s / sqrt(3.0);
    TEST_ASSERT_DOUBLE_WITHIN(expected_r * RTOL, expected_r, r);
    assert_circle_passes_through(x1, z1, x2, z2, c, r);
}

void test_circle_symmetric_geometry(void) {
    /* Symmetric monochromator: source and sink both 1 m from the crystal,
       scattering angle 2θ = 80° (θ = 40°).
       The circumradius of a triangle with two equal sides L and included
       angle α = 180°-2θ = 100° is R = L / (2·sin(α)).
       But here we work in the mono frame (origin = crystal center),
       so we just verify the circle passes through all three points. */
    double theta = 40.0 * M_PI / 180.0;
    double L = 1.0;
    /* Source direction in mono frame (before ScatAngle rotation). */
    double xs = sin(theta) * L, zs = -cos(theta) * L;
    /* Sink direction (mirror of source for symmetric geometry). */
    double xk = -sin(theta) * L, zk = -cos(theta) * L;
    double c[2];
    double r = mono_rowland_circle_xz(xs, zs, xk, zk, c);
    TEST_ASSERT_TRUE(r > 0.0);
    assert_circle_passes_through(xs, zs, xk, zk, c, r);
}

/* ─── mono_rowland_coverage_limit_point ─────────────────────────────────── */

void test_coverage_limit_zero_angle_returns_antipodal(void) {
    /* At rangle=0 the function follows the source-to-center direction until
       it hits the circle again, yielding the antipodal point (0, +1) for a
       source at (0, -1) with the center at the origin. */
    double center[2] = {0.0, 0.0};
    double x = 0.0, z = -1.0;
    double point[2];
    mono_rowland_coverage_limit_point(0, x, z, 0.0, center, 1.0, point);
    TEST_ASSERT_DOUBLE_WITHIN(TOL,  0.0, point[0]);
    TEST_ASSERT_DOUBLE_WITHIN(TOL, +1.0, point[1]);  /* antipodal, not source */
}

void test_coverage_limit_mirror_symmetry(void) {
    /* Calling with which=0 and which=1 from the same source point should
       return two points that are mirror-images of each other about the line
       through the origin and the source. */
    double center[2] = {0.5, -0.5};
    double radius = sqrt(0.5 * 0.5 + 0.5 * 0.5); /* distance from center to (0,0) */
    double x = 0.707, z = -0.707;                 /* 45° from -z */
    double angle = 20.0 * M_PI / 180.0;
    double p0[2], p1[2];
    mono_rowland_coverage_limit_point(0, x, z, angle, center, radius, p0);
    mono_rowland_coverage_limit_point(1, x, z, angle, center, radius, p1);

    /* The two points must be equidistant from the source. */
    double d0 = sqrt((p0[0]-x)*(p0[0]-x) + (p0[1]-z)*(p0[1]-z));
    double d1 = sqrt((p1[0]-x)*(p1[0]-x) + (p1[1]-z)*(p1[1]-z));
    TEST_ASSERT_DOUBLE_WITHIN(d0 * RTOL, d0, d1);

    /* The two points must be on opposite sides of the source-direction axis. */
    /* Cross-product of (x,z) with (pi-x, pi-z) should have opposite signs. */
    double cross0 = x * (p0[1] - z) - z * (p0[0] - x);
    double cross1 = x * (p1[1] - z) - z * (p1[0] - x);
    TEST_ASSERT_TRUE(cross0 * cross1 <= 0.0);
}

/* ─── mono_rowland_exact_focus_angle ────────────────────────────────────── */

void test_exact_focus_angle_symmetric_geometry(void) {
    /* When the slab is at the origin (0, 0) the vectors n0 and np are
       identical, so the required tilt is exactly 0 regardless of source/sink
       directions. */
    double ax = 0.866, az = -0.5;   /* source direction */
    double bx = -0.866, bz = -0.5;  /* sink direction (symmetric) */
    double point[2] = {0.0, 0.0};   /* slab at origin → n0 == np → phi = 0 */
    double angle = mono_rowland_exact_focus_angle(ax, az, bx, bz, point);
    TEST_ASSERT_DOUBLE_WITHIN(RTOL, 0.0, angle);
}

void test_exact_focus_angle_no_oob_access(void) {
    /* The old code accessed point[2] on a 2-element array (UB).
       This test just verifies that the function returns a finite value when
       given a valid 2-element point array, exercising the fixed code path. */
    double point[2] = {0.5, -0.866}; /* only 2 elements — old code would OOB */
    double ax = 0.5, az = -0.866;
    double bx = -0.5, bz = -0.866;
    double angle = mono_rowland_exact_focus_angle(ax, az, bx, bz, point);
    TEST_ASSERT_TRUE(isfinite(angle));
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_gauss_peak_value);
    RUN_TEST(test_gauss_symmetry);

    RUN_TEST(test_circle_right_angle);
    RUN_TEST(test_circle_collinear_x_axis);
    RUN_TEST(test_circle_collinear_z_axis);
    RUN_TEST(test_circle_abs_bug_regression);
    RUN_TEST(test_circle_equilateral_triangle);
    RUN_TEST(test_circle_symmetric_geometry);

    RUN_TEST(test_coverage_limit_zero_angle_returns_antipodal);
    RUN_TEST(test_coverage_limit_mirror_symmetry);

    RUN_TEST(test_exact_focus_angle_symmetric_geometry);
    RUN_TEST(test_exact_focus_angle_no_oob_access);

    return UNITY_END();
}
