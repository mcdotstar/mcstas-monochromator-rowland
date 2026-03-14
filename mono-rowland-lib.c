/*******************************************************************************
 * mono-rowland-lib.c
 *
 * Pure-C geometry and math utilities for the Monochromator_Rowland McStas
 * component.  See mono-rowland-lib.h for the public interface.
 ******************************************************************************/
#ifndef MONO_ROWLAND_LIB_H
// header include guarded for source inclusion via McCode %include
#include "mono-rowland-lib.h"
#endif 

double mono_rowland_gauss(double x, double mean, double rms) {
    double d = (x) - (mean);
    return exp(-d * d / (2.0 * rms * rms)) / (sqrt(2.0 * M_PI) * rms);
}

double mono_rowland_circle_xz(double x0, double z0, double x1, double z1,
                               double *c) {
    /* Find the circumscribed circle of the triangle formed by (x0, z0),
       the origin (0, 0), and (x1, z1).

       A point (cx, cz) is equidistant from (0,0) and (x0, z0) iff it lies
       on the perpendicular bisector:
           x0*cx + z0*cz = (x0^2 + z0^2)/2
       Likewise for (x1, z1):
           x1*cx + z1*cz = (x1^2 + z1^2)/2
       Solve the resulting 2x2 linear system. */

    double det = x0 * z1 - z0 * x1;
    if (det == 0.0)
        return -1.0; /* collinear */

    double r0_half = 0.5 * (x0 * x0 + z0 * z0);
    double r1_half = 0.5 * (x1 * x1 + z1 * z1);

    c[0] = (r0_half * z1 - r1_half * z0) / det;
    c[1] = (x0 * r1_half - x1 * r0_half) / det;

    return sqrt((c[0] - x0) * (c[0] - x0) + (c[1] - z0) * (c[1] - z0));
}

void mono_rowland_coverage_limit_point(int which, double x, double z,
                                       double rangle, double *center,
                                       double radius, double *point) {
    (void)radius; /* radius is implicit in the circle; unused here */
    double vl = sqrt(x * x + z * z);
    double v[2]  = {x / vl, z / vl};
    double xi[2] = {center[0] - x, center[1] - z};
    double n[2]  = {v[1], v[0]};
    /* Two choices for the perpendicular direction; select via *which*. */
    n[which > 0] *= -1.0;
    /* Direction toward the coverage-limit point on the circle. */
    double s[2];
    s[0] = n[0] * sin(rangle) - v[0] * cos(rangle);
    s[1] = n[1] * sin(rangle) - v[1] * cos(rangle);
    /* Both (x,z) and *point* lie on the circle, so the formula simplifies. */
    double proj = s[0] * xi[0] + s[1] * xi[1];
    point[0] = x + s[0] * 2.0 * proj;
    point[1] = z + s[1] * 2.0 * proj;
}

double mono_rowland_exact_focus_angle(double ax, double az,
                                      double bx, double bz,
                                      double *point) {
    /* Normal to the bisector of source->slab and sink->slab vectors, evaluated
       with and without the slab point.  The angle between the two normals is
       the required crystal tilt. */
    double n0[2] = {ax + bx,                  az + bz};
    double np[2] = {ax + bx - 2.0 * point[0], az + bz - 2.0 * point[1]};

    double l0 = sqrt(n0[0] * n0[0] + n0[1] * n0[1]);
    double lp = sqrt(np[0] * np[0] + np[1] * np[1]);
    n0[0] /= l0;  n0[1] /= l0;
    np[0] /= lp;  np[1] /= lp;

    double cphi = n0[0] * np[0] + n0[1] * np[1];
    /* Clamp to [-1, 1] to guard against floating-point overshoot. */
    if (cphi >  1.0) cphi =  1.0;
    if (cphi < -1.0) cphi = -1.0;
    double phi = acos(cphi);

    /* Determine sign: positive if the source-sink cross-product and the
       normal cross-product point in the same direction. */
    double cab = -(ax * bz - az * bx);
    double cnn = -(n0[0] * np[1] - n0[1] * np[0]);
    return (cab * cnn < 0.0) ? -phi : phi;
}
