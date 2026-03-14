/*******************************************************************************
 * mono-rowland-lib.h
 *
 * Pure-C geometry and math utilities for the Monochromator_Rowland McStas
 * component.  These functions have no dependency on McStas runtime types and
 * can therefore be compiled and unit-tested independently.
 *
 * Included in Monochromator_Rowland.comp via:
 *   SHARE %{ %include "mono-rowland-lib" %}
 *
 * For standalone use (e.g. C unit tests), compile mono-rowland-lib.c and
 * include this header.
 ******************************************************************************/
#ifndef MONO_ROWLAND_LIB_H
#define MONO_ROWLAND_LIB_H

#include <math.h>
#include <stdio.h>

/* Gaussian probability density with mean *mean* and standard deviation *rms*.
   Equivalent to the McStas GAUSS() macro but with an explicit normalisation
   factor so that the integral over all x equals 1. */
double mono_rowland_gauss(double x, double mean, double rms);

/* Find the unique circle passing through (x0, z0), the origin (0, 0), and
   (x1, z1) in the xz-plane.  Stores the center as c[0] (x) and c[1] (z).
   Returns the radius, or -1 when the three points are collinear. */
double mono_rowland_circle_xz(double x0, double z0, double x1, double z1, double *c);

/* Compute one of the two points on the Rowland circle at angular distance
   *rangle* (radians) from the point (x, z) as seen from *center*.
   *which*  0 -> "left" limit point, non-zero -> "right" limit point.
   Result stored in point[0] (x) and point[1] (z). */
void mono_rowland_coverage_limit_point(int which, double x, double z,
                                       double rangle, double *center,
                                       double radius, double *point);

/* Compute the tilt angle (radians) that a crystal slab at position *point*
   (xz-pair) must have so that it focuses neutrons from (ax, az) toward
   (bx, bz) using the exact Rowland-circle reflection condition.
   Returns a signed angle; positive -> tilt toward the source side. */
double mono_rowland_exact_focus_angle(double ax, double az,
                                      double bx, double bz,
                                      double *point);

/* Crystal slab properties used by crystal_interaction (defined in the McStas
   SHARE block of Monochromator_Rowland.comp). */
struct crystal_properties {
    int    flag;  /* bitfield: bit0=verbose, bit1=rTableFlag, bit2=tTableFlag */
    double tau;   /* scattering vector magnitude (AA^-1) */
    int    n;     /* diffraction order (0 = auto) */
    double r;     /* maximum reflectivity r0 */
    double t;     /* transmission efficiency t0 */
    double rms_y; /* horizontal mosaic RMS (radians) */
    double rms_z; /* vertical mosaic RMS (radians) */
    double rms;   /* max(rms_y, rms_z) */
};

#endif /* MONO_ROWLAND_LIB_H */
