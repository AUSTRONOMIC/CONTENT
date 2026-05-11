#!/usr/bin/env python3
"""
eq_mount_sim.py  --  Equatorial Mount Simulator  v4
Austronomic | Engineering the Cosmos

Changes from v3 (all pre-verified analytically):
  1. Removed RA / Dec axis arrowhead markers.
  2. Two disk-scale sliders: RA and Dec gauge disks resize dynamically.
     Axis-through-centre preserved at every scale (xz-scale for RA,
     yz-scale for Dec; fixed coordinate unchanged in each case).
  3. Slider renamed "Hour Angle" (HA = LST - RA; more precise).
  4. OTA tapered: objective ring r=0.165 at z=+OTA_LEN faces target;
     eyepiece ring r=0.075 at z=-OTA_LEN faces observer.
  5. Latitude range extended to -89.5 ... +89.5.  Snap-to ±0.3 deg at
     the equator (lat=0 is a geometric singularity: both poles on horizon).
  6. RangeSlider selects start/end GIF frame (0-360, one full sky rotation).
  7. "Save GIF" button exports the selected frame range to
     austronomic_eqmount.gif using PIL directly (reliable on all backends).
  v4 -> v5 additions:
  8. View Azimuth slider (-180..180 deg) rotates camera around vertical axis.
     View Elevation slider (-10..85 deg) tilts camera up/down.
     Both update ax.view_init() immediately on drag.
  9. GIF generator fully rewritten: stops FuncAnimation timer before capture,
     renders frames via fig.savefig()+BytesIO+PIL (no PillowWriter, no
     flush_events), restarts timer in finally block.

Physics (unchanged from v3, analytically proven exact):
  RA drive counter-rotates at sidereal rate -> exact tracking.
  CS centred at Dec bearing; sight-line residual < 1.1 deg.
"""

import os
import io
import time as _time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.widgets import Slider, Button, RangeSlider
from mpl_toolkits.mplot3d import Axes3D          # noqa

np.random.seed(42)

# ===============================================================================
#  CONSTANTS
# ===============================================================================

CS_R       = 18.0    # celestial-sphere radius, centred at Dec bearing
OMEGA      = 0.45    # accelerated sidereal rate [rad s^-1]
MOUNT_BASE = 0.9     # world-Y of mount base (tripod head junction)
FLOOR_Y    = -0.9    # world-Y of tripod floor / Earth surface
RA_ELEV    = 2.2     # RA axis length in mountPivot frame (Dec bearing height)
DEC_XOFF   = 0.30    # Dec arm offset in raGroup +X
OTA_LEN    = 1.0     # OTA half-length in decGroup Z
ALT_H_AA   = 1.3     # height of alt-az altitude bearing above MOUNT_BASE
LAT_EPS    = 0.30    # equatorial snap-zone [deg]: avoids polar singularity
GIF_FPS    = 20      # frames per second for exported GIF
GIF_FRAMES = 360     # total frames representing one full sky rotation

# Palette -- white background, all elements high-contrast
C_BG     = '#f5f7fc'
C_RA     = '#cc1111'
C_DEC    = '#117722'
C_PS     = '#1133cc'
C_OTA    = '#445566'
C_BODY   = '#3a5060'
C_CW     = '#2a4050'
C_SIGHT  = '#3355aa'
C_PAXL   = '#883333'
C_POLE   = '#cc8800'
C_TARGET = '#cc11cc'
C_GRID   = '#2060a0'
C_EQ     = '#1050d0'
C_STARS  = '#1a2060'
C_GND    = '#2a6a3c'
C_GND2   = '#1a4a2c'
C_TRIPOD = '#3a4a58'
C_RA_D   = '#cc1111'
C_DC_D   = '#117722'

# ===============================================================================
#  MATH UTILITIES
# ===============================================================================

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

def aR(R, pts):
    return (R @ np.atleast_2d(pts).T).T

def _safe_lat(lat_deg):
    """Snap near-equatorial latitudes to avoid R_pivot singularity."""
    if abs(lat_deg) < LAT_EPS:
        return LAT_EPS * (1. if lat_deg >= 0. else -1.)
    return lat_deg

# Polar alignment error (module-level globals; set by slider callbacks before polar_dir is called)
_polar_off_x = 0.  # azimuth error [deg]
_polar_off_y = 0.  # elevation error [deg]

def polar_dir(lat_deg):
    lat_deg = _safe_lat(lat_deg)
    phi  = np.radians(abs(lat_deg))
    zsgn = 1. if lat_deg < 0 else -1.
    nom  = np.array([0., np.sin(phi), zsgn * np.cos(phi)])
    ox, oy = np.radians(_polar_off_x), np.radians(_polar_off_y)
    if abs(ox) > 1e-9 or abs(oy) > 1e-9:
        # Small tilt: ox = east-west azimuth screw, oy = elevation screw
        cz, sz = np.cos(ox), np.sin(ox)
        cx, sx = np.cos(oy), np.sin(oy)
        Rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
        Rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
        nom = Rx @ Rz @ nom
        nom = nom / np.linalg.norm(nom)
    return nom

def R_pivot(lat_deg):
    """Full 3-D rotation: maps mount local +Y -> polar_dir(lat_deg).
    The old Rx-only formula silently dropped the X component of polar_dir,
    which is non-zero when a polar azimuth error is applied.
    This version correctly handles azimuth AND elevation polar offsets."""
    pd   = polar_dir(lat_deg)
    zref = np.array([0.,0.,1.]) if abs(pd[2]) < 0.99 else np.array([0.,1.,0.])
    ex   = np.cross(pd, zref);  ex  /= np.linalg.norm(ex)
    ez   = np.cross(ex,  pd);   ez  /= np.linalg.norm(ez)
    return np.column_stack([ex, pd, ez])   # R @ [0,1,0] = pd (polar axis) ✓

def polar_dir_true(lat_deg):
    """True celestial pole direction — zero alignment error, always."""
    lat_deg = _safe_lat(lat_deg)
    phi  = np.radians(abs(lat_deg))
    zsgn = 1. if lat_deg < 0 else -1.
    return np.array([0., np.sin(phi), zsgn * np.cos(phi)])

def R_pivot_true(lat_deg):
    """Same full 3-D construction as R_pivot but using the TRUE pole direction."""
    pd   = polar_dir_true(lat_deg)
    zref = np.array([0.,0.,1.]) if abs(pd[2]) < 0.99 else np.array([0.,1.,0.])
    ex   = np.cross(pd, zref);  ex  /= np.linalg.norm(ex)
    ez   = np.cross(ex,  pd);   ez  /= np.linalg.norm(ez)
    return np.column_stack([ex, pd, ez])

# ===============================================================================
#  COORDINATE CHAINS  (MOUNT_BASE world-Y applied in every chain)
# ===============================================================================

_MB = np.array([0., MOUNT_BASE, 0.])

def pivot_to_world(pts, lat):
    return aR(R_pivot(lat), np.atleast_2d(pts).astype(float)) + _MB

def ra_to_world(pts, lat, ra_ang):
    pts = np.atleast_2d(pts).astype(float)
    return pivot_to_world(aR(Ry(ra_ang), pts) + [0., RA_ELEV, 0.], lat)

def dec_to_world(pts, lat, ra_ang, dec_deg):
    pts = np.atleast_2d(pts).astype(float)
    return ra_to_world(
        aR(Rx(-np.radians(dec_deg)), pts) + [DEC_XOFF, 0., 0.], lat, ra_ang)

def dec_bearing_world(lat):
    return RA_ELEV * polar_dir(lat) + _MB

def dec_bearing_world_true(lat):
    """Reference point on the true polar axis at the same height as the Dec bearing."""
    return RA_ELEV * polar_dir_true(lat) + _MB

def cs_to_world(pts, lat, sky):
    """CS rotates around the TRUE celestial pole (not the mount's offset axis).
    When polar offset is non-zero the mount drifts off-target over time — the
    same drift that polar misalignment produces in real observations."""
    pts = np.atleast_2d(pts).astype(float)
    return aR(R_pivot_true(lat), aR(Ry(-sky), pts)) + dec_bearing_world_true(lat)

# ─── Parallactic angle ────────────────────────────────────────────────────────
# q: angle at the star between the great circle to the zenith and the great
# circle to the north celestial pole.
# q = 0: zenith is north of star (for NH transit)
# q = ±180: zenith is south (typical for SH observer tracking a northern target)
# For a GEM EQ mount: HA is fixed during tracking, so q is CONSTANT -> no
# field rotation in the eyepiece.
# For an alt-az mount: HA advances, q changes -> field rotates.

def parallactic_angle(lat_deg, dec_deg, ha_deg):
    phi   = np.radians(_safe_lat(lat_deg))
    delta = np.radians(dec_deg)
    H     = np.radians(ha_deg)
    return np.degrees(np.arctan2(np.sin(H),
                                  np.tan(phi)*np.cos(delta) - np.sin(delta)*np.cos(H)))

def _safe_dec(dec_deg):
    """Snap ±90 slightly inward to keep cos(delta) nonzero."""
    return np.clip(dec_deg, -89.9, 89.9)

def ha_dec_to_altaz(lat_deg, ha_deg, dec_deg):
    """Convert (latitude, HA, Dec) -> (altitude, azimuth) all in degrees.
    Azimuth: 0 = North, 90 = East, 180 = South, 270 = West (clockwise from N).
    """
    phi   = np.radians(_safe_lat(lat_deg))
    delta = np.radians(_safe_dec(dec_deg))
    H     = np.radians(ha_deg)
    sinh  = np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(H)
    h_rad = np.arcsin(np.clip(sinh, -1., 1.))
    cos_h = np.cos(h_rad)
    if cos_h < 1e-6:               # very close to zenith: azimuth undefined
        return np.degrees(h_rad), 0.0
    sinA  = -np.cos(delta)*np.sin(H) / cos_h
    cosA  = (np.sin(delta) - np.sin(h_rad)*np.sin(phi)) / (cos_h * np.cos(phi))
    return np.degrees(h_rad), np.degrees(np.arctan2(sinA, cosA)) % 360.

def look_at_rotation(fwd_w):
    """Rotation matrix R such that R @ [0,0,1] = normalise(fwd_w).
    The local X axis (R @ [1,0,0]) points rightward, Y (R @ [0,1,0]) upward."""
    fwd = np.array(fwd_w, dtype=float); fwd /= np.linalg.norm(fwd)
    up  = np.array([0., 1., 0.])
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0., 0., 1.])
    right = np.cross(fwd, up);  right /= np.linalg.norm(right)
    tup   = np.cross(right, fwd)
    return np.column_stack([right, tup, fwd])   # R@[0,0,1]=fwd ✓

# Eyepiece stationary-mode FOV projection scale:
# 1.0 eyepiece unit = 1/FOV_SCALE radians ≈ FOV half-angle
FOV_SCALE = 16.0   # => half-angle = 1/16 rad ≈ 3.6 deg

# Dense star sphere — 15000 uniform points on full CS sphere.
# Shown ONLY in the eyepiece FOV (not in the 3-D scene).
# 15000 × (FOV solid angle / sphere solid angle) ≈ 15 stars typically in FOV.
np.random.seed(88)
_nt3   = 15000
_u_ds  = np.random.uniform(-1., 1.,         _nt3)
_f_ds  = np.random.uniform(0.,  2.*np.pi,   _nt3)
_r_ds  = np.sqrt(np.maximum(1. - _u_ds**2,  0.))
DENSE_SPHERE_3D = CS_R * np.c_[_r_ds*np.cos(_f_ds), _u_ds, _r_ds*np.sin(_f_ds)]
DENSE_SPHERE_SIZES = np.random.uniform(1.0, 5.0, _nt3)   # varied dot sizes
np.random.seed(42)  # restore original seed for subsequent geometry


def _build_aa_az_disk_base():
    """Azimuth gauge disk in az_base XZ plane at y=0 (Ry applied externally)."""
    _t = np.linspace(0, 2*np.pi, 64, endpoint=True)
    items = []
    items.append(('ring',  np.c_[0.54*np.cos(_t), np.zeros(64), 0.54*np.sin(_t)]))
    items.append(('inner', np.c_[0.42*np.cos(_t), np.zeros(64), 0.42*np.sin(_t)]))
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('maj', np.array([[0.44*np.cos(a), 0., 0.44*np.sin(a)],
                                        [0.54*np.cos(a), 0., 0.54*np.sin(a)]])))
    for i in range(60):
        if i % 5 != 0:
            a = i/60*2*np.pi
            items.append(('min', np.array([[0.50*np.cos(a), 0., 0.50*np.sin(a)],
                                            [0.54*np.cos(a), 0., 0.54*np.sin(a)]])))
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('spk', np.array([[0., 0., 0.],
                                        [0.42*np.cos(a), 0., 0.42*np.sin(a)]])))
    return items


def _build_aa_alt_disk_base():
    """Altitude gauge disk in OTA local YZ plane at x=0 (R_look applied externally)."""
    _t = np.linspace(0, 2*np.pi, 64, endpoint=True)
    items = []
    items.append(('ring',  np.c_[np.zeros(64), 0.44*np.cos(_t), 0.44*np.sin(_t)]))
    items.append(('inner', np.c_[np.zeros(64), 0.34*np.cos(_t), 0.34*np.sin(_t)]))
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('maj', np.array([[0., 0.36*np.cos(a), 0.36*np.sin(a)],
                                        [0., 0.44*np.cos(a), 0.44*np.sin(a)]])))
    for i in range(60):
        if i % 5 != 0:
            a = i/60*2*np.pi
            items.append(('min', np.array([[0., 0.40*np.cos(a), 0.40*np.sin(a)],
                                            [0., 0.44*np.cos(a), 0.44*np.sin(a)]])))
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('spk', np.array([[0., 0., 0.],
                                        [0., 0.34*np.cos(a), 0.34*np.sin(a)]])))
    return items


def _scale_aa_az_disk(base_pts, s):
    """Scale az disk x,z by s; y=0 stays fixed."""
    p = base_pts.copy(); p[:, [0, 2]] *= s; return p

def _scale_aa_alt_disk(base_pts, s):
    """Scale alt disk y,z by s; x=0 stays fixed."""
    p = base_pts.copy(); p[:, [1, 2]] *= s; return p


AA_AZ_DISK_BASE  = _build_aa_az_disk_base()
AA_ALT_DISK_BASE = _build_aa_alt_disk_base()

# ===============================================================================
#  GEOMETRY PRIMITIVES
# ===============================================================================

def _ring_xy(r, z, n=32):
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return np.c_[r * np.cos(t), r * np.sin(t), np.full(n, z)]

def _ring_xz(r, y, n=32):
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return np.c_[r * np.cos(t), np.full(n, y), r * np.sin(t)]

def _ring_yz(r, x, n=28):
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return np.c_[np.full(n, x), r * np.cos(t), r * np.sin(t)]

def cyl_y(r, y0, y1, nl=6, nr=24):
    segs = [_ring_xz(r, y0, nr), _ring_xz(r, y1, nr)]
    for i in range(nl):
        a = 2 * np.pi * i / nl
        segs.append(np.array([[r*np.cos(a), y0, r*np.sin(a)],
                               [r*np.cos(a), y1, r*np.sin(a)]]))
    return segs

def cyl_x(r, x0, x1, nl=4, nr=22):
    segs = [_ring_yz(r, x0, nr), _ring_yz(r, x1, nr)]
    for i in range(nl):
        a = 2 * np.pi * i / nl
        segs.append(np.array([[x0, r*np.cos(a), r*np.sin(a)],
                               [x1, r*np.cos(a), r*np.sin(a)]]))
    return segs

def _box_edges(w, h, d, cx, cy, cz):
    hw, hh, hd = w/2, h/2, d/2
    v = np.array([
        [cx-hw, cy-hh, cz-hd], [cx+hw, cy-hh, cz-hd],
        [cx+hw, cy+hh, cz-hd], [cx-hw, cy+hh, cz-hd],
        [cx-hw, cy-hh, cz+hd], [cx+hw, cy-hh, cz+hd],
        [cx+hw, cy+hh, cz+hd], [cx-hw, cy+hh, cz+hd]])
    return [v[[a, b]] for a, b in
            [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]]

# ===============================================================================
#  DISK GEOMETRY HELPERS  (base scale = 1.0)
# ===============================================================================
# RA disk lives in mountPivot XZ plane at y=RA_ELEV.
# All pts: y = RA_ELEV (fixed); x,z to be multiplied by ra_disk_scale.
# Dec disk lives in raGroup YZ plane at x=DEC_XOFF.
# All pts: x = DEC_XOFF (fixed); y,z to be multiplied by dec_disk_scale.

def _build_ra_disk_base():
    """Return list of (kind, base_pts) in mountPivot frame at scale=1."""
    _t = np.linspace(0, 2*np.pi, 64, endpoint=True)
    items = []
    # outer ring
    items.append(('ring', np.c_[0.54*np.cos(_t), np.full(64, RA_ELEV), 0.54*np.sin(_t)]))
    # inner ring
    items.append(('inner', np.c_[0.42*np.cos(_t), np.full(64, RA_ELEV), 0.42*np.sin(_t)]))
    # major ticks (12)
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('maj', np.array([
            [0.44*np.cos(a), RA_ELEV, 0.44*np.sin(a)],
            [0.54*np.cos(a), RA_ELEV, 0.54*np.sin(a)]])))
    # minor ticks (48)
    for i in range(60):
        if i % 5 != 0:
            a = i/60*2*np.pi
            items.append(('min', np.array([
                [0.50*np.cos(a), RA_ELEV, 0.50*np.sin(a)],
                [0.54*np.cos(a), RA_ELEV, 0.54*np.sin(a)]])))
    # spokes (12)
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('spk', np.array([
            [0., RA_ELEV, 0.],
            [0.42*np.cos(a), RA_ELEV, 0.42*np.sin(a)]])))
    return items

def _build_dec_disk_base():
    """Return list of (kind, base_pts) in raGroup frame at scale=1."""
    _t = np.linspace(0, 2*np.pi, 64, endpoint=True)
    items = []
    items.append(('ring', np.c_[np.full(64, DEC_XOFF), 0.44*np.cos(_t), 0.44*np.sin(_t)]))
    items.append(('inner', np.c_[np.full(64, DEC_XOFF), 0.34*np.cos(_t), 0.34*np.sin(_t)]))
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('maj', np.array([
            [DEC_XOFF, 0.36*np.cos(a), 0.36*np.sin(a)],
            [DEC_XOFF, 0.44*np.cos(a), 0.44*np.sin(a)]])))
    for i in range(60):
        if i % 5 != 0:
            a = i/60*2*np.pi
            items.append(('min', np.array([
                [DEC_XOFF, 0.40*np.cos(a), 0.40*np.sin(a)],
                [DEC_XOFF, 0.44*np.cos(a), 0.44*np.sin(a)]])))
    for i in range(12):
        a = i/12*2*np.pi
        items.append(('spk', np.array([
            [DEC_XOFF, 0., 0.],
            [DEC_XOFF, 0.34*np.cos(a), 0.34*np.sin(a)]])))
    return items

def _scale_ra_disk(base_pts, s):
    """Scale RA disk pts: multiply x,z by s (y=RA_ELEV stays fixed)."""
    p = base_pts.copy()
    p[:, [0, 2]] *= s
    return p

def _scale_dec_disk(base_pts, s):
    """Scale Dec disk pts: multiply y,z by s (x=DEC_XOFF stays fixed)."""
    p = base_pts.copy()
    p[:, [1, 2]] *= s
    return p

RA_DISK_BASE  = _build_ra_disk_base()
DEC_DISK_BASE = _build_dec_disk_base()

# ===============================================================================
#  PRE-COMPUTED STATIC GEOMETRY
# ===============================================================================

# -- Stars in CS local frame --------------------------------------------------
def _sphere_rand(n):
    u = np.random.uniform(-1, 1, n)
    f = np.random.uniform(0, 2*np.pi, n)
    r = np.sqrt(np.maximum(1 - u*u, 0))
    return np.c_[CS_R*r*np.cos(f), CS_R*u, CS_R*r*np.sin(f)]

STARS      = _sphere_rand(270)
BRIGHT     = _sphere_rand(12)
POLE_LOCAL = np.array([[0., CS_R, 0.]])

# -- Celestial grid -----------------------------------------------------------
def _cs_grid():
    segs, kinds = [], []
    t = np.linspace(0, 2*np.pi, 80, endpoint=True)
    for d in (-60, -30, 0, 30, 60):
        rd = np.radians(d)
        segs.append(np.c_[CS_R*np.cos(rd)*np.cos(t),
                           np.full(80, CS_R*np.sin(rd)),
                           CS_R*np.cos(rd)*np.sin(t)])
        kinds.append('eq' if d == 0 else 'dec')
    for h in range(6):
        ang = h/6*2*np.pi
        phi = np.linspace(-np.pi, np.pi, 80)
        segs.append(np.c_[CS_R*np.cos(phi)*np.cos(ang),
                           CS_R*np.sin(phi),
                           CS_R*np.cos(phi)*np.sin(ang)])
        kinds.append('ra')
    return segs, kinds

CS_SEGS, CS_KINDS = _cs_grid()

# -- mountPivot frame ---------------------------------------------------------
PIVOT_HOUSING = cyl_y(0.14, 0., RA_ELEV)
PIVOT_RA_LINE = np.array([[0., 0., 0.], [0., RA_ELEV, 0.]])
_PSO          = np.array([0.17, 0., 0.17])
PIVOT_PS_LINE = np.array([_PSO + [0, 0.10, 0], _PSO + [0, RA_ELEV*0.90, 0]])

# -- raGroup frame ------------------------------------------------------------
RA_DEC_HSG  = cyl_x(0.19, -0.34, DEC_XOFF + 0.08)
RA_DEC_LINE = np.array([[-0.80, 0., 0.], [DEC_XOFF + 0.80, 0., 0.]])
RA_MOTOR    = _box_edges(0.22, 0.18, 0.24, DEC_XOFF+0.16, -0.08, 0.18)

# -- decGroup OTA: TAPERED tube (objective at z=+OTA_LEN faces target) -------
#    objective ring (z=+OTA_LEN): r=0.165  >>  eyepiece (z=-OTA_LEN): r=0.075
#    Sight line goes from dec_to_world([0,0,+OTA_LEN]) -- unchanged.
DEC_OTA_AXIS  = np.array([[0., 0., -OTA_LEN], [0., 0., OTA_LEN]])
DEC_OTA_FRONT = _ring_xy(0.165,  OTA_LEN)     # objective -- large, toward target
DEC_OTA_BACK  = _ring_xy(0.075, -OTA_LEN)     # eyepiece  -- small
DEC_OTA_B1    = _ring_xy(0.148,  0.60)         # barrel band near objective
DEC_OTA_B2    = _ring_xy(0.108, -0.60)         # barrel band near eyepiece
DEC_DEW       = _ring_xy(0.140,  OTA_LEN+0.22) # dew-shield extension
DEC_FOC       = np.array([[0.090, 0., -0.32], [0.090, 0., -0.56]])  # focuser
DEC_CW_SHAFT  = np.array([[0., 0., 0.], [0., 0., -1.45]])
DEC_CW_RING   = _ring_xy(0.235, -1.26)
DEC_CW_RIM    = _ring_xy(0.235, -1.09)

# -- Tripod (world-space static) ----------------------------------------------
def _make_tripod():
    legs, details = [], []
    top_r, bot_r = 0.18, 1.28
    for i in range(3):
        ang    = i/3*2*np.pi + np.pi/6
        ca, sa = np.cos(ang), np.sin(ang)
        top = np.array([top_r*ca, MOUNT_BASE, top_r*sa])
        bot = np.array([bot_r*ca, FLOOR_Y,    bot_r*sa])
        legs.append(np.array([top, bot]))
        px, pz = -sa, ca
        details.append(np.array([bot + 0.20*np.array([px, 0, pz]),
                                  bot - 0.20*np.array([px, 0, pz])]))
        brc = np.array([top_r*0.90*ca, FLOOR_Y, top_r*0.90*sa])
        details.append(np.array([brc, bot]))
    return legs, details

TRIPOD_LEGS, TRIPOD_DTLS = _make_tripod()

_spr_y   = FLOOR_Y + 0.42*(MOUNT_BASE - FLOOR_Y)
_st      = np.linspace(0, 2*np.pi, 56, endpoint=True)
SPREADER = np.c_[0.88*np.cos(_st), np.full(56, _spr_y), 0.88*np.sin(_st)]
_at      = np.linspace(0, 2*np.pi, 40, endpoint=True)
AZ_BASE  = np.c_[0.56*np.cos(_at), np.full(40, MOUNT_BASE), 0.56*np.sin(_at)]

# -- Alt-Az mount geometry templates ----------------------------------------
# aa_post: vertical cylinder rotated by azimuth around Y (az_frame).
# aa_ota_*: OTA parts oriented by look_at rotation matrix.
ALT_H_AA_ELEV = ALT_H_AA   # altitude bearing height above MOUNT_BASE (same as constant)
AA_POST       = cyl_y(0.14, 0., ALT_H_AA, nl=6, nr=24)
_tt_r         = np.linspace(0, 2*np.pi, 32, endpoint=True)
AA_AZ_RING    = np.c_[0.38*np.cos(_tt_r), np.zeros(32), 0.38*np.sin(_tt_r)]
AA_ALT_RING   = np.c_[0.24*np.cos(_tt_r), 0.24*np.sin(_tt_r), np.zeros(32)]
# OTA templates reused: DEC_OTA_AXIS, DEC_OTA_FRONT, DEC_OTA_BACK, DEC_OTA_B1, DEC_OTA_B2, DEC_DEW

# -- Earth ground (world-space static) ----------------------------------------
GND_R = 9.5

def _earth():
    circles, spokes = [], []
    for r in [1.5, 3.0, 4.5, 6.0, 7.5, GND_R]:
        t = np.linspace(0, 2*np.pi, 64, endpoint=True)
        circles.append(np.c_[r*np.cos(t), np.full(64, FLOOR_Y), r*np.sin(t)])
    for i in range(12):
        a = i/12*2*np.pi
        spokes.append(np.array([[0., FLOOR_Y, 0.],
                                 [GND_R*np.cos(a), FLOOR_Y, GND_R*np.sin(a)]]))
    return circles, spokes

EARTH_C, EARTH_S = _earth()

# ===============================================================================
#  SIMULATION STATE
# ===============================================================================

S = dict(lat=-37.9, dec=50.0, ha=0.0, speed=4.0,
         playing=True, sky=0.0, ra=0.0, t_prev=None,
         gauge_scale=1.0,           # single scale for all gauge disks (EQ + AltAz)
         gif_start=0, gif_end=180,
         view_azim=-45., view_elev=18.,
         mode='eq_gem',
         field_rot=0.0, prev_q=None,
         stat_d_w=None, stat_e_r=None, stat_e_u=None,
         stat_R_look=None, stat_az_rad=None, stat_fov_stars=None,
         )

def _ra_ang():
    return -(S['ra'] + np.radians(S['ha']))

def _tgt_cs():
    d = np.radians(S['dec']); h = np.radians(S['ha'])
    return np.array([[-CS_R*np.cos(d)*np.sin(h),
                       CS_R*np.sin(d),
                       CS_R*np.cos(d)*np.cos(h)]])

# ===============================================================================
#  FIGURE LAYOUT
# ===============================================================================

matplotlib.rcParams.update({'font.family': 'monospace'})
fig = plt.figure(figsize=(14, 9.5), facecolor=C_BG)
try:
    fig.canvas.manager.set_window_title('EQ Mount Simulator  v4 — Austronomic')
except Exception:
    pass

# 3D axes — bottom 35% reserved for controls
ax = fig.add_axes([0.00, 0.35, 0.73, 0.65], projection='3d')
ax.set_facecolor(C_BG)
ax.grid(False)
ax.set_axis_off()
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.set_pane_color((1., 1., 1., 0.))

LM = 11.5
ax.set_xlim(-LM, LM)
ax.set_ylim(-2.5, LM*1.2)
ax.set_zlim(-LM, LM)
try:
    ax.set_box_aspect([1, 1, 1])
except AttributeError:
    pass
ax.view_init(elev=18, azim=-45, vertical_axis='y')   # orbit around world-Y = vertical

# ── Slider / Button helpers ──────────────────────────────────────────────────
_SC = '#e8ecf5'    # slider background
_SL = '#2a6aaa'    # slider track colour

def _sax(left, bot, w=0.290, h=0.021):
    return fig.add_axes([left, bot, w, h], facecolor=_SC)

def _mk_slider(rect, label, vmin, vmax, init, step=None):
    sl = Slider(_sax(*rect), label, vmin, vmax, valinit=init,
                color=_SL, valstep=step)
    sl.label.set_color('#203050')
    sl.label.set_fontsize(7.5)
    sl.valtext.set_color('#1040a0')
    sl.valtext.set_fontsize(7.5)
    return sl

# Left column: x=0.030, w=0.290
_LC = 0.030
sl_lat   = _mk_slider((_LC, 0.300, 0.290, 0.021), 'Latitude (deg)',   -89.5, 89.5, S['lat'],  0.5)
sl_dec   = _mk_slider((_LC, 0.250, 0.290, 0.021), 'Target Dec (deg)', -90.0, 90.0, S['dec'],  0.5)
sl_ha    = _mk_slider((_LC, 0.200, 0.290, 0.021), 'Hour Angle (deg)', -90,   90,   S['ha'],   1.0)
sl_gauge = _mk_slider((_LC, 0.150, 0.290, 0.021), 'Axis Gauge Scale',  0.30,  3.0,  1.0,      0.05)
sl_time  = _mk_slider((_LC, 0.100, 0.290, 0.021), 'Time Scrub (deg)',  0.0, 360.0,  0.0,      0.5)

# Mount mode radio buttons
from matplotlib.widgets import RadioButtons as _RB
fig.text(_LC, 0.083, 'Mount Mode:', fontsize=7.5, color='#203050',
         fontfamily='monospace', fontweight='bold')
ax_mode = fig.add_axes([_LC, 0.007, 0.290, 0.068], facecolor='#eef0f8')
ax_mode.set_frame_on(True)
for sp in ax_mode.spines.values():
    sp.set_edgecolor('#9ab0c8'); sp.set_linewidth(0.8)
radio_mode = _RB(ax_mode, ['EQ-GEM', 'Alt-Az', 'Stationary'], activecolor='#2a6aaa')
radio_mode.set_active(0)
for lbl in radio_mode.labels:
    lbl.set_fontsize(8); lbl.set_color('#203050'); lbl.set_fontfamily('monospace')

# Right column: x=0.370, w=0.270
_RC = 0.370
sl_spd  = _mk_slider((_RC, 0.300, 0.270, 0.021), 'Speed  x',            0.5, 30,  S['speed'], 0.5)
sl_azim = _mk_slider((_RC, 0.250, 0.270, 0.021), 'View Azimuth (deg)', -180, 180,  S['view_azim'], 1.0)
sl_elev = _mk_slider((_RC, 0.200, 0.270, 0.021), 'View Elevation (deg)', -10,  85, S['view_elev'], 1.0)
sl_polx = _mk_slider((_RC, 0.150, 0.270, 0.021), 'Polar Az Error (deg)', -3.0, 3.0, 0.0, 0.05)
sl_poly = _mk_slider((_RC, 0.100, 0.270, 0.021), 'Polar El Error (deg)', -3.0, 3.0, 0.0, 0.05)
fig.text(_RC, 0.089, '  Polar align errors (EQ-GEM only)',
         fontsize=6.5, color='#804020', fontfamily='monospace')

# RangeSlider for GIF frame range
ax_gif = fig.add_axes([_RC, 0.032, 0.270, 0.042], facecolor=_SC)
sl_gif = RangeSlider(ax_gif, 'GIF Frames', 0, GIF_FRAMES,
                     valinit=(S['gif_start'], S['gif_end']), valstep=1)
sl_gif.label.set_color('#203050');  sl_gif.label.set_fontsize(7.5)
sl_gif.valtext.set_color('#1040a0'); sl_gif.valtext.set_fontsize(7.5)
fig.text(_RC, 0.020, '  frame 0 = 0 deg sky,  360 = full rotation  (20 fps)',
         fontsize=6.5, color='#506080', fontfamily='monospace')

# Buttons
_BX = 0.688; _BW = 0.062; _BH = 0.040
def _bax(y): return fig.add_axes([_BX, y, _BW, _BH])
btn_play  = Button(_bax(0.275), 'PAUSE',    color='#d8dff0', hovercolor='#b8c8e8')
btn_reset = Button(_bax(0.225), 'RESET',    color='#d8dff0', hovercolor='#b8c8e8')
btn_gif   = Button(_bax(0.060), 'SAVE GIF', color='#dff0d8', hovercolor='#b8e8b8')
for b in (btn_play, btn_reset, btn_gif):
    b.label.set_color('#1040a0'); b.label.set_fontsize(8)

# ── Panel text ───────────────────────────────────────────────────────────────
def _ft(x, y, txt, **kw):
    return fig.text(x, y, txt, transform=fig.transFigure,
                    fontfamily='monospace', **kw)

_ft(0.800, 0.958, 'MOUNT TELEMETRY', fontsize=9, color='#1040a0',
    fontweight='bold', ha='center')
for k, y in zip(['Sky drift:', 'RA drive:', 'Dec angle:', 'Hour angle:', 'Latitude:'],
                 [0.914, 0.873, 0.832, 0.791, 0.750]):
    _ft(0.748, y, k, fontsize=8, color='#405068')
s_vals = [_ft(0.862, y, '---', fontsize=8, color='#1040a0', fontweight='bold')
          for y in [0.914, 0.873, 0.832, 0.791, 0.750]]

_ft(0.800, 0.695, 'LEGEND', fontsize=9, color='#1040a0',
    fontweight='bold', ha='center')
for c, lbl, y in [
    (C_RA,     'RA axis (polar)',  0.660),
    (C_DEC,    'Dec axis',         0.628),
    (C_PS,     'Polar scope',      0.596),
    (C_PAXL,   'Axis to Pole',     0.564),
    (C_OTA,    'OTA tube',         0.532),
    (C_POLE,   'Celestial pole',   0.500),
    (C_TARGET, 'Target object',    0.468),
    (C_SIGHT,  'Sight line',       0.436),
    (C_RA_D,   'RA gauge disk',    0.404),
    (C_DC_D,   'Dec gauge disk',   0.372),
]:
    _ft(0.748, y, '===', fontsize=10, color=c)
    _ft(0.774, y, lbl,   fontsize=8,  color='#304050')

_ft(0.748, 0.342,
    'RA drive counter-rotates\nat sidereal rate, cancelling\n'
    "Earth's rotation.\n\n"
    'CS centred at Dec bearing.\n'
    'Sight-line residual < 1.1 deg.\n'
    'HA = LST - RA.\n\n'
    'Drag 3D: click + drag\n'
    'Zoom: scroll wheel\n'
    'Space = pause   R = reset\n\n'
    'GIF saved to:\naustronomic_eqmount.gif',
    fontsize=7.5, color='#406080', va='top')

# ===============================================================================
#  EYEPIECE INSET  — bottom-right corner of the 3D view region
# ===============================================================================
# Shows the target star as seen through the telescope eyepiece.
# Key physics:
#   GEM EQ mount (this simulation): HA is fixed during tracking, so the
#   parallactic angle q is constant -> zero field rotation in the eyepiece.
#   Alt-Az mount (future): HA advances with Earth rotation, q changes,
#   the field rotates at rate dq/dt around the centred target star.
# The zenith direction arrow (orange) shows which way is "up" toward the zenith.

_EYE_L, _EYE_B, _EYE_W, _EYE_H = 0.510, 0.356, 0.210, 0.245
ax_eye = fig.add_axes([_EYE_L, _EYE_B, _EYE_W, _EYE_H])
ax_eye.set_facecolor('#04040e')
ax_eye.set_aspect('equal')
ax_eye.set_xlim(-1.42, 1.42)
ax_eye.set_ylim(-1.55, 1.30)
ax_eye.set_xticks([]); ax_eye.set_yticks([])
for sp in ax_eye.spines.values():
    sp.set_edgecolor('#303060'); sp.set_linewidth(1.5)

# Eyepiece aperture circle
_et = np.linspace(0, 2*np.pi, 120)
ax_eye.plot(np.cos(_et), np.sin(_et), color='#6070a0', lw=2.0, zorder=10)
ax_eye.fill_between(np.cos(_et), np.sin(_et), -2, color='#04040e', zorder=9)

# Cross-hairs (reticle)
ax_eye.plot([-1, 1], [0, 0], color='#252545', lw=0.8, zorder=11)
ax_eye.plot([0, 0], [-1, 1], color='#252545', lw=0.8, zorder=11)

# Small reference circle at 50% radius
ax_eye.plot(0.5*np.cos(_et), 0.5*np.sin(_et), color='#1c1c3c', lw=0.6, ls='--', zorder=11)

# Dense star field — 15000 CS-sphere points projected gnomonically each frame.
# Initialise with dummy positions; real values computed in first update().
sc_eye_dense  = ax_eye.scatter(np.full(15000, 99.), np.full(15000, 99.),
                                s=2.0, c='#5a6888', alpha=0.60, zorder=11, linewidths=0)
# Sparse layer — the same STARS+BRIGHT projected into eyepiece FOV.
sc_eye_sparse = ax_eye.scatter([99.]*282, [99.]*282,
                                s=[3.5]*270+[9.]*12,
                                c=['#8090b0']*270+['#c0d0ff']*12,
                                alpha=0.85, zorder=12, linewidths=0)

# Target star — drawn as a small asymmetric cross whose orientation
# is anchored to the celestial sphere (rotates in Alt-Az, drifts in Stationary).
# N-S arm (long): points toward celestial north in FOV.
# E-W arm (short): points toward celestial east in FOV.
ln_spike1, = ax_eye.plot([0, 0], [-0.28, 0.28], color='#ffdd00',
                          lw=2.5, alpha=0.90, zorder=15, solid_capstyle='round')
ln_spike2, = ax_eye.plot([-0.16, 0.16], [0, 0], color='#ffdd00',
                          lw=1.6, alpha=0.70, zorder=15, solid_capstyle='round')
sc_eye_target, = ax_eye.plot(0, 0, 'o', ms=4, color='#ffdd00',
                              mew=0, zorder=16, alpha=1.0)
# Telescope FOV centre — always at (0,0); red ring shows where scope points.
_et_c = np.linspace(0, 2*np.pi, 40)
ax_eye.plot(0.06*np.cos(_et_c), 0.06*np.sin(_et_c), color='#ff3333',
            lw=1.6, alpha=0.95, zorder=17)  # red centre ring
ax_eye.plot([-0.06, 0.06], [0, 0], color='#ff3333', lw=0.8, alpha=0.7, zorder=17)
ax_eye.plot([0, 0], [-0.06, 0.06], color='#ff3333', lw=0.8, alpha=0.7, zorder=17)
# Mount polar axis marker in eyepiece — green circle (visible in stationary mode)
sc_eye_mount_pole = ax_eye.scatter([99.], [99.], s=80, color='#00cc44',
                                    marker='o', linewidths=1.5,
                                    edgecolors='#00ff55', alpha=0.0, zorder=14)

# North orientation needle: direction of celestial north at target in the FOV.
# EQ: always points +y (up, toward N label).
# AltAz: rotates with cumulative field rotation -> shows non-zero rotation.
# Stationary: drifts with target and rotates.
ln_eye_north, = ax_eye.plot([0, 0], [0, 0.28], color='#ff4488',
                              lw=2.2, zorder=14, alpha=0.90)

ax_eye.text(0, 1.12, 'N', ha='center', va='bottom', color='#7090e0',
            fontsize=7.5, fontfamily='monospace', fontweight='bold', zorder=16)
ax_eye.text(-1.12, 0, 'E', ha='right', va='center', color='#7090e0',
            fontsize=7.5, fontfamily='monospace', fontweight='bold', zorder=16)

# Zenith direction arrow (dynamic — angle = parallactic angle q)
_q0  = parallactic_angle(S['lat'], S['dec'], S['ha'])
_qr0 = np.radians(_q0)
_zx, _zy = 0.75 * np.sin(_qr0), 0.75 * np.cos(_qr0)
ln_eye_zenith, = ax_eye.plot([0, _zx], [0, _zy],
                              color='#ff8833', lw=2.0, zorder=14)
sc_eye_zenith  = ax_eye.scatter([_zx], [_zy], s=35, color='#ff8833',
                                 marker='^', zorder=15, linewidths=0)
txt_eye_z      = ax_eye.text(_zx * 1.22, _zy * 1.22, 'Z', ha='center', va='center',
                              color='#ff8833', fontsize=7, fontfamily='monospace', zorder=16)

# Celestial pole marker (stationary mode only)
sc_eye_pole    = ax_eye.scatter([0], [0], s=55, color='#cc8800',
                                 marker='o', zorder=14, linewidths=1,
                                 edgecolors='#ffcc00', alpha=0.0)

txt_eye_q    = ax_eye.text(0, -1.21, f'q = {_q0:.1f}\u00b0',
                            ha='center', va='top', color='#a0b0d0',
                            fontsize=7, fontfamily='monospace', zorder=16)
txt_eye_mode = ax_eye.text(0, -1.40, 'GEM EQ:  field rot = 0',
                            ha='center', va='top', color='#50a050',
                            fontsize=6.5, fontfamily='monospace', zorder=16)

ax_eye.set_title('Eyepiece FOV', color='#8090b0', fontsize=7.5,
                  fontfamily='monospace', pad=3)

# ===============================================================================
#  CREATE PLOT OBJECTS
# ===============================================================================

def _ln(pts, c, lw=1.0, alpha=1.0, ls='-'):
    ln, = ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=c, lw=lw, alpha=alpha, ls=ls)
    return ln

def _sc(pts, s, c, alpha=1.0, marker='o', ds=False):
    return ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2],
                        s=s, c=c, alpha=alpha, marker=marker, depthshade=ds)

# Earth ground (drawn first)
for circ in EARTH_C:
    ax.plot3D(*circ.T, color=C_GND2, lw=0.65, alpha=0.50)
for spk in EARTH_S:
    ax.plot3D(*spk.T,  color=C_GND2, lw=0.50, alpha=0.38)
ax.plot3D(*EARTH_C[-1].T, color=C_GND, lw=2.0, alpha=0.85)

# Tripod (static)
for seg in TRIPOD_LEGS:
    ax.plot3D(*seg.T, color=C_TRIPOD, lw=3.2, alpha=0.95)
for seg in TRIPOD_DTLS:
    ax.plot3D(*seg.T, color=C_TRIPOD, lw=1.8, alpha=0.78)
ax.plot3D(*SPREADER.T, color=C_TRIPOD, lw=1.4, alpha=0.62)
ax.plot3D(*AZ_BASE.T,  color=C_BODY,   lw=2.2, alpha=0.85)

# Dynamic line groups
piv_lines, ra_lines, dec_lines, cs_lines = [], [], [], []
ra_disk_lines  = []   # (line_obj, base_pts) — scaled by ra_disk_scale
dec_disk_lines = []   # (line_obj, base_pts) — scaled by dec_disk_scale

# mountPivot: housing + RA axis + polar scope
for seg in PIVOT_HOUSING:
    piv_lines.append((_ln(seg, C_BODY, lw=0.9, alpha=0.62), seg))
piv_lines.append((_ln(PIVOT_RA_LINE, C_RA, lw=6.0), PIVOT_RA_LINE))
piv_lines.append((_ln(PIVOT_PS_LINE, C_PS, lw=3.2), PIVOT_PS_LINE))

# RA gauge disk (mountPivot frame, dynamic scale)
_ra_lw = {'ring': 2.6, 'inner': 0.7, 'maj': 2.0, 'min': 0.8, 'spk': 0.4}
_ra_al = {'ring': 1.0, 'inner': 0.28,'maj': 1.0, 'min': 0.52,'spk': 0.18}
for kind, base_pts in RA_DISK_BASE:
    w0 = pivot_to_world(base_pts, S['lat'])
    ln = _ln(w0, C_RA_D, lw=_ra_lw[kind], alpha=_ra_al[kind])
    ra_disk_lines.append((ln, base_pts))

# raGroup: Dec bearing housing + Dec axis + motor
for seg in RA_DEC_HSG:
    ra_lines.append((_ln(seg, C_BODY, lw=0.9, alpha=0.62), seg))
ra_lines.append((_ln(RA_DEC_LINE, C_DEC, lw=6.0), RA_DEC_LINE))
for seg in RA_MOTOR:
    ra_lines.append((_ln(seg, '#445566', lw=0.6, alpha=0.45), seg))

# Dec gauge disk (raGroup frame, dynamic scale)
_dc_lw = {'ring': 2.6, 'inner': 0.7, 'maj': 2.0, 'min': 0.8, 'spk': 0.4}
_dc_al = {'ring': 1.0, 'inner': 0.28,'maj': 1.0, 'min': 0.52,'spk': 0.18}
for kind, base_pts in DEC_DISK_BASE:
    w0 = ra_to_world(base_pts, S['lat'], _ra_ang())
    ln = _ln(w0, C_DC_D, lw=_dc_lw[kind], alpha=_dc_al[kind])
    dec_disk_lines.append((ln, base_pts))

# decGroup: tapered OTA (no struts) + counterweight
dec_lines.append((_ln(DEC_OTA_AXIS,  C_OTA, lw=5.0), DEC_OTA_AXIS))
dec_lines.append((_ln(DEC_OTA_FRONT, C_OTA, lw=3.0), DEC_OTA_FRONT))  # big objective
dec_lines.append((_ln(DEC_OTA_BACK,  C_OTA, lw=1.2), DEC_OTA_BACK))   # small eyepiece
dec_lines.append((_ln(DEC_OTA_B1,    C_OTA, lw=1.8), DEC_OTA_B1))
dec_lines.append((_ln(DEC_OTA_B2,    C_OTA, lw=1.5), DEC_OTA_B2))
dec_lines.append((_ln(DEC_DEW,   C_OTA, lw=1.2, alpha=0.55), DEC_DEW))
dec_lines.append((_ln(DEC_FOC,   '#556677', lw=2.0), DEC_FOC))
dec_lines.append((_ln(DEC_CW_SHAFT, C_BODY, lw=2.2), DEC_CW_SHAFT))
dec_lines.append((_ln(DEC_CW_RING,  C_CW,   lw=3.0), DEC_CW_RING))
dec_lines.append((_ln(DEC_CW_RIM,   C_CW,   lw=1.5), DEC_CW_RIM))

# ── Alt-Az and Stationary mount line objects (hidden when in EQ-GEM mode) ────
aa_post_lines = []   # az column (Ry(-az) in az_frame)
aa_ota_lines  = []   # alt bearing + OTA parts (R_look oriented)

_aa_init = np.zeros((2, 3))  # placeholder geometry (updated in first frame)
for seg in AA_POST:
    aa_post_lines.append((_ln(seg, C_BODY, lw=0.9, alpha=0.62), seg))
aa_post_lines.append((_ln(AA_AZ_RING,  C_BODY, lw=1.8, alpha=0.80), AA_AZ_RING))
aa_ota_lines.append((_ln(AA_ALT_RING,  C_DEC,  lw=3.0), AA_ALT_RING))
aa_ota_lines.append((_ln(DEC_OTA_AXIS, C_OTA,  lw=5.0), DEC_OTA_AXIS))
aa_ota_lines.append((_ln(DEC_OTA_FRONT,C_OTA,  lw=3.0), DEC_OTA_FRONT))
aa_ota_lines.append((_ln(DEC_OTA_BACK, C_OTA,  lw=1.2), DEC_OTA_BACK))
aa_ota_lines.append((_ln(DEC_OTA_B1,   C_OTA,  lw=1.8), DEC_OTA_B1))
aa_ota_lines.append((_ln(DEC_OTA_B2,   C_OTA,  lw=1.5), DEC_OTA_B2))
aa_ota_lines.append((_ln(DEC_DEW,      C_OTA,  lw=1.2, alpha=0.55), DEC_DEW))
aa_ota_lines.append((_ln(DEC_FOC,      '#556677', lw=2.0), DEC_FOC))

# Initially hide alt-az lines (EQ-GEM is default)
for ln, _ in aa_post_lines + aa_ota_lines:
    ln.set_visible(False)

# Sight line (aa mode uses separate ln_aa_sight drawn identically to ln_sight)
ln_aa_sight = _ln(np.zeros((2, 3)), C_SIGHT, lw=2.0, alpha=0.90, ls='--')
ln_aa_sight.set_visible(False)

# ── Alt-Az gauge disk line objects ------------------------------------------
_aa_lw = {'ring': 2.6, 'inner': 0.7, 'maj': 2.0, 'min': 0.8, 'spk': 0.4}
_aa_al = {'ring': 1.0, 'inner': 0.28,'maj': 1.0, 'min': 0.52,'spk': 0.18}

aa_az_disk_lines  = []   # Az gauge disk (Ry(-A_rad) applied, base y=0)
aa_alt_disk_lines = []   # Alt gauge disk (R_look applied, base x=0)

_az_d0 = np.zeros((2, 3))
for kind, base_pts in AA_AZ_DISK_BASE:
    ln = _ln(base_pts, C_RA_D, lw=_aa_lw[kind], alpha=_aa_al[kind])
    ln.set_visible(False)
    aa_az_disk_lines.append((ln, base_pts))

for kind, base_pts in AA_ALT_DISK_BASE:
    ln = _ln(base_pts, C_DC_D, lw=_aa_lw[kind], alpha=_aa_al[kind])
    ln.set_visible(False)
    aa_alt_disk_lines.append((ln, base_pts))

# Az pointer (sweeps az disk face; Ry(-A_rad) applied in update)
ln_aa_az_ptr  = _ln(np.zeros((2, 3)), C_RA_D, lw=3.8, alpha=0.95)
ln_aa_az_ptr.set_visible(False)
# Alt pointer (sweeps alt disk face; R_look applied in update)
ln_aa_alt_ptr = _ln(np.zeros((2, 3)), C_DC_D, lw=3.8, alpha=0.95)
ln_aa_alt_ptr.set_visible(False)
_rp0 = ra_to_world([[0., 0., 0.], [0.48, 0., 0.]], S['lat'], _ra_ang())
ln_ra_ptr  = _ln(_rp0, C_RA_D, lw=3.8, alpha=0.95)

_dp0 = dec_to_world([[0., 0., 0.], [0., 0., 0.40]], S['lat'], _ra_ang(), S['dec'])
ln_dec_ptr = _ln(_dp0, C_DC_D, lw=3.8, alpha=0.95)

# Sight line (OTA front tip -> target star)
ln_sight   = _ln(np.zeros((2, 3)), C_SIGHT, lw=2.0, alpha=0.90, ls='--')

# Polar axis extension -> celestial pole
ln_pole_ax = _ln(np.zeros((2, 3)), C_PAXL, lw=1.8, alpha=0.75, ls='--')

# Celestial grid
for seg, kind in zip(CS_SEGS, CS_KINDS):
    c  = C_EQ   if kind == 'eq'  else C_GRID
    lw = 1.2    if kind == 'eq'  else 0.65
    cs_lines.append((_ln(seg, c, lw=lw, alpha=0.82), seg))

# Scatter objects
_l0, _s0 = S['lat'], 0.
sc_stars  = _sc(cs_to_world(STARS,      _l0, _s0),  2.5, C_STARS, alpha=0.65)
sc_bright = _sc(cs_to_world(BRIGHT,     _l0, _s0), 14.0, C_STARS, alpha=0.92)
sc_pole   = _sc(cs_to_world(POLE_LOCAL, _l0, _s0), 200., C_POLE,  marker='*')
sc_target = _sc(cs_to_world(_tgt_cs(),  _l0, _s0), 190., C_TARGET, marker='*')

# Red ring on CS sphere = where telescope is actually pointing.
_p0 = cs_to_world(POLE_LOCAL, _l0, _s0)   # placeholder; updated in update()
sc_fov_center = _sc(_p0, 280., '#ff3333', marker='o', alpha=0.0)

# Green ring on CS sphere = where mount polar axis points.
# Zero error: coincides with sc_pole (true pole). Non-zero: shifts away.
_pd0 = dec_bearing_world_true(_l0) + CS_R * polar_dir(_l0)
sc_mount_pole = _sc(_pd0.reshape(1,3), 140., '#00cc44', marker='o', alpha=0.85)

# ===============================================================================
#  UPDATE FUNCTION
# ===============================================================================

def update(frame):
    now = _time.time()
    if S['t_prev'] is None:
        S['t_prev'] = now
    dt = min(now - S['t_prev'], 0.05)
    S['t_prev'] = now

    mode = S['mode']
    if S['playing']:
        dang = OMEGA * S['speed'] * dt
        S['sky'] += dang
        if mode == 'eq_gem':
            S['ra'] += dang          # RA drive counter-rotates: net HA = const

    lat  = S['lat'];  sky = S['sky'];  dec = _safe_dec(S['dec'])
    gs   = S['gauge_scale']
    raa  = _ra_ang()

    # ── Visibility: EQ or Alt-Az/Stationary ──────────────────────────────────
    eq_vis = (mode == 'eq_gem')
    aa_vis = (mode in ('alt_az', 'stationary'))

    # EQ structure — dec_lines MUST be included to hide OTA/CW in non-EQ modes
    for ln, _ in (piv_lines + ra_lines + dec_lines +
                  list(ra_disk_lines) + list(dec_disk_lines)):
        ln.set_visible(eq_vis)
    ln_ra_ptr.set_visible(eq_vis);  ln_dec_ptr.set_visible(eq_vis)
    ln_sight.set_visible(eq_vis);   ln_pole_ax.set_visible(eq_vis)

    # Alt-az structure
    for ln, _ in aa_post_lines + aa_ota_lines:
        ln.set_visible(aa_vis)
    ln_aa_sight.set_visible(aa_vis)
    for ln, _ in aa_az_disk_lines + aa_alt_disk_lines:
        ln.set_visible(aa_vis)
    ln_aa_az_ptr.set_visible(aa_vis);  ln_aa_alt_ptr.set_visible(aa_vis)

    # ── EQ-GEM mode ──────────────────────────────────────────────────────────
    if mode == 'eq_gem':
        for ln, pts in piv_lines:
            w = pivot_to_world(pts, lat);  ln.set_data_3d(*w.T)
        for ln, bp in ra_disk_lines:
            w = pivot_to_world(_scale_ra_disk(bp, gs), lat);  ln.set_data_3d(*w.T)
        for ln, pts in ra_lines:
            w = ra_to_world(pts, lat, raa);  ln.set_data_3d(*w.T)
        for ln, bp in dec_disk_lines:
            w = ra_to_world(_scale_dec_disk(bp, gs), lat, raa);  ln.set_data_3d(*w.T)
        for ln, pts in dec_lines:
            w = dec_to_world(pts, lat, raa, dec);  ln.set_data_3d(*w.T)
        rp = ra_to_world([[0.,0.,0.],[0.48*gs,0.,0.]], lat, raa)
        ln_ra_ptr.set_data_3d(*rp.T)
        dp = dec_to_world([[0.,0.,0.],[0.,0.,0.40*gs]], lat, raa, dec)
        ln_dec_ptr.set_data_3d(*dp.T)
        tip_w = dec_to_world([[0., 0., OTA_LEN]], lat, raa, dec)[0]
        tgt_w = cs_to_world(_tgt_cs(), lat, sky)[0]
        ln_sight.set_data_3d([tip_w[0], tgt_w[0]],
                              [tip_w[1], tgt_w[1]], [tip_w[2], tgt_w[2]])
        db  = dec_bearing_world(lat)        # mount Dec bearing (offset axis)
        pol = CS_R * polar_dir_true(lat) + dec_bearing_world_true(lat)  # TRUE pole
        ln_pole_ax.set_data_3d([db[0],pol[0]], [db[1],pol[1]], [db[2],pol[2]])

        # EQ FOV frame from OTA orientation
        dec_r   = np.radians(dec)
        _Rlook  = R_pivot(lat) @ Ry(raa) @ Rx(-dec_r)
        _dp, _er, _eu = _Rlook[:,2], _Rlook[:,0], _Rlook[:,1]

    # ── Alt-Az and Stationary modes ──────────────────────────────────────────
    else:
        ha_eff_deg = S['ha'] + np.degrees(sky)   # no RA drive: HA grows

        if mode == 'alt_az':
            h_deg, A_deg = ha_dec_to_altaz(lat, ha_eff_deg, dec)
            A_rad   = np.radians(A_deg)
            dec_r   = np.radians(dec); ha_r = np.radians(ha_eff_deg)
            tgt_unit = np.array([-np.cos(dec_r)*np.sin(ha_r),
                                   np.sin(dec_r),
                                   np.cos(dec_r)*np.cos(ha_r)])
            R_look   = look_at_rotation(R_pivot(lat) @ tgt_unit)
        else:
            A_rad  = S['stat_az_rad']  if S['stat_az_rad']  is not None else 0.
            R_look = S['stat_R_look']  if S['stat_R_look']  is not None else np.eye(3)

        _dp = R_look[:,2]; _er = R_look[:,0]; _eu = R_look[:,1]
        aa_origin = _MB + np.array([0., ALT_H_AA, 0.])

        # Az post
        for ln, bp in aa_post_lines:
            ln.set_data_3d(*(aR(Ry(-A_rad), bp) + _MB).T)

        # Alt bearing + OTA (no EQ geometry — only aa_ota_lines)
        for ln, bp in aa_ota_lines:
            ln.set_data_3d(*(aR(R_look, bp) + aa_origin).T)

        # Az gauge disk
        for ln, bp in aa_az_disk_lines:
            ln.set_data_3d(*(aR(Ry(-A_rad), _scale_aa_az_disk(bp, gs)) + _MB).T)
        ap = aR(Ry(-A_rad), np.array([[0.,0.,0.],[0.48*gs,0.,0.]])) + _MB
        ln_aa_az_ptr.set_data_3d(*ap.T)

        # Alt gauge disk
        for ln, bp in aa_alt_disk_lines:
            ln.set_data_3d(*(aR(R_look, _scale_aa_alt_disk(bp, gs)) + aa_origin).T)
        alp = aR(R_look, np.array([[0.,0.,0.],[0.,0.40*gs,0.]])) + aa_origin
        ln_aa_alt_ptr.set_data_3d(*alp.T)

        # Sight line
        tip_w = (R_look @ np.array([0., 0., OTA_LEN])) + aa_origin
        tgt_w = cs_to_world(_tgt_cs(), lat, sky)[0]
        ln_aa_sight.set_data_3d([tip_w[0], tgt_w[0]],
                                  [tip_w[1], tgt_w[1]], [tip_w[2], tgt_w[2]])

    # ── Celestial sphere ──────────────────────────────────────────────────────
    for ln, pts in cs_lines:
        w = cs_to_world(pts, lat, sky);  ln.set_data_3d(*w.T)
    sw = cs_to_world(STARS,      lat, sky)
    bw = cs_to_world(BRIGHT,     lat, sky)
    pw = cs_to_world(POLE_LOCAL, lat, sky)
    tw = cs_to_world(_tgt_cs(),  lat, sky)
    sc_stars._offsets3d  = (*sw.T,)
    sc_bright._offsets3d = (*bw.T,)
    sc_pole._offsets3d   = ([pw[0,0]], [pw[0,1]], [pw[0,2]])
    sc_target._offsets3d = ([tw[0,0]], [tw[0,1]], [tw[0,2]])

    # ── Eyepiece: full gnomonic projection from celestial sphere ────────────────
    # RP uses the TRUE pole (CS rotates around true axis, not offset axis).
    RP  = R_pivot_true(lat)

    def _set_spikes(tx, ty, rot_deg):
        """Draw N-S (long) and E-W (short) spikes anchored to celestial sphere."""
        ar = np.radians(rot_deg); ca, sa = np.cos(ar), np.sin(ar)
        nx, ny = -sa, ca        # celestial north direction in this FOV orientation
        ex_, ey_ = ca, sa       # celestial east direction
        ln_spike1.set_xdata([tx - 0.28*nx, tx + 0.28*nx])
        ln_spike1.set_ydata([ty - 0.28*ny, ty + 0.28*ny])
        ln_spike2.set_xdata([tx - 0.16*ex_, tx + 0.16*ex_])
        ln_spike2.set_ydata([ty - 0.16*ey_, ty + 0.16*ey_])
        sc_eye_target.set_data([tx], [ty])

    # Pick FOV frame for the eyepiece projection
    if mode == 'stationary' and S['stat_d_w'] is not None:
        fov_dp, fov_er, fov_eu = S['stat_d_w'], S['stat_e_r'], S['stat_e_u']
    else:
        fov_dp, fov_er, fov_eu = _dp, _er, _eu

    # Project dense sphere + sparse layers into eyepiece
    dxy  = _gnomonic_batch(DENSE_SPHERE_3D, fov_dp, fov_er, fov_eu, RP, sky)
    sxy  = _gnomonic_batch(np.vstack([STARS, BRIGHT]), fov_dp, fov_er, fov_eu, RP, sky)
    sc_eye_dense.set_offsets(dxy)
    sc_eye_sparse.set_offsets(sxy)

    # Project target yellow star (its CS-local position via gnomonic)
    tgt_xy = _gnomonic_one(_tgt_cs()[0], fov_dp, fov_er, fov_eu, RP, sky)

    # Project mount polar axis onto eyepiece FOV
    # mount_pole_dir_world = polar_dir(lat), convert to CS-local via Ry(sky)@RP.T
    _mp_cs  = CS_R * Ry(sky) @ RP.T @ polar_dir(lat)
    mp_xy   = _gnomonic_one(_mp_cs, fov_dp, fov_er, fov_eu, RP, sky)
    if abs(mp_xy[0]) < 1.5 and abs(mp_xy[1]) < 1.5:
        sc_eye_mount_pole.set_offsets([[mp_xy[0], mp_xy[1]]])
        sc_eye_mount_pole.set_alpha(0.90)
    else:
        sc_eye_mount_pole.set_alpha(0.)

    # Update FOV-centre red ring on 3D CS sphere (where telescope is pointing)
    # OTA pointing direction: last column of the FOV rotation matrix (world frame)
    ota_world_dir = fov_dp / np.linalg.norm(fov_dp)
    fov_ctr_world  = dec_bearing_world_true(lat) + CS_R * ota_world_dir
    sc_fov_center._offsets3d = ([fov_ctr_world[0]], [fov_ctr_world[1]], [fov_ctr_world[2]])
    sc_fov_center.set_alpha(0.88)

    # Update mount pole green ring on 3D CS sphere
    mp_world   = dec_bearing_world_true(lat) + CS_R * polar_dir(lat)
    sc_mount_pole._offsets3d = ([mp_world[0]], [mp_world[1]], [mp_world[2]])

    if mode == 'eq_gem':
        # In EQ-GEM: telescope tracks the target.  With perfect polar alignment
        # tgt_xy ≈ (0,0); with polar error it drifts away — showing the mount is
        # no longer centred on the intended star.
        sc_eye_pole.set_alpha(0.)
        q_now = parallactic_angle(lat, dec, S['ha'])
        qr    = np.radians(q_now)
        _zx_n, _zy_n = 0.75*np.sin(qr), 0.75*np.cos(qr)
        ln_eye_zenith.set_xdata([0, _zx_n]); ln_eye_zenith.set_ydata([0, _zy_n])
        sc_eye_zenith.set_offsets([_zx_n, _zy_n]); sc_eye_zenith.set_alpha(1.)
        txt_eye_z.set_position((_zx_n*1.22, _zy_n*1.22)); txt_eye_z.set_text('Z')
        txt_eye_z.set_color('#ff8833')
        # Spikes anchored to celestial sphere — no field rotation in EQ.
        # Target position uses tgt_xy: drifts off-centre when polar error is non-zero.
        tx, ty = tgt_xy
        _set_spikes(tx, ty, 0.)
        # North needle: from target toward celestial north (always upward in EQ)
        ln_eye_north.set_xdata([tx, tx]);     ln_eye_north.set_ydata([ty, ty + 0.28])
        txt_eye_q.set_text(f'q = {q_now:.1f}\u00b0')
        txt_eye_mode.set_text('GEM EQ:  field rot = 0'); txt_eye_mode.set_color('#50a050')
        S['prev_q'] = None; S['field_rot'] = 0.

    elif mode == 'alt_az':
        ha_eff = S['ha'] + np.degrees(sky)
        q_now  = parallactic_angle(lat, dec, ha_eff)
        if S['prev_q'] is not None:
            dq = (q_now - S['prev_q'] + 180.) % 360. - 180.
            S['field_rot'] += dq
        S['prev_q'] = q_now
        sc_eye_pole.set_alpha(0.)
        qr = np.radians(q_now)
        _zx_n, _zy_n = 0.75*np.sin(qr), 0.75*np.cos(qr)
        ln_eye_zenith.set_xdata([0, _zx_n]); ln_eye_zenith.set_ydata([0, _zy_n])
        sc_eye_zenith.set_offsets([_zx_n, _zy_n]); sc_eye_zenith.set_alpha(1.)
        txt_eye_z.set_position((_zx_n*1.22, _zy_n*1.22)); txt_eye_z.set_text('Z')
        txt_eye_z.set_color('#ff8833')
        # Spikes rotate with field — the celestial N-S direction sweeps around the target.
        # Target position at tgt_xy (centre when tracking perfectly, off-centre otherwise).
        fr = S['field_rot']
        tx, ty = tgt_xy
        _set_spikes(tx, ty, fr)
        # North needle: rotated by field_rot, anchored at target position
        frad = np.radians(fr)
        ln_eye_north.set_xdata([tx, tx - np.sin(frad)*0.28])
        ln_eye_north.set_ydata([ty, ty + np.cos(frad)*0.28])
        txt_eye_q.set_text(f'q = {q_now:.1f}\u00b0')
        txt_eye_mode.set_text(f'Alt-Az: field rot = {fr:.0f}\u00b0')
        txt_eye_mode.set_color('#cc6622')

    else:  # stationary
        if S['stat_d_w'] is None:
            _init_stationary_state(lat, dec, S['ha'], sky)
        if S['stat_d_w'] is not None:
            d_p = S['stat_d_w']; e_r = S['stat_e_r']; e_u = S['stat_e_u']
            # Target drifts as HA grows
            ha_r = np.radians(S['ha'] + np.degrees(sky)); d_r = np.radians(dec)
            tgt_cs = CS_R * np.array([-np.cos(d_r)*np.sin(ha_r),
                                        np.sin(d_r), np.cos(d_r)*np.cos(ha_r)])
            tp = _gnomonic_one(tgt_cs, d_p, e_r, e_u, RP, sky)
            sc_eye_target.set_data([tp[0]], [tp[1]])
            # Pole: invariant under Ry(-sky) — stays fixed in FOV
            pp = _gnomonic_one(np.array([0., CS_R, 0.]), d_p, e_r, e_u, RP, sky)
            if abs(pp[0]) < 5 and abs(pp[1]) < 5:
                sc_eye_pole.set_offsets([[pp[0], pp[1]]]); sc_eye_pole.set_alpha(1.)
                txt_eye_z.set_position((pp[0]*1.18, pp[1]*1.18))
                txt_eye_z.set_text('P'); txt_eye_z.set_color('#cc8800')
                dvx, dvy = pp[0]-tp[0], pp[1]-tp[1]
                dn = max(np.hypot(dvx, dvy), 1e-6)
                # North needle points from target toward pole in FOV
                ln_eye_north.set_xdata([tp[0], tp[0]+dvx/dn*0.28])
                ln_eye_north.set_ydata([tp[1], tp[1]+dvy/dn*0.28])
                # Spikes at target position, oriented toward pole (celestial north)
                spike_rot = np.degrees(np.arctan2(dvx, dvy))
                _set_spikes(tp[0], tp[1], spike_rot)
            else:
                sc_eye_pole.set_alpha(0.); txt_eye_z.set_text('')
                ln_eye_north.set_xdata([tp[0], tp[0]]); ln_eye_north.set_ydata([tp[1], tp[1]])
                _set_spikes(tp[0], tp[1], 0.)
            ln_eye_zenith.set_xdata([0,0]); ln_eye_zenith.set_ydata([0,0])
            sc_eye_zenith.set_alpha(0.)
            txt_eye_q.set_text('No tracking')
            txt_eye_mode.set_text('Stationary: full field drift')
            txt_eye_mode.set_color('#aa3333')

    # ── Telemetry ─────────────────────────────────────────────────────────────
    sky_d = np.degrees(sky) % 360.
    ra_d  = np.degrees(S['ra']) % 360.
    ns    = 'N' if lat >= 0 else 'S'
    for sv, txt in zip(s_vals, [f'{sky_d:.1f} deg', f'{ra_d:.1f} deg',
                                  f'{dec:.0f} deg',   f"{S['ha']:.0f} deg",
                                  f'{abs(lat):.1f} deg {ns}']):
        sv.set_text(txt)

    return []


# ── Gnomonic projection helpers ────────────────────────────────────────────────

def _gnomonic_one(cs_pt, d_p, e_r, e_u, RP, sky):
    dw  = RP @ (Ry(-sky) @ np.array(cs_pt, dtype=float)) / CS_R
    den = float(np.dot(dw, d_p))
    if den < 0.05:
        return (99., 99.)
    return (float(np.dot(dw, e_r))/den*FOV_SCALE,
            float(np.dot(dw, e_u))/den*FOV_SCALE)

def _gnomonic_batch(cs_pts, d_p, e_r, e_u, RP, sky):
    Rsky  = Ry(-sky)
    dws   = (RP @ (Rsky @ cs_pts.T)).T / CS_R
    dens  = dws @ d_p
    xs    = dws @ e_r
    ys    = dws @ e_u
    result = np.full((len(cs_pts), 2), 99.)
    mask   = dens > 0.05
    result[mask, 0] = xs[mask] / dens[mask] * FOV_SCALE
    result[mask, 1] = ys[mask] / dens[mask] * FOV_SCALE
    return result



def _gen_fov_stars(tgt_cs_unit, n=120, fov_half=0.20):
    """Generate n CS-local point vectors within fov_half rad of tgt_cs_unit.
    Used for stationary-mode dense field (larger n and radius than EQ/AltAz cluster)."""
    rng = np.random.RandomState(77)
    tgt = np.array(tgt_cs_unit, dtype=float); tgt /= np.linalg.norm(tgt)
    up  = np.array([0., 1., 0.])
    if abs(np.dot(tgt, up)) > 0.99:
        up = np.array([1., 0., 0.])
    r_ax = np.cross(tgt, up);  r_ax /= np.linalg.norm(r_ax)
    u_ax = np.cross(r_ax, tgt)
    cos_min = np.cos(fov_half)
    cosv = rng.uniform(cos_min, 1.0, n)
    phi  = rng.uniform(0., 2.*np.pi, n)
    sinv = np.sqrt(1. - cosv**2)
    return CS_R * np.column_stack([
        cosv[:,None]*tgt + sinv[:,None]*(np.cos(phi)[:,None]*r_ax +
                                          np.sin(phi)[:,None]*u_ax)]).reshape(n, 3)


def _init_stationary_state(lat, dec, ha, sky):
    """Capture pointing direction and FOV frame when entering Stationary mode."""
    ha_eff_r = np.radians(ha + np.degrees(sky))
    d_r      = np.radians(_safe_dec(dec))
    tgt_unit = np.array([-np.cos(d_r)*np.sin(ha_eff_r),
                           np.sin(d_r),
                           np.cos(d_r)*np.cos(ha_eff_r)])
    tgt_dir_w = R_pivot(lat) @ tgt_unit          # world unit-vector of OTA axis

    e_r = np.cross(tgt_dir_w, np.array([0.,1.,0.]))
    if np.linalg.norm(e_r) < 1e-6:
        e_r = np.cross(tgt_dir_w, np.array([0.,0.,1.]))
    e_r /= np.linalg.norm(e_r)
    e_u  = np.cross(e_r, tgt_dir_w)

    _, A_deg   = ha_dec_to_altaz(lat, ha + np.degrees(sky), dec)
    R_look_now = look_at_rotation(tgt_dir_w)

    S['stat_d_w']       = tgt_dir_w
    S['stat_e_r']       = e_r
    S['stat_e_u']       = e_u
    S['stat_az_rad']    = np.radians(A_deg)
    S['stat_R_look']    = R_look_now
    S['stat_fov_stars'] = DENSE_SPHERE_3D  # use the full dense sphere (projected each frame)

# ===============================================================================
#  GIF EXPORT
# ===============================================================================

def save_gif(event):
    """
    Capture selected GIF frame range and save with PIL directly.

    Fixes vs v4:
      - Stops FuncAnimation timer before capture (prevents re-entrant updates)
      - Uses fig.savefig() -> BytesIO -> Image.copy() (reliable on all backends)
      - No PillowWriter, no flush_events() in the loop
      - Restarts animation timer in finally block regardless of errors
    """
    import io
    try:
        from PIL import Image
    except ImportError:
        print('ERROR: Pillow not installed.  Run:  pip install Pillow')
        return

    start_f = int(S['gif_start'])
    end_f   = int(S['gif_end'])
    if end_f <= start_f:
        print('GIF ERROR: end frame must exceed start frame.')
        return

    n = end_f - start_f
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'austronomic_eqmount.gif')

    # ── Stop the animation timer so it cannot fire during capture ────────────
    ani.event_source.stop()
    old_playing  = S['playing']
    old_sky      = S['sky']
    old_ra       = S['ra']
    old_t_prev   = S['t_prev']
    S['playing'] = False

    btn_gif.label.set_text('0 %')
    fig.canvas.draw()

    frames = []
    try:
        # ── Compute 3D-axes-only bbox (excludes control panel/legend) ─────────
        # The 3D axes occupies figure coords [0.00, 0.35, 0.73, 0.65].
        # bbox_inches clips savefig to that region so every GIF frame contains
        # only the 3D view and eyepiece inset, with no sliders or legend.
        import matplotlib.transforms as _mxt
        _fw, _fh  = fig.get_size_inches()
        _ax_bbox  = _mxt.Bbox([[0.0, 0.35 * _fh], [0.73 * _fw, _fh]])

        # Lock view angle to whatever was set before pressing Save GIF
        ax.view_init(elev=S['view_elev'], azim=S['view_azim'], vertical_axis='y')

        for i, fidx in enumerate(range(start_f, end_f)):
            # Advance sky to exactly this frame's angle
            S['sky']   = 2.0 * np.pi * fidx / GIF_FRAMES
            S['ra']    = S['sky']
            S['t_prev'] = None          # prevents stale dt accumulation

            update(fidx)
            fig.canvas.draw()           # render to canvas (no event dispatch)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, facecolor=C_BG,
                        bbox_inches=_ax_bbox, pad_inches=0)
            buf.seek(0)
            frames.append(Image.open(buf).copy())  # .copy() detaches from buf
            buf.close()

            # Progress label every ~5%
            if n > 0 and ((i + 1) % max(1, n // 20) == 0 or i == n - 1):
                btn_gif.label.set_text(f'{100*(i+1)//n} %')
                fig.canvas.draw()

        # ── Save ─────────────────────────────────────────────────────────────
        frames[0].save(
            outpath, save_all=True, append_images=frames[1:],
            loop=0, duration=1000 // GIF_FPS, optimize=False)
        print(f'Saved {n}-frame GIF ({GIF_FPS} fps)  ->  {outpath}')

    except Exception as exc:
        import traceback
        print(f'GIF export error: {exc}')
        traceback.print_exc()

    finally:
        # ── Always restore state and restart animation ────────────────────────
        S['playing'] = old_playing
        S['sky']     = old_sky
        S['ra']      = old_ra
        S['t_prev']  = old_t_prev
        btn_gif.label.set_text('SAVE GIF')
        fig.canvas.draw()
        ani.event_source.start()

# ===============================================================================
#  CALLBACKS
# ===============================================================================

def _on_lat(v):
    raw = float(v)
    S['lat'] = _safe_lat(raw)
    S['sky'] = 0.; S['ra'] = 0.
    S['stat_d_w'] = None

def _on_dec(v):
    S['dec'] = float(v)
    S['stat_d_w'] = None   # invalidate stationary state when target changes

def _on_ha(v):
    S['ha']  = float(v)
    S['stat_d_w'] = None
def _on_spd(v): S['speed'] = float(v)

def _on_gauge(v):
    S['gauge_scale'] = float(v)

def _on_time(v):
    """Time scrubber: jump animation to this sky position."""
    sky_val = np.radians(float(v))
    S['sky'] = sky_val
    if S['mode'] == 'eq_gem':
        S['ra'] = sky_val    # keep RA drive in sync
    S['prev_q'] = None       # reset field rotation tracking
    S['t_prev'] = None

def _on_polx(v):
    global _polar_off_x; _polar_off_x = float(v)
    S['sky'] = 0.; S['ra'] = 0.; S['t_prev'] = None   # restart drift from zero

def _on_poly(v):
    global _polar_off_y; _polar_off_y = float(v)
    S['sky'] = 0.; S['ra'] = 0.; S['t_prev'] = None

def _on_mode(label):
    S['mode'] = {'EQ-GEM': 'eq_gem', 'Alt-Az': 'alt_az',
                 'Stationary': 'stationary'}[label]
    S['field_rot'] = 0.; S['prev_q'] = None
    if S['mode'] == 'stationary':
        _init_stationary_state(S['lat'], S['dec'], S['ha'], S['sky'])
    else:
        S['stat_d_w'] = None

def _on_azim(v):
    S['view_azim'] = float(v)
    ax.view_init(elev=S['view_elev'], azim=S['view_azim'], vertical_axis='y')
    fig.canvas.draw_idle()

def _on_elev(v):
    S['view_elev'] = float(v)
    ax.view_init(elev=S['view_elev'], azim=S['view_azim'], vertical_axis='y')
    fig.canvas.draw_idle()

def _on_gif(val):
    lo, hi = int(val[0]), int(val[1])
    S['gif_start'] = min(lo, hi - 1)
    S['gif_end']   = max(hi, lo + 1)

sl_lat.on_changed(_on_lat);   sl_dec.on_changed(_on_dec)
sl_ha.on_changed(_on_ha);     sl_spd.on_changed(_on_spd)
sl_gauge.on_changed(_on_gauge); sl_time.on_changed(_on_time)
sl_polx.on_changed(_on_polx); sl_poly.on_changed(_on_poly)
sl_azim.on_changed(_on_azim); sl_elev.on_changed(_on_elev)
sl_gif.on_changed(_on_gif);   radio_mode.on_clicked(_on_mode)

def _on_play(ev):
    S['playing'] = not S['playing']
    btn_play.label.set_text('PAUSE' if S['playing'] else 'PLAY ')
    fig.canvas.draw_idle()

def _on_reset(ev):
    S['sky'] = 0.; S['ra'] = 0.

btn_play.on_clicked(_on_play)
btn_reset.on_clicked(_on_reset)
btn_gif.on_clicked(save_gif)

def _on_key(event):
    if event.key == ' ':
        _on_play(None)
    elif event.key in ('r', 'R'):
        _on_reset(None)

fig.canvas.mpl_connect('key_press_event', _on_key)

# ===============================================================================
#  LAUNCH
# ===============================================================================

ani = manim.FuncAnimation(fig, update, interval=50, blit=False)
plt.show()