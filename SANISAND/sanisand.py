"""
After installing libraries:
-SELECT GPU RENDER (important if you see glitched render)

Linux command
Select the GPU to render with OPEN-GL

    sudo apt install -y mesa-utils pciutils
    lspci | grep -Ei "VGA|3D|Display"
    glxinfo -B
    echo 'export __NV_PRIME_RENDER_OFFLOAD=1' >> ~/.bashrc
    echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ~/.bashrc
    source ~/.bashrc

Local Interpreter
    python3 -m manim -p --renderer=opengl sanisand.py SANISAND

Virtual enviroment (in this example using uv)
    uv run manim -p --renderer=opengl sanisand.py SANISAND

Run preview
    manim render --renderer=opengl -p  sanisand.py SANISAND

Render video and save file
    manim render --renderer=opengl --write_to_movie -qh sanisand.py SANISAND

Quality
    -ql low
    -qh full-hd
    -qk 4k

FPS control
choose number of fps for faster render and test scene (example "--fps 8")
    manim render --renderer=opengl --write_to_movie -ql --fps 8 sanisand.py SANISAND
--------------------------------------------------------------------------------------

--- keyboard timeline instructions---
SPACE   play / pause
LEFT    step back
RIGHT   step forward   (SHIFT = x10, CTRL = single)
HOME    start
END     fine

--- 3D ---
Z       zoom IN
X       zoom OUT
F       follow on/off
V       reset

--- MODE KEYS ---
 Q = SOLO
 W = SEQUENCE
 E = ALL / SIM

--- Info ---
I = Info
--------------------------------------------------------------------------------------
"""

from manim import *
from manim.opengl import *
from manim.utils.rate_functions import (
    ease_in_out_cubic,
    ease_in_cubic,
    ease_out_cubic,
    linear,
)
import glob
import os
import numpy as np
import pandas as pd
import re

# ================================================================
# ================================================================
#
#   P A R A M E T E R S
#
#   WARNING: never reassign these names inside construct. If a
#   function contains even one assignment to a name, Python treats
#   it as local for the WHOLE function and the global value becomes
#   unreachable (UnboundLocalError). To change a value at runtime,
#   use a ValueTracker instead.
#
# ================================================================

# ----------------------------------------------------------------
# INPUT FILES
# ----------------------------------------------------------------
FILE_GLOB_PATTERN = "CU_*.out"
FILE_LIST = []                    # if non-empty, overrides the glob

# per-dataset colours, in order
PALETTE = [
    GREEN_B, BLUE_B, YELLOW_B, RED_B,
    PURPLE_B, TEAL_B, ORANGE, PINK, LIGHT_BROWN,
]

# ----------------------------------------------------------------
# CONSTITUTIVE MODEL  (SANISAND 2004, Dafalias & Manzari)
# ----------------------------------------------------------------
M = 1.25            # critical state stress ratio in compression
c = 0.712           # M_extension / M_compression
e0 = 0.934          # reference void ratio of the CSL
lambda_c = 0.019    # CSL slope
xi = 0.7            # CSL exponent
p_atm = 100.0       # reference atmospheric pressure
m_yield = 0.01    # yield surface half-aperture

# magnification of the stress point relative to alpha:
# 1.0 = true geometry, >1 exaggerates the offset r - alpha
M_YIELD_DISPLAY_SCALE = 1

# ----------------------------------------------------------------
# KEYBOARD
# ----------------------------------------------------------------
KEY_ARROW_STEP = 5          # data frames per arrow keypress
KEY_ZOOM_FACTOR = 1.25      # step of the Z / X keys
KEY_DETAIL_STEP = 1.25      # step of the + / - keys
KEY_DETAIL_MIN = 0.02
KEY_DETAIL_MAX = 20.0

# ----------------------------------------------------------------
# TIMELINE AND PLAYBACK
# ----------------------------------------------------------------
PLAYBACK_SPEED = 0.05       # datasets per second: 0.05 -> 20 s each

# playback rate function, not the choreography one
# https://docs.manim.community/en/stable/reference/manim.utils.rate_functions.html
RATE = ease_in_cubic

# ----------------------------------------------------------------
# 3D AXES
#
# MARGIN is the headroom beyond the maximum p' in the data. The tick
# step is chosen on the DATA value, not on the margin-inflated one:
# otherwise the margin pushes into the next decade and the rounding
# overshoots badly (p'max 3000 -> axis at 4000).
# With TIGHT_RANGE the limit is exactly p'max * (1 + MARGIN).
# ----------------------------------------------------------------
P3D_MARGIN = 0.0
P3D_TICKS = 6
P3D_TIGHT_RANGE = True

AXES3D_LENGTH = 4.0         # axis cube side, in manim units
AXES3D_SCALE = 0.5
AXES3D_LABEL_OUT = 1.35     # label distance, as a fraction of p'max

CONE_GROUP_SHIFT = LEFT * 1 + DOWN * 3
CONE_GROUP_SCALE = 1.5

# initial camera
PHI0 = 65 * DEGREES
THETA0 = 80 * DEGREES
LIGHT_SOURCE_POS = np.array([-8.0, -8.0, 12.0])

# ----------------------------------------------------------------
# Z-FIGHTING
#
# Coplanar geometries (the yield circle lies EXACTLY on the yield
# surface cone, the sections lie on their own cones) compete for the
# same depth value. The depth buffer has finite precision: when
# zoomed in far, fragments fall in the same quantisation interval
# and one or the other wins from frame to frame, i.e. the surface
# flickers.
#
# False = always drawn on top, in insertion order.
# ----------------------------------------------------------------
DEPTH_TEST_YIELD = False
DEPTH_TEST_SECTIONS = False
DEPTH_TEST_CRITICAL = False

# ----------------------------------------------------------------
# CRITICAL STATE CONE
# ----------------------------------------------------------------
CRITICAL_COLOR = BLUE
CRITICAL_SURF_RES = (24, 32)    # surface sampling
CRITICAL_MESH_SW = 1.0
CRITICAL_MESH_OPACITY = 0.5

HYDRO_COLOR = YELLOW
HYDRO_SW = 3.0                  # hydrostatic axis (space diagonal)

# ----------------------------------------------------------------
# YIELD SURFACE
#
# OpenGLSurfaceMesh does not draw the surface: it draws its
# WIREFRAME. Its resolution is INDEPENDENT of the source
# OpenGLSurface one (which only controls sampling) and defaults to
# (21, 21): a dense cage that in perspective reads as two
# overlapping surfaces.
# If the effect comes out inverted, swap the two values.
# ----------------------------------------------------------------
YIELD_SECTION_COLOR = WHITE
YIELD_SECTION_SW = 3.0

YIELD_MESH_COLOR = GRAY_B
YIELD_MESH_RES = (2, 24)        # (rings along p', generatrices)
YIELD_MESH_SW = 1.2
YIELD_MESH_OPACITY = 0.35
YIELD_SURF_RES = (16, 24)

# ----------------------------------------------------------------
# CONE SECTIONS (PI-plane)
# ----------------------------------------------------------------
SECTION_RESOLUTION = 121
SECTION_FILL_OPACITY = 0.18

SECTION_CRITICAL_SW = 3.0
SECTION_BOUND_SW = 2.0
SECTION_DIL_SW = 2.0
SECTION_BOUND_DASHES = 60
SECTION_DIL_DASHES = 110

# ----------------------------------------------------------------
# MERIDIAN PLANE
# Both FACTORs are fractions of p'max.
# ----------------------------------------------------------------
MERIDIAN_THETA = 0 * DEGREES
MERIDIAN_HALF_FACTOR = 1.5
MERIDIAN_P_MAX_FACTOR = 1.2
MERIDIAN_P_MIN = 0.0
MERIDIAN_FILL = 0.12
MERIDIAN_HALF_PLANE = False
MERIDIAN_HIDE_ON_ZOOM = True    # hidden during the mega zoom

# ----------------------------------------------------------------
# 3D LINE WIDTHS AND RADII
#
# Base values, all multiplied by sc3d() with the `detail` tracker:
# the choreography shrinks it during the mega zoom, the + / - keys
# adjust it by hand.
# ----------------------------------------------------------------
PATH3D_SW = 6.0             # 3D load path
DOT3D_R = 0.07              # current point
ALPHA_CENTER_R = 0.05       # yield surface centre
RATIO_DOT_R = 0.05          # tips of alpha^b and alpha^d
N_VECTOR_LENGTH = 250.0     # length of n, in stress units
N_VECTOR_SW = 3.0
N_VECTOR_TIP = 0.18
N_VECTOR_BASE_R = 0.04
RATIO_LINE_SW = 2.0         # alpha, alpha^b, alpha^d, alpha^c lines
R_LINE_SW = 2.5             # r line

# ----------------------------------------------------------------
# 2D PANELS
# These do not follow sc3d: they are fixed in frame and must stay
# stable.
# ----------------------------------------------------------------
PATH2D_SW = 4.0
DOT2D_R = 0.07

PQ_X_LEN = 4.2
PQ_Y_LEN = 3.2
PQ_SHIFT = RIGHT * 3.6 + UP * 1.9

QEPS_X_LEN = 4.2
QEPS_Y_LEN = 2.8
QEPS_SHIFT = RIGHT * 3.6 + DOWN * 2.1

PQ_BG_COLOR = RED
PQ_BG_OPACITY = 0.2
GRID_COLOR = GRAY_B
GRID_OPACITY = 0.8

CSL_SW = 3.0
WEDGE_FILL_OPACITY = 0.15
BOUND_LINE_SW = 2.0
DIL_LINE_SW = 2.0

READOUT_FONT = 20
READOUT_BUFF = 0.14


# ----------------------------------------------------------------
# INTRO AND OVERLAY
#
# Write (DrawBorderThenFill) breaks on VGroups of Tex under the
# OpenGL renderer: it interpolates mobject, starting copy and
# outline together, and the submobject families do not match.
#   "fade" -> soft appearance     "create" -> the text is drawn
#   "write" -> nicer but may crash
# ----------------------------------------------------------------
INTRO_TIME = 1.6
INTRO_TEXT_KIND = "fade"
INTRO_HOLD = 0.3

INFO_IN_RENDER = False      # filename and key hints in the saved video


# ----------------------------------------------------------------
# RENDER CHOREOGRAPHY
#
# ZOOM < 1 tightens. The three factors are cumulative.
# ----------------------------------------------------------------
ZOOM_A = 0.70               # initial tightening
ZOOM_B = 0.60               # tightening as follow engages
ZOOM_MEGA = 0.30            # mega zoom on the load point

DTHETA = 15 * DEGREES
THETA_FOLLOW = -2.0         # rotations, in multiples of DTHETA
THETA_MEGA = 1.5

DETAIL_ZOOM = 0.05          # 3D line widths during the mega zoom

# Weights, NOT fractions: they get normalised, so only their ratio
# matters. Total duration is fixed by the data.
W_ZOOM_IN = 0.08
W_HOLD_1 = 0.10
W_FOLLOW = 0.08
W_HOLD_2 = 0.10
W_MEGA = 0.10
W_HOLD_3 = 0.25
W_OUT = 0.14
W_TAIL = 0.05               # final tail, playback stopped


# ================================================================
# end of parameters
# ================================================================

_EASE_X = np.linspace(0.0, 1.0, 2001)
_EASE_Y = np.array([RATE(float(x)) for x in _EASE_X])

def ease_inv(y):
    return float(np.interp(np.clip(y, 0.0, 1.0), _EASE_Y, _EASE_X))

# ================================================================
# PATCH MOUSE
#
# manim/renderer/opengl_renderer_window.py
# {1: LEFT, 2: MOUSE, 4: RIGHT}.
# ================================================================

try:
    from manim.renderer.opengl_renderer_window import Window as _MWindow

    _KNOWN_BUTTONS = (1, 2, 4)

    def _guard_mouse(name):

        orig = getattr(_MWindow, name, None)

        if orig is None or getattr(orig, "_button_guarded", False):
            return

        def patched(self, x, y, button, modifiers, *args, **kwargs):

            if button not in _KNOWN_BUTTONS:
                return None

            return orig(self, x, y, button, modifiers, *args, **kwargs)

        patched._button_guarded = True

        setattr(_MWindow, name, patched)

    for _n in ("on_mouse_press", "on_mouse_release"):
        _guard_mouse(_n)

except Exception as _e:
    print(f"[warn] mouse patch not applied: {_e}")


class SANISAND(ThreeDScene):


    FILES = FILE_LIST
    FILE_GLOB = FILE_GLOB_PATTERN

    ARROW_STEP = KEY_ARROW_STEP
    ZOOM_FACTOR = KEY_ZOOM_FACTOR
    DETAIL_STEP = KEY_DETAIL_STEP
    DETAIL_MIN = KEY_DETAIL_MIN
    DETAIL_MAX = KEY_DETAIL_MAX

    def _cam(self):
        cam = getattr(self, "camera", None)
        frame = getattr(cam, "frame", None)

        return frame if frame is not None else cam

    def _zoom_at_point(self, factor):
        cam = self._cam()

        if cam is None or not hasattr(cam, "scale"):
            return

        cam.scale(factor, about_point=self._current_point3d())

        self._view_state["zoom"] *= factor


    def _toggle_follow(self):

        st = self._view_state

        st["follow"] = not st["follow"]

        if st["follow"]:

            cam = self._cam()

            if cam is not None:
                cam.move_to(self._current_point3d())



    def _reset_view(self):

        cam = self._cam()

        if cam is None or not hasattr(cam, "scale"):
            return

        st = self._view_state
        cam.scale(1.0 / st["zoom"], about_point=cam.get_center())
        cam.move_to(self._cam_home_center)

        st["zoom"] = 1.0
        st["follow"] = False

    def _scale_detail(self, factor):


        tracker = getattr(self, "_detail", None)

        if tracker is None:
            return

        val = float(np.clip(
            tracker.get_value() * factor,
            self.DETAIL_MIN,
            self.DETAIL_MAX,
        ))

        tracker.set_value(val)

    

    def _toggle_info(self):

        mobs = getattr(self, "_info_mobs", None)

        if not mobs:
            return

        self._info_visible = not getattr(self, "_info_visible", True)

        if self._info_visible:

            self.add_fixed_in_frame_mobjects(*mobs)

            fixer = getattr(self, "_fix_in_frame_deep", None)

            if fixer is not None:
                for m in mobs:
                    fixer(m)

        else:
            self.remove(*mobs)
    # ================================================================
    # TIMELINE
    # ================================================================

    def _seek(self, t_disp):

        if not getattr(self, "_timeline_ready", False):
            return

        t_disp = float(
            np.clip(t_disp, 0.0, self._n_datasets - 1e-6)
        )

        self._master_time.set_value(t_disp)

        if self._ui_state["mode"] == "sim":
            raw = self._n_datasets * ease_inv(
                t_disp / self._n_datasets
            )
        else:
            ds = int(np.floor(t_disp))
            raw = ds + ease_inv(t_disp - ds)

        self._raw_time.set_value(
            float(np.clip(raw, 0.0, self._n_datasets - 1e-6))
        )

    def _step_master_time(self, direction, modifiers=0):

        if not getattr(self, "_timeline_ready", False):
            return

        import pyglet.window.key as K

        # SHIFT -> x10
        # CTRL  -> single frame
        n_steps = self.ARROW_STEP

        if modifiers & K.MOD_SHIFT:
            n_steps *= 10

        if modifiers & K.MOD_CTRL:
            n_steps = 1

        st = self._ui_state
        mode = st["mode"]

        # ============================================================
        # SOLO
        # arrow inside dataset
        # ============================================================

        if mode == "solo":
            ds = int(st["active"])

            t = float(self._master_time.get_value())

            # tempo locale 0..1 del dataset
            local_t = np.clip(t - ds, 0.0, 1.0)

            step = n_steps / max(self._seg_lens[ds], 1)

            local_t += direction * step

            local_t = float(np.clip(local_t, 0.0, 1.0))

            self._seek(ds + local_t)

            return

        # ============================================================
        # SEQUENCE / ALL
        # ============================================================

        t = float(
            np.clip(
                self._master_time.get_value(),
                0,
                self._n_datasets - 1e-6
            )
        )

        if mode == "sim":
            ds = int(np.clip(st["active"], 0, self._n_datasets - 1))
            step = (
                self._n_datasets
                * n_steps
                / max(self._seg_lens[ds], 1)
            )

        else:

            ds = int(np.floor(t))

            step = n_steps / max(self._seg_lens[ds], 1)

        new_t = float(
            np.clip(
                t + direction * step,
                0,
                self._n_datasets - 1e-6
            )
        )

        self._seek(new_t)

    def _set_mode(self, mode, active=None):

        if not getattr(self, "_timeline_ready", False):
            return

        st = self._ui_state

        if active is not None:

            if not (0 <= active < self._n_datasets):
                return
            st["active"] = active

        st["mode"] = mode
        st["playing"] = False

        if mode == "solo":
            self._seek(float(st["active"]))
            st["stop_at"] = st["active"] + 1.0

        elif mode == "seq":
            self._seek(float(st["active"]))
            st["stop_at"] = None
        else:   # sim
            self._seek(0.0)
            st["stop_at"] = None

    def _select_dataset(self, active):

        if not getattr(self, "_timeline_ready", False):
            return

        if not (0 <= active < self._n_datasets):
            return

        st = self._ui_state

        st["active"] = active

        if st["mode"] == "sim":
            return

        st["playing"] = False

        st["stop_at"] = None

        self._seek(float(active))


    # ================================================================
    # KEYBOARD
    # ================================================================

    def on_key_press(self, symbol, modifiers):

        if not getattr(self, "_timeline_ready", False):
            return super().on_key_press(symbol, modifiers)

        import pyglet.window.key as K

        if symbol == K.I:
            self._toggle_info()
            return

        if symbol in (
                K.PLUS,
                K.EQUAL,
                getattr(K, "NUM_ADD", -1),
        ):
            self._scale_detail(self.DETAIL_STEP)
            return

        if symbol in (
                K.MINUS,
                getattr(K, "NUM_SUBTRACT", -1),
        ):
            self._scale_detail(1.0 / self.DETAIL_STEP)
            return

        if symbol in (K._0, getattr(K, "NUM_0", -1)):
            self._set_mode("sim")
            return

        # ================================================================
        # MODE KEYS
        # Q = SOLO
        # W = SEQUENCE
        # E = ALL / SIM
        # ================================================================

        if symbol == K.Q:
            self._set_mode(
                "solo",
                active=self._ui_state["active"],
            )

            return

        if symbol == K.W:
            self._set_mode(
                "seq",
                active=self._ui_state["active"],
            )

            return

        if symbol == K.E:
            self._set_mode("sim")

            return


        for k in range(9):

            if symbol == getattr(K, f"_{k + 1}", None) or \
                    symbol == getattr(K, f"NUM_{k + 1}", None):
                self._select_dataset(k)
                return

        # 3D ----------------

        if symbol == K.Z:
            self._zoom_at_point(1.0 / self.ZOOM_FACTOR)
            return

        if symbol == K.X:
            self._zoom_at_point(self.ZOOM_FACTOR)
            return

        if symbol == K.F:
            self._toggle_follow()
            return

        if symbol == K.V:
            self._reset_view()
            return

        #  timeline ----------------

        if symbol == K.SPACE:

            st = self._ui_state

            st["playing"] = not st["playing"]

            if st["playing"] and st.get("stop_at") is not None:

                if self._master_time.get_value() >= st["stop_at"] - 1e-5:
                    st["stop_at"] = None
            return

        if symbol == K.RIGHT:
            self._ui_state["playing"] = False
            self._step_master_time(+1, modifiers)
            return

        if symbol == K.LEFT:
            self._ui_state["playing"] = False
            self._step_master_time(-1, modifiers)
            return

        if symbol == K.HOME:

            self._ui_state["playing"] = False

            if self._ui_state["mode"] == "solo":

                ds = self._ui_state["active"]

                self._seek(float(ds))

            else:

                self._seek(0.0)

            return

        if symbol == K.END:

            self._ui_state["playing"] = False

            if self._ui_state["mode"] == "solo":

                ds = self._ui_state["active"]

                self._seek(float(ds) + 1.0 - 1e-6)

            else:

                self._seek(self._n_datasets - 1e-6)

            return

        return super().on_key_press(symbol, modifiers)

    # ================================================================
    # SCENE
    # ================================================================

    def construct(self):

        view_state = {
            "zoom": 1.0,
            "follow": False,
        }
        self._view_state = view_state

        # ============================================================
        # LOAD DATA
        # ============================================================

        def load_data(file):
            raw = pd.read_csv(
                file,
                sep=r"\s+",
                engine="python",
                header=None,
                skiprows=1
            )

            n_cols = raw.shape[1]
            # COLUMNS UMAT:
            # time, stran, stress, statev...
            base_cols = (
                ["time(1)", "time(2)"]
                + [f"stran({i})" for i in range(1, 7)]
                + [f"stress({i})" for i in range(1, 7)]
            )

            n_statev = n_cols - len(base_cols)
            statev_cols = [
                f"statev({i})"
                for i in range(1, n_statev + 1)
            ]

            raw.columns = base_cols + statev_cols
            data = raw

            def sv(i, default=0.0):
                col = f"statev({i})"
                if col in data.columns:
                    return data[col].to_numpy(dtype=float)
                return np.full(len(data), float(default))

            e = sv(1)
            pc = sv(8)
            q = sv(70)

            s11 = -data["stress(1)"].to_numpy(dtype=float)
            s22 = -data["stress(2)"].to_numpy(dtype=float)
            s33 = -data["stress(3)"].to_numpy(dtype=float)

            p = (s11 + s22 + s33) / 3.0

            eps1 = -data["stran(1)"].to_numpy(dtype=float)
            eps2 = -data["stran(2)"].to_numpy(dtype=float)
            eps3 = -data["stran(3)"].to_numpy(dtype=float)

            TENSOR_SIGN = -1.0
            def tensor3(i1, i2, i3):
                return TENSOR_SIGN * np.column_stack(
                    (sv(i1), sv(i2), sv(i3))
                )

            # n33
            N = tensor3(80, 81, 82)
            # alpha : back-stress ratio
            A = tensor3(83, 84, 85)
            # r33  : stress ratio
            R = tensor3(86, 87, 88)
            # alphab : bounding back-stress ratio
            AB = tensor3(75, 76, 77)
            # alphad : dilatancy back-stress ratio
            AD = tensor3(90, 91, 92)
            # alphac : critical back-stress ratio
            AC = tensor3(60, 61, 62)

            m_data = sv(89, default=0.0125)

            Mb_data = sv(72)
            Md_data = sv(71)
            cos3t = sv(93)

            return {
                "e": e,
                "p": p,
                "q": q,
                "pc": pc,

                "eps1": eps1,
                "eps2": eps2,
                "eps3": eps3,

                "s11": s11,
                "s22": s22,
                "s33": s33,

                "N": N,
                "R": R,
                "A": A,
                "AB": AB,
                "AD": AD,
                "AC": AC,

                "m": m_data,
                "Mb": Mb_data,
                "Md": Md_data,
                "cos3t": cos3t,
            }

        files = list(self.FILES)

        def sort_key(f):
            name = os.path.basename(f)

            numbers = [
                int(x)
                for x in re.findall(r"\d+", name)
            ]

            return numbers


        if not files:
            files = sorted(
                glob.glob(self.FILE_GLOB),
                key=sort_key
            )
        if len(files) > 9:
            print(f"  NOTA: {len(files)} key limit 9")
        for k, f in enumerate(files):
            print(f"    [{k + 1}] {f}")

        colors = [
            PALETTE[i % len(PALETTE)] for i in range(len(files))
        ]
        datasets = [load_data(f) for f in files]
        ds_labels = [
            os.path.splitext(os.path.basename(f))[0] for f in files
        ]
        self._ds_labels = ds_labels

        # AUTO RANGE

        def _step(raw, allowed=(1, 2, 2.5, 5, 10)):
            if raw <= 0:
                return 1.0

            exp = np.floor(np.log10(raw))
            frac = raw / 10 ** exp

            for a in allowed:
                if frac <= a * 1.000001:
                    return a * 10 ** exp

            return 10 * 10 ** exp

        def _range(values, n_ticks=5, from_zero=None,
                   margin=0.05, tight=True):
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v)]

            if v.size == 0:
                return 0.0, 1.0, 0.5

            vmin = float(v.min())
            vmax = float(v.max())

            span = vmax - vmin

            if span <= 0:
                span = max(abs(vmax), 1e-9)

            if from_zero is None:
                from_zero = (
                    (vmin >= 0.0 and vmin <= 0.5 * span)
                    or (vmax <= 0.0 and abs(vmax) <= 0.5 * span)
                )

            lo_t = vmin - span * margin
            hi_t = vmax + span * margin

            if from_zero:
                if vmin >= 0:
                    lo_t = 0.0
                else:
                    hi_t = 0.0

            step = _step((hi_t - lo_t) / max(n_ticks, 1))

            lo = (
                0.0 if (from_zero and vmin >= 0)
                else np.floor(lo_t / step) * step
            )

            if tight:
                hi = 0.0 if (from_zero and vmax <= 0) else hi_t
            else:
                hi = (
                    0.0 if (from_zero and vmax <= 0)
                    else np.ceil(hi_t / step) * step
                )

            if hi <= lo:
                hi = lo + step

            return float(lo), float(hi), float(step)

        def tick_decimals(step):

            if step <= 0:
                return 0

            return int(max(0, np.ceil(-np.log10(step)) +
                           (1 if abs(step / 10 ** np.floor(np.log10(step))
                                     - 2.5) < 1e-9 else 0)))

        def tick_values(lo, hi, step):

            n = int(np.floor((hi - lo) / step + 1e-9))

            return [lo + k * step for k in range(n + 1)]

        all_e = np.concatenate([d["e"] for d in datasets])
        all_p = np.concatenate([d["p"] for d in datasets])
        all_q = np.concatenate([d["q"] for d in datasets])
        all_eps1 = np.concatenate([d["eps1"] for d in datasets])

        p_axis_min, p_axis_max, p_step = _range(
            all_p, 4, from_zero=True
        )

        q_axis_min, q_axis_max, q_step = _range(
            all_q, 5, from_zero=True
        )
        eps_axis_min, eps_axis_max, eps_step = _range(all_eps1, 5)
        e_axis_min, e_axis_max, e_step = _range(all_e, 5)

        p_labels = tick_values(p_axis_min, p_axis_max, p_step)
        q_labels = tick_values(q_axis_min, q_axis_max, q_step)

        p_dec = tick_decimals(p_step)
        q_dec = tick_decimals(q_step)
        eps_dec = tick_decimals(eps_step)

        _p_data = float(all_p.max())
        _p_needed = (1.0 + P3D_MARGIN) * _p_data

        p_step_3d = _step(_p_data / max(P3D_TICKS, 1))

        if P3D_TIGHT_RANGE:
            p_max_yield = _p_needed
        else:
            p_max_yield = float(
                np.ceil(_p_needed / p_step_3d) * p_step_3d
            )

        def g_theta(theta, c):
            return (
                (2 * c)
                / ((1 + c) - (1 - c) * np.cos(3 * theta))
            )

        # ============================================================
        # 3D SURFACE PLOT
        # ============================================================

        axes3d = ThreeDAxes(
            x_range=[0, p_max_yield, p_step_3d],
            y_range=[0, p_max_yield, p_step_3d],
            z_range=[0, p_max_yield, p_step_3d],
            x_length=AXES3D_LENGTH,
            y_length=AXES3D_LENGTH,
            z_length=AXES3D_LENGTH,
            axis_config={
                "include_tip": False,
            },
        ).scale(AXES3D_SCALE)

        def swap_xz(v):
            v = np.asarray(v, dtype=float)
            return v[[2, 1, 0]]

        labels3d = axes3d.get_axis_labels(
            Tex(r"$\sigma_3$", font_size=26),
            Tex(r"$\sigma_2$", font_size=26),
            Tex(r"$\sigma_1$", font_size=26)
        )
        labels3d[2].rotate(
            80 * DEGREES,
            axis=OUT,
        )

        self.set_camera_orientation(phi=PHI0, theta=THETA0)

        self.camera.light_source.move_to(LIGHT_SOURCE_POS)

        for k, lab in enumerate(labels3d):

            pos = np.zeros(3)
            pos[k] = AXES3D_LABEL_OUT * p_max_yield

            lab.move_to(axes3d.c2p(*pos))

        self.add(axes3d, labels3d)

        # ============================================================
        # PARAMETRIC SURFACE
        #
        # sigma = p' I + s
        #
        # s = q * u(theta)
        #
        # u(theta) = sqrt(2/3) *
        #            [cos(theta),
        #             cos(theta-2pi/3),
        #             cos(theta+2pi/3)]
        #
        # ||u|| = 1, tr(u) = 0
        #
        # quindi q = ||s||.
        # ============================================================

        def dev_dir(theta):

            return np.sqrt(2 / 3) * np.array([
                np.cos(theta),
                np.cos(theta - 2 * np.pi / 3),
                np.cos(theta + 2 * np.pi / 3),
            ])

        # ============================================================
        # CONE CRITICAL / BOUNDING
        # ============================================================
        SQ23 = np.sqrt(2.0 / 3.0)
        def cone_surface(p, theta, M_val):

            return (
                    p * np.ones(3)
                    + p * SQ23 * M_val * g_theta(theta, c) * dev_dir(theta)
            )
        # ============================================================
        # YIELD CONE
        #
        # f = ||s - p*alpha|| - sqrt(2/3)*p*m = 0
        # ============================================================
        def cone_surface_circular(p, theta, m_val, alpha):

            alpha = np.asarray(alpha)

            return (
                    p * (np.ones(3) + alpha)
                    + p * SQ23 * m_val * dev_dir(theta)
            )
        # SCALE YIELD

        def magnify_r(r_vec, alpha_vec, scale=M_YIELD_DISPLAY_SCALE):

            r_vec = np.asarray(r_vec, dtype=float)
            alpha_vec = np.asarray(alpha_vec, dtype=float)

            return alpha_vec + scale * (r_vec - alpha_vec)

        def stress_point_display(p, r_vec, alpha_vec):

            return p * (1.0 + magnify_r(r_vec, alpha_vec))



        def set_depth_test_deep(mob, enabled):

            for sub in mob.get_family():

                if enabled:
                    if hasattr(sub, "apply_depth_test"):
                        sub.apply_depth_test()

                else:
                    if hasattr(sub, "deactivate_depth_test"):
                        sub.deactivate_depth_test()

            return mob

        critical_surface_cone = OpenGLSurface(
            lambda u, v: axes3d.c2p(*cone_surface(u, v, M)),
            u_range=[0, p_max_yield],
            v_range=[0, TAU],
            resolution=CRITICAL_SURF_RES,
            fill_opacity=0.4,
            stroke_width=0,
            checkerboard_colors=False,
            fill_color=CRITICAL_COLOR,
            gloss=0.35,
            shadow=0.4,
        )

        critical_surface_mesh = OpenGLSurfaceMesh(
            critical_surface_cone,
            stroke_width=CRITICAL_MESH_SW,
            stroke_color=CRITICAL_COLOR,
            stroke_opacity=CRITICAL_MESH_OPACITY,
        )

        set_depth_test_deep(critical_surface_mesh, DEPTH_TEST_CRITICAL)

        hydro_axis = Line(
            axes3d.c2p(0, 0, 0),
            axes3d.c2p(p_max_yield, p_max_yield, p_max_yield),
            color=HYDRO_COLOR,
            stroke_width=HYDRO_SW,
        )


        self.add(critical_surface_mesh, hydro_axis)

        cone_group = Group(
            axes3d,
            labels3d,
            critical_surface_mesh,
            hydro_axis,
        )

        cone_group.move_to(CONE_GROUP_SHIFT)
        cone_group.scale(CONE_GROUP_SCALE)

        # ============================================================
        # P'-Q
        # ============================================================

        pq_axes = Axes(
            x_range=[p_axis_min, p_axis_max, p_step],
            y_range=[q_axis_min, q_axis_max, q_step],
            x_length=PQ_X_LEN,
            y_length=PQ_Y_LEN,
            axis_config={
                "include_tip": False,
                "font_size": 16
            },
        ).shift(PQ_SHIFT)

        pq_axes.add_coordinates(
            p_labels,
            q_labels,
            num_decimal_places=max(p_dec, q_dec),
            font_size=14
        )

        # red background
        pq_background = Polygon(
            pq_axes.c2p(p_axis_min, q_axis_min),
            pq_axes.c2p(p_axis_max, q_axis_min),
            pq_axes.c2p(p_axis_max, q_axis_max),
            pq_axes.c2p(p_axis_min, q_axis_max),
            color=PQ_BG_COLOR,
            fill_color=PQ_BG_COLOR,
            fill_opacity=PQ_BG_OPACITY,
            stroke_width=1,
            stroke_color=PQ_BG_COLOR,
            stroke_opacity=0.5,
        )

        pq_grid_v = VGroup()

        for x in p_labels:
            line = Line(
                pq_axes.c2p(x, q_axis_min),
                pq_axes.c2p(x, q_axis_max),
                stroke_color=GRID_COLOR,
                stroke_width=1,
                stroke_opacity=GRID_OPACITY,
            )
            pq_grid_v.add(line)

        pq_grid_h = VGroup()

        for y in q_labels:
            line = Line(
                pq_axes.c2p(p_axis_min, y),
                pq_axes.c2p(p_axis_max, y),
                stroke_color=GRID_COLOR,
                stroke_width=1,
                stroke_opacity=GRID_OPACITY,
            )
            pq_grid_h.add(line)

        pq_grid = VGroup(
            pq_grid_v,
            pq_grid_h,
        )

        pq_labels = pq_axes.get_axis_labels(
            Tex("p'", font_size=28),
            Tex("q", font_size=28)
        )

        def e_c(p):
            return (
                e0 - lambda_c * (np.maximum(p, 1e-6) / p_atm) ** xi
            )

        # ============================================================
        # Q - EPS_A
        # ============================================================

        qeps_axes = Axes(
            x_range=[eps_axis_min, eps_axis_max, eps_step],
            y_range=[q_axis_min, q_axis_max, q_step],
            x_length=QEPS_X_LEN,
            y_length=QEPS_Y_LEN,
            axis_config={
                "include_tip": False,
                "font_size": 16
            },
        ).shift(QEPS_SHIFT)

        qeps_axes.x_axis.add_numbers(
            font_size=14,
            num_decimal_places=eps_dec
        )

        qeps_axes.y_axis.add_numbers(
            font_size=14,
            num_decimal_places=q_dec
        )

        qeps_labels = qeps_axes.get_axis_labels(
            Tex(r"$\varepsilon_a$ (\%)", font_size=24),
            Tex("q", font_size=24)
        )

        # -----  q - eps_a -----

        eps_labels = tick_values(
            eps_axis_min, eps_axis_max, eps_step
        )

        qeps_grid_v = VGroup()

        for x in eps_labels:
            line = Line(
                qeps_axes.c2p(x, q_axis_min),
                qeps_axes.c2p(x, q_axis_max),
                stroke_color=GRID_COLOR,
                stroke_width=1,
                stroke_opacity=GRID_OPACITY,
            )
            qeps_grid_v.add(line)

        qeps_grid_h = VGroup()

        for y in q_labels:
            line = Line(
                qeps_axes.c2p(eps_axis_min, y),
                qeps_axes.c2p(eps_axis_max, y),
                stroke_color=GRID_COLOR,
                stroke_width=1,
                stroke_opacity=GRID_OPACITY,
            )
            qeps_grid_h.add(line)

        qeps_grid = VGroup(
            qeps_grid_v,
            qeps_grid_h,
        )

        # ============================================================
        # LEGEND
        # ============================================================

        def legend_entry(style, label_text, color=GRAY_B):
            if style == "solid":
                sample = Line(
                    ORIGIN, RIGHT * 0.5,
                    color=color, stroke_width=3
                )

            elif style == "dashed_wide":

                sample = DashedLine(
                    ORIGIN, RIGHT * 0.5,
                    color=color, stroke_width=2, dash_length=0.12
                )

            else:

                sample = DashedLine(
                    ORIGIN, RIGHT * 0.5,
                    color=color, stroke_width=2, dash_length=0.04
                )

            label = Tex(
                label_text, font_size=18, color=WHITE
            ).next_to(sample, RIGHT, buff=0.15)

            return VGroup(sample, label)

        legend_group = VGroup(
            legend_entry("solid", "$M$", color=BLUE),
            legend_entry("dashed_wide", "$M_b$ ", color=ORANGE),
            legend_entry("dashed_narrow", "$M_d$", color=PURPLE_B),
        )

        legend_group.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend_group.scale(1)
        legend_group.next_to(pq_axes, RIGHT, buff=0.2)
        legend_group.shift(LEFT * 0.4)

        # ============================================================
        # LEGEND 3D
        # ============================================================

        legend3d_group = VGroup(
            legend_entry("solid", r"$r$", color=GREEN_B),
            legend_entry("dashed_wide", r"$\alpha$", color=WHITE),
            legend_entry("dashed_wide", r"$\alpha^b$", color=ORANGE),
            legend_entry("dashed_narrow", r"$\alpha^d$", color=PURPLE_B),
            legend_entry("solid", r"$n$", color=WHITE),
        )

        legend3d_group.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend3d_group.scale(1)
        legend3d_group.to_corner(UL, buff=0.35)

        ds_title = Tex(
            f"[1/{len(ds_labels)}] "
            + ds_labels[0].replace("_", r"\_"),
            font_size=10,
            color=colors[0],
        ).to_edge(UP, buff=0.25)

        ds_title_state = {"ds": None}

        MODE_TAG = {
            "solo": "single",
            "seq": "sequence",
            "sim": "all",
        }

        def update_ds_title(mob, dt=0):
            ds, _ = ds_and_idx()
            mode = ui_state["mode"]
            key = (ds, mode)
            if key == ds_title_state["ds"]:
                return

            ds_title_state["ds"] = key

            if mode == "sim":
                text = (
                    f"all ({len(ds_labels)}) --- "
                    + ds_labels[ds].replace("_", r"\_")
                )
            else:
                text = (
                    f"[{ds + 1}/{len(ds_labels)}] "
                    + ds_labels[ds].replace("_", r"\_")
                )

            new_tex = Tex(
                text + r"  \small (" + MODE_TAG[mode] + ")",
                font_size=10,
                color=colors[ds],
            )

            new_tex.move_to(mob)

            mob.become(new_tex)


            fix_in_frame_deep(mob)

        keys_hint = Tex(
            r"1--9 \quad 0 all "
            r"\quad SPACE play \quad "
            r"$\leftarrow\rightarrow$ step \quad "
            r"Z/X zoom \quad F follow \quad V reset "
            r"\quad $+/-$ detail \quad I info",
            font_size=10,
            color=GRAY_B,
        ).to_edge(DOWN, buff=0.15)

        d0 = datasets[0]

        readout_specs = []

        def add_readout_row(
            column, tex, init_vals, getter,
            color=WHITE, dec=4, sign=False
        ):

            init_vals = np.atleast_1d(np.asarray(init_vals, dtype=float))

            lab = Tex(tex, font_size=READOUT_FONT, color=color)

            nums = [
                DecimalNumber(
                    float(v),
                    num_decimal_places=dec,
                    font_size=READOUT_FONT,
                    color=color,
                    include_sign=sign,
                )
                for v in init_vals
            ]

            row = VGroup(lab, *nums).arrange(RIGHT, buff=READOUT_BUFF)

            column.append((row, nums))

            readout_specs.append((nums, getter))

            return row


        col_left = []

        add_readout_row(
            col_left, r"$e$", d0["e"][0],
            lambda d, i: [d["e"][i]],
            color=YELLOW, dec=4,
        )

        add_readout_row(
            col_left, r"$p'$", d0["p"][0],
            lambda d, i: [d["p"][i]],
            dec=1,
        )

        add_readout_row(
            col_left, r"$q$", d0["q"][0],
            lambda d, i: [d["q"][i]],
            dec=1,
        )

        add_readout_row(
            col_left, r"$\alpha$", d0["A"][0],
            lambda d, i: d["A"][i],
            dec=4, sign=True,
        )

        add_readout_row(
            col_left, r"$\alpha^b$", d0["AB"][0],
            lambda d, i: d["AB"][i],
            color=ORANGE, dec=4, sign=True,
        )

        add_readout_row(
            col_left, r"$\alpha^d$", d0["AD"][0],
            lambda d, i: d["AD"][i],
            color=PURPLE_B, dec=4, sign=True,
        )

        col_right = []

        add_readout_row(
            col_right, r"$M_b$", M,
            lambda d, i: [
                Mb_at(d, i)
            ],
            color=ORANGE, dec=4,
        )

        add_readout_row(
            col_right, r"$M_d$", M,
            lambda d, i: [
                Md_at(d, i)
            ],
            color=PURPLE_B, dec=4,
        )

        add_readout_row(
            col_right, r"$n$", d0["N"][0],
            lambda d, i: d["N"][i],
            dec=4, sign=True,
        )

        add_readout_row(
            col_right, r"$r$", d0["R"][0],
            lambda d, i: d["R"][i],
            color=GREEN_B, dec=4, sign=True,
        )

        add_readout_row(
            col_right, r"$\varepsilon_a$", d0["eps1"][0],
            lambda d, i: [d["eps1"][i]],
            dec=5, sign=True,
        )

        def build_column(rows):
            group = VGroup(*[r for r, _ in rows])

            group.arrange(DOWN, aligned_edge=LEFT, buff=0.13)

            x_num = max(nums[0].get_left()[0] for _, nums in rows)

            for _, nums in rows:

                shift = x_num - nums[0].get_left()[0]

                for num in nums:
                    num.shift(RIGHT * shift)

            return group

        readout_left = build_column(col_left)
        readout_right = build_column(col_right)

        readout = VGroup(readout_left, readout_right)

        readout.arrange(RIGHT, aligned_edge=UP, buff=0.55)

        readout.to_corner(DL, buff=0.35)

        def update_readout(mob, dt=0):

            ds, idx = ds_and_idx()

            d = datasets[ds]

            for nums, getter in readout_specs:

                vals = getter(d, idx)

                for num, v in zip(nums, vals):
                    num.set_value(float(v))


            fix_in_frame_deep(readout)

        self.add_fixed_in_frame_mobjects(
            pq_background,
            pq_grid,
            pq_axes,
            pq_labels,
            qeps_grid,
            qeps_axes,
            qeps_labels,
            readout,
            legend_group,
            legend3d_group,
            ds_title,
            keys_hint,
        )

        if (
            str(config.renderer) == "RendererType.OPENGL"
            or str(config.renderer).lower() == "opengl"
        ):

            from manim.mobject.opengl.opengl_vectorized_mobject import (
                OpenGLVMobject
            )

            PathBase = OpenGLVMobject

        else:

            PathBase = VMobject

        # ============================================================
        # ALWAYS REDRAW + FIXED IN FRAME
        # ============================================================

        def fix_in_frame_deep(mob):

            for sub in mob.get_family():

                if hasattr(sub, "fix_in_frame"):
                    sub.fix_in_frame()

            return mob

        self._fix_in_frame_deep = fix_in_frame_deep

        def always_redraw_fixed(func):

            mob = fix_in_frame_deep(func())

            def updater(m):

                m.become(func())
                fix_in_frame_deep(m)

            mob.add_updater(updater)

            return mob

        def always_rebuild(func):
            """
            Variante di always_redraw senza become().

            become() passa da align_data, che PAREGGIA le due famiglie
            inserendo sottomobject nulli quando le lunghezze non
            combaciano, invece di sostituirle. Su una OpenGLSurfaceMesh
            (che e' un VGroup di linee) restano cosi' appese le linee
            del passo precedente e la superficie appare sdoppiata.

            Qui i sottomobject vengono rimpiazzati di netto.
            """

            container = func()

            def updater(m):

                new = func()

                if not new.submobjects:
                    # niente figli: non c'e' altro modo che become
                    m.become(new)
                    return

                subs = list(new.submobjects)

                if hasattr(m, "set_submobjects"):
                    m.set_submobjects(subs)
                else:
                    m.submobjects = subs

            container.add_updater(updater)

            return container

        # ============================================================
        # MASTER TIME
        #
        # raw_time     : avanza LINEARMENTE con dt
        # master_time  : RATE(raw_time), e' quello che indicizza i dati
        # ============================================================

        n_datasets = len(datasets)

        seg_lens = [len(d["p"]) - 1 for d in datasets]

        raw_time = ValueTracker(0)
        master_time = ValueTracker(0)

        # Miscelazione continua del follow, 0 = camera libera,
        # 1 = camera agganciata al punto di carico. Il booleano
        # view_state["follow"] resta per il tasto F, che aggancia
        # di scatto: e' quello che serve in interattivo.
        follow_blend = ValueTracker(0.0)


        ui_state = {
            "playing": False,
            "mode": "seq",
            "active": 0,
            "stop_at": None,
        }

        def apply_rate(r):
            """Tempo grezzo -> tempo visualizzato."""

            r = float(np.clip(r, 0.0, n_datasets - 1e-6))

            if ui_state["mode"] == "sim":
                # easing su tutta la durata
                return n_datasets * RATE(r / n_datasets)

            # easing su ogni singolo dataset
            ds = int(np.floor(r))

            return ds + RATE(r - ds)

        def _clock():

            return float(
                np.clip(
                    master_time.get_value(), 0, n_datasets - 1e-6
                )
            )

        def ds_and_idx():

            t = _clock()
            mode = ui_state["mode"]

            if mode == "sim":

                prog = t / n_datasets

                ds = int(np.clip(ui_state["active"], 0, n_datasets - 1))

                idx = int(
                    np.clip(round(prog * seg_lens[ds]),
                            0, seg_lens[ds])
                )

                return ds, idx

            if mode == "solo":

                ds = int(np.clip(ui_state["active"], 0, n_datasets - 1))

                local_t = np.clip(t - ds, 0, 1)

                idx = int(
                    np.clip(local_t * seg_lens[ds], 0, seg_lens[ds])
                )

                return ds, idx

            # seq
            ds = int(np.floor(t))

            local_t = np.clip(t - ds, 0, 1)

            idx = int(
                np.clip(local_t * seg_lens[ds], 0, seg_lens[ds])
            )

            ui_state["active"] = ds

            return ds, idx

        def n_shown_for(ds_index):

            t = _clock()

            mode = ui_state["mode"]

            if mode == "sim":

                prog = t / n_datasets

                return max(
                    int(round(prog * seg_lens[ds_index])) + 1, 2
                )

            if mode == "solo":

                if ds_index != ui_state["active"]:
                    return 0

                local_t = np.clip(t - ds_index, 0, 1)

                return max(
                    int(round(local_t * seg_lens[ds_index])) + 1, 2
                )

            # seq
            cur_ds = int(np.floor(t))

            if ds_index < cur_ds:
                return seg_lens[ds_index] + 1

            if ds_index > cur_ds:
                return 0

            local_t = np.clip(t - cur_ds, 0, 1)

            return max(
                int(round(local_t * seg_lens[ds_index])) + 1, 2
            )

        def m_at(d, idx):

            val = float(d["m"][idx])

            return val if val > 1e-12 else m_yield

        def psi_at(d, idx):
            return d["e"][idx] - e_c(d["p"][idx])

        def current_point3d():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            point = stress_point_display(
                d["p"][idx],
                d["R"][idx],
                d["A"][idx],
            )

            return axes3d.c2p(*swap_xz(point))

        self._current_point3d = current_point3d


        def Mb_at(d, idx):
            v = float(d["Mb"][idx])
            return v

        def Md_at(d, idx):
            v = float(d["Md"][idx])

            return v

        # ============================================================
        # CSL PLANE p'-q
        # ============================================================

        def make_csl_line():

            ds, idx = ds_and_idx()


            Mth = M

            return Line(
                pq_axes.c2p(0, 0),
                pq_axes.c2p(p_axis_max, max(Mth * p_axis_max, 0)),
                color=CRITICAL_COLOR,
                stroke_width=CSL_SW,
            )


        detail = ValueTracker(1.0)

        _sc3d_seen = set()

        self.add(detail)
        self._detail = detail

        def sc3d(base, floor=1e-3):


            return max(base * float(detail.get_value()), floor)

        path3d_list = []

        for i, ds in enumerate(datasets):

            color = colors[i]

            e = ds["e"]
            p = ds["p"]
            q = ds["q"]
            eps1 = ds["eps1"]

            s11 = ds["s11"]
            s22 = ds["s22"]
            s33 = ds["s33"]

            pts3d_full = [
                axes3d.c2p(*swap_xz([s11[j], s22[j], s33[j]]))
                for j in range(len(s11))
            ]

            pts_qp_full = [
                pq_axes.c2p(p[j], q[j]) for j in range(len(p))
            ]

            pts_qeps_full = [
                qeps_axes.c2p(eps1[j], q[j])
                for j in range(len(eps1))
            ]

            def make_path_updater(
                points_full,
                ds_index=i,
                color=color,
                sw=PATH2D_SW,
                scaled=False,
            ):

                def updater():


                    width = sc3d(sw) if scaled else sw

                    n_show = n_shown_for(ds_index)
                    if n_show < 1:

                        path = PathBase(
                            color=color,
                            stroke_width=width,
                            stroke_opacity=0,
                        )

                        path.set_points_as_corners(
                            [points_full[0], points_full[0]]
                        )

                        return path

                    pts = points_full[:max(n_show, 2)]

                    path = PathBase(color=color, stroke_width=width)
                    path.set_points_as_corners(pts)

                    return path

                return updater

            path3d = always_redraw(
                make_path_updater(
                    pts3d_full,
                    sw=PATH3D_SW,
                    scaled=True,
                )
            )

            path_qp = always_redraw_fixed(
                make_path_updater(pts_qp_full)
            )

            path_qeps = always_redraw_fixed(
                make_path_updater(pts_qeps_full)
            )

            self.add(path3d)

            self.add_fixed_in_frame_mobjects(path_qp, path_qeps)

            path3d_list.append(path3d)

        # ============================================================
        # CURRENT POSITION DOTS
        # ============================================================

        def make_pos_dot_3d():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            return Dot3D(
                axes3d.c2p(*swap_xz([
                    d["s11"][idx],
                    d["s22"][idx],
                    d["s33"][idx],
                ])),
                radius=sc3d(DOT3D_R),
                color=colors[ds]
            )

        def make_pos_dot_qp():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            dot = Dot(
                pq_axes.c2p(d["p"][idx], d["q"][idx]),
                radius=DOT2D_R,
                color=colors[ds]
            )

            dot.z_index = 4

            return dot

        def make_pos_dot_qeps():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            dot = Dot(
                qeps_axes.c2p(d["eps1"][idx], d["q"][idx]),
                radius=DOT2D_R,
                color=colors[ds]
            )

            dot.z_index = 4

            return dot

        # ============================================================
        # WEDGE
        # ============================================================

        def make_pos_wedge():

            ds, idx = ds_and_idx()
            d = datasets[ds]
            p_cur = d["p"][idx]
            q_cur = d["q"][idx]
            eta_cur = q_cur / p_cur if p_cur != 0 else 0.0
            m_cur = m_at(d, idx)
            color = colors[ds]
            origin_pt = pq_axes.c2p(0, 0)

            up_end = pq_axes.c2p(
                p_axis_max,
                max((eta_cur + m_cur) * p_axis_max, 0)
            )

            dn_end = pq_axes.c2p(
                p_axis_max,
                max((eta_cur - m_cur) * p_axis_max, 0)
            )

            fill = Polygon(
                origin_pt, up_end, dn_end,
                color=color,
                fill_color=color,
                fill_opacity=WEDGE_FILL_OPACITY,
                stroke_width=0
            )

            line_up = Line(
                origin_pt, up_end, color=color, stroke_width=1.5
            )

            line_dn = Line(
                origin_pt, dn_end, color=color, stroke_width=1.5
            )

            g = VGroup(fill, line_up, line_dn)

            g.z_index = 2

            return g

        # ============================================================
        # BOUNDING LINE (plane p'-q)
        #
        # M_b = M * g(theta) * exp(-n_b * psi)
        # ============================================================

        def make_pos_bound_line():

            ds, idx = ds_and_idx()
            d = datasets[ds]
            Mb_curr = Mb_at(d, idx)
            line = DashedVMobject(
                Line(
                    pq_axes.c2p(0, 0),
                    pq_axes.c2p(
                        p_axis_max, max(Mb_curr * p_axis_max, 0)
                    ),
                    color=ORANGE,
                    stroke_width=BOUND_LINE_SW,
                    stroke_opacity=0.7
                ),
                num_dashes=20,
                dashed_ratio=0.5,
            )

            line.z_index = 2

            return line

        # ============================================================
        # DILATANCY LINE (plane p'-q)
        #
        # M_d = M * g(theta) * exp(+n_d * psi)
        # ============================================================

        def make_pos_dil_line():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            Md_curr = Md_at(d, idx)

            line = DashedVMobject(
                Line(
                    pq_axes.c2p(0, 0),
                    pq_axes.c2p(
                        p_axis_max, max(Md_curr * p_axis_max, 0)
                    ),
                    color=PURPLE_B,
                    stroke_width=DIL_LINE_SW,
                    stroke_opacity=0.7
                ),
                num_dashes=40,
                dashed_ratio=0.25,
            )

            line.z_index = 2

            return line

        def make_pos_yield_mesh():

            ds, idx = ds_and_idx()
            d = datasets[ds]
            alpha_cur = d["A"][idx]
            m_cur = m_at(d, idx) * M_YIELD_DISPLAY_SCALE

            surf = OpenGLSurface(
                lambda u, v: axes3d.c2p(
                    *swap_xz(
                        cone_surface_circular(
                            u, v, m_cur, alpha_cur
                        )
                    )
                ),
                u_range=[0, p_max_yield],
                v_range=[0, TAU],
                resolution=YIELD_SURF_RES,
                fill_opacity=0,
                stroke_width=0,
                checkerboard_colors=False,
                gloss=0,
                shadow=0,
            )

            mesh = OpenGLSurfaceMesh(
                surf,
                resolution=YIELD_MESH_RES,
                stroke_width=sc3d(YIELD_MESH_SW),
                stroke_color=YIELD_MESH_COLOR,
                stroke_opacity=YIELD_MESH_OPACITY,
            )

            return set_depth_test_deep(mesh, DEPTH_TEST_YIELD)

        # ============================================================
        # MERIDIAN PLANE

        meridian_half = MERIDIAN_HALF_FACTOR * p_max_yield
        meridian_p_max = MERIDIAN_P_MAX_FACTOR * p_max_yield

        def make_meridian_plane():

            u = dev_dir(MERIDIAN_THETA)
            hyd = np.array([1.0, 1.0, 1.0])

            w_neg = 0.0 if MERIDIAN_HALF_PLANE else -meridian_half

            corners = [
                axes3d.c2p(*swap_xz(MERIDIAN_P_MIN * hyd + w_neg * u)),
                axes3d.c2p(*swap_xz(meridian_p_max * hyd + w_neg * u)),
                axes3d.c2p(*swap_xz(meridian_p_max * hyd + meridian_half * u)),
                axes3d.c2p(*swap_xz(MERIDIAN_P_MIN * hyd + meridian_half * u)),
            ]

            fill = Polygon(
                *corners,
                color=RED_A,
                fill_color=RED_A,
                fill_opacity=MERIDIAN_FILL,

            )

            edge = Polygon(
                *corners,
                color=RED,
                stroke_width=2,
                stroke_opacity=0.8,
                fill_opacity=0,
            )

            return VGroup(fill, edge)

        # ============================================================
        # Section of cones
        #
        #     x(theta) = cone_surface(p', theta, M_eff)
        #
        #     M_eff = M               -> critical
        #     M_eff = M*exp(-nb*psi)  -> bounding
        #     M_eff = M*exp(+nd*psi)  -> dilatancy
        #
        # ============================================================

        SECTION_THETAS = np.linspace(0, TAU, SECTION_RESOLUTION)

        def section_points(p_cur, M_eff):

            return [
                axes3d.c2p(
                    *swap_xz(cone_surface(p_cur, t, M_eff))
                )
                for t in SECTION_THETAS
            ]

        def section_curve(pts, color, stroke_width=SECTION_CRITICAL_SW,
                          opacity=1.0):

            curve = PathBase(
                color=color,
                stroke_width=sc3d(stroke_width),
                stroke_opacity=opacity,
            )

            curve.set_points_as_corners(pts)

            return set_depth_test_deep(curve, DEPTH_TEST_SECTIONS)

        def make_critical_section():

            ds, idx = ds_and_idx()

            p_cur = datasets[ds]["p"][idx]

            pts = section_points(p_cur, M)

            outline = section_curve(
                pts, CRITICAL_COLOR, stroke_width=SECTION_CRITICAL_SW
            )

            if SECTION_FILL_OPACITY <= 0:
                return outline

            fill = Polygon(
                *pts[:-1],
                color=BLUE,
                fill_color=BLUE,
                fill_opacity=SECTION_FILL_OPACITY,
                stroke_width=0,
            )

            return Group(fill, outline)

        def make_bounding_section():

            ds, idx = ds_and_idx()
            d = datasets[ds]
            Mb = float(d["Mb"][idx])

            return DashedVMobject(
                section_curve(
                    section_points(d["p"][idx], Mb),
                    ORANGE,
                    stroke_width=SECTION_BOUND_SW,
                    opacity=0.9,
                ),
                num_dashes=SECTION_BOUND_DASHES,
                dashed_ratio=0.55,
            )

        def make_dilatancy_section():

            ds, idx = ds_and_idx()
            d = datasets[ds]

            Md = float(d["Md"][idx])

            return DashedVMobject(
                section_curve(
                    section_points(d["p"][idx], Md),
                    PURPLE_B,
                    stroke_width=SECTION_DIL_SW,
                    opacity=0.9,
                ),
                num_dashes=SECTION_DIL_DASHES,
                dashed_ratio=0.3,
            )

        # ============================================================
        # YIELD CIRCLE
        # ============================================================

        def make_pi_yield_circle():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            p_cur = d["p"][idx]
            alpha_cur = d["A"][idx]

            m_cur = m_at(d, idx) * M_YIELD_DISPLAY_SCALE

            thetas = np.linspace(0, TAU, 60)

            pts = [
                axes3d.c2p(
                    *swap_xz(
                        cone_surface_circular(
                            p_cur, theta, m_cur, alpha_cur
                        )
                    )
                )
                for theta in thetas
            ]

            circle = PathBase(
                color=YIELD_SECTION_COLOR,
                stroke_width=sc3d(YIELD_SECTION_SW),
            )

            circle.set_points_as_corners(pts)

            return set_depth_test_deep(circle, DEPTH_TEST_YIELD)

        def make_alpha_center_dot():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            center = d["p"][idx] * (1.0 + d["A"][idx])

            return Dot3D(
                axes3d.c2p(*swap_xz(center)),
                radius=sc3d(ALPHA_CENTER_R),
                color=WHITE
            )


        def make_ratio_line_factory(
            key,
            color=None,
            dashed=False,
            num_dashes=40,
            dashed_ratio=0.5,
            stroke_width=RATIO_LINE_SW,
            stroke_opacity=0.8,
            magnified=False,
        ):

            def maker():

                ds, idx = ds_and_idx()

                d = datasets[ds]

                vec = d[key][idx]

                if magnified:
                    vec = magnify_r(vec, d["A"][idx])

                direction = 1.0 + vec

                col = color if color is not None else colors[ds]

                line = Line(
                    axes3d.c2p(0, 0, 0),
                    axes3d.c2p(*swap_xz(p_max_yield * direction)),
                    color=col,
                    stroke_width=sc3d(stroke_width),
                    stroke_opacity=stroke_opacity,
                )

                if dashed:

                    return DashedVMobject(
                        line,
                        num_dashes=num_dashes,
                        dashed_ratio=dashed_ratio,
                    )

                return line

            return maker

        make_r_line = make_ratio_line_factory(
            "R",
            color=GREEN_B,
            dashed=False,
            stroke_width=R_LINE_SW,
            magnified=True,
        )

        make_alpha_axis_line = make_ratio_line_factory(
            "A",
            color=WHITE,
            dashed=True,
            num_dashes=40,
            dashed_ratio=0.5,
            stroke_opacity=0.6,
        )

        make_alphab_line = make_ratio_line_factory(
            "AB",
            color=ORANGE,
            dashed=True,
            num_dashes=20,
            dashed_ratio=0.5,
            stroke_opacity=0.85,
        )

        make_alphad_line = make_ratio_line_factory(
            "AD",
            color=PURPLE_B,
            dashed=True,
            num_dashes=50,
            dashed_ratio=0.25,
            stroke_opacity=0.85,
        )

        make_alphac_line = make_ratio_line_factory(
            "AC",
            color=BLUE,
            dashed=False,
            num_dashes=50,
            dashed_ratio=0.25,
            stroke_opacity=0.85,
        )

        def make_ratio_dot_factory(key, color, radius=RATIO_DOT_R):

            def maker():

                ds, idx = ds_and_idx()

                d = datasets[ds]

                point = d["p"][idx] * (1.0 + d[key][idx])

                return Dot3D(
                    axes3d.c2p(*swap_xz(point)),
                    radius=sc3d(radius),
                    color=color
                )

            return maker

        make_alphab_dot = make_ratio_dot_factory("AB", ORANGE)
        make_alphad_dot = make_ratio_dot_factory("AD", PURPLE_B)

        def make_n_vector():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            r_point = stress_point_display(
                d["p"][idx],
                d["R"][idx],
                d["A"][idx]
            )

            n_cur = d["N"][idx]

            norm = np.linalg.norm(n_cur)

            n_hat = n_cur / norm if norm > 1e-9 else n_cur

            tip_point = r_point + N_VECTOR_LENGTH * n_hat

            start = axes3d.c2p(*swap_xz(r_point))
            end = axes3d.c2p(*swap_xz(tip_point))


            arrow = Arrow(
                start,
                end,
                buff=0,
                color=WHITE,
                stroke_width=sc3d(N_VECTOR_SW),
                tip_length=sc3d(N_VECTOR_TIP),
                max_tip_length_to_length_ratio=0.3,
                max_stroke_width_to_length_ratio=100,
            )

            base_dot = Dot3D(
                start,
                radius=sc3d(N_VECTOR_BASE_R),
                color=colors[ds]
            )

            return Group(arrow, base_dot)

        # ============================================================
        # ALWAYS REDRAW
        #
        # 3D  -> always_redraw
        # 2D  -> always_redraw_fixed
        # ============================================================

        pos_dot_3d = always_redraw(make_pos_dot_3d)
        pos_yield_mesh = always_rebuild(make_pos_yield_mesh)

        pos_pi_yield_circle = always_redraw(make_pi_yield_circle)
        pos_alpha_center = always_redraw(make_alpha_center_dot)

        pos_critical_section = always_redraw(make_critical_section)
        pos_bounding_section = always_redraw(make_bounding_section)
        pos_dilatancy_section = always_redraw(make_dilatancy_section)

        pos_r_line = always_redraw(make_r_line)
        pos_alpha_axis = always_redraw(make_alpha_axis_line)
        pos_alphab_line = always_redraw(make_alphab_line)
        pos_alphad_line = always_redraw(make_alphad_line)
        pos_alphac_line = always_redraw(make_alphac_line)
        pos_alphab_dot = always_redraw(make_alphab_dot)
        pos_alphad_dot = always_redraw(make_alphad_dot)

        pos_n_vector = always_redraw(make_n_vector)

        meridian_plane = make_meridian_plane()
        csl_line = always_redraw_fixed(make_csl_line)

        pos_dot_qp = always_redraw_fixed(make_pos_dot_qp)
        pos_dot_qeps = always_redraw_fixed(make_pos_dot_qeps)

        pos_wedge = always_redraw_fixed(make_pos_wedge)
        pos_bound_line = always_redraw_fixed(make_pos_bound_line)
        pos_dil_line = always_redraw_fixed(make_pos_dil_line)

        self.add(
            meridian_plane,
            pos_dot_3d,
            pos_yield_mesh,
            pos_critical_section,
            pos_bounding_section,
            pos_dilatancy_section,
            pos_pi_yield_circle,
            pos_alpha_center,
            pos_r_line,
            pos_alpha_axis,
            pos_alphab_line,
            pos_alphad_line,
            pos_alphac_line,
            pos_alphab_dot,
            pos_alphad_dot,
            pos_n_vector,
        )

        self.add_fixed_in_frame_mobjects(
            csl_line,
            pos_wedge,
            pos_bound_line,
            pos_dil_line,
            pos_dot_qp,
            pos_dot_qeps,
        )


        update_readout(None)


        def master_time_updater(mob, dt):

            if ui_state["playing"]:

                new_raw = raw_time.get_value() + dt * PLAYBACK_SPEED

                limit = ui_state["stop_at"]

                if limit is None:
                    limit = float(n_datasets)

                if new_raw >= limit:

                    new_raw = limit - 1e-6
                    ui_state["playing"] = False

                raw_time.set_value(new_raw)

            mob.set_value(apply_rate(raw_time.get_value()))

            w = float(np.clip(follow_blend.get_value(), 0.0, 1.0))

            if view_state["follow"]:
                w = 1.0

            if w > 1e-6:

                cam = self._cam()

                if cam is not None:

                    base = cam.get_center()
                    target = current_point3d()

                    cam.move_to(base + w * (target - base))

        master_time.add_updater(master_time_updater)

        self.add(raw_time)
        self.add(master_time)
        self.add(follow_blend)

        cam0 = self._cam()

        self._cam_home_center = (
            cam0.get_center() if cam0 is not None else ORIGIN
        )
        self._master_time = master_time
        self._raw_time = raw_time
        self._ui_state = ui_state
        self._n_datasets = n_datasets
        self._seg_lens = seg_lens
        self._timeline_ready = True


        info_mobs = [ds_title, keys_hint]

        info_visible = (not config.write_to_movie) or INFO_IN_RENDER

        if not info_visible:
            self.remove(*info_mobs)

        self._info_mobs = info_mobs
        self._info_visible = info_visible

        intro_items = [
            # --- 3D ---
            (axes3d, "create", False),
            (hydro_axis, "create", False),
            (labels3d, "text", False),
            (meridian_plane, "create", False),
            (critical_surface_mesh, "create", False),
            (pos_critical_section, "create", False),
            (pos_bounding_section, "create", False),
            (pos_dilatancy_section, "create", False),
            (pos_yield_mesh, "create", False),
            (pos_pi_yield_circle, "create", False),
            (pos_alpha_center, "create", False),
            (pos_alpha_axis, "create", False),
            (pos_r_line, "create", False),
            (pos_alphab_line, "create", False),
            (pos_alphad_line, "create", False),
            (pos_alphac_line, "create", False),
            (pos_alphab_dot, "create", False),
            (pos_alphad_dot, "create", False),
            (pos_n_vector, "create", False),
            (pos_dot_3d, "create", False),
            # ---  2D (fixed in frame) ---
            (pq_background, "create", True),
            (pq_grid, "create", True),
            (pq_axes, "create", True),
            (pq_labels, "text", True),
            (qeps_grid, "create", True),
            (qeps_axes, "create", True),
            (qeps_labels, "text", True),
            (csl_line, "create", True),
            (pos_wedge, "create", True),
            (pos_bound_line, "create", True),
            (pos_dil_line, "create", True),
            (pos_dot_qp, "create", True),
            (pos_dot_qeps, "create", True),
            # --- HUD (fixed in frame) ---
            (readout, "fade", True),
            (legend_group, "fade", True),
            (legend3d_group, "fade", True),
            (ds_title, "fade", True),
            (keys_hint, "fade", True),
        ]

        if not info_visible:
            intro_items = [
                it for it in intro_items
                if it[0] not in info_mobs
            ]


        def intro_anim(mob, kind):

            if kind == "text":
                kind = INTRO_TEXT_KIND

            if kind == "fade":
                return FadeIn(mob)

            if kind == "write":
                return Write(mob)

            if not hasattr(mob, "pointwise_become_partial"):

                if mob.submobjects:
                    return AnimationGroup(
                        *[
                            intro_anim(sub, kind)
                            for sub in mob.submobjects
                        ]
                    )

                return FadeIn(mob)

            return Create(mob)

        def intro_restore():

            plain = [m for m, _, f in intro_items if not f]
            fixed = [m for m, _, f in intro_items if f]

            self.add(*plain)

            self.add_fixed_in_frame_mobjects(*fixed)

            for m in fixed:
                fix_in_frame_deep(m)

        if config.write_to_movie:

            self.remove(*[m for m, _, _ in intro_items])

            ghosts = []

            for m, kind, fixed in intro_items:

                ghost = m.copy()
                ghost.clear_updaters()

                if fixed:
                    fix_in_frame_deep(ghost)

                ghosts.append((ghost, kind))

            try:
                self.play(
                    *[
                        intro_anim(g, kind)
                        for g, kind in ghosts
                    ],
                    run_time=INTRO_TIME,
                    rate_func=RATE,
                    lag_ratio=0.0,
                )

            except Exception as exc:

                print(
                    f"[warn] intro failed "
                    f"({type(exc).__name__}: {exc}), "
                    f" FadeIn"
                )

                self.play(
                    *[FadeIn(g) for g, _ in ghosts],
                    run_time=INTRO_TIME,
                    rate_func=RATE,
                    lag_ratio=0.0,
                )

            self.remove(*[g for g, _ in ghosts])

            intro_restore()

            self.wait(INTRO_HOLD)


        readout_left.add_updater(update_readout)
        ds_title.add_updater(update_ds_title)

        # ============================================================
        # RECORDING SESSION
        # ============================================================

        if not config.write_to_movie:

            self.interactive_embed()
            return

        ui_state["mode"] = "solo"
        ui_state["active"] = 0
        ui_state["stop_at"] = None
        ui_state["playing"] = False
        view_state["follow"] = False
        raw_time.set_value(0.0)
        master_time.set_value(0.0)

        cam = self._cam()


        theta_t = ValueTracker(THETA0)

        master_time.add_updater(
            lambda m, dt: self.set_camera_orientation(theta=theta_t.get_value())
        )

        self.add(theta_t)


        seg = 1.0 / PLAYBACK_SPEED  # time dataset

        T = n_datasets * seg  # time total video

        W = {
            "zoom_in": W_ZOOM_IN,
            "hold_1": W_HOLD_1,
            "follow": W_FOLLOW,
            "hold_2": W_HOLD_2,
            "mega": W_MEGA,
            "hold_3": W_HOLD_3,
            "out": W_OUT,
            "tail": W_TAIL,
        }

        _w_tot = sum(W.values())

        def dur(key):
            return T * W[key] / _w_tot

        ui_state["playing"] = True

        # ------------------------------------------------------------

        cam.save_state()

        # 1)
        self.next_section("cam_zoom_in")
        self.play(
            cam.animate.scale(ZOOM_A),
            theta_t.animate.set_value(THETA0 + DTHETA),
            run_time=dur("zoom_in"),
            rate_func=RATE,
        )
        self.wait(dur("hold_1"))


        # 2)
        self.next_section("cam_follow")
        self.play(
            cam.animate.scale(ZOOM_B),
            follow_blend.animate.set_value(1.0),
            theta_t.animate.set_value(THETA0 + THETA_FOLLOW * DTHETA),
            run_time=dur("follow"),
            rate_func=RATE,
        )
        self.wait(dur("hold_2"))

        # 3)
        self.next_section("cam_mega_zoom")

        zoom_anims = [
            cam.animate.scale(ZOOM_MEGA),
            detail.animate.set_value(DETAIL_ZOOM),
            theta_t.animate.set_value(THETA0 + THETA_MEGA * DTHETA),
        ]

        if MERIDIAN_HIDE_ON_ZOOM:
            zoom_anims.append(FadeOut(meridian_plane))

        self.play(
            *zoom_anims,
            run_time=dur("mega"),
            rate_func=RATE,
        )
        self.wait(dur("hold_3"))

        # 4)
        self.next_section("cam_out")
        out_anims = [
            follow_blend.animate.set_value(0.0),
            Restore(cam),
            theta_t.animate.set_value(THETA0),
            detail.animate.set_value(1.0),
        ]

        if MERIDIAN_HIDE_ON_ZOOM:
            out_anims.append(FadeIn(meridian_plane))

        self.play(
            *out_anims,
            run_time=dur("out"),
            rate_func=RATE,
        )

        ui_state["playing"] = False

        self.wait(dur("tail"))
