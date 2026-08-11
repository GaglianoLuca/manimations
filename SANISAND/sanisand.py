"""

--- keyboard timeline ---
SPACE   play / pause
LEFT    step back   (ARROW_STEP step alla volta)
RIGHT   step forward   (SHIFT = x10, CTRL = passo singolo)
HOME    start
END     fine

--- 3D ---
Z       zoom IN
X       zoom OUT
F       follow on/off
V       reset

##  STATEV

statev(1)   e
statev(8)   pc
statev(69)  p'
statev(70)  q

statev(75)  alphab(1,1)      bounding back-stress ratio
statev(76)  alphab(2,2)
statev(77)  alphab(3,3)

statev(80)  n33(1,1)         direzione di carico (deviatorica, unitaria)
statev(81)  n33(2,2)
statev(82)  n33(3,3)

statev(83)  alpha(1,1)       back-stress ratio
statev(84)  alpha(2,2)
statev(85)  alpha(3,3)

statev(86)  r33(1,1)         stress ratio r = s / p'
statev(87)  r33(2,2)
statev(88)  r33(3,3)

statev(89)  m
statev(90)  alphad(1,1)      dilatancy back-stress ratio
statev(91)  alphad(2,2)
statev(92)  alphad(3,3)
statev(93)  cos(3*theta)
"""

from manim import *
from manim.opengl import *
import glob
import os

import numpy as np
import pandas as pd


class SANISAND(ThreeDScene):

    FILES = []

    FILE_GLOB = "CU_*.out"
    ARROW_STEP = 5
    ZOOM_FACTOR = 1.25

    # ================================================================
    # CAMERA
    # ================================================================
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

        print(f"  zoom = {1.0 / self._view_state['zoom']:7.2f}x")

    def _toggle_follow(self):

        st = self._view_state

        st["follow"] = not st["follow"]

        if st["follow"]:

            cam = self._cam()

            if cam is not None:
                cam.move_to(self._current_point3d())

        print("  follow " + ("ON" if st["follow"] else "OFF"))

    def _reset_view(self):

        cam = self._cam()

        if cam is None or not hasattr(cam, "scale"):
            return

        st = self._view_state
        cam.scale(1.0 / st["zoom"], about_point=cam.get_center())
        cam.move_to(self._cam_home_center)

        st["zoom"] = 1.0
        st["follow"] = False

    # ================================================================
    # TIMELINE
    # ================================================================

    def _step_master_time(self, direction, modifiers=0):

        if not getattr(self, "_timeline_ready", False):
            return

        import pyglet.window.key as K

        # SHIFT ->  x10,  CTRL ->  single
        n_steps = self.ARROW_STEP

        if modifiers & K.MOD_SHIFT:
            n_steps *= 10

        if modifiers & K.MOD_CTRL:
            n_steps = 1

        t = float(
            np.clip(
                self._master_time.get_value(),
                0,
                self._n_datasets - 1e-6
            )
        )

        ds = int(np.floor(t))
        step = n_steps / max(self._seg_lens[ds], 1)

        new_t = float(
            np.clip(
                t + direction * step,
                0,
                self._n_datasets - 1e-6
            )
        )

        self._master_time.set_value(new_t)

    def _set_mode(self, mode, active=None):

        if not getattr(self, "_timeline_ready", False):
            return

        st = self._ui_state

        if active is not None:

            if not (0 <= active < self._n_datasets):
                return
            st["active"] = active

        st["mode"] = mode
        st["playing"] = True

        if mode == "solo":
            self._master_time.set_value(float(st["active"]))
            st["stop_at"] = st["active"] + 1.0

        elif mode == "seq":
            self._master_time.set_value(float(st["active"]))
            st["stop_at"] = None
        else:   # sim
            self._master_time.set_value(0.0)
            st["stop_at"] = None

    # ================================================================
    # KEYBOARD
    # ================================================================

    def on_key_press(self, symbol, modifiers):

        if not getattr(self, "_timeline_ready", False):
            return super().on_key_press(symbol, modifiers)

        import pyglet.window.key as K

        if symbol in (K._0, getattr(K, "NUM_0", -1)):
            self._set_mode("sim")
            return

        for k in range(9):

            if symbol == getattr(K, f"_{k + 1}") or \
               symbol == getattr(K, f"NUM_{k + 1}", -1):

                self._set_mode(
                    "seq" if (modifiers & K.MOD_SHIFT) else "solo",
                    active=k,
                )
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
            self._master_time.set_value(0)
            return

        if symbol == K.END:
            self._ui_state["playing"] = False
            self._master_time.set_value(self._n_datasets - 1e-6)
            return

        return super().on_key_press(symbol, modifiers)

    # ================================================================
    # SCENE
    # ================================================================

    def construct(self):

        M_YIELD_DISPLAY_SCALE = 1.0
        # --- PLANE PI ---
        PI_PLANE_HALF = 700
        PI_PLANE_FILL = 0.25

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

            eps1 = data["stran(1)"].to_numpy(dtype=float)
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

            m_data = sv(89, default=0.0125)
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

                "m": m_data,
                "cos3t": cos3t,
            }

        files = list(self.FILES)

        if not files:
            files = sorted(glob.glob(self.FILE_GLOB))

        if len(files) > 9:
            print(f"  NOTA: {len(files)} key limit 9")


        for k, f in enumerate(files):
            print(f"    [{k + 1}] {f}")

        palette = [
            GREEN_B, BLUE_B, YELLOW_B, RED_B,
            PURPLE_B, TEAL_B, ORANGE, PINK, LIGHT_BROWN,
        ]

        colors = [palette[i % len(palette)] for i in range(len(files))]
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

        def _range(values, n_ticks=5, from_zero=None, margin=0.05):
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

            n = int(round((hi - lo) / step))

            return [lo + k * step for k in range(n + 1)]

        # --- data ---

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

        # ============================================================
        # SANISAND PARAMETERS
        # ============================================================

        M = 1.25
        c = 0.688
        e0 = 0.934
        lambda_c = 0.019
        xi = 0.712
        p_atm = 100
        n_b = 1.1
        n_d = 3.5
        m_yield = 0.0125

        _p_needed = 1.15 * float(all_p.max())
        p_step_3d = _step(_p_needed / 4)
        p_max_yield = float(np.ceil(_p_needed / p_step_3d) * p_step_3d)

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
            x_length=4,
            y_length=4,
            z_length=4,
            axis_config={
                "include_tip": False,
            },
        ).scale(0.5)

        labels3d = axes3d.get_axis_labels(
            Tex(r"$\sigma_1$", font_size=26),
            Tex(r"$\sigma_2$", font_size=26),
            Tex(r"$\sigma_3$", font_size=26)
        )

        self.set_camera_orientation(
            phi=65 * DEGREES,
            theta=90 * DEGREES,
            distance=100
        )

        self.camera.light_source.move_to(np.array([-8, -8, 12]))

        LABEL_OUT = 1.35

        for k, lab in enumerate(labels3d):

            pos = np.zeros(3)
            pos[k] = LABEL_OUT * p_max_yield

            lab.move_to(axes3d.c2p(*pos))

        self.add(axes3d, labels3d)

        # ============================================================
        # PARAMETRIC SURFACE
        # Sigma = pI + q
        # u(theta) = sqrt(2/3)*[cos(t), cos(t-2pi/3), cos(t+2pi/3)]
        #
        # ||u|| = 1, tr(u) = 0.
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
        # SCALE YIELD FOR BETTER VISUALIZATION

        def magnify_r(r_vec, alpha_vec, scale=M_YIELD_DISPLAY_SCALE):

            r_vec = np.asarray(r_vec, dtype=float)
            alpha_vec = np.asarray(alpha_vec, dtype=float)

            return alpha_vec + scale * (r_vec - alpha_vec)

        def stress_point_display(p, r_vec, alpha_vec):

            return p * (1.0 + magnify_r(r_vec, alpha_vec))

        critical_surface_cone = OpenGLSurface(
            lambda u, v: axes3d.c2p(*cone_surface(u, v, M)),
            u_range=[0, p_max_yield],
            v_range=[0, TAU],
            resolution=(24, 32),
            fill_opacity=0.4,
            stroke_width=0,
            checkerboard_colors=False,
            fill_color=BLUE,
            gloss=0.35,
            shadow=0.4,
        )

        critical_surface_mesh = OpenGLSurfaceMesh(
            critical_surface_cone,
            stroke_width=1,
            stroke_color=BLUE,
            stroke_opacity=0.5
        )

        hydro_axis = Line(
            axes3d.c2p(0, 0, 0),
            axes3d.c2p(p_max_yield, p_max_yield, p_max_yield),
            color=YELLOW,
            stroke_width=3
        )

        self.add(critical_surface_mesh)

        cone_group = Group(
            axes3d,
            labels3d,
            critical_surface_mesh,
            hydro_axis,
        )

        cone_group.move_to(LEFT * 1 + DOWN * 3)
        cone_group.scale(2.7)

        # ============================================================
        # P'-Q
        # ============================================================

        pq_axes = Axes(
            x_range=[p_axis_min, p_axis_max, p_step],
            y_range=[q_axis_min, q_axis_max, q_step],
            x_length=4.2,
            y_length=3.2,
            axis_config={
                "include_tip": False,
                "font_size": 16
            },
        ).shift(RIGHT * 3.6 + UP * 1.9)

        pq_axes.add_coordinates(
            p_labels,
            q_labels,
            num_decimal_places=max(p_dec, q_dec),
            font_size=14
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
            x_length=4.2,
            y_length=2.8,
            axis_config={
                "include_tip": False,
                "font_size": 16
            },
        ).shift(RIGHT * 3.6 + DOWN * 2.1)

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
            legend_entry("solid", "$M$ (critical)", color=RED),
            legend_entry("dashed_wide", "$M_b$ (bounding)"),
            legend_entry("dashed_narrow", "$M_d$ (dilatancy)"),
        )

        legend_group.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend_group.scale(1)
        legend_group.next_to(pq_axes, RIGHT, buff=0.2)
        legend_group.shift(LEFT * 1)

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
        legend3d_group.scale(2)
        legend3d_group.to_corner(UL, buff=0.35)

        ds_title = Tex(
            f"[1/{len(ds_labels)}] "
            + ds_labels[0].replace("_", r"\_"),
            font_size=22,
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
                font_size=22,
                color=colors[ds],
            )

            new_tex.move_to(mob)

            mob.become(new_tex)

            if hasattr(mob, "fix_in_frame"):
                mob.fix_in_frame()

        keys_hint = Tex(
            r"1--9 solo \quad SHIFT+n seq \quad 0 all "
            r"\quad SPACE play \quad "
            r"$\leftarrow\rightarrow$ step \quad "
            r"Z/X zoom \quad F follow \quad V reset",
            font_size=14,
            color=GRAY_B,
        ).to_edge(DOWN, buff=0.15)

        READOUT_FONT = 20
        READOUT_BUFF = 0.14

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
                M_theta_at(d, i) * np.exp(-n_b * psi_at(d, i))
            ],
            color=ORANGE, dec=4,
        )

        add_readout_row(
            col_right, r"$M_d$", M,
            lambda d, i: [
                M_theta_at(d, i) * np.exp(n_d * psi_at(d, i))
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
            """Impila le righe e allinea le colonne numeriche.

            arrange(aligned_edge=LEFT) allinea solo il bordo della
            riga: siccome le etichette hanno larghezze diverse, i
            numeri vanno riallineati a mano.
            """

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

        self.add_fixed_in_frame_mobjects(
            pq_axes,
            pq_labels,
            qeps_axes,
            qeps_labels,
            readout,
            legend_group,
            legend3d_group,
            ds_title,
            keys_hint,
        )

        self.play(
            Create(pq_axes),
            Write(pq_labels),
            Create(qeps_axes),
            Write(qeps_labels),
            Create(hydro_axis),
            run_time=0.8,
        )

        # ============================================================
        # PATH BASE (OpenGL vs Cairo)
        # ============================================================

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

        def always_redraw_fixed(func):

            mob = fix_in_frame_deep(func())

            def updater(m):

                m.become(func())
                fix_in_frame_deep(m)

            mob.add_updater(updater)

            return mob

        # ============================================================
        # MASTER TIME
        # ============================================================

        n_datasets = len(datasets)

        seg_lens = [len(d["p"]) - 1 for d in datasets]

        master_time = ValueTracker(0)

        playback_speed = 0.1

        ui_state = {
            "playing": False,
            "mode": "seq",
            "active": 0,
            "stop_at": None,
        }

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

            return axes3d.c2p(*stress_point_display(
                d["p"][idx],
                d["R"][idx],
                d["A"][idx],
            ))

        self._current_point3d = current_point3d

        # ============================================================
        # LODE
        #
        # theta da  cos(3*theta) = sqrt(6)*tr(n^3), in [0, 60] gradi.
        # La pendenza critica nel piano p'-q e' M*g(theta), NON M.
        # ============================================================

        def lode_theta(d, idx):

            n_vec = d["N"][idx]
            nn = np.linalg.norm(n_vec)

            if nn < 1e-9:
                return 0.0

            n_hat = n_vec / nn

            cos3t = np.sqrt(6.0) * float(np.sum(n_hat ** 3))

            return float(
                np.arccos(np.clip(cos3t, -1.0, 1.0)) / 3.0
            )

        def M_theta_at(d, idx):
            return M * g_theta(lode_theta(d, idx), c)

        # ============================================================
        # CSL PLANE p'-q
        # ============================================================

        def make_csl_line():

            ds, idx = ds_and_idx()

            Mth = M_theta_at(datasets[ds], idx)

            return Line(
                pq_axes.c2p(0, 0),
                pq_axes.c2p(p_axis_max, max(Mth * p_axis_max, 0)),
                color=RED,
                stroke_width=3,
            )

        # ============================================================
        # PATHS

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
                axes3d.c2p(s11[j], s22[j], s33[j])
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
                sw=4
            ):

                def updater():

                    n_show = n_shown_for(ds_index)

                    # zero punti = dataset nascosto (modalita' solo)
                    if n_show < 1:

                        path = PathBase(
                            color=color,
                            stroke_width=sw,
                            stroke_opacity=0,
                        )

                        path.set_points_as_corners(
                            [points_full[0], points_full[0]]
                        )

                        return path

                    pts = points_full[:max(n_show, 2)]

                    path = PathBase(color=color, stroke_width=sw)
                    path.set_points_as_corners(pts)

                    return path

                return updater

            path3d = always_redraw(
                make_path_updater(pts3d_full, sw=6)
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
                axes3d.c2p(
                    d["s11"][idx], d["s22"][idx], d["s33"][idx]
                ),
                radius=0.07,
                color=colors[ds]
            )

        def make_pos_dot_qp():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            dot = Dot(
                pq_axes.c2p(d["p"][idx], d["q"][idx]),
                radius=0.07,
                color=colors[ds]
            )

            dot.z_index = 4

            return dot

        def make_pos_dot_qeps():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            dot = Dot(
                qeps_axes.c2p(d["eps1"][idx], d["q"][idx]),
                radius=0.07,
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
                fill_opacity=0.15,
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

            Mb_curr = M_theta_at(d, idx) * np.exp(
                -n_b * psi_at(d, idx)
            )

            line = DashedVMobject(
                Line(
                    pq_axes.c2p(0, 0),
                    pq_axes.c2p(
                        p_axis_max, max(Mb_curr * p_axis_max, 0)
                    ),
                    color=colors[ds],
                    stroke_width=2,
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

            Md_curr = M_theta_at(d, idx) * np.exp(
                n_d * psi_at(d, idx)
            )

            line = DashedVMobject(
                Line(
                    pq_axes.c2p(0, 0),
                    pq_axes.c2p(
                        p_axis_max, max(Md_curr * p_axis_max, 0)
                    ),
                    color=colors[ds],
                    stroke_width=2,
                    stroke_opacity=0.7
                ),
                num_dashes=40,
                dashed_ratio=0.25,
            )

            line.z_index = 2

            return line

        # ============================================================
        # DYNAMIC YIELD SURFACE (3D)
        # ============================================================

        def make_pos_yield_mesh():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            alpha_cur = d["A"][idx]

            m_cur = m_at(d, idx) * M_YIELD_DISPLAY_SCALE

            surf = OpenGLSurface(
                lambda u, v: axes3d.c2p(
                    *cone_surface_circular(u, v, m_cur, alpha_cur)
                ),
                u_range=[0, p_max_yield],
                v_range=[0, TAU],
                resolution=(16, 24),
                fill_opacity=0,
                stroke_width=0,
                checkerboard_colors=False,
                gloss=0,
                shadow=0,
            )

            return OpenGLSurfaceMesh(
                surf,
                stroke_width=1,
                stroke_color=colors[ds],
                stroke_opacity=0.6
            )

        # ============================================================
        # PI PLANE

        def pi_plane_basis():

            e1 = np.array([1, -1, 0]) / np.sqrt(2)
            e2 = np.array([1, 1, -2]) / np.sqrt(6)

            return e1, e2

        def pi_square_corners(p_cur):

            e1, e2 = pi_plane_basis()

            center = np.array([p_cur, p_cur, p_cur])

            return [
                axes3d.c2p(*(center + PI_PLANE_HALF * (+e1 + e2))),
                axes3d.c2p(*(center + PI_PLANE_HALF * (-e1 + e2))),
                axes3d.c2p(*(center + PI_PLANE_HALF * (-e1 - e2))),
                axes3d.c2p(*(center + PI_PLANE_HALF * (+e1 - e2))),
            ]

        def make_pi_plane():

            ds, idx = ds_and_idx()

            corners = pi_square_corners(datasets[ds]["p"][idx])

            fill = Polygon(
                *corners,
                color=GRAY_B,
                fill_color=GRAY_B,
                fill_opacity=PI_PLANE_FILL,
                stroke_width=0,
            )

            edge = Polygon(
                *corners,
                color=GRAY_B,
                stroke_width=2,
                stroke_opacity=0.8,
                fill_opacity=0,
            )

            return VGroup(fill, edge)

        # ============================================================
        # MERIDIAN PLANE

        MERIDIAN_THETA = 0 * DEGREES
        MERIDIAN_HALF = PI_PLANE_HALF
        MERIDIAN_P_MIN = 0.0
        MERIDIAN_P_MAX = p_max_yield
        MERIDIAN_FILL = 0.18

        MERIDIAN_HALF_PLANE = False

        def make_meridian_plane():

            u = dev_dir(MERIDIAN_THETA)
            hyd = np.array([1.0, 1.0, 1.0])

            w_neg = 0.0 if MERIDIAN_HALF_PLANE else -MERIDIAN_HALF

            corners = [
                axes3d.c2p(*(MERIDIAN_P_MIN * hyd + w_neg * u)),
                axes3d.c2p(*(MERIDIAN_P_MAX * hyd + w_neg * u)),
                axes3d.c2p(*(MERIDIAN_P_MAX * hyd + MERIDIAN_HALF * u)),
                axes3d.c2p(*(MERIDIAN_P_MIN * hyd + MERIDIAN_HALF * u)),
            ]

            fill = Polygon(
                *corners,
                color=RED,
                fill_color=RED,
                fill_opacity=MERIDIAN_FILL,
                stroke_width=0,
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

        SECTION_RESOLUTION = 121
        SECTION_FILL_OPACITY = 0.18

        SECTION_THETAS = np.linspace(0, TAU, SECTION_RESOLUTION)

        def section_points(p_cur, M_eff):

            return [
                axes3d.c2p(*cone_surface(p_cur, t, M_eff))
                for t in SECTION_THETAS
            ]

        def section_curve(pts, color, stroke_width=3, opacity=1.0):

            curve = PathBase(
                color=color,
                stroke_width=stroke_width,
                stroke_opacity=opacity,
            )

            curve.set_points_as_corners(pts)

            return curve

        def make_critical_section():

            ds, idx = ds_and_idx()

            p_cur = datasets[ds]["p"][idx]

            pts = section_points(p_cur, M)

            outline = section_curve(pts, BLUE, stroke_width=3)

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

            Mb = M * np.exp(-n_b * psi_at(d, idx))

            return DashedVMobject(
                section_curve(
                    section_points(d["p"][idx], Mb),
                    ORANGE,
                    stroke_width=2,
                    opacity=0.9,
                ),
                num_dashes=60,
                dashed_ratio=0.55,
            )

        def make_dilatancy_section():

            ds, idx = ds_and_idx()
            d = datasets[ds]

            Md = M * np.exp(n_d * psi_at(d, idx))

            return DashedVMobject(
                section_curve(
                    section_points(d["p"][idx], Md),
                    PURPLE_B,
                    stroke_width=2,
                    opacity=0.9,
                ),
                num_dashes=110,
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
                    *cone_surface_circular(
                        p_cur, theta, m_cur, alpha_cur
                    )
                )
                for theta in thetas
            ]

            circle = PathBase(color=colors[ds], stroke_width=3)

            circle.set_points_as_corners(pts)

            return circle

        def make_alpha_center_dot():

            ds, idx = ds_and_idx()

            d = datasets[ds]

            center = d["p"][idx] * (1.0 + d["A"][idx])

            return Dot3D(
                axes3d.c2p(*center),
                radius=0.05,
                color=WHITE
            )


        def make_ratio_line_factory(
            key,
            color=None,
            dashed=False,
            num_dashes=40,
            dashed_ratio=0.5,
            stroke_width=2,
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
                    axes3d.c2p(*(p_max_yield * direction)),
                    color=col,
                    stroke_width=stroke_width,
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
            stroke_width=2.5,
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


        def make_ratio_dot_factory(key, color, radius=0.05):

            def maker():

                ds, idx = ds_and_idx()

                d = datasets[ds]

                point = d["p"][idx] * (1.0 + d[key][idx])

                return Dot3D(
                    axes3d.c2p(*point),
                    radius=radius,
                    color=color
                )

            return maker

        make_alphab_dot = make_ratio_dot_factory("AB", ORANGE)
        make_alphad_dot = make_ratio_dot_factory("AD", PURPLE_B)

        N_VECTOR_LENGTH = 250

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

            start = axes3d.c2p(*r_point)
            end = axes3d.c2p(*tip_point)


            arrow = Arrow(
                start,
                end,
                buff=0,
                color=WHITE,
                stroke_width=3,
                tip_length=0.18,
                max_tip_length_to_length_ratio=0.3,
                max_stroke_width_to_length_ratio=100,
            )

            base_dot = Dot3D(
                start,
                radius=0.04,
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
        pos_yield_mesh = always_redraw(make_pos_yield_mesh)
        pos_pi_plane = always_redraw(make_pi_plane)
        pos_pi_yield_circle = always_redraw(make_pi_yield_circle)
        pos_alpha_center = always_redraw(make_alpha_center_dot)

        pos_critical_section = always_redraw(make_critical_section)
        pos_bounding_section = always_redraw(make_bounding_section)
        pos_dilatancy_section = always_redraw(make_dilatancy_section)

        pos_r_line = always_redraw(make_r_line)
        pos_alpha_axis = always_redraw(make_alpha_axis_line)
        pos_alphab_line = always_redraw(make_alphab_line)
        pos_alphad_line = always_redraw(make_alphad_line)

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

        readout_left.add_updater(update_readout)
        update_readout(None)

        ds_title.add_updater(update_ds_title)


        def master_time_updater(mob, dt):

            if ui_state["playing"]:

                new_val = mob.get_value() + dt * playback_speed


                limit = ui_state["stop_at"]

                if limit is None:
                    limit = float(n_datasets)

                if new_val >= limit:

                    new_val = limit - 1e-6
                    ui_state["playing"] = False

                mob.set_value(new_val)

            if view_state["follow"]:

                cam = self._cam()

                if cam is not None:
                    cam.move_to(current_point3d())

        master_time.add_updater(master_time_updater)

        self.add(master_time)

        cam0 = self._cam()

        self._cam_home_center = (
            cam0.get_center() if cam0 is not None else ORIGIN
        )

        self._master_time = master_time
        self._ui_state = ui_state
        self._n_datasets = n_datasets
        self._seg_lens = seg_lens
        self._timeline_ready = True

        self.interactive_embed()
