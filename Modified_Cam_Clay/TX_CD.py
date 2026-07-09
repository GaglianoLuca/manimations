from manim import *
import pandas as pd
import numpy as np
from manim.utils.rate_functions import ease_in_sine


class MCC(ThreeDScene):
    def construct(self):

        title = Tex("MCC: Triaxial-CD load path new", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        vettore_scarti = [-0.33, -0.392, -0.457]

        def load_data(file):
            data = pd.read_csv(file, sep=r"\s+", engine="python")
            e = data['statev(1)']
            if file == "CD_MCC_OCR1.out":
                e = e + vettore_scarti[0]
            if file == "CD_MCC_OCR2.out":
                e = e + vettore_scarti[1]
            if file == "CD_MCC_OCR4.out":
                e = e + vettore_scarti[2]
            s11 = -data['stress(1)']
            s22 = -data['stress(2)']
            s33 = -data['stress(3)']
            pc  = data['statev(8)']
            p   = (s11 + s22 + s33) / 3.0
            q   = s11 - s22
            return e.to_numpy(), p.to_numpy(), q.to_numpy(), pc.to_numpy()

        files    = ["CD_MCC_OCR1.out", "CD_MCC_OCR2.out", "CD_MCC_OCR4.out"]
        colors   = [PURPLE, GREEN, RED]
        colors_e = [RED, GREEN, PURPLE]
        datasets = [load_data(f) for f in files]
        tracker  = ValueTracker(0)

        self.set_camera_orientation(
            phi=90 * DEGREES,
            theta=-90 * DEGREES,
            gamma=-90 * DEGREES,
            zoom=0.6,
        )

        p_max = max([p.max() for _, p, _, _ in datasets])
        q_max = max([q.max() for _, _, q, _ in datasets])
        e_max = max([e.max() for e, _, _, _ in datasets])
        e_min = min([e.min() for e, _, _, _ in datasets])

        axes = ThreeDAxes(
            x_range=[0, p_max, 50],
            y_range=[0.3, e_max, 50],
            z_range=[0, q_max, 50],
            x_length=7,
            y_length=7,
            z_length=7,
        ).rotate(-90 * DEGREES, UP).rotate(45 * DEGREES, RIGHT).shift(OUT * -2)

        lab_z = Tex("$q$").scale(1.7)
        lab_z.move_to(axes.z_axis.get_end() + OUT * 1).rotate(55 * DEGREES, RIGHT)
        lab_y = Tex("$e$").scale(1.7)
        lab_y.move_to(axes.y_axis.get_end() + OUT * 1).rotate(-90 * DEGREES, UP).rotate(70 * DEGREES, RIGHT)
        lab_x = Tex("$p'$").scale(1.7)
        lab_x.move_to(axes.x_axis.get_end() + OUT * 1).rotate(-90 * DEGREES, UP).rotate(70 * DEGREES, RIGHT)
        labels = VGroup(lab_y, lab_x, lab_z)
        self.play(Create(axes), Write(labels))

        M                = 1
        N                = 2
        lambda_cam_clay  = 0.1
        kappa_s          = 0.01

        def surface_func(p0_local, p_ratio):
            v_plot = (
                N - lambda_cam_clay * np.log(p0_local)
                + kappa_s * np.log(1 / p_ratio)
            )
            q_val = M * np.sqrt(
                p0_local * p_ratio * (p0_local - p0_local * p_ratio)
            )
            return np.array([
                p0_local * p_ratio,
                v_plot - 1,
                q_val if not np.isnan(q_val) else 0,
            ])

        sbs = Surface(
            lambda u, v: axes.c2p(*surface_func(u, v)),
            u_range=[10, 820],
            v_range=[0.01, 0.99],
            resolution=(20, 20),
            fill_opacity=0.3,
            stroke_width=0.4,
            checkerboard_colors=False,
            fill_color=BLUE,
        )

        CSL = ParametricFunction(
            lambda t: axes.c2p(*surface_func(t, 0.5)),
            t_range=[10, 800],
            color=YELLOW,
            stroke_width=3,
        )

        loadPath = []
        for i, (e, p, q, pc) in enumerate(datasets):
            points = [axes.c2p(p[j], e[j], q[j]) for j in range(len(e))]
            curve  = VMobject(color=colors[i])
            curve.set_points_as_corners(points)
            loadPath.append(curve)

        self.play(Create(sbs), run_time=3, rate_func=ease_in_sine)
        self.wait(1)

        def get_yield(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            p_c   = pc[idx]
            e_c   = e[idx]
            p_vals = np.linspace(0, p_c, 200)
            q_vals = M * np.sqrt(p_vals * (p_c - p_vals))
            points = [axes.c2p(p_i, e_c, q_i) for p_i, q_i in zip(p_vals, q_vals)]
            curve  = VMobject(color=colors[dataset_idx % len(colors)])
            curve.set_points_as_corners(points)
            return curve

        mcc_curves = []
        for i in range(len(datasets)):
            curve = get_yield(i)
            curve.add_updater(lambda m, idx=i: m.become(get_yield(idx)))

            mcc_curves.append(curve)
        self.play(Create(mcc_curves[1]),Create(mcc_curves[2]),Create(mcc_curves[0]))

        def get_vector_tex(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            color = colors[dataset_idx % len(colors_e)]
            return MathTex(
                r"\left[ e, p^{\prime}, q \right] = "
                f"[{e[idx]:.2f}, {p[idx]:.2f}, {q[idx]:.2f}]",
                color=color,
                font_size=30,
            ).to_corner(UL).shift(DOWN * dataset_idx * 0.5)

        vector_labels = []
        for i in range(len(datasets)):
            label = get_vector_tex(i)
            self.add_fixed_in_frame_mobjects(label)
            vector_labels.append(label)

        def update_vectors():
            for i, label in enumerate(vector_labels):
                label.become(get_vector_tex(i))
                self.add_fixed_in_frame_mobjects(label)

        for label in vector_labels:
            label.add_updater(lambda m: update_vectors())

     
        MINI_W = 2.8   # larghezza di ciascun mini-asse 
        MINI_H = 1.8   # altezza
        MINI_X = -5.2  # centro orizzontale 
        MINI_Y_TOP = -0.9  # centro verticale piano q-p'
        MINI_Y_BOT = -3.0  # centro verticale piano e-p'

        def make_mini_axes_qp():
            ax = Axes(
                x_range=[0, p_max, p_max / 4],
                y_range=[0, q_max, q_max / 4],
                x_length=MINI_W,
                y_length=MINI_H,
              #  axis_config={"stroke_width": 1.5, "tip_length": 0.06, "tip_width": 0.06 },
                x_axis_config={"color": WHITE},
                y_axis_config={"color": WHITE},
            )
            ax.move_to([MINI_X, MINI_Y_TOP, 0])
            lx = Tex(r"$p'$", font_size=18).next_to(ax.x_axis.get_right(), DOWN, buff=0.08)
            ly = Tex(r"$q$",  font_size=18).next_to(ax.y_axis.get_top(),   LEFT, buff=0.08)
            return VGroup(ax, lx, ly), ax

        def make_mini_axes_ep():
            ax = Axes(
                x_range=[0, p_max, p_max / 4],
                y_range=[e_min * 0.95, e_max * 1.05, (e_max - e_min) / 4],
                x_length=MINI_W,
                y_length=MINI_H,  
               # axis_config={"stroke_width": 1.5, "tip_length": 0.06, "tip_width": 0.06},
                x_axis_config={"color": WHITE},
                y_axis_config={"color": WHITE},
            )
            ax.move_to([MINI_X, MINI_Y_BOT, 0])
            lx = Tex(r"$p'$", font_size=18).next_to(ax.x_axis.get_right(), DOWN, buff=0.08)
            ly = Tex(r"$e$", font_size=18).next_to(ax.y_axis.get_top(), LEFT, buff=0.08)
            return VGroup(ax, lx, ly), ax

        mini_qp_group, mini_ax_qp = make_mini_axes_qp()
        mini_ep_group, mini_ax_ep = make_mini_axes_ep()

        self.add_fixed_in_frame_mobjects(mini_qp_group)
        self.add_fixed_in_frame_mobjects(mini_ep_group)
        mini_csl_qp = Line(
            start=mini_ax_qp.c2p(0, 0),
            end=mini_ax_qp.c2p(p_max, M * p_max),
            color=YELLOW,
            stroke_width=1.8,
        )
        self.add_fixed_in_frame_mobjects(mini_csl_qp)
        #self.play(Create(mini_csl_qp))
        #self.play(Create(mini_qp_group), Create(mini_ep_group), run_time=1)

        def get_mini_path_qp(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            pts   = [mini_ax_qp.c2p(p[j], q[j]) for j in range(idx + 1)]
            if len(pts) < 2:
                pts = [mini_ax_qp.c2p(p[0], q[0])] * 2
            curve = VMobject(color=colors[dataset_idx % len(colors)], stroke_width=1.5)
            curve.set_points_as_corners(pts)
            return curve

        def get_mini_path_ep(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            pts   = [mini_ax_ep.c2p(p[j], e[j]) for j in range(idx + 1)]
            if len(pts) < 2:
                pts = [mini_ax_ep.c2p(p[0], e[0])] * 2
            curve = VMobject(color=colors[dataset_idx % len(colors)], stroke_width=1.5)
            curve.set_points_as_corners(pts)
            return curve

        mini_paths_qp = []
        mini_paths_ep = []

        for i in range(len(datasets)):
            c_qp = get_mini_path_qp(i)
            c_qp.add_updater(lambda m, idx=i: m.become(get_mini_path_qp(idx)))
            self.add_fixed_in_frame_mobjects(c_qp)
            mini_paths_qp.append(c_qp)

            c_ep = get_mini_path_ep(i)
            c_ep.add_updater(lambda m, idx=i: m.become(get_mini_path_ep(idx)))
            self.add_fixed_in_frame_mobjects(c_ep)
            mini_paths_ep.append(c_ep)

        def get_dot_qp(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            return Dot(
                mini_ax_qp.c2p(p[idx], q[idx]),
                radius=0.05,
                color=colors[dataset_idx % len(colors)],
            )

        def get_dot_ep(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            return Dot(
                mini_ax_ep.c2p(p[idx], e[idx]),
                radius=0.05,
                color=colors[dataset_idx % len(colors)],
            )

        dots_qp = []
        dots_ep = []
        for i in range(len(datasets)):
            d_qp = get_dot_qp(i)
            d_qp.add_updater(lambda m, idx=i: m.become(get_dot_qp(idx)))
            self.add_fixed_in_frame_mobjects(d_qp)
            dots_qp.append(d_qp)

            d_ep = get_dot_ep(i)
            d_ep.add_updater(lambda m, idx=i: m.become(get_dot_ep(idx)))
            self.add_fixed_in_frame_mobjects(d_ep)
            dots_ep.append(d_ep)

        def get_mini_yield_qp(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            n_pts = len(e)
            idx   = min(int(tracker.get_value() * (n_pts - 1)), n_pts - 1)
            p_c   = pc[idx]
            p_vals = np.linspace(0, p_c, 200)
            q_vals = M * np.sqrt(np.maximum(p_vals * (p_c - p_vals), 0))
            pts = [mini_ax_qp.c2p(p_i, q_i) for p_i, q_i in zip(p_vals, q_vals)]
            if len(pts) < 2:
                pts = pts * 2
            curve = VMobject(
                color=colors[dataset_idx % len(colors)],
                stroke_width=1.2,
                stroke_opacity=0.85,
            )
            curve.set_points_as_corners(pts)
            return curve

        mini_yield_qp = []
        for i in range(len(datasets)):
            c = get_mini_yield_qp(i)
            c.add_updater(lambda m, idx=i: m.become(get_mini_yield_qp(idx)))
            self.add_fixed_in_frame_mobjects(c)
            mini_yield_qp.append(c)

        self.play(
            *[Create(curve) for curve in loadPath],
            tracker.animate.set_value(1),
            run_time=7,
            rate_func=ease_in_sine,
        )

        self.play(Create(CSL), run_time=1)
        self.wait(1)

        full_system = VGroup(axes, labels, CSL, sbs, *loadPath)
        self.play(
            Rotate(full_system, 45 * DEGREES, about_point=ORIGIN, axis=LEFT),
            run_time=2,
        )
        self.wait(2)


        self.play(
            full_system.animate
                .rotate(PI / 2, axis=OUT)
                .shift(OUT * -1)
                .shift(RIGHT * -8)
                .shift(DOWN * 6),
            run_time=5,
        )
        self.play(
            sbs.animate.set_fill(opacity=0.1),
            sbs.animate.set_stroke(width=0.1),
            run_time=1,
        )
        self.wait(5)
