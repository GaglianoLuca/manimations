from manim import *
import pandas as pd
import numpy as np
from manim.utils.rate_functions import ease_in_sine


class MCC(ThreeDScene):
    def construct(self):

        #Load from file
        title = Tex("MCC: Triaxial-CD load path", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        vettore_scarti=[-0.33,-0.392,-0.457] #adapt the e-void from the file


        def load_data(file):
            data = pd.read_csv(file, sep=r"\s+", engine="python")
            e = data['statev(1)'] #void-index
            if file=="CD_MCC_OCR1.out":
                i=0
                e = e  + vettore_scarti[0]
            if file=="CD_MCC_OCR2.out":
                i = 1
                e = e + vettore_scarti[1]
            if file == "CD_MCC_OCR4.out":
                i = 1
                e = e + vettore_scarti[2]

            s11 = -data['stress(1)']
            s22 = -data['stress(2)']
            s33 = -data['stress(3)']
            pc = data['statev(8)']
            p = (s11 + s22 + s33) / 3.0
            q = s11 - s22

            return e.to_numpy(), p.to_numpy(), q.to_numpy(), pc.to_numpy()

        # Files available from https://github.com/MeritaTafili/REV-wIncDr/tree/main/Session%203
        files = ["CD_MCC_OCR1.out", "CD_MCC_OCR2.out", "CD_MCC_OCR4.out"]
        colors = [PURPLE, GREEN, RED]
        datasets = [load_data(f) for f in files]
        tracker = ValueTracker(0)
        colors_e = [RED, GREEN, PURPLE]

        self.set_camera_orientation(
            phi=90 * DEGREES,
            theta=-90 * DEGREES,
            gamma=-90 * DEGREES,
            zoom=0.6
        )
        p_max = max([p.max() for _, p, _,_ in datasets])
        q_max = max([q.max() for _, _, q,_ in datasets])
        e_max = max([e.max() for e, _, _ ,_ in datasets])

        axes = ThreeDAxes(
            x_range=[0, p_max * 1.1, 50],
            y_range=[0.3, e_max * 1.1, 50],
            z_range=[0, q_max * 1.1, 50],
            x_length=7,
            y_length=7,
            z_length=7,
        ).rotate(-90 * DEGREES, UP).rotate(70 * DEGREES, RIGHT)

        lab_z = Tex("$q$").scale(1.7)
        lab_z.move_to(axes.z_axis.get_end() + OUT * 1).rotate(70 * DEGREES, RIGHT) #not the best, need to update
        lab_y = Tex("$e$").scale(1.7)
        lab_y.move_to(axes.y_axis.get_end() + OUT * 1).rotate(-90 * DEGREES, UP).rotate(70 * DEGREES, RIGHT)
        lab_x = Tex("$p'$").scale(1.7)
        lab_x.move_to(axes.x_axis.get_end() + OUT * 1).rotate(-90 * DEGREES, UP).rotate(70 * DEGREES, RIGHT)
        labels = VGroup(lab_y, lab_x, lab_z)
        self.play(Create(axes), Write(labels))

        #MCC parameters
        M = 1
        N = 2
        lambda_cam_clay = 0.1
        kappa_s = 0.01
        
        # STATE BOUNDARY SURFACE (SBS) OF THE MODIFIED CAM CLAY MODEL
        # p'-q-e space
        # Project: Animating Soil Models https://soilmodels.com/soilanim
        # by: Gertraud Medicus, University of Innbruck, 09/2021

        def surface_func(p0_local, p_ratio):
            v_plot = (
                    N - lambda_cam_clay * np.log(p0_local)
                    + kappa_s * np.log(1 / p_ratio)
            )
            q = M * np.sqrt(
                p0_local * p_ratio
                * (p0_local - p0_local * p_ratio)
            )
            return np.array([
                p0_local*p_ratio,
                v_plot - 1,
                q if not np.isnan(q) else 0
            ])

        sbs = Surface(
            lambda u, v: axes.c2p(
                *surface_func(u, v)
            ),
            u_range=[10, 820],
            v_range=[0.01, 0.99],
            resolution=(50, 50),
            fill_opacity=0.3,
            stroke_width=0.4,
            checkerboard_colors=False,
            fill_color=BLUE
        )

        u_val = 0.5
        CSL = ParametricFunction(
            lambda t: axes.c2p(
                *surface_func(t, u_val)
            ),
            t_range=[10, 800],
            color=YELLOW,
            stroke_width=3
        )

        loadPath = []
        for i, (e, p, q, pc) in enumerate(datasets):
            points = [axes.c2p(p[j], e[j], q[j]) for j in range(len(e))]
            curve = VMobject(color=colors[i])
            curve.set_points_as_corners(points)
            loadPath.append(curve)

        self.play( Create(sbs), run_time=3, rate_func=ease_in_sine)
        self.wait(1)

        def get_mcc_curve(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            N = len(e)
            idx = min(int(tracker.get_value() * (N - 1)), N - 1)
            p_c = pc[idx] 
            e_c = e[idx] 
            num_points=200
            p_vals = np.linspace(0, p_c, num_points)
            q_vals = M * np.sqrt(p_vals * (p_c - p_vals))
            points = [axes.c2p(p_i, e_c, q_i) for p_i, q_i in zip(p_vals, q_vals)]
            curve = VMobject(color=colors[dataset_idx % len(colors)])
            curve.set_points_as_corners(points)
            return curve

        mcc_curves = []
        for i in range(len(datasets)):
            curve = get_mcc_curve(i)
            curve.add_updater(lambda m, idx=i: m.become(get_mcc_curve(idx)))
            self.play(Create(curve))
            mcc_curves.append(curve)
            
        #print the value e,p',q
        def get_vector_tex(dataset_idx):
            e, p, q, pc = datasets[dataset_idx]
            N = len(e)
            idx = min(int(tracker.get_value() * (N - 1)), N - 1)
            color = colors[dataset_idx % len(colors_e)]
            return MathTex(
                r"\left[ e, p^{\prime}, q \right] = "
                f"[{e[idx]:.2f}, {p[idx]:.2f}, {q[idx]:.2f}]",
                color=color,
                font_size=30
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
    
        self.play(*[Create(curve) for curve in loadPath],
                  tracker.animate.set_value(1),run_time=7, rate_func=ease_in_sine)
        self.play(Create(CSL),run_time=1)
        self.wait(1)

        full_system = VGroup(axes, labels,CSL,sbs,loadPath,mcc_curves)
        for axis in [LEFT]:
            self.play(
                Rotate(
                    full_system,
                    70 * DEGREES,
                    about_point=ORIGIN,
                    axis=axis
                ),
                run_time=2
            )
        self.wait(2)
        u_val1 = 530.12
        EL1 = ParametricFunction(
            lambda t: axes.c2p(
                *surface_func(u_val1, t)
            ),
            t_range=[0.01, 0.99],
            color=WHITE,
            stroke_width=3
        )
        u_val1 = 567.08
        EL2 = ParametricFunction(
            lambda t: axes.c2p(
                *surface_func(u_val1, t)
            ),
            t_range=[0.01, 0.99],
            color=WHITE,
            stroke_width=3
        )
        u_val1 = 626.52
        EL3 = ParametricFunction(
            lambda t: axes.c2p(
                *surface_func(u_val1, t)
            ),
            t_range=[0.01, 0.99],
            color=WHITE,
            stroke_width=3
        )
        self.play(
            ReplacementTransform(mcc_curves[0], EL1),
            ReplacementTransform(mcc_curves[1], EL2),
            ReplacementTransform(mcc_curves[2], EL3)
        )
        self.remove(mcc_curves[0],mcc_curves[1],mcc_curves[2])
        
        for axis in [OUT]:
            self.play(
                full_system.animate
                .rotate(PI / 2, axis=OUT) 
                .shift(OUT * 2) 
                .shift(RIGHT * -4)
                .shift(DOWN * 8),
                run_time=2
            )
        self.play(sbs.animate.set_fill(opacity=0.1), sbs.animate.set_stroke(width=0.1), run_time=1)
        self.wait(5)
