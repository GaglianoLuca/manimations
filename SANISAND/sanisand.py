from manim import *
import numpy as np
import pandas as pd


class SANISAND(ThreeDScene):
    def construct(self):

        def load_data(file):

            raw = pd.read_csv(file, sep=r"\s+", engine="python", header=None, skiprows=1)
            n_cols = raw.shape[1]

            # column UMAT: time, stran, stress, statev...
            base_cols = ["time(1)", "time(2)"] \
                        + [f"stran({i})" for i in range(1, 7)] \
                        + [f"stress({i})" for i in range(1, 7)]

            n_statev = n_cols - len(base_cols)
            statev_cols = [f"statev({i})" for i in range(1, n_statev + 1)]
            raw.columns = base_cols + statev_cols
            data = raw

            e = data['statev(1)']
            s11 = -data['stress(1)']
            s22 = -data['stress(2)']
            s33 = -data['stress(3)']
            pc = data['statev(8)']
            eps1 = data['stran(1)']
            eps2 = -data['stran(2)']
            eps3 = -data['stran(3)']
            p = data['statev(69)']
            q = data['statev(70)']

            return (e.to_numpy(), p.to_numpy(), q.to_numpy(), pc.to_numpy(),
                    eps1.to_numpy(), eps2.to_numpy(), eps3.to_numpy(),
                    s11.to_numpy(), s22.to_numpy(), s33.to_numpy())

        files = ["output_CU_1A.out","output_CU_1B.out","output_CU_1C.out","output_CU_2A.out","output_CU_2B.out","output_CU_2C.out"]
        colors = [GREEN_A, GREEN_B, GREEN_C, RED_A, RED_B, RED_C]

        datasets = [load_data(f) for f in files]
        
        #DEBUG
        '''
        for file, data in zip(files, datasets):
            print(f"\n===== {file} =====")
            e, p, q, pc, eps1, eps2, eps3, s11, s22, s33 = data

            print("s22:", s22[:100], "...", "len =", len(e))
            print("s11:", s11[:100], "...", "len =", len(e))
            print("s33:", s33[:100], "...", "len =", len(e))
            print("p:", p[:500], "...", "len =", len(p))
            print("q:", q[:500], "...", "len =", len(q))
            print("pc:", pc[:50], "...", "len =", len(pc))
            print("eps1:", eps1[:5])
            print("eps2:", eps2[:5])
            print("eps3:", eps3[:5])
            print("s11:", s11[:5])
            print("s22:", s22[:5])
            print("s33:", s33[:5])
            print("e:", e[:100])
        '''


        ##############################
        # AUTO AXIS RANGES FROM DATA
        ##############################
        def auto_max_range(values, margin_frac=0.1):
            vmax = float(values.max())
            margin = vmax * margin_frac if vmax > 0 else 1e-3
            axis_max = vmax + margin
            step = axis_max / 5
            return 0, axis_max, step

        def auto_range(values, margin_frac=0.1):
            vmin, vmax = float(values.min()), float(values.max())
            span = vmax - vmin
            margin = span * margin_frac if span > 0 else max(abs(vmax), 1e-3) * margin_frac
            axis_min = vmin - margin
            axis_max = vmax + margin
            step = (axis_max - axis_min) / 5
            return axis_min, axis_max, step

        all_e = np.concatenate([d[0] for d in datasets])
        all_p = np.concatenate([d[1] for d in datasets])
        all_q = np.concatenate([d[2] for d in datasets])
        all_eps1 = np.concatenate([d[4] for d in datasets])

        e_axis_min, e_axis_max, e_step = auto_range(all_e)
        p_axis_min, p_axis_max, p_step = auto_max_range(all_p)
        q_axis_min, q_axis_max, q_step = auto_max_range(all_q)
        eps_axis_min, eps_axis_max, eps_step = auto_max_range(all_eps1)
        print("p_min:", p_axis_min)

        ##############################
        # SANISAND PARAMETERS
        ##############################
        M = 1.25
        c = 0.688
        e0 = 0.934
        lambda_c = 0.019
        xi = 0.712
        p_atm = 100
        n_b = 1.1
        n_d = 3.5
        m_yield = 0.01
        p_max_yield = 2000

        def g_theta(theta, c):
            return (2 * c) / ((1 + c) - (1 - c) * np.cos(3 * theta))

        ##############################
        # 3D SURFACE PLOT (S1-S2-S3)
        ##############################

        axes3d = ThreeDAxes(
            x_range=[0, p_max_yield, 200], y_range=[0, p_max_yield, 200], z_range=[0, p_max_yield, 200],
            x_length=3.8, y_length=3.8, z_length=3.8,
        ).shift(LEFT * 5 + UP * 1.5)

        labels3d = axes3d.get_axis_labels(
            Tex(r"$\sigma_1$", font_size=26), Tex(r"$\sigma_2$", font_size=26), Tex(r"$\sigma_3$", font_size=26)
        )
        self.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES, distance=500)
        self.add(axes3d, labels3d)

        def cone_surface(p, theta, M_val):
            M_theta = M_val * g_theta(theta, c)
            r = np.sqrt(2 / 3) * M_theta * p

            s1 = r * np.sqrt(2 / 3) * np.cos(theta)
            s2 = r * np.sqrt(2 / 3) * np.cos(theta - 2 * np.pi / 3)
            s3 = r * np.sqrt(2 / 3) * np.cos(theta + 2 * np.pi / 3)

            return np.array([p + s1, p + s2, p + s3])

        critical_surface_cone = Surface(
            lambda u, v: axes3d.c2p(*cone_surface(u, v, M)),
            u_range=[0, p_max_yield],
            v_range=[0, TAU],
            resolution=(20, 20),
            fill_opacity=0.3,
            stroke_width=0.4,
            checkerboard_colors=False,
            fill_color=BLUE,
        )

        hydro_axis = Line(axes3d.c2p(0, 0, 0), axes3d.c2p(p_max_yield, p_max_yield, p_max_yield), color=YELLOW, stroke_width=3)

        cone_group = VGroup(
            axes3d,
            labels3d,
            critical_surface_cone,
            hydro_axis,
        )
        cone_group.move_to(RIGHT + DOWN * 5)
        cone_group.scale(0.7)

        #self.add(cone_group)
        rotation_center = axes3d.c2p(0, 0, 0)

        # ================================================================
        # P'-Q  (top right)
        # ================================================================
        p_axis_min, p_axis_max, p_step = 0, 2000, 500
        p_labels = [500, 1000,1500, 2000]
        q_axis_min, q_axis_max, q_step = 0, 2500, 500
        q_labels = [500, 1000,1500, 2000,2500]

        pq_axes = Axes(
            x_range=[p_axis_min, p_axis_max, p_step],
            y_range=[q_axis_min, q_axis_max, q_step],
            x_length=4.2, y_length=3.2,
            axis_config={"include_tip": False, "font_size": 16},
        ).shift(RIGHT * 3.6 + UP * 1.9)
        pq_axes.add_coordinates(p_labels, q_labels, num_decimal_places=0, font_size=14)
        pq_labels = pq_axes.get_axis_labels(Tex("p'", font_size=28), Tex("q", font_size=28))
        print("p_min_2:", p_axis_min)
        csl_line = pq_axes.plot(lambda p: M * p, x_range=[0, p_axis_max], color=RED, stroke_width=3)
        csl_line.set_z_index(1)


        def e_c(p):
            return e0 - lambda_c * (np.maximum(p, 1e-6) / p_atm) ** xi

        # ================================================================
        # Q - EPS_A  (bottom right)
        # ================================================================

        qeps_axes = Axes(
            x_range=[eps_axis_min, eps_axis_max,0.05],
            y_range=[q_axis_min, q_axis_max, q_step],
            x_length=4.2, y_length=2.8,
            axis_config={"include_tip": False, "font_size": 16},
        ).shift(RIGHT * 3.6 + DOWN * 2.1)
        qeps_axes.add_coordinates(font_size=14)
        qeps_axes.x_axis.add_numbers(
            font_size=14,
            num_decimal_places=2
        )

        qeps_axes.y_axis.add_numbers(
            font_size=14,
            num_decimal_places=0
        )
        qeps_labels = qeps_axes.get_axis_labels(
            Tex(r"$\varepsilon_a$ (\%)", font_size=24), Tex("q", font_size=24)
        )

        # ================================================================
        # Q - e (void ratio)  (bottom left)
        # ================================================================

        qe_axes = Axes(
            x_range=[q_axis_min, q_axis_max, q_step],
            y_range=[e_axis_min, e_axis_max, e_step],
            x_length=4.2,
            y_length=2.8,
            axis_config={"include_tip": False, "font_size": 16},
        ).shift(LEFT * 4.5 + DOWN * 2.2)

        qe_axes.x_axis.add_numbers(
            font_size=14,
            num_decimal_places=0
        )

        qe_axes.y_axis.add_numbers(
            font_size=14,
            num_decimal_places=2
        )

        qe_labels = qe_axes.get_axis_labels(
            Tex("q", font_size=24),
            Tex("e", font_size=24)
        )


        def legend_line(style, y_offset, label_text):
            if style == "solid":
                sample = Line(ORIGIN, RIGHT * 0.5, color=GRAY_B, stroke_width=3)
            elif style == "dashed_wide":
                sample = DashedLine(ORIGIN, RIGHT * 0.5, color=GRAY_B, stroke_width=2, dash_length=0.12)
            else:  # dashed_narrow
                sample = DashedLine(ORIGIN, RIGHT * 0.5, color=GRAY_B, stroke_width=2, dash_length=0.04)
            label = Tex(label_text, font_size=18, color=WHITE).next_to(sample, RIGHT, buff=0.15)
            group = VGroup(sample, label).shift(UP * y_offset)
            return group


        legend_M = legend_line("solid", 0, "$M$ (yield)")
        legend_Mb = legend_line("dashed_wide", -0.35, "$M_b$ (bounding)")
        legend_Md = legend_line("dashed_narrow", -0.70, "$M_d$ (dilatancy)")

        legend_group = VGroup( legend_M, legend_Mb, legend_Md)
        legend_group.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend_group.scale(0.65)
        legend_group.next_to(pq_axes, RIGHT, buff=0.2)  # centro-dx del grafico p-q
        legend_group.shift(LEFT * 0.5)

        self.add_fixed_in_frame_mobjects(
            pq_axes, pq_labels, csl_line,
            qeps_axes, qeps_labels, qe_axes, qe_labels, legend_group

        )


        self.play(
            Create(pq_axes), Write(pq_labels),
            Create(csl_line),
            Create(qeps_axes), Write(qeps_labels),
            Create(qe_axes), Write(qe_labels),
            Create(hydro_axis),
            run_time=0.8,
        )

        ##############################
        # BUILD PATHS
        ##############################

        def build_path(points, color, stroke_width=4):
            path = VMobject(color=color, stroke_width=stroke_width)
            path.set_points_as_corners(points)
            return path

        max_len = max(len(e) for e, *_ in datasets) - 1

        path3d_list = []
        dot3d_list = []
        all_dynamic_mobjects = []  

        for i, (e, p, q, pc, eps1, eps2, eps3, s11, s22, s33) in enumerate(datasets):
            color = colors[i]
            n_i = len(p) - 1

            progress = ValueTracker(0)  
            create_anims = [] 
            dynamic_mobjects = []
            fading_mobjects = []  

            def idx_of(n_local=n_i, progress=progress):
                return int(np.clip(progress.get_value() * n_local, 0, n_local))

            # 3D 
            pts3d = [axes3d.c2p(s11[j], s22[j], s33[j]) for j in range(len(s11))]
            path3d = build_path(pts3d, color)
            path3d_list.append(path3d)
            self.add(path3d)
            create_anims.append(Create(path3d, run_time=8, rate_func=linear))

            # P'-Q
            pts_qp = [pq_axes.c2p(p[j], q[j]) for j in range(len(p))]
            path_qp = build_path(pts_qp, color)
            path_qp.set_z_index(3)
            self.add_fixed_in_frame_mobjects(path_qp)
            create_anims.append(Create(path_qp, run_time=8, rate_func=linear))

            # Q - eps_a
            pts_qeps = [qeps_axes.c2p(eps1[j], q[j]) for j in range(len(eps1))]
            path_qeps = build_path(pts_qeps, color)
            self.add_fixed_in_frame_mobjects(path_qeps)
            create_anims.append(Create(path_qeps, run_time=8, rate_func=linear))

            # Q - e
            pts_qe = [qe_axes.c2p(q[j], e[j]) for j in range(len(e))]
            path_qe = build_path(pts_qe, color)

            self.add_fixed_in_frame_mobjects(path_qe)
            create_anims.append(Create(path_qe, run_time=8, rate_func=linear))

            def make_wedge_mobject(p_arr=p, q_arr=q, idx_fn=idx_of, color=color):
                idx = idx_fn()
                p_cur, q_cur = p_arr[idx], q_arr[idx]
                r_cur = q_cur / p_cur if p_cur != 0 else 0.0
                origin_pt = pq_axes.c2p(0, 0)
                up_end = pq_axes.c2p(p_max_yield, max((r_cur + m_yield) * p_max_yield, 0))
                dn_end = pq_axes.c2p(p_max_yield, max((r_cur - m_yield) * p_max_yield, 0))
                fill = Polygon(origin_pt, up_end, dn_end, color=color,
                               fill_color=color, fill_opacity=0.15, stroke_width=0)
                line_up = Line(origin_pt, up_end, color=color, stroke_width=1.5)
                line_dn = Line(origin_pt, dn_end, color=color, stroke_width=1.5)
                return VGroup(fill, line_up, line_dn)
    
            wedge = always_redraw(make_wedge_mobject)
            wedge.set_z_index(2)
            self.add_fixed_in_frame_mobjects(wedge)
            fading_mobjects.append(wedge)  
            
            def make_bound_mobject(e_arr=e, p_arr=p, idx_fn=idx_of, color=color):
                idx = idx_fn()
                psi = e_arr[idx] - e_c(p_arr[idx])
                Mb_curr = M * np.exp(-n_b * psi)
                origin_pt = pq_axes.c2p(0, 0)
                end_pt = pq_axes.c2p(p_axis_max, max(Mb_curr * p_axis_max, 0))
                return DashedVMobject(
                    Line(
                        origin_pt,
                        end_pt,
                        color=color,
                        stroke_width=2,
                        stroke_opacity=0.7,
                    ),
                    num_dashes=20,
                    dashed_ratio=0.5,
                )

            bound_line = always_redraw(make_bound_mobject)
            bound_line.set_z_index(2)
            self.add_fixed_in_frame_mobjects(bound_line)
            fading_mobjects.append(bound_line)  


            def make_dil_mobject(e_arr=e, p_arr=p, idx_fn=idx_of, color=color):
                idx = idx_fn()
                psi = e_arr[idx] - e_c(p_arr[idx])
                Md_curr = M * np.exp(n_d * psi)
                origin_pt = pq_axes.c2p(0, 0)
                end_pt = pq_axes.c2p(p_axis_max, max(Md_curr * p_axis_max, 0))
                return DashedVMobject(
                    Line(
                        origin_pt,
                        end_pt,
                        color=color,
                        stroke_width=2,
                        stroke_opacity=0.7,
                    ),
                    num_dashes=40,
                    dashed_ratio=0.25,
                )
            dil_line = always_redraw(make_dil_mobject)
            dil_line.set_z_index(2)
            self.add_fixed_in_frame_mobjects(dil_line)
            fading_mobjects.append(dil_line)  

            def make_bounding_surface_3d(e_arr=e, p_arr=p, idx_fn=idx_of, color=color):
                idx = idx_fn()
                psi = e_arr[idx] - e_c(p_arr[idx])
                Mb_curr = M * np.exp(-n_b * psi)
                return Surface(
                    lambda u, v: axes3d.c2p(*cone_surface(u, v, Mb_curr)),
                    u_range=[0, 2500],
                    v_range=[0, TAU],
                    resolution=(14, 14),
                    fill_opacity=0.2,
                    stroke_width=0.3,
                    checkerboard_colors=False,
                    fill_color=color,
                )

            bounding_surface_3d = always_redraw(make_bounding_surface_3d)
            self.add(bounding_surface_3d)
            fading_mobjects.append(bounding_surface_3d)

            def make_dot_3d(s11_arr=s11, s22_arr=s22, s33_arr=s33, idx_fn=idx_of, color=color):
                idx = idx_fn()
                return Dot3D(axes3d.c2p(s11_arr[idx], s22_arr[idx], s33_arr[idx]), radius=0.06, color=color)

            dot_3d = always_redraw(make_dot_3d)
            self.add(dot_3d)
            dynamic_mobjects.append(dot_3d)
            dot3d_list.append(dot_3d)

            def make_dot_qp(p_arr=p, q_arr=q, idx_fn=idx_of, color=color):
                idx = idx_fn()
                dot = Dot(pq_axes.c2p(p_arr[idx], q_arr[idx]), radius=0.06, color=color)
                dot.set_z_index(4)
                return dot

            dot_qp = always_redraw(make_dot_qp)
            self.add_fixed_in_frame_mobjects(dot_qp)
            dynamic_mobjects.append(dot_qp)

            def make_dot_qeps(eps_arr=eps1, q_arr=q, idx_fn=idx_of, color=color):
                idx = idx_fn()
                dot = Dot(qeps_axes.c2p(eps_arr[idx], q_arr[idx]), radius=0.06, color=color)
                dot.set_z_index(4)
                return dot

            dot_qeps = always_redraw(make_dot_qeps)
            self.add_fixed_in_frame_mobjects(dot_qeps)
            dynamic_mobjects.append(dot_qeps)

            def make_dot_qe(e_arr=e, q_arr=q, idx_fn=idx_of, color=color):
                idx = idx_fn()
                dot = Dot(qe_axes.c2p( q_arr[idx],e_arr[idx]), radius=0.06, color=color)
                dot.set_z_index(4)
                return dot

            dot_qe = always_redraw(make_dot_qe)
            self.add_fixed_in_frame_mobjects(dot_qe)
            dynamic_mobjects.append(dot_qe)

            self.play(*create_anims, progress.animate.set_value(1), run_time=8, rate_func=linear)
           
            for mob in fading_mobjects:
                mob.clear_updaters()
            self.play(*[FadeOut(mob) for mob in fading_mobjects], run_time=0.5)

            all_dynamic_mobjects.extend(dynamic_mobjects)

        cone_group.add(path3d_list, dot3d_list)
        Create(critical_surface_cone)
        self.play(
            Rotate(
                cone_group,
                angle=TAU,
                axis=OUT,
                about_point=rotation_center,
                run_time=10,
                rate_func=smooth
            )
        )

        for mob in all_dynamic_mobjects:
            mob.clear_updaters()

        self.wait(2)
