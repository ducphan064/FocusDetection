import sys, os, time, math
import numpy as np
import cv2
import mediapipe as mp
import platform
from utils import *
import streamlit as st

# 5) CALIBRATION (2s)
def calibrate_neutral(cap, face, seconds=2.0, stframe=None):
    ys, ps, ears, mars = [], [], [], []
    t0 = time.time()

    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face.process(rgb)

        if not res.multi_face_landmarks:
            cv2.putText(frame, "Calibration: look straight...",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
        else:
            lms = res.multi_face_landmarks[0].landmark
            leye = get_eye_points(lms, LEFT_EYE, w, h)
            reye = get_eye_points(lms, RIGHT_EYE, w, h)
            ear = 0.5*(eye_aspect_ratio(leye)[-1]+eye_aspect_ratio(reye)[-1]) if len(leye)==6 and len(reye)==6 else None
            mar_val = mouth_aspect_ratio(lms, w, h)

            yaw, pitch,_,_,_,_ = solve_head_pose(lms, w, h)
            if yaw is not None:
                ys.append(yaw)
                ps.append(pitch)
            if ear is not None:
                ears.append(ear)
            if mar_val is not None:
                mars.append(mar_val)

            cv2.putText(frame, "Calibrating... Keep looking straight",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)

        # --- Render to Streamlit instead of OpenCV GUI ---
        if stframe is not None:
            stframe.image(frame, channels="BGR")

        # VERY IMPORTANT: stop Streamlit from freezing
        time.sleep(0.01)

    yaw0 = float(np.mean(ys)) if ys else 0.0
    pitch0 = float(np.mean(ps)) if ps else 0.0
    ear0 = float(np.mean(ears)) if ears else 0.28
    mar0 = float(np.mean(mars)) if mars else 0.02

    return yaw0, pitch0, ear0, mar0

def main():
    print("[DEBUG] Python:", sys.version)
    print("[DEBUG] OpenCV:", cv2.__version__)
    st.markdown(
            "<h1 style='text-align:center; color:white;'>Focus Monitor Dashboard</h1>",
            unsafe_allow_html=True
        )
    
    st.set_page_config(
        page_title="Focus Monitor",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    col_main, col_right = st.columns([2, 2])

    # ---- MAIN SCREEN (big video frame) ----


    with col_main:
        stframe = st.empty()

        # Nội dung chính canh giữa và chữ lớn


        st.markdown(
        "<div style='text-align:center; font-size:36px; font-weight:bold;'>Main monitor</div>",
        unsafe_allow_html=True
    )

        # Status box với chữ lớn
        status_box = st.empty()
        status_box.markdown(
            "<div style='text-align:center; font-size:28px; color:blue;'>Status: Ready</div>",
            unsafe_allow_html=True
        )


    # ---- RIGHT SIDE (2 rows) ----
    with col_right:
        # Hàng trên: 2 screen mắt
        row1 = st.columns(2)
        left_eye_box = row1[0].empty()
        left_eye_offset = row1[0].empty()
        right_eye_offset = row1[0].empty()
        right_eye_box = row1[1].empty()
        left_eye_ear = row1[1].empty()
        right_eye_ear = row1[1].empty()
        
        offset_final = row1[0].empty()
        ear_final = row1[1].empty()

        # Hàng dưới: 1 screen head
        row2 = st.columns(2)
        head_box = row2[0].empty()
        pitch_eq = row2[0].empty()
        yaw_eq = row2[0].empty()
        mouth_box = row2[1].empty()
        mar_eq = row2[1].empty()

   
    # st_calibration = st.empty()
    cap = open_camera()

    
    # if not cap.isOpened():
    #     st.error("Camera open failed. Try index 1...")
    #     cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    #     if not cap.isOpened():
    #         print("[FATAL] No camera. Check macOS Camera permissions.")
    #         return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_face = mp.solutions.face_mesh
    fsm = FocusFSM(th_on=0.75, th_off=0.5, dwell_on=1.2, dwell_off=0.8)
    flt = FeatureSmoother(alpha=0.75)

    # === Panel & Hotkeys state ===
    # sens = {"th_on": fsm.th_on, "dwell_on": fsm.dwell_on}
    
    mp_drawing = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles
    with mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as model:

        yaw0, pitch0, ear0, mar0_calib = calibrate_neutral(cap, model, seconds=2.0, stframe=stframe)
        print(f"[CALIB] yaw0={yaw0:.2f}, pitch0={pitch0:.2f}, ear0={ear0:.3f}, mar0={mar0_calib:.3f}")

        #bổ sung ngày 3/11/25
        yawn = YawnTracker(ema_alpha=0.75, ratio_on=8.0, ratio_off=1.5,
                   dwell_on=0.8, dwell_off=0.25, cap_update=1.8)
        yawn.baseline = mar0_calib
        
        last_state = fsm.state
        beep_path = "/System/Library/Sounds/Pop.aiff" if sys.platform=="darwin" else None

        while True:
            ok, frame = cap.read()
            # frame = digital_zoom(frame)
            if not ok: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = model.process(rgb) #Equal to y_pred = model.predict

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                lms_draw = res.multi_face_landmarks[0]
            # DRAW FULL FACEMESH
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lms_draw,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

                # DRAW EYES + IRIS with nicer colors
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lms_draw,
                    connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lms_draw,
                    connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )

                # IRIS OUTLINE
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lms_draw,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                )

                # EAR
                leye = get_eye_points(lms, LEFT_EYE, w, h)
                reye = get_eye_points(lms, RIGHT_EYE, w, h)

                d26_r, d35_r, d14_r, ear_right = eye_aspect_ratio(reye) if len(leye)==6 and len(reye)==6 else None
                d26_l, d35_l, d14_l, ear_left = eye_aspect_ratio(leye) if len(leye)==6 and len(reye)==6 else None
                ear_val = 0.5*(ear_left+ear_right) if len(leye)==6 and len(reye)==6 else None

                yaw, pitch, rvec, tvec, cam_mtx, dist = solve_head_pose(lms, w, h)
                L, offL, R, offR, gaze_off = compute_gaze_offset(lms, w, h)
                # print("Gaze offset", gaze_off)

######################## BỔ SUNG 3/11/25

                # # === MAR (mouth) & yawn
                d_v1, d_v2, d_v3, horizontal, mar_val = mouth_aspect_ratio(lms, w, h)
                yinfo = yawn.update(mar_val, tnow=time.time())

                # # UI: hiển thị MAR/baseline/ratio & trạng thái
                # cv2.putText(frame, f"MAR={0.0 if mar_val is None else mar_val:.2f} "
                #                 f"MAR0={0.0 if yinfo['baseline'] is None else yinfo['baseline']:.2f} "
                #                 f"r={yinfo['ratio']:.2f}  YAWN={yinfo['state'].upper()}",
                #             (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

                # # Vòng tròn tiến trình dwell của YAWN
                # yc = (300, 150); yr = 18
                # cv2.circle(frame, yc, yr, (200,200,200), 2)
                # cv2.ellipse(frame, yc, (yr, yr), -90, 0, yinfo['progress']*360,
                #             (0,165,255), 3)  # cam
                # cv2.putText(frame, "yawn dwell", (yc[0]-45, yc[1]+40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                # # Border flash cam khi đang (hoặc sắp) YAWN
                # if yinfo['state'] == "yawn" or (yinfo['ratio'] >= yawn.r_on and yinfo['progress'] > 0):
                #     cv2.rectangle(frame, (4,4), (w-4, h-4), (0,165,255), 6)  # cam
###################################
                # Smoothing
                yaw_s, pitch_s, ear_s, gaze_s, mar_s = flt.update(yaw, pitch, ear_val, gaze_off, mar_val)
                left_eye_crop = crop_region(frame, lms, LEFT_EYE)
                right_eye_crop = crop_region(frame, lms, RIGHT_EYE)
                face_crop= crop_region(frame, lms, FACE_OVAL, padding=200)
                mouth_crop = crop_region(frame, lms, MOUTH)
                # print('Eye crop shape:', left_eye_crop.shape)
                # print('face crop shape:', face_crop.shape)
                # print('mouth crop shape:', mouth_crop.shape)
                if right_eye_crop is not None and right_eye_crop.size != 0:
                    right_eye_crop = cv2.resize(right_eye_crop, EYE_DISPLAY)
                    right_eye_box.image(right_eye_crop, channels="BGR", caption="Right Eye")
                    right_eye_ear.latex(fr"""
                                        \mathrm{{Right EAR}} =
                                        \frac{{
                                        \lVert P_2 - P_6 \rVert
                                        + \lVert P_3 - P_5 \rVert
                                        }}{{
                                        2 \cdot \lVert P_1 - P_4 \rVert
                                        }}\\
                                        =
                                        \frac{{
                                        {d26_r:.3f}
                                        +
                                        {d35_r:.3f}
                                        }}{{
                                        2 \cdot {d14_r:.3f}
                                        }}
                                        = \mathrm{ear_right:.3f}
                                        """)
                    right_eye_offset.latex(fr"""
                                            \mathrm{{Right\ Offset}} =
                                            \frac{{\lVert v_R \rVert}}{{\lVert L_1 - L_2 \rVert}} \\
                                            =
                                            \frac{{{R[0]:.3f}}}{{{R[1]:.3f}}}
                                            = {offR:.3f}
                                            """)
                else:
                    right_eye_ear.write("Right eye out of frame")
                if left_eye_crop is not None and left_eye_crop.size != 0:
                    left_eye_crop = cv2.resize(left_eye_crop, EYE_DISPLAY)
                    left_eye_box.image(left_eye_crop, channels="BGR", caption="Left Eye")
                    left_eye_ear.latex(fr"""
                                        \mathrm{{Left EAR}} =
                                        \frac{{
                                        \lVert P_2 - P_6 \rVert
                                        + \lVert P_3 - P_5 \rVert
                                        }}{{
                                        2 \cdot \lVert P_1 - P_4 \rVert
                                        }}\\
                                        =
                                        \frac{{
                                        {d26_l:.3f}
                                        +
                                        {d35_l:.3f}
                                        }}{{
                                        2 \cdot {d14_l:.3f}
                                        }}
                                        = \mathrm{ear_left:.3f}
                                        """)
                    left_eye_offset.latex(fr"""
                                        \mathrm{{Left\ Offset}} =
                                        \frac{{\lVert v_L \rVert}}{{\lVert L_1 - L_2 \rVert}} \\
                                        =
                                        \frac{{{L[0]:.3f}}}{{{L[1]:.3f}}}
                                        = {offL:.3f}
                                        """)

                else:
                    left_eye_box.write("Left eye out of frame")
                if (left_eye_crop is not None and right_eye_crop is not None):
                    ear_final.latex(fr"""
                                        \mathrm{{EAR}} =
                                        \frac{{
                                        Right EAR
                                        + Left EAR
                                        }}{{
                                        2
                                        }}\\
                                        =
                                        \frac{{
                                        {ear_left:.3f}
                                        +
                                        {ear_right:.3f}
                                        }}{{
                                        2
                                        }}
                                        = \mathrm{ear_val:.3f}
                                        """)
                    offset_final.latex(fr"""
                                        \mathrm{{Offset}} =
                                        \frac{{
                                        Right Offset
                                        + Left Offset
                                        }}{{
                                        2
                                        }}\\
                                        =
                                        \frac{{
                                        {offR:.3f}
                                        +
                                        {offL:.3f}
                                        }}{{
                                        2
                                        }}
                                        = \mathrm{ear_val:.3f}
                                        """)
                if face_crop is not None and face_crop.size != 0:
                    head_box.image(face_crop, channels="BGR", caption="Face")
                    pitch_eq.latex(fr"""
                                \mathrm{{pitch}} =
                                \arctan2\!\left(-R_{{2,0}},\, \sqrt{{R_{{0,0}}^2 + R_{{1,0}}^2}}\right) \cdot \frac{{180}}{{\pi}}
                                \\
                                = {pitch:.3f}^\circ
                                """)
                    yaw_eq.latex(fr"""
                                \mathrm{{yaw}} =
                                \arctan2\!\left(R_{{1,0}},\, R_{{0,0}}\right) \cdot \frac{{180}}{{\pi}}
                                \\
                                = {yaw:.3f}^\circ
                                """)
                else:
                    head_box.write("Face out of frame")
                if mouth_crop is not None and mouth_crop.size != 0:
                    mouth_crop = cv2.resize(mouth_crop, EYE_DISPLAY)
                    mouth_box.image(mouth_crop, channels="BGR", caption="Mouth")
                    mar_eq.latex(fr"""
                                \mathrm{{MAR}} = 
                                \frac{{
                                \lVert P_2 - P_8 \rVert 
                                + \lVert P_3 - P_7 \rVert 
                                + \lVert P_4 - P_6 \rVert
                                }}{{
                                3 \cdot \lVert P_1 - P_5 \rVert
                                }}
                                \\
                                = \frac{{
                                {d_v1:.3f}
                                + {d_v2:.3f}
                                + {d_v3:.3f}
                                }}{{
                                3 {horizontal:.3f} }} = \mathrm{mar_val:.3f}
                                """)
                else:
                    mouth_box.write("Mouth out of frame")
                # Score + FSM
                mar0 = yinfo['baseline']
                # Sửa lệnh gọi hàm (cần 8 tham số)
                score = compute_score(yaw_s, pitch_s, ear_s, mar_s, gaze_s, yaw0, pitch0, ear0, mar0)              
                state_before = fsm.state
                state, progress = fsm.update(score)

                # ===== UI: trạng thái, số liệu
                # cv2.putText(frame, f"STATE: {state.upper()}",
                #             (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                #             (0,255,0) if state=="focused" else (0,0,255), 2)
                # cv2.putText(frame, f"yaw={0.0 if yaw_s is None else yaw_s:.1f}  "
                #                    f"pitch={0.0 if pitch_s is None else pitch_s:.1f}  "
                #                    f"EAR={0.0 if ear_s is None else ear_s:.2f}  "
                #                    f"gaze_off={0.0 if gaze_s is None else gaze_s:.2f}",
                #             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

                # # SCORE bar + ngưỡng
                # bar_x, bar_y, bar_w, bar_h = 10, 90, 220, 18
                # cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), 1)
                # fill_w = int(bar_w * max(0.0, min(1.0, score)))
                # cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h),
                #               (0,0,255) if score>=fsm.th_on else (0,200,0), -1)
                # x_on  = bar_x + int(bar_w * fsm.th_on)
                # x_off = bar_x + int(bar_w * fsm.th_off)
                # cv2.line(frame, (x_on, bar_y), (x_on, bar_y+bar_h), (0,0,255), 2)
                # cv2.line(frame, (x_off, bar_y), (x_off, bar_y+bar_h), (0,255,0), 2)
                # cv2.putText(frame, f"SCORE {score:.2f}", (bar_x, bar_y-5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # # Vòng tròn dwell + border & countdown
                # center=(150, 150); radius=18
                # cv2.circle(frame, center, radius, (200,200,200), 2)
                # cv2.ellipse(frame, center, (radius, radius), -90, 0, progress*360, (0,255,255), 3)
                # cv2.putText(frame, "dwell", (center[0]-25, center[1]+40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                # ứng viên chuyển?
                candidate, target, need = False, None, None
                if state == "focused" and score >= fsm.th_on:
                    candidate, target, need = True, "UNFOCUSED", fsm.dwell_on
                elif state == "unfocused" and score <= fsm.th_off:
                    candidate, target, need = True, "FOCUSED", fsm.dwell_off

                def draw_border(img, color, t=6):
                    cv2.rectangle(img, (2,2), (w-2, h-2), color, t)

                if candidate and need is not None:
                    time_left = max(0.0, need * (1.0 - progress))
                    draw_border(frame, (0,255,255), 8)  # vàng
                    cv2.putText(frame, f"{target} in {time_left:.1f}s", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                else:
                    draw_border(frame, (0,255,0) if state=="focused" else (0,0,255), 6)

                # Beep khi vừa chuyển
                if fsm.state != state_before and beep_path and os.path.exists(beep_path):
                    if fsm.state == "unfocused":
                        os.system(f"afplay '{beep_path}' &")

            else:
                cv2.putText(frame, "No face detected", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            stframe.image(
                            frame,
                            channels="BGR",
                        )

                        # OPTIONAL: show debug state
            # Cập nhật status với style
            status_box.markdown(
                f"<div style='text-align:center; font-size:28px; color:blue;'>FSM state: {state.upper()}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.5)

            # Streamlit-controlled stop button
            # if st.button("Stop"):
            #     break

            # --- Panel & Hotkeys ---
            # put_panel(frame, sens)
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:  # ESC
            #     break
            # elif key == ord(']'):
            #     sens["th_on"] = min(0.99, sens["th_on"] + 0.02);  fsm.th_on = sens["th_on"]
            # elif key == ord('['):
            #     sens["th_on"] = max(0.50, sens["th_on"] - 0.02);  fsm.th_on = sens["th_on"]
            # elif key == ord('='):  # tăng dwell_on (khó vào UNFOCUS)
            #     sens["dwell_on"] = min(3.0, sens["dwell_on"] + 0.1);  fsm.dwell_on = sens["dwell_on"]
            # elif key == ord('-'):  # giảm dwell_on (dễ vào UNFOCUS)
            #     sens["dwell_on"] = max(0.3, sens["dwell_on"] - 0.1);  fsm.dwell_on = sens["dwell_on"]

            # cv2.imshow("Focus Monitor", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
