import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import supervision as sv
from src.detector import TrafficDetector
from src.tracker import TrafficTracker
from src.analytics import TrafficAnalytics
from src.collision_detector import CollisionDetector
from src.traffic_predictor import TrafficPredictor
from src.speed_estimator import SpeedEstimator
from src.incident_detector import IncidentDetector
from src.heatmap_generator import HeatmapGenerator
import config
import logging

# Suppress harmless Tornado asyncio noise
logging.getLogger("tornado.web").setLevel(logging.WARNING)

# Page Config
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Smart Traffic Analysis System v2.0"},
)

# Reduce Streamlit rerun frequency
st.session_state.update(st.session_state)  # Initialize session state safely


# Initialize components
@st.cache_resource
def load_models():
    return {
        "detector": TrafficDetector(),
        "tracker": TrafficTracker(),
        "analytics": TrafficAnalytics(),
        "collision_detector": CollisionDetector(),
        "traffic_predictor": TrafficPredictor(),
        "speed_estimator": SpeedEstimator(),
        "incident_detector": IncidentDetector(),
        "heatmap_generator": HeatmapGenerator(),
    }


models = load_models()

# Sidebar
st.sidebar.title("🚦 Smart Traffic System")
st.sidebar.divider()

app_mode = st.sidebar.selectbox(
    "📊 Choose Mode",
    [
        "Live Dashboard",
        "Collision Alerts",
        "Traffic Prediction",
        "Speed Analysis",
        "Incident Report",
        "Heatmap Analytics",
        "Statistics Dashboard",
    ],
)

source_option = st.sidebar.radio("📹 Input Source", ["Upload Video", "Webcam (Demo)"])

uploaded_file = None
if source_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Traffic Video", type=["mp4", "avi", "mov"]
    )

st.sidebar.divider()
st.sidebar.write("### 🎛️ System Settings")
enable_heatmap = st.sidebar.checkbox("Enable Heatmap Overlay", value=True)
enable_speed_display = st.sidebar.checkbox("Show Speed Vectors", value=True)
enable_trajectory = st.sidebar.checkbox("Show Vehicle Trajectories", value=True)
collision_sensitivity = st.sidebar.slider(
    "Collision Detection Sensitivity", 0.3, 1.0, 0.6
)
enable_decision_lab = st.sidebar.checkbox("Enable Decision Lab", value=True)

# Main Dashboard
st.title(f"🏙️ {config.DASHBOARD_TITLE}")

# State management
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Time", "Risk", "Pollution", "Vehicle_Count", "Speed"]
    )
if "collision_alerts" not in st.session_state:
    st.session_state.collision_alerts = []


def process_video(video_path):
    """Enhanced video processing with all new features."""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reinitialize models with frame dimensions
    models["collision_detector"] = CollisionDetector(frame_width, frame_height)
    models["heatmap_generator"] = HeatmapGenerator(frame_width, frame_height)
    models["speed_estimator"].fps = fps

    frame_count = 0
    frame_display_count = 0
    last_frame_time = 0
    target_frame_time = 1.0 / 15  # Limit display to 15 fps to prevent cache overflow

    metrics_update_interval = 10  # Update metrics every 10 frames
    analysis_update_interval = 20  # Update analysis every 20 frames
    last_metrics_update = 0
    last_analysis_update = 0

    # Placeholder containers
    frame_placeholder = st.empty()
    metrics_container = st.empty()
    analysis_container = st.empty()

    frame_times = []
    collision_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_time = frame_count / fps
        frame_times.append(time.time())

        try:
            # 1. Detection
            results = models["detector"].detect(frame)
            detections = sv.Detections.from_ultralytics(results)

            if len(detections) == 0:
                # No detections, still show the frame but process it
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Only display if enough time has passed
                current_time = time.time()
                if current_time - last_frame_time >= target_frame_time:
                    frame_placeholder.image(display_frame, use_container_width=True)
                    last_frame_time = current_time
                    frame_display_count += 1
                time.sleep(0.01)  # Small delay to prevent overwhelming Streamlit
                continue

            # Add class names
            detections.data["class_name"] = np.array(
                [models["detector"].get_class_name(c) for c in detections.class_id]
            )

            # 2. Tracking
            detections = models["tracker"].update(detections)

            # 3. Speed Estimation
            models["speed_estimator"].update(detections)

            # 4. Analytics
            models["analytics"].update_analytics(detections, frame_time)

            # 5. Collision Detection
            models["collision_detector"].update(detections, frame_time)

            # 6. Incident Detection
            models["incident_detector"].update(detections, models["speed_estimator"])

            # 7. Traffic Prediction
            avg_speed = models["speed_estimator"].get_average_speed()
            congestion_level = len(models["analytics"].vehicle_history) / 20.0
            models["traffic_predictor"].update(
                len(models["analytics"].vehicle_history),
                avg_speed,
                min(congestion_level, 1.0),
                frame_time,
            )

            # 8. Heatmap
            models["heatmap_generator"].update(detections)

            # 9. Visualization
            annotated_frame = models["tracker"].annotate_frame(
                frame, detections, show_trajectory=enable_trajectory
            )

            # Add speed vectors if enabled
            if enable_speed_display:
                annotated_frame = models["speed_estimator"].draw_speed_annotations(
                    annotated_frame, detections
                )

            # Add collision warnings
            collision_alerts = models["collision_detector"].get_alerts()
            if (
                collision_alerts
                and collision_alerts[0]["risk_score"] > collision_sensitivity
            ):
                annotated_frame = models["collision_detector"].draw_collision_warnings(
                    annotated_frame, detections
                )
                collision_history.extend(collision_alerts)

            # Add heatmap overlay
            if enable_heatmap:
                annotated_frame = models["heatmap_generator"].render_heatmap_on_frame(
                    annotated_frame, use_temporal=False
                )

            # Convert for display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display frame with throttling to prevent media cache overflow
            current_time = time.time()
            if current_time - last_frame_time >= target_frame_time:
                frame_placeholder.image(display_frame, use_container_width=True)
                last_frame_time = current_time
                frame_display_count += 1

            time.sleep(0.01)  # Small delay to allow Streamlit to serve media

            # Update Metrics periodically (not every frame to prevent flickering)
            if frame_count - last_metrics_update >= metrics_update_interval:
                last_metrics_update = frame_count

                risk_score = models["analytics"].calculate_risk_index()
                emissions = models["analytics"].estimate_emissions()
                vehicle_count = len(models["analytics"].vehicle_history)
                avg_vehicle_speed = models["speed_estimator"].get_average_speed()
                collision_alerts = models["collision_detector"].get_alerts()

                with metrics_container.container():
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                    with metric_col1:
                        st.metric(
                            "🔴 Risk Index",
                            f"{risk_score:.1f}",
                            delta=f"{('High' if risk_score > 50 else 'Medium' if risk_score > 25 else 'Low')}",
                        )

                    with metric_col2:
                        st.metric(
                            "🚗 Active Vehicles",
                            vehicle_count,
                            delta=f"Peak: {models['analytics'].peak_vehicles}",
                        )

                    with metric_col3:
                        all_speeds = models["speed_estimator"].get_all_speeds()
                        max_speed = (
                            max([v["speed_ms"] for v in all_speeds.values()], default=0)
                            if all_speeds
                            else 0
                        )
                        st.metric(
                            "⚡ Avg Speed",
                            f"{avg_vehicle_speed:.1f} m/s",
                            delta=f"Max: {max_speed:.1f}",
                        )

                    with metric_col4:
                        collision_risk_count = len(
                            [a for a in collision_alerts if a["risk_score"] > 0.6]
                        )
                        st.metric(
                            "⚠️ Collision Alerts",
                            collision_risk_count,
                            delta=f"High Risk: {collision_risk_count > 0}",
                        )

            # Update analysis tabs periodically
            if frame_count - last_analysis_update >= analysis_update_interval:
                last_analysis_update = frame_count

                with analysis_container.container():
                    st.divider()

                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                        [
                            "📊 Real-Time",
                            "⚠️ Alerts",
                            "🗺️ Spatial",
                            "📈 Trends",
                            "🚗 Vehicles",
                            "🧠 Decision Lab",
                        ]
                    )

                    # Real-time tab
                    with tab1.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**📊 Current Metrics**")
                            stats = models["analytics"].get_statistics()
                            st.json(
                                {
                                    "Current Vehicles": stats["current_vehicles"],
                                    "Peak Vehicles": stats["peak_vehicles"],
                                    "Violations": stats["violations"],
                                    "Risk Index": f"{stats['risk_index']:.1f}",
                                    "Emissions": f"{stats['emissions']:.1f} g/min",
                                }
                            )

                        with col2:
                            st.write("**🚦 Signal Recommendation**")
                            st.success(
                                models["analytics"].get_signal_recommendation(),
                                icon="✅",
                            )

                            forecast = models[
                                "traffic_predictor"
                            ].get_congestion_forecast()
                            risk_level = forecast.get("risk_level", "Low")
                            status = forecast.get("status", "Data not available")
                            color = (
                                "🔴"
                                if risk_level == "High"
                                else "🟡"
                                if risk_level == "Medium"
                                else "🟢"
                            )
                            st.write(f"{color} {status}")

                    # Alerts tab
                    with tab2.container():
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**⚠️ Collision Risks**")
                            collision_alerts = models["collision_detector"].get_alerts()
                            if collision_alerts:
                                for alert in collision_alerts[:3]:
                                    if alert["risk_score"] > collision_sensitivity:
                                        st.warning(
                                            f"Risk: {alert['risk_score']:.2f} | "
                                            f"Distance: {alert['distance']:.0f}px | "
                                            f"Vehicles: #{alert['vehicle_1']} ⚡ #{alert['vehicle_2']}"
                                        )
                            else:
                                st.info("No collision risks detected ✅")

                        with col2:
                            st.write("**🚨 Incidents**")
                            incidents = models["incident_detector"].get_incidents()
                            if incidents:
                                for incident in incidents[:3]:
                                    st.error(
                                        f"**{incident['type'].upper()}** | Severity: {incident['severity']}"
                                    )
                            else:
                                st.info("No incidents detected ✅")

                    # Spatial tab
                    with tab3.container():
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**🗺️ Traffic Hotspots**")
                            hotspots = models["heatmap_generator"].get_hotspots(
                                threshold=0.4
                            )
                            if hotspots:
                                for spot in hotspots[:3]:
                                    st.warning(
                                        f"Density: {spot['density']:.2f} at ({spot['pixel_x']}, {spot['pixel_y']})"
                                    )
                            else:
                                st.info("No high-density areas detected")

                        with col2:
                            st.write("**📍 Regional Congestion**")
                            regions = models[
                                "heatmap_generator"
                            ].get_congestion_index_by_region()
                            region_data = pd.DataFrame(
                                list(regions.items()), columns=["Region", "Congestion"]
                            )
                            st.bar_chart(region_data.set_index("Region"))

                    # Trends tab
                    with tab4.container():
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**📈 Congestion Forecast**")
                            forecast = models[
                                "traffic_predictor"
                            ].get_congestion_forecast()
                            if "predictions" in forecast and forecast["predictions"]:
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        y=forecast["predictions"],
                                        mode="lines+markers",
                                        name="Predicted Congestion",
                                        line=dict(color="orange"),
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Collecting data for forecast...")

                        with col2:
                            st.write("**🔍 Anomalies**")
                            anomalies = models[
                                "traffic_predictor"
                            ].get_anomaly_detection()
                            if anomalies["alert"]:
                                for anom in anomalies["anomalies"]:
                                    st.warning(
                                        f"**{anom['type']}** - Severity: {anom['severity']}"
                                    )
                            else:
                                st.success("No anomalies detected ✅")

                    # Vehicles tab
                    with tab5.container():
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**🚗 Vehicle Distribution**")
                            vehicle_dist = pd.DataFrame(
                                list(models["analytics"].total_counts.items()),
                                columns=["Type", "Count"],
                            )
                            fig = px.pie(vehicle_dist, values="Count", names="Type")
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.write("**⚡ Speed Distribution**")
                            speed_hist = models["speed_estimator"].get_speed_histogram()
                            if speed_hist["counts"]:
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Bar(
                                        x=speed_hist["bins"][:-1],
                                        y=speed_hist["counts"],
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Collecting speed data...")

                    # Decision Lab tab
                    with tab6.container():
                        if enable_decision_lab:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**🧩 Explainable Risk Breakdown**")
                                risk_breakdown = models["analytics"].get_risk_breakdown(
                                    collision_alerts=models[
                                        "collision_detector"
                                    ].get_alerts(),
                                    incidents=models[
                                        "incident_detector"
                                    ].get_incidents(),
                                )

                                component_df = pd.DataFrame(
                                    {
                                        "Component": list(
                                            risk_breakdown[
                                                "component_percentages"
                                            ].keys()
                                        ),
                                        "Contribution_%": list(
                                            risk_breakdown[
                                                "component_percentages"
                                            ].values()
                                        ),
                                    }
                                )
                                st.bar_chart(component_df.set_index("Component"))
                                st.caption(
                                    f"Total risk index: {risk_breakdown['total_risk_index']:.1f}"
                                )

                                st.write("**🔎 Forecast Confidence**")
                                confidence = models[
                                    "traffic_predictor"
                                ].get_prediction_confidence()
                                st.metric(
                                    "Prediction Confidence",
                                    f"{confidence['confidence_score']:.1f}%",
                                    delta=confidence["label"],
                                )
                                for reason in confidence["reasons"]:
                                    st.write(f"- {reason}")

                            with col2:
                                st.write("**⏱️ Near-Miss Analytics (TTC)**")
                                near_miss = models[
                                    "collision_detector"
                                ].get_near_miss_summary()
                                ncol1, ncol2, ncol3 = st.columns(3)
                                ncol1.metric(
                                    "Near-Miss Pairs", near_miss["near_miss_count"]
                                )
                                ncol2.metric(
                                    "Critical TTC", near_miss["critical_count"]
                                )
                                ncol3.metric(
                                    "Mean Risk", f"{near_miss['mean_risk']:.2f}"
                                )

                                top_events = near_miss.get("top_events", [])
                                if top_events:
                                    top_df = pd.DataFrame(top_events)
                                    st.dataframe(top_df, use_container_width=True)
                                else:
                                    st.info("No near-miss events right now")

                                st.write("**🚦 Signal What-If Simulator**")
                                forecast = models[
                                    "traffic_predictor"
                                ].get_congestion_forecast()
                                current_congestion = float(
                                    forecast.get("current_congestion", 0.0)
                                )
                                current_vehicles = len(
                                    models["analytics"].vehicle_history
                                )

                                what_if = models["analytics"].simulate_signal_what_if(
                                    predicted_congestion=current_congestion,
                                    active_vehicles=current_vehicles,
                                )
                                st.success(f"Best action now: {what_if['best_action']}")
                                sim_df = pd.DataFrame(what_if["scenarios"])
                                st.dataframe(sim_df, use_container_width=True)
                        else:
                            st.info("Decision Lab is disabled from sidebar settings")

            # Update history periodically
            if frame_count - last_metrics_update <= metrics_update_interval:
                risk_score = models["analytics"].calculate_risk_index()
                emissions = models["analytics"].estimate_emissions()
                vehicle_count = len(models["analytics"].vehicle_history)
                avg_vehicle_speed = models["speed_estimator"].get_average_speed()

                new_row = {
                    "Time": frame_time,
                    "Risk": risk_score,
                    "Pollution": emissions,
                    "Vehicle_Count": vehicle_count,
                    "Speed": avg_vehicle_speed,
                }
                st.session_state.history = pd.concat(
                    [st.session_state.history, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

        except Exception as e:
            # Log error but continue
            print(f"Error on frame {frame_count}: {str(e)}")
            import traceback

            traceback.print_exc()
            # Still display the frame with throttling
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = time.time()
            if current_time - last_frame_time >= target_frame_time:
                frame_placeholder.image(display_frame, use_container_width=True)
                last_frame_time = current_time
                frame_display_count += 1
            time.sleep(0.01)
            continue

    cap.release()

    # Final summary
    st.divider()
    st.subheader("📋 Session Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    stats = models["analytics"].get_statistics()
    with col1:
        st.metric("Total Vehicles Detected", stats["total_vehicles_detected"])
    with col2:
        st.metric("Peak Congestion", stats["peak_vehicles"])
    with col3:
        st.metric("Violations Recorded", stats["violations"])
    with col4:
        st.metric(
            "Total Incidents", len(models["incident_detector"].get_incident_history())
        )
    with col5:
        st.metric("Frames Processed", frame_count)


# Main Execution

if app_mode == "Live Dashboard":
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)
    elif source_option == "Webcam (Demo)":
        st.warning("Webcam mode is active. Please ensure camera access.")
        # process_video(0)  # Commented to avoid hanging
        st.info("Webcam streaming would start here")
    else:
        st.info("📹 Please upload a video file to analyze")

elif app_mode == "Statistics Dashboard":
    st.subheader("📊 Performance Analytics")

    if not st.session_state.history.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.line_chart(
                st.session_state.history.set_index("Time")["Risk"], height=300
            )
            st.caption("Safety Risk Over Time")

        with col2:
            st.area_chart(
                st.session_state.history.set_index("Time")["Vehicle_Count"], height=300
            )
            st.caption("Vehicle Count Over Time")

        with col3:
            st.line_chart(
                st.session_state.history.set_index("Time")["Speed"], height=300
            )
            st.caption("Average Speed Over Time")
    else:
        st.info("Run live analysis first to see statistics")

else:
    st.info(f"📊 Selected mode: {app_mode}")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.info(f"Processing in {app_mode} mode...")
        # You can customize processing per mode here
        process_video(tfile.name)
    else:
        st.info("📹 Please upload a video file to analyze")
