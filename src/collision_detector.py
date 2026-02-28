import numpy as np
import cv2
from typing import List, Tuple, Dict
import supervision as sv


class CollisionDetector:
    """
    Advanced collision detection and avoidance warning system.
    Predicts potential collisions based on trajectory and speed.
    """

    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.vehicle_trajectories = {}  # tracker_id -> list of positions
        self.collision_alerts = []
        self.safety_distance = 50  # pixels (minimum safe distance)

    def update(self, detections: sv.Detections, frame_time: float):
        """
        Update collision detector with new detections.

        Args:
            detections: Latest detections from tracker
            frame_time: Current frame timestamp
        """
        current_positions = {}

        # Extract current positions
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            if tracker_id is not None:
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                current_positions[tracker_id] = (center_x, center_y, box)

        # Update trajectories
        for tracker_id, pos in current_positions.items():
            if tracker_id not in self.vehicle_trajectories:
                self.vehicle_trajectories[tracker_id] = []
            self.vehicle_trajectories[tracker_id].append(
                {"pos": pos[:2], "time": frame_time, "box": pos[2]}
            )

            # Keep only last 30 frames
            if len(self.vehicle_trajectories[tracker_id]) > 30:
                self.vehicle_trajectories[tracker_id].pop(0)

        # Detect potential collisions
        self.collision_alerts = self._detect_collisions(current_positions, detections)

        # Clean up old trajectories
        self.vehicle_trajectories = {
            k: v for k, v in self.vehicle_trajectories.items() if k in current_positions
        }

    def _detect_collisions(
        self, positions: Dict, detections: sv.Detections
    ) -> List[Dict]:
        """
        Detect potential collisions between vehicles.
        """
        alerts = []
        tracker_ids = list(positions.keys())

        for i, id1 in enumerate(tracker_ids):
            for id2 in tracker_ids[i + 1 :]:
                # Get trajectories
                traj1 = self.vehicle_trajectories.get(id1, [])
                traj2 = self.vehicle_trajectories.get(id2, [])

                if len(traj1) < 3 or len(traj2) < 3:
                    continue

                # Calculate velocity vectors
                vel1 = self._estimate_velocity(traj1)
                vel2 = self._estimate_velocity(traj2)

                # Check if on collision course
                pos1 = positions[id1][:2]
                pos2 = positions[id2][:2]

                collision_risk = self._calculate_collision_risk(
                    pos1, vel1, pos2, vel2, positions[id1][2], positions[id2][2]
                )

                if collision_risk > 0.6:  # High collision risk threshold
                    alerts.append(
                        {
                            "vehicle_1": id1,
                            "vehicle_2": id2,
                            "risk_score": collision_risk,
                            "position_1": pos1,
                            "position_2": pos2,
                            "distance": np.sqrt(
                                (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
                            ),
                        }
                    )

        return alerts

    def _estimate_velocity(self, trajectory: List[Dict]) -> Tuple[float, float]:
        """
        Estimate velocity from trajectory history.
        """
        if len(trajectory) < 2:
            return (0, 0)

        # Use last 5 frames for better estimation
        recent = trajectory[-5:] if len(trajectory) >= 5 else trajectory

        dx = recent[-1]["pos"][0] - recent[0]["pos"][0]
        dy = recent[-1]["pos"][1] - recent[0]["pos"][1]

        # Normalize by time
        dt = recent[-1]["time"] - recent[0]["time"]
        if dt == 0:
            dt = 0.033  # Default 30fps

        return (dx / dt, dy / dt)

    def _calculate_collision_risk(self, pos1, vel1, pos2, vel2, box1, box2) -> float:
        """
        Calculate collision risk score (0-1) between two vehicles.
        """
        # Distance between vehicles
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Size of bounding boxes
        size1 = np.sqrt((box1[2] - box1[0]) ** 2 + (box1[3] - box1[1]) ** 2)
        size2 = np.sqrt((box2[2] - box2[0]) ** 2 + (box2[3] - box2[1]) ** 2)
        min_safe_dist = size1 + size2 + self.safety_distance

        # Check if vehicles are moving towards each other
        vel_mag1 = np.sqrt(vel1[0] ** 2 + vel1[1] ** 2)
        vel_mag2 = np.sqrt(vel2[0] ** 2 + vel2[1] ** 2)

        if vel_mag1 < 0.1 and vel_mag2 < 0.1:
            return 0.0  # Both stationary

        # Relative velocity
        rel_vel_x = vel2[0] - vel1[0]
        rel_vel_y = vel2[1] - vel1[1]

        # Dot product to check if approaching
        approaching_factor = (rel_vel_x * dx + rel_vel_y * dy) / (distance + 0.1)

        # Risk calculation
        distance_risk = (
            1.0 - min(distance / min_safe_dist, 1.0)
            if distance < min_safe_dist
            else 0.0
        )
        approaching_risk = max(approaching_factor, 0) / (vel_mag1 + vel_mag2 + 0.1)

        collision_risk = distance_risk * 0.6 + approaching_risk * 0.4

        return min(collision_risk, 1.0)

    def get_alerts(self) -> List[Dict]:
        """
        Get current collision alerts.
        """
        return sorted(
            self.collision_alerts, key=lambda x: x["risk_score"], reverse=True
        )

    def draw_collision_warnings(self, frame, detections: sv.Detections) -> np.ndarray:
        """
        Draw collision warnings on frame.
        """
        annotated = frame.copy()

        for alert in self.collision_alerts:
            if alert["risk_score"] > 0.6:
                pos1 = tuple(map(int, alert["position_1"]))
                pos2 = tuple(map(int, alert["position_2"]))

                # Draw line between vehicles
                cv2.line(annotated, pos1, pos2, (0, 0, 255), 3)

                # Draw warning circles
                cv2.circle(annotated, pos1, 30, (0, 0, 255), 3)
                cv2.circle(annotated, pos2, 30, (0, 0, 255), 3)

                # Draw risk score
                mid_point = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                cv2.putText(
                    annotated,
                    f"COLLISION RISK: {alert['risk_score']:.2f}",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        return annotated

    def get_near_miss_summary(self) -> Dict:
        """
        Summarize near-miss events with approximate TTC values.
        TTC is estimated from pair distance and relative speed projection.
        """
        if not self.collision_alerts:
            return {
                "near_miss_count": 0,
                "critical_count": 0,
                "mean_risk": 0.0,
                "top_events": [],
            }

        events = []
        for alert in self.collision_alerts:
            p1 = np.array(alert["position_1"])
            p2 = np.array(alert["position_2"])
            relative_distance = float(np.linalg.norm(p2 - p1))

            id1 = alert["vehicle_1"]
            id2 = alert["vehicle_2"]
            traj1 = self.vehicle_trajectories.get(id1, [])
            traj2 = self.vehicle_trajectories.get(id2, [])
            vel1 = self._estimate_velocity(traj1) if len(traj1) >= 2 else (0.0, 0.0)
            vel2 = self._estimate_velocity(traj2) if len(traj2) >= 2 else (0.0, 0.0)

            rel_vel = np.array([vel2[0] - vel1[0], vel2[1] - vel1[1]])
            line_vec = (p2 - p1) / (relative_distance + 1e-6)
            closing_speed = float(np.dot(rel_vel, line_vec))

            if closing_speed > 1e-3:
                ttc = relative_distance / closing_speed
            else:
                ttc = float("inf")

            events.append(
                {
                    "vehicle_1": id1,
                    "vehicle_2": id2,
                    "risk_score": float(alert["risk_score"]),
                    "distance_px": relative_distance,
                    "ttc_sec": float(ttc) if np.isfinite(ttc) else None,
                    "critical": bool(
                        np.isfinite(ttc) and ttc < 2.0 and alert["risk_score"] >= 0.7
                    ),
                }
            )

        critical_count = len([e for e in events if e["critical"]])
        mean_risk = float(np.mean([e["risk_score"] for e in events]))
        top_events = sorted(events, key=lambda x: x["risk_score"], reverse=True)[:5]

        return {
            "near_miss_count": len(events),
            "critical_count": critical_count,
            "mean_risk": mean_risk,
            "top_events": top_events,
        }
