import numpy as np
from typing import Dict
from collections import deque


class TrafficPredictor:
    """
    Traffic flow prediction using time-series analysis.
    Predicts future congestion and vehicle counts.
    """

    def __init__(self, history_length: int = 60, prediction_horizon: int = 30):
        """
        Initialize traffic predictor.

        Args:
            history_length: Number of time steps to use for history (frames)
            prediction_horizon: How many frames ahead to predict
        """
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.vehicle_count_history = deque(maxlen=history_length)
        self.average_speed_history = deque(maxlen=history_length)
        self.congestion_history = deque(maxlen=history_length)
        self.timestamp_history = deque(maxlen=history_length)
        self.predictions = {}

    def update(
        self,
        vehicle_count: int,
        average_speed: float,
        congestion_level: float,
        timestamp: float,
    ):
        """
        Update predictor with current metrics.

        Args:
            vehicle_count: Number of vehicles in frame
            average_speed: Average speed of vehicles (pixels/frame)
            congestion_level: Congestion metric (0-1)
            timestamp: Current timestamp
        """
        self.vehicle_count_history.append(vehicle_count)
        self.average_speed_history.append(average_speed)
        self.congestion_history.append(congestion_level)
        self.timestamp_history.append(timestamp)

        if len(self.vehicle_count_history) >= self.history_length:
            self._predict_future()

    def _predict_future(self):
        """
        Generate predictions for future time steps.
        """
        if len(self.vehicle_count_history) < 10:
            return

        # Convert to numpy arrays
        counts = np.array(list(self.vehicle_count_history))
        speeds = np.array(list(self.average_speed_history))
        congestion = np.array(list(self.congestion_history))

        # Simple exponential smoothing + trend
        self.predictions = {
            "vehicle_count": self._exponential_smoothing(counts, alpha=0.3),
            "average_speed": self._exponential_smoothing(speeds, alpha=0.2),
            "congestion": self._exponential_smoothing(congestion, alpha=0.25),
        }

    def _exponential_smoothing(self, data: np.ndarray, alpha: float = 0.3) -> Dict:
        """
        Simple exponential smoothing with trend for predictions.
        """
        if len(data) < 3:
            return {
                "current": float(data[-1]),
                "predictions": [float(data[-1])] * self.prediction_horizon,
            }

        # Fit exponential model
        level = data[-1]
        trend = (data[-1] - data[-2]) if len(data) > 1 else 0

        predictions = []
        for h in range(1, self.prediction_horizon + 1):
            forecast = level + h * trend * (1 - alpha)
            predictions.append(max(0, forecast))  # Ensure non-negative

        return {
            "current": float(data[-1]),
            "mean": float(np.mean(data[-10:])),
            "std": float(np.std(data[-10:])),
            "predictions": predictions,
            "trend": float(trend),
        }

    def get_congestion_forecast(self) -> Dict:
        """
        Get congestion forecast for next N frames.
        """
        if "congestion" not in self.predictions:
            return {"status": "Low Traffic", "risk_level": "Low"}

        congestion_pred = self.predictions["congestion"]
        avg_future_congestion = np.mean(congestion_pred["predictions"])

        if avg_future_congestion > 0.7:
            status = "Heavy Congestion Expected"
            risk_level = "High"
        elif avg_future_congestion > 0.4:
            status = "Moderate Congestion Building"
            risk_level = "Medium"
        else:
            status = "Light Traffic Ahead"
            risk_level = "Low"

        return {
            "status": status,
            "risk_level": risk_level,
            "current_congestion": congestion_pred["current"],
            "predicted_peak": max(congestion_pred["predictions"]),
            "predictions": congestion_pred["predictions"],
            "trend": congestion_pred["trend"],
        }

    def get_vehicle_count_forecast(self) -> Dict:
        """
        Get vehicle count forecast.
        """
        if "vehicle_count" not in self.predictions:
            return {"predictions": []}

        count_pred = self.predictions["vehicle_count"]
        return {
            "current": count_pred["current"],
            "average": count_pred["mean"],
            "predictions": count_pred["predictions"],
            "trend": count_pred["trend"],
        }

    def get_anomaly_detection(self) -> Dict:
        """
        Detect unusual traffic patterns.
        """
        anomalies = []

        if len(self.vehicle_count_history) < 5:
            return {"anomalies": anomalies}

        counts = np.array(list(self.vehicle_count_history))[-20:]
        mean = np.mean(counts)
        std = np.std(counts)

        if std > 0:
            z_scores = np.abs((counts - mean) / std)
            if np.any(z_scores > 2.5):
                anomalies.append(
                    {
                        "type": "unusual_vehicle_count",
                        "severity": "high" if np.max(z_scores) > 3 else "medium",
                        "z_score": float(np.max(z_scores)),
                    }
                )

        # Check for sudden speed changes
        speeds = np.array(list(self.average_speed_history))[-20:]
        if len(speeds) > 2:
            speed_changes = np.diff(speeds)
            if np.any(np.abs(speed_changes) > 5):
                anomalies.append(
                    {
                        "type": "sudden_speed_change",
                        "severity": "medium",
                        "change": float(np.max(np.abs(speed_changes))),
                    }
                )

        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "alert": len(anomalies) > 0,
        }

    def get_prediction_confidence(self) -> Dict:
        """
        Return heuristic confidence score for current forecasts.
        Higher score means more stable recent history and sufficient data.
        """
        history_size = len(self.vehicle_count_history)
        if history_size < 10:
            return {
                "confidence_score": 0.0,
                "label": "Low",
                "reasons": ["Insufficient history for reliable forecasting"],
            }

        recent_counts = np.array(list(self.vehicle_count_history)[-20:])
        recent_speeds = np.array(list(self.average_speed_history)[-20:])

        count_cv = float(np.std(recent_counts) / (np.mean(recent_counts) + 1e-6))
        speed_cv = (
            float(np.std(recent_speeds) / (np.mean(recent_speeds) + 1e-6))
            if np.mean(recent_speeds) > 0
            else 1.0
        )

        stability_score = max(0.0, 1.0 - min((count_cv + speed_cv) / 2.0, 1.0))
        data_score = min(history_size / float(self.history_length), 1.0)
        anomaly_score = 1.0 - min(self.get_anomaly_detection()["count"] / 3.0, 1.0)

        confidence = (
            (stability_score * 0.5) + (data_score * 0.3) + (anomaly_score * 0.2)
        )
        confidence_score = float(confidence * 100.0)

        if confidence_score >= 75:
            label = "High"
        elif confidence_score >= 45:
            label = "Medium"
        else:
            label = "Low"

        reasons = []
        if data_score < 0.6:
            reasons.append("Limited historical window")
        if stability_score < 0.5:
            reasons.append("High short-term volatility")
        if anomaly_score < 0.7:
            reasons.append("Recent anomalies detected")
        if not reasons:
            reasons.append("Stable trends and sufficient history")

        return {
            "confidence_score": confidence_score,
            "label": label,
            "reasons": reasons,
        }
