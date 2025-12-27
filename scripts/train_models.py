"""
Train ML Models Script

Train anomaly detection and other ML models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.model_trainer import AnomalyModelTrainer


def main():
    """Train ML models."""
    print("Training anomaly detection model...")

    trainer = AnomalyModelTrainer(contamination=0.05)

    # Train from synthetic data (for initial deployment)
    output_path = "./models/anomaly_detector.pkl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    success = trainer.train_from_synthetic(output_path, n_samples=1000)

    if success:
        print(f"Model saved to {output_path}")
    else:
        print("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
