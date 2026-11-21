import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import models
from app.scenarios.fault_scenarios import (
    FeatureDistributionShiftScenario,
    InvalidCategoricalValuesScenario,
    MissingValueSpikeScenario,
    PerformanceDegradationScenario,
    PredictionDistributionShiftScenario,
    ServingFailureScenario,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    session.add(models.DeploymentEnvironment(name="staging"))
    session.add(models.DeploymentEnvironment(name="production"))
    session.commit()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "holiday": ["None"] * 60,
            "temp": np.linspace(280.0, 300.0, 60),
            "rain_1h": [0.0] * 60,
            "snow_1h": [0.0] * 60,
            "clouds_all": [40] * 60,
            "weather_main": ["Clouds"] * 60,
            "weather_description": ["scattered clouds"] * 60,
            "date_time": ["2026-11-21 09:00:00"] * 60,
            "traffic_volume": np.linspace(1000, 5000, 60),
        }
    )


def test_scenario_1_feature_distribution_shift(db_session, base_df):
    scenario = FeatureDistributionShiftScenario(seed=42)
    res = scenario.run(db_session, base_df)
    assert res.seed == 42
    assert res.actual_detected is True
    assert "temp" in res.evidence["drifting_features"]
    assert "Synthetic demonstration" in res.limitations


def test_scenario_2_missing_value_spike(db_session, base_df):
    scenario = MissingValueSpikeScenario(seed=42)
    res = scenario.run(db_session, base_df)
    assert res.seed == 42
    assert res.actual_detected is True
    assert "clouds_all" in res.evidence["drifting_features"]


def test_scenario_3_invalid_categorical_values(db_session, base_df):
    scenario = InvalidCategoricalValuesScenario(seed=42)
    res = scenario.run(db_session, base_df)
    assert res.seed == 42
    assert res.actual_detected is True
    assert "weather_main" in res.evidence["drifting_features"]


def test_scenario_4_prediction_distribution_shift(db_session):
    scenario = PredictionDistributionShiftScenario(seed=42)
    res = scenario.run(db_session)
    assert res.seed == 42
    assert res.actual_detected is True
    assert "Evaluates prediction output shift independently" in res.limitations


def test_scenario_5_performance_degradation_with_delayed_labels(db_session):
    scenario = PerformanceDegradationScenario(seed=42)
    res = scenario.run(db_session)
    assert res.seed == 42
    assert res.actual_detected is True
    assert res.evidence["mae"] > 450.0


def test_scenario_6_serving_failure(db_session):
    scenario = ServingFailureScenario(seed=42)
    res = scenario.run(db_session)
    assert res.seed == 42
    assert res.actual_detected is True
    assert res.evidence["load_failed"] is True
