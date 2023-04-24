import pytest
import pandas as pd 

from ddql_optimal_execution.preprocessing._preprocessor import Preprocessor

data_path = "tests/data"

def test_init():
    preprocessor = Preprocessor(n_periods=10)
    assert preprocessor is not None


#parametrize several parameters with pytest

@pytest.mark.parametrize(["QV", "volume"], [(True, True), (False, False), (True, False), (False, True)])
def test_call(QV : bool, volume : bool):
    df = pd.read_csv(f"{data_path}/historical_data.csv")

    preprocessor = Preprocessor(n_periods=10, QV=QV, volume=volume)
    df1, raw_prices = preprocessor(df)

    assert df1 is not None
    assert raw_prices is not None

    assert df1['period'].max() == 9

    if QV:
        assert 'QV' in df1.columns

    if volume:
        assert 'volume' in df1.columns

    assert df1.isna().sum().sum() == 0



