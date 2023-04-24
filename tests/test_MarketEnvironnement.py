import pytest

from ddql_optimal_execution.environnement._env import MarketEnvironnement
from ddql_optimal_execution.environnement._exceptions import EpisodeIndexError

data_path = "tests/data"

@pytest.mark.parametrize("multi_episodes", [True, False])
def test_init(multi_episodes):
    env = MarketEnvironnement(data_path=data_path, multi_episodes=multi_episodes)
    assert env is not None


@pytest.mark.parametrize("multi_episodes", [True, False])
def test_reset(multi_episodes):
    env = MarketEnvironnement(data_path=data_path, multi_episodes=multi_episodes)
    env.reset()
    assert env is not None


def test_swap_data():
    env = MarketEnvironnement(data_path=data_path, multi_episodes=True)
    env.reset()
    assert env is not None
    env.swap_episode(1)
    assert env is not None


def test_swap_data_error():
    env = MarketEnvironnement(data_path=data_path, multi_episodes=True)
    env.reset()
    assert env is not None
    with pytest.raises(EpisodeIndexError):
        env.swap_episode(10)
    assert env is not None

def test_step():
    env = MarketEnvironnement(data_path=data_path, multi_episodes=True, initial_inventory=100)
    rw = env.step(1)

    assert env is not None
    assert env.state['inventory'] == 99

    assert rw is not None




