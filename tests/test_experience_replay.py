import pytest

import numpy as np

from ddql_optimal_execution.experience_replay._experience_replay import ExperienceReplay
from ddql_optimal_execution.experience_replay._experience_dict import ExperienceDict
from ddql_optimal_execution.experience_replay._exceptions import (
    ReplayMemorySamplingError,
)

from ddql_optimal_execution.state._state import State


def test_init():
    experience_replay = ExperienceReplay(capacity=10)
    assert experience_replay is not None


def test_is_empty():
    experience_replay = ExperienceReplay(capacity=10)
    assert experience_replay.is_empty == True


def test_push():
    experience_replay = ExperienceReplay(capacity=10)

    state = State({"inventory": 100})

    experience_replay.push(state, 1, 1, state, 3)
    assert experience_replay is not None

    assert experience_replay.is_empty == False
    assert experience_replay.is_full == False


def test_push_full():
    experience_replay = ExperienceReplay(capacity=10)

    state = State({"inventory": 100})

    for _ in range(10):
        experience_replay.push(state, 1, 1, state, 3)

    assert experience_replay.is_empty == False
    assert experience_replay.is_full == True


def test_sample():
    experience_replay = ExperienceReplay(capacity=10)

    state = State({"inventory": 100})

    for _ in range(10):
        experience_replay.push(state, 1, 1, state, 3)

    sample = experience_replay.get_sample(5)

    assert len(sample) == 5


def test_sample_error():
    experience_replay = ExperienceReplay(capacity=10)

    state = State({"inventory": 100})

    for _ in range(10):
        experience_replay.push(state, 1, 1, state, 3)

    with pytest.raises(ReplayMemorySamplingError):
        experience_replay.get_sample(11)


@pytest.mark.parametrize("capacity", [10, 20, 16])
def test_make_room(capacity):
    experience_replay = ExperienceReplay(capacity=capacity)

    state = State({"inventory": 100})

    for i in range(capacity):
        experience_replay.push(state, i, 1, state, 3)

    initial_arr = experience_replay.memory.copy()

    experience_replay.push(state, 23, 1, state, 5)

    assert experience_replay.memory[-1] == ExperienceDict(
        state=state,
        action=23,
        reward=1,
        next_state=state,
        dist2Horizon=5,
    )

    assert np.argwhere(initial_arr != experience_replay.memory)[0] <= capacity // 2
