import pytest 

from ddql_optimal_execution.state._state import State

def test_init():
    state = State()
    assert state is not None

def test_init_with_dict():
    state = State({'inventory': 100})
    assert state is not None
    assert state['inventory'] == 100



def test_update_state():
    state = State({'inventory': 100})

    state.update_state(inventory=99)

    assert state['inventory'] == 99


def test_update_state_error():
    state = State({'inventory': 100})

    with pytest.raises(KeyError):
        state.update_state(inventory=99, price=100)



def test_copy():
    state_1 = State({'inventory': 100})
    state_2 = state_1.copy()

    assert state_1 is not state_2


#requires to install torch for CI 

# def test_astensor():
#     state = State({'inventory': 100})

#     tensor = state.astensor

#     assert tensor is not None
#     assert tensor.shape == (1,)
#     assert tensor.dtype == torch.float32
#     assert tensor[0] == 100
