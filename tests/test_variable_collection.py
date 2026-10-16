from dataclasses import dataclass

import numpy as np
import pytest

from gundam_interface.internal.samples import VariableCollectionView


@dataclass
class VariableHolder:
    value: int | float

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value


class VariableCollection:
    def __init__(self):
        self.names = ["X", "identifier"]
        self.variables = [VariableHolder(0.5), VariableHolder(2**53 + 1)]

    def findVarIndex(self, name, throwIfNotFound):
        return self.names.index(name) if name in self.names else -1

    def getVarList(self):
        return self.variables

    def getNameList(self):
        return self.names


def test_variable_assignment_by_name_and_index_updates_shared_views():
    handle = VariableCollection()
    first = VariableCollectionView(handle)
    second = VariableCollectionView(handle)
    first["X"] = 1.25
    second[np.int64(1)] = 2**63 + 123

    assert second[0] == 1.25
    assert first[-1] == 2**63 + 123
    assert first.toDict() == {"X": 1.25, "identifier": 2**63 + 123}


@pytest.mark.parametrize(
    ("key", "error"),
    [("missing", KeyError), (2, IndexError), (-3, IndexError), (0.5, TypeError)],
)
def test_invalid_variable_keys_do_not_modify_another_variable(key, error):
    variables = VariableCollectionView(VariableCollection())
    original = variables.toDict()

    with pytest.raises(error):
        _ = variables[key]
    with pytest.raises(error):
        variables[key] = 42

    assert variables.toDict() == original
