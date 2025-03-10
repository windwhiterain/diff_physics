from abc import abstractmethod
from typing import Any

import taichi


@taichi.data_oriented
class System:
    @abstractmethod
    def set_data(self, data: Any) -> None: ...
