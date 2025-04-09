from abc import abstractmethod
from typing import Any

import taichi


@taichi.data_oriented
class System:

    def set_data(self, data: Any) -> None:
        self.data = data
