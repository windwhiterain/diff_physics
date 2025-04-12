from diff_physics.common.entity import Frame


class Objective:
    def update(self, frame: Frame) -> tuple[Frame, float]: ...
