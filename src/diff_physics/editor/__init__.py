from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import taichi as ti
import taichi.math as tm

from diff_physics.editor.renderable import Edges, Points, Renderable
from taichi_hint.wrap.linear_algbra import Vec

up = Vec(0, 0, 1)


class Action:
    pass


@dataclass
class SaveImage(Action):
    name: str


class Editor:

    def __init__(self, directory: Path) -> None:
        self.renderables = list[Renderable]()
        self.directory = directory

    def run(self, func: Iterator):
        pause = False
        time = 0
        dt = 0.5

        window = ti.ui.Window("diff_physics", res=(1024, 1024))
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.up(*up)
        camera.lookat(0, 0, 0)
        camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        canvas = window.get_canvas()

        camera_position = Vec(0, -30, 0)
        move_a = False
        move_d = False
        move_w = False
        move_s = False

        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.ESCAPE:
                    break
                if window.event.key == 'a':
                    move_a = True
                if window.event.key == 'd':
                    move_d = True
                if window.event.key == 'w':
                    move_w = True
                if window.event.key == 's':
                    move_s = True
            if window.get_event(ti.ui.RELEASE):
                if window.event.key == 'a':
                    move_a = False
                if window.event.key == 'd':
                    move_d = False
                if window.event.key == 'w':
                    move_w = False
                if window.event.key == 's':
                    move_s = False

            if move_a:
                camera_position -= camera_position.cross(up).normalized()*dt
            if move_d:
                camera_position += camera_position.cross(up).normalized()*dt
            if move_w:
                camera_position -= camera_position.normalized()*dt
            if move_s:
                camera_position += camera_position.normalized()*dt

            if window.is_pressed(ti.ui.SPACE):
                pause = not pause
                if pause:
                    ti.profiler.print_kernel_profiler_info()
                    ti.profiler.clear_kernel_profiler_info()

            if not pause:
                time += dt

            action = next(func)
            camera.position(camera_position[0],
                            camera_position[1], camera_position[2])
            scene.set_camera(camera)
            scene.point_light(pos=camera_position, color=(1, 1, 1))
            for renderable in self.renderables:
                renderable.update()
                color = (renderable.color[0], renderable.color[1], renderable.color[2])
                if isinstance(renderable, Edges):
                    scene.lines(
                        renderable.positions(),
                        indices=renderable.indices().to_numpy().flatten(),
                        color=color,
                        width=1,
                    )
                elif isinstance(renderable, Points):
                    scene.particles(renderable.positions(), 0.02, color=color)
            canvas.set_background_color((1, 1, 1))
            canvas.scene(scene)

            if isinstance(action, SaveImage):
                path = self.directory / "images_window" / f"{action.name}.png"
                window.save_image(str(path))
                print(f"save image window at {path}")

            window.show()
