import numpy as np
import random
from manim import *

# for reproducibility
np.random.seed(20241202)    

def dumb_predict(target: np.array, dim: int = 3) -> np.array:
    return np.array([random.choice([1, -1]) for i in range(dim)])

def simple_boosting(target: np.array, steps: int = 1000, learning_rate: float = 0.001, dim: int = 3) -> np.array:

    boosted_learners = np.zeros(shape=(steps+1,dim))
    for i in range(1, steps + 1):
        learner = dumb_predict(target=target, dim=dim)
        print(f"step {i}: new prediction {learner}")
        
        boosted_learners[i,] = boosted_learners[i - 1,] + learning_rate * learner if np.inner(target, learner) >= 0 else boosted_learners[i - 1,] - learning_rate * learner
        loss = np.sum(np.square(np.subtract(target, boosted_learners[i, ])))
        print(f"step {i}: boosted learner {boosted_learners[i, ]}. loss: {loss}")

        if loss < 0.05: 
            break


    return boosted_learners[1:i]


boosting_steps = 10000
learning_rate = 0.01
shape = 3
# target = np.ones(shape=shape)
source = np.array([0, 0, 0])
target = np.array([1, -1, -1])


print(f"predict target {target} with {boosting_steps} steps")

# predictions = simple_boosting(target=target, steps=boosting_steps, learning_rate=learning_rate)
# sampling_step = int(predictions.shape[0] / 15)

# predictions = np.flip(predictions[-1::-sampling_step,], 0)

# for prediction in predictions:
#     print(prediction)


# class GetAxisLabelsExample(ThreeDScene):
#     def construct(self):
#         self.set_camera_orientation(phi=50 * DEGREES, theta=-30 * DEGREES, zoom=0.8)
#         axes = ThreeDAxes(
#             x_range=[-2, 2, 0.5],
#             y_range=[-2, 2, 0.5],
#             z_range=[-2, 2, 0.5]
#         )
#         labels = axes.get_axis_labels(
#             Text("x").scale(0.5), Text("y").scale(0.5), Text("z").scale(0.5)
#         )

#         source_x, source_y, source_z = source
#         dot_1 = Dot3D(point=axes.coords_to_point(source_x, source_y, source_z), radius=0.05, color=RED)
#         target_x, target_y, target_z = target
#         dot_2 = Dot3D(point=axes.coords_to_point(target_x, target_y, target_z), radius=0.1, color=BLUE)

#         self.add(axes, labels)
#         self.play(Create(axes))
#         self.wait(1)
#         self.play(Create(dot_1), Create(dot_2))
        
#         line_start = source
#         for prediction in predictions:
#             x_0_, y_0, z_0 = line_start
#             x, y, z = prediction
#             # destination = Dot3D(point=axes.coords_to_point(x, y, z), radius=0.1, color=RED)
#             line_end = prediction
#             line_3d = Line3D(start=axes.coords_to_point(x_0_, y_0, z_0), end=axes.coords_to_point(x, y, z), color=RED, thickness=0.01)
#             # self.play(dot_1.animate.move_to(destination))
#             self.play(Create(line_3d))
#             self.wait(0.0001)
#             line_start = line_end


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

# def random_walk(num_steps, max_step=0.05):
#     """Return a 3D random walk as (num_steps, 3) array."""
#     start_pos = np.random.random(3)
#     steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
#     walk = start_pos + np.cumsum(steps, axis=0)
#     return walk


def update_path(num, prediction, line):
    line.set_data_3d(prediction[:num, :].T)
    return line

boosted_path = simple_boosting(target=target, steps=boosting_steps, learning_rate=learning_rate)

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
# Create line initially without data
line = ax.plot([], [], [])[0]


print(line)
# Setting the Axes properties
ax.set(xlim3d=(-2, 2), xlabel='X')
ax.set(ylim3d=(-2, 2), ylabel='Y')
ax.set(zlim3d=(-2, 2), zlabel='Z')
ax.scatter([1], [-1], [-1])

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_path, 2000, fargs=(boosted_path, line), interval=1)

ani.save("matplotlib_animation.gif", writer="pillow")

plt.show()