import numpy as np
import random
from scipy.optimize import line_search
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def dumb_predict(target: np.array, dim: int = 3) -> np.array:
    return np.array([random.choice([1, -1]) for i in range(dim)])

def simple_boosting(target: np.array, steps: int = 1000, learner_weight: float = 1, dim: int = 3) -> np.array:

    boosted_learners = np.zeros(shape=(steps+1,dim))
    for i in range(1, steps + 1):
        print(f"iteration {i}")
        starting_learner = dumb_predict(target=target, dim=dim)
        
        def select_directional_boosted_learner(target, learner, current_boosted_learner):
            print(f"boosting learner {learner}")
            scaled_learner = learner_weight * learner
            # choose lower loss between learner and learner with flipped direction
            if (np.sum((target - (current_boosted_learner - scaled_learner))**2) < np.sum((target - (current_boosted_learner + scaled_learner))**2)):
                boosted_learner = current_boosted_learner - scaled_learner
                loss = np.sum((target - boosted_learner)**2)
                print(f"flipped learner has lower loss {loss}. boosted learner {boosted_learner}")
                return boosted_learner
            
            boosted_learner = current_boosted_learner + scaled_learner
            loss = np.sum((target - boosted_learner)**2)
            print(f"unflipped learner has lower loss {loss}. boosted learner {boosted_learner}")
            return boosted_learner


            boosted_learner = current_boosted_learner + learner_weight * learner
            boosted_learner_flipped = current_boosted_learner - learner_weight * learner
            boosted_learner_loss = 0.5 * np.sum((target - boosted_learner)**2)
            boosted_learner_flipped_loss = 0.5 * np.sum((target - boosted_learner_flipped)**2)
            print(f"boosted learner {boosted_learner} with loss {boosted_learner_loss}")
            print(f"flipped boosted learner {boosted_learner_flipped} with loss {boosted_learner_flipped_loss}")
            return boosted_learner if (boosted_learner_loss - boosted_learner_flipped_loss) <= 0 else boosted_learner_flipped


        boosted_learners[i,] = select_directional_boosted_learner(target, starting_learner, boosted_learners[i - 1,])
        loss = np.sum((target - boosted_learners[i, ])**2)
        print(f"step {i}: boosted learner {boosted_learners[i, ]}. loss: {loss}")

        # if loss < 0.0001: 
        #     break


    return boosted_learners


boosting_steps = 500
learner_weight = 0.01
shape = 3
# target = np.ones(shape=shape)
source = np.array([0, 0, 0])
target = np.array([1, 1, -1])


print(f"predict target {target} with {boosting_steps} steps")

def update_lines(num, predicted_path, loss, lines):
    lines[0].set_data_3d(predicted_path[:num, :].T)
    lines[1].set_data(loss[:num,].T)
    return lines

boosting_paths = simple_boosting(target=target, steps=boosting_steps, learner_weight=learner_weight)
print(f"paths: {boosting_paths}")
boosting_rmse = np.array([[i, np.sqrt(np.sum((target - boosted_path)**2))] for i, boosted_path in enumerate(boosting_paths)])
print(f"losses: {boosting_rmse[:5, :]}")

# Attaching 3D axis to the figure
fig = plt.figure(figsize=plt.figaspect(0.3))
ax_path = fig.add_subplot(1, 2, 1, projection="3d")
ax_path.set(xlim3d=(0, 1.2), xlabel='x')
ax_path.set(ylim3d=(0, 1.2), ylabel='y')
ax_path.set(zlim3d=(0, -1.2), zlabel='z')
ax_path.scatter(source[0], source[1], source[2])
ax_path.scatter(target[0], target[1], target[2])
ax_path.set_title('Boosting Path')

# Create line initially without data
line_boost = ax_path.plot([], [], [])[0]

ax_loss = fig.add_subplot(1, 2, 2)
# Create line initially without data
line_loss = ax_loss.plot([0], [1])[0]

ax_loss.set(xlim=(0,boosting_steps), xlabel='iteration')
ax_loss.set(ylim=(0,np.sqrt(3)), ylabel='y (RMSE)')
ax_loss.set_title('Root Mean Squared Error')
# Creating the Animation object
ani_boosting = animation.FuncAnimation(
    fig, update_lines, 1000, fargs=(boosting_paths, boosting_rmse, [line_boost, line_loss]), interval=10)

plt.show()
ani_boosting.save("boosting_intuition.gif", writer="pillow")
