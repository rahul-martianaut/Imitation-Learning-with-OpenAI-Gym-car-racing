import gym
import numpy as np
import keyboard
from PIL import Image



def get_discrete_action(action):

    if action[0] == -0.5:
        disc_action = [1, 0, 0, 0, 0]

    elif action[0]== 0.5:
        disc_action = [0, 1, 0, 0, 0]

    elif action[1]== 0.5:
        disc_action = [0, 0, 1, 0, 0]

    elif action[2]== 0.5:
        disc_action = [0, 0, 0, 1, 0]

    else:
        disc_action = [0, 0, 0, 0, 1]

    return disc_action


def manual_control(env, num_frames):
    env.reset()
    images = []
    labels = []

    for i in range(num_frames):
        action = [0.0, 0.0, 0.0] #no action
        if keyboard.is_pressed('left'):
            action[0] = -0.5  # Steer left
        elif keyboard.is_pressed('right'):
            action[0] = 0.5  # Steer right
        elif keyboard.is_pressed('up'):
            action[1] = 0.5  # Accelerate
        elif keyboard.is_pressed('down'):
            action[2] = 0.5  # Brake

        obs, _, done, _, _ = env.step(action)

        env.render()

        obs = obs[0:82, 0:96]  # Cropping
        images.append(obs)
        disc_action = get_discrete_action(action)
        labels.append(disc_action)
        if done or keyboard.is_pressed('q'):
            break

    images = images[30:]
    labels = labels[30:]
    np.savez("manual_control_data_new.npz", images=images, labels=labels)
    env.close()

if __name__ == "__main__":
    env = gym.make('CarRacing-v2', render_mode='human')
    manual_control(env, 500)

    data = np.load("manual_control_data_new.npz")
    imgs = data['images']
    labels = data['labels'].astype(float)
    image_size = (96, 96)
    images = np.zeros((len(labels), *image_size, 3), dtype=np.uint8)
    for i in range(len(images)):
        img = Image.fromarray(imgs[i])
        images[i] = np.array(img.resize(image_size))

    # 4 class data
    pos = []
    for idx, value in enumerate(labels):
        if value[4] != 1:
            pos.append(idx)

    labels = labels[pos]
    labels = labels[:, :4].astype(float)
    images = images[pos]

    np.savez("training_data_4class_new.npz", images=images, labels=labels)