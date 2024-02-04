import gym
import numpy as np
import torch
from torchvision.transforms import transforms
from model import CNN_RNN_Classifier
import keyboard

# CNN- LSTM
model_path = 'weights/modelLSTM.pth'
model = CNN_RNN_Classifier(in_channels=3, out_size=4).to("cpu")


model.load_state_dict(torch.load(model_path))
model.eval()

# transform for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def preprocess_observation(observation, prev_obs_buffer):
    obs_array = observation[0:82, 0:96]  # crop
    obs_tensor = transform(obs_array)
    obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim

    # Append current observation to buffer
    prev_obs_buffer = torch.cat([prev_obs_buffer[1:], obs_tensor], dim=0)

    return prev_obs_buffer


def get_model_action(observation):
    with torch.no_grad():

        output = model(observation.unsqueeze(0))  # Add batch dim
        actions = torch.argmax(output, dim=2).squeeze().tolist()
        for action in actions:
            if action == 0:
                return [-0.2, 0.0, 0.0]
            if action == 1:
                return [0.2, 0.0, 0.0]
            if action == 2:
                return [0.0, 0.2, 0.0]
            if action == 3:
                return [0.0, 0.0, 0.2]
            else:
                return [0.0, 0.0, 0.0]


def main():
    env = gym.make('CarRacing-v2', render_mode='human')

    obs = env.reset()
    obs_buffer = torch.zeros(5, 3, 96, 96)

    for frame in range(1, 5000):  # frames
        if frame <= 30:
            # random actions for the first 30 frames
            action = env.action_space.sample()
        else:

            obs_buffer = preprocess_observation(obs, obs_buffer)
            action = get_model_action(obs_buffer)

        obs, _, done, _, _ = env.step(action)
        env.render()

        if keyboard.is_pressed('q' or 'Q'):
            break

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
