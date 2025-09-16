import matplotlib.pyplot as plt
import numpy as np


class ContactForcesLivePlotter:
    plt.ion()

    def __init__(self, env):
        self.env = env
        
        self.fig, self.axes = plt.subplots(2, 3)
        ylim = [-50, 1000]
        self.axes[0,0].set(xlabel='time', ylabel='rf_x [N]', ylim=ylim)
        self.axes[0,1].set(xlabel='time', ylabel='rf_y [N]', ylim=ylim)
        self.axes[0,2].set(xlabel='time', ylabel='rf_z [N]', ylim=ylim)

        self.axes[1,0].set(xlabel='time', ylabel='lf_x [N]', ylim=ylim)
        self.axes[1,1].set(xlabel='time', ylabel='lf_y [N]', ylim=ylim)
        self.axes[1,2].set(xlabel='time', ylabel='lf_z [N]', ylim=ylim)

        self.right_foot_contact_forces_x = list()
        self.right_foot_contact_forces_y = list()
        self.right_foot_contact_forces_z = list()
        self.left_foot_contact_forces_x = list()
        self.left_foot_contact_forces_y = list()
        self.left_foot_contact_forces_z = list()

        self.episode_length = 0

    def log(self, contact_forces):
        self.right_foot_contact_forces_x.append(contact_forces[self.env.feet_ids[0], 0].item())
        self.right_foot_contact_forces_y.append(contact_forces[self.env.feet_ids[0], 1].item())
        self.right_foot_contact_forces_z.append(contact_forces[self.env.feet_ids[0], 2].item())
        self.left_foot_contact_forces_x.append(contact_forces[self.env.feet_ids[1], 0].item())
        self.left_foot_contact_forces_y.append(contact_forces[self.env.feet_ids[1], 1].item())
        self.left_foot_contact_forces_z.append(contact_forces[self.env.feet_ids[1], 2].item())

        self.episode_length += 1

    def plot(self):
        self.axes[0,0].plot(np.arange(self.episode_length), self.right_foot_contact_forces_x, 'r-')
        self.axes[0,1].plot(np.arange(self.episode_length), self.right_foot_contact_forces_y, 'r-')
        self.axes[0,2].plot(np.arange(self.episode_length), self.right_foot_contact_forces_z, 'r-')
        self.axes[0,2].axhline(y=self.env.total_weight, color='r', linestyle='--')
        self.axes[0,2].legend(['measured', 'total weight'])

        self.axes[1,0].plot(np.arange(self.episode_length), self.left_foot_contact_forces_x, 'b-')
        self.axes[1,1].plot(np.arange(self.episode_length), self.left_foot_contact_forces_y, 'b-')
        self.axes[1,2].plot(np.arange(self.episode_length), self.left_foot_contact_forces_z, 'b-')
        self.axes[1,2].axhline(y=self.env.total_weight, color='b', linestyle='--')
        self.axes[1,2].legend(['measured', 'total weight'])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()