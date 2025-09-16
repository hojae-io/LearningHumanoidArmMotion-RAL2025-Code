import os
from extensions import ISAACLAB_BRL_ROOT_DIR
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from PIL import Image

def moving_average(x, w):
    """ Moving average filter 
        x: input signal
        w: window size
    """
    nan = np.array([np.nan] * (w-1))
    mvag = np.convolve(x, np.ones(w), 'valid') / w
    return np.concatenate((nan, mvag))

class ScreenShotter():

    def __init__(self, env, resume_path):
        self.env = env
        path_split = resume_path.split('/')
        self.checkpoint = path_split[-1][:-3]
        log_root_path = '/'.join(path_split[:-1])
        self.folderpath = os.path.join(log_root_path, 'analysis')
        os.makedirs(self.folderpath, exist_ok=True)
        self.screenshot_cnt = 0

    def screenshot(self, image):
        # Create a PIL Image from our NumPy array
        img = Image.fromarray(image, 'RGB')
        # Save the image as PDF
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_{self.screenshot_cnt}.pdf")
        img.save(filepath, "PDF", resolution=100.0)
        self.screenshot_cnt += 1


class AnalysisRecorder():

    def __init__(self, env, resume_path):
        self.env = env
        path_split = resume_path.split('/')
        self.checkpoint = path_split[-1][:-3]
        log_root_path = '/'.join(path_split[:-1])
        self.frames = []
        self.states_dict = defaultdict(list) # defaultdict(lambda: defaultdict(list)), defaultdict(list)
        self.commands_dict = defaultdict(list)
        self.fps = int(1/self.env.step_dt)
        self.episode_length = 0
        self.folderpath = os.path.join(log_root_path, 'analysis')
        os.makedirs(self.folderpath, exist_ok=True)

    def log(self, image, states_dict, commands_dict):
        # * Log images
        self.frames.append(image)

        # * Log states and commands
        for key, value in states_dict.items():
            if key == 'root_lin_vel_w':
                self.states_dict['COM_vel_x'].append(value[0].item())
                self.states_dict['COM_vel_y'].append(value[1].item())
            elif key == 'step_length':
                self.states_dict['step_length'].append(value.item())
            elif key == 'step_width':
                self.states_dict['step_width'].append(value.item())
            elif key == "contact_forces":
                self.states_dict['rf_contact_forces_x'].append(value[self.env.feet_ids[0],0].item())
                self.states_dict['rf_contact_forces_y'].append(value[self.env.feet_ids[0],1].item())
                self.states_dict['rf_contact_forces_z'].append(value[self.env.feet_ids[0],2].item())
                self.states_dict['lf_contact_forces_x'].append(value[self.env.feet_ids[1],0].item())
                self.states_dict['lf_contact_forces_y'].append(value[self.env.feet_ids[1],1].item())
                self.states_dict['lf_contact_forces_z'].append(value[self.env.feet_ids[1],2].item())

        for key, value in commands_dict.items():
            if key == 'vel_command':
                self.commands_dict['COM_dvel_x'].append(value[0].item())
                self.commands_dict['COM_dvel_y'].append(value[1].item())
            elif key == 'dstep_length':
                self.commands_dict['dstep_length'].append(value.item())
            elif key == 'dstep_width':
                self.commands_dict['dstep_width'].append(value.item())
            elif key == 'step_commands':
                self.commands_dict['dstep_left_x'].append(value[0,0].item())
                self.commands_dict['dstep_left_y'].append(value[0,1].item())
                self.commands_dict['dstep_right_x'].append(value[1,0].item())
                self.commands_dict['dstep_right_y'].append(value[1,1].item())

        self.episode_length += 1

    def save_animation_and_video(self):
        # self.make_animation()
        # self.make_animation_vel_tracking()
        # self.make_contact_forces_animation()
        self.make_video()

    def make_animation(self):
        """ Make animation for states_dict and commands_dict using FuncAnimation from matplotlib """
        print("Creating animation...")
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])

        # * Define the animation for COM tracking
        ax = fig.add_subplot(spec[0])

        def COM_vel_2D_init():
            COM_vel_x_ani.set_data(np.linspace(0, 1, 0), COM_vel_x[0:0])
            COM_vel_y_ani.set_data(np.linspace(0, 1, 0), COM_vel_y[0:0])
            return [COM_vel_x_ani, COM_vel_y_ani]

        def COM_vel_2D_update(i):
            COM_vel_x_ani.set_data(np.linspace(0, i-1, i), COM_vel_x[0:i])
            COM_vel_y_ani.set_data(np.linspace(0, i-1, i), COM_vel_y[0:i])
            COM_dvel_x_ani.set_data(np.linspace(0, i-1, i), COM_dvel_x[0:i])
            COM_dvel_y_ani.set_data(np.linspace(0, i-1, i), COM_dvel_y[0:i])
            return [COM_vel_x_ani, COM_vel_y_ani, ax]

        COM_vel_x = self.states_dict['COM_vel_x']
        COM_vel_y = self.states_dict['COM_vel_y']
        COM_dvel_x = self.commands_dict['COM_dvel_x']
        COM_dvel_y = self.commands_dict['COM_dvel_y']
        ax.set_xlim(0, self.episode_length)
        ax.set_ylim(min(min(COM_vel_x), min(COM_vel_y), min(COM_dvel_x), min(COM_dvel_y))-0.1, max(max(COM_vel_x), max(COM_vel_y), max(COM_dvel_x), max(COM_dvel_y))+0.1)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('CoM velocity (m/s)')
        tick_loc = np.arange(0, self.episode_length, 100)
        tick_labels = tick_loc / 100
        ax.xaxis.set_ticks(tick_loc)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.grid(ls='--')

        COM_vel_x_ani, = ax.plot([], [], color='k', label='CoM velocity x')
        COM_dvel_x_ani, = ax.plot([], [], color='k', linestyle='--', label='desired CoM velocity x')
        COM_vel_y_ani, = ax.plot([], [], color='purple', label='CoM velocity y')
        COM_dvel_y_ani, = ax.plot([], [], color='purple', linestyle='--', label='desired CoM velocity y')
        ax.legend(loc='upper right')

        # COM_vel_2D = FuncAnimation(fig=fig, init_func=COM_vel_2D_init, func=COM_vel_2D_update, frames=range(self.episode_length), interval=50, blit=False)

        # * Define the animation for step length and step width
        bx = fig.add_subplot(spec[1])

        step_length = self.states_dict['step_length']
        step_width = self.states_dict['step_width']
        dstep_length = self.commands_dict['dstep_length']
        dstep_width = self.commands_dict['dstep_width']

        def step_params_2D_init():
            step_length_ani.set_data(np.linspace(0, 1, 0), step_length[0:0])
            step_width_ani.set_data(np.linspace(0, 1, 0), step_width[0:0])
            dstep_length_ani.set_data(np.linspace(0, 1, 0), dstep_length[0:0])
            dstep_width_ani.set_data(np.linspace(0, 1, 0), dstep_width[0:0])
            return [step_length_ani, step_width_ani]

        def step_params_2D_update(i):
            step_length_ani.set_data(np.linspace(0, i-1, i), step_length[0:i])
            step_width_ani.set_data(np.linspace(0, i-1, i), step_width[0:i])
            dstep_length_ani.set_data(np.linspace(0, i-1, i), dstep_length[0:i])
            dstep_width_ani.set_data(np.linspace(0, i-1, i), dstep_width[0:i])
            return [step_length_ani, step_width_ani, bx]

        bx.set_xlim(0, self.episode_length)
        bx.set_ylim(min(min(step_length), min(step_width), min(dstep_length), min(dstep_width))-0.1, max(max(step_length), max(step_width), max(dstep_length), max(dstep_width))+0.1)
        bx.set_xlabel('time (s)')
        bx.set_ylabel('step length/width (m)')
        tick_loc = np.arange(0, self.episode_length, 100)
        tick_labels = tick_loc / 100
        bx.xaxis.set_ticks(tick_loc)
        bx.xaxis.set_ticklabels(tick_labels)
        bx.grid(ls='--')

        step_length_ani, = bx.plot([], [], color='gray', label='step length')
        dstep_length_ani, = bx.plot([], [], color='gray', linestyle='--', label='desired step length')
        step_width_ani, = bx.plot([], [], color='cyan', label='step width')
        dstep_width_ani, = bx.plot([], [], color='cyan', linestyle='--', label='desired step width')
        bx.legend(loc='upper right')

        # step_params_2D = FuncAnimation(fig=fig, init_func=step_params_2D_init, func=step_params_2D_update, frames=range(self.episode_length), interval=50, blit=False)

        # * Combine all the animations
        def _init_func():
            artist1 = COM_vel_2D_init()
            artist2 = step_params_2D_init()
            return artist1 + artist2
        
        def _update_func(i):
            artist1 = COM_vel_2D_update(i)
            artist2 = step_params_2D_update(i)
            return artist1 + artist2

        anim = FuncAnimation(fig=fig, init_func=_init_func, func=_update_func, frames=range(self.episode_length), interval=50, blit=False)

        # * Save the animation
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_plot.mp4")
        # COM_vel_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
        # step_params_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
        anim.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])

        print(f"Animation saved to {filepath}")

    def make_animation_vel_tracking(self):
        """ Make animation for states_dict and commands_dict using FuncAnimation from matplotlib """
        print("Creating animation...")
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])

        # * Define the animation for COM tracking
        ax = fig.add_subplot(spec[0])

        def COM_vel_2D_init():
            COM_vel_x_ani.set_data(np.linspace(0, 1, 0), COM_vel_x[0:0])
            COM_vel_y_ani.set_data(np.linspace(0, 1, 0), COM_vel_y[0:0])
            return [COM_vel_x_ani, COM_vel_y_ani]

        def COM_vel_2D_update(i):
            COM_vel_x_ani.set_data(np.linspace(0, i-1, i), COM_vel_x[0:i])
            COM_vel_y_ani.set_data(np.linspace(0, i-1, i), COM_vel_y[0:i])
            COM_dvel_x_ani.set_data(np.linspace(0, i-1, i), COM_dvel_x[0:i])
            COM_dvel_y_ani.set_data(np.linspace(0, i-1, i), COM_dvel_y[0:i])
            return [COM_vel_x_ani, COM_vel_y_ani, ax]

        COM_vel_x = self.states_dict['COM_vel_x']
        COM_vel_y = self.states_dict['COM_vel_y']
        COM_dvel_x = self.commands_dict['COM_dvel_x']
        COM_dvel_y = self.commands_dict['COM_dvel_y']
        ax.set_xlim(0, self.episode_length)
        ax.set_ylim(min(min(COM_vel_x), min(COM_vel_y), min(COM_dvel_x), min(COM_dvel_y))-0.1, max(max(COM_vel_x), max(COM_vel_y), max(COM_dvel_x), max(COM_dvel_y))+0.1)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('CoM velocity (m/s)')
        tick_loc = np.arange(0, self.episode_length, self.fps)
        tick_labels = tick_loc / self.fps
        ax.xaxis.set_ticks(tick_loc)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.grid(ls='--')

        COM_vel_x_ani, = ax.plot([], [], color='k', label='CoM velocity x')
        COM_dvel_x_ani, = ax.plot([], [], color='k', linestyle='--', label='desired CoM velocity x')
        COM_vel_y_ani, = ax.plot([], [], color='purple', label='CoM velocity y')
        COM_dvel_y_ani, = ax.plot([], [], color='purple', linestyle='--', label='desired CoM velocity y')
        ax.legend(loc='upper right')

        COM_vel_2D = FuncAnimation(fig=fig, init_func=COM_vel_2D_init, func=COM_vel_2D_update, frames=range(self.episode_length), interval=50, blit=False)
        
        # * Save the animation
        filepath = os.path.join(self.folderpath, "velocity_tracking_plot.mp4")
        COM_vel_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])

        print(f"Animation saved to {filepath}")

    def make_contact_forces_animation(self):
        """ Make animation for contact forces using FuncAnimation from matplotlib """
        print("Creating contact forces animation...")
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])

        # * Define the animation for right foot contact forces
        ax = fig.add_subplot(spec[0])

        def rf_contact_forces_2D_init():
            rf_contact_forces_x_ani.set_data(np.linspace(0, 1, 0), rf_contact_forces_x[0:0])
            rf_contact_forces_y_ani.set_data(np.linspace(0, 1, 0), rf_contact_forces_y[0:0])
            rf_contact_forces_z_ani.set_data(np.linspace(0, 1, 0), rf_contact_forces_z[0:0])
            return [rf_contact_forces_x_ani, rf_contact_forces_y_ani, rf_contact_forces_z_ani, total_weight_line] 

        def rf_contact_forces_2D_update(i):
            rf_contact_forces_x_ani.set_data(np.linspace(0, i-1, i), rf_contact_forces_x[0:i])
            rf_contact_forces_y_ani.set_data(np.linspace(0, i-1, i), rf_contact_forces_y[0:i])
            rf_contact_forces_z_ani.set_data(np.linspace(0, i-1, i), rf_contact_forces_z[0:i])
            return [rf_contact_forces_x_ani, rf_contact_forces_y_ani, rf_contact_forces_z_ani, total_weight_line, ax]
        
        rf_contact_forces_x = self.states_dict['rf_contact_forces_x']
        rf_contact_forces_y = self.states_dict['rf_contact_forces_y']
        rf_contact_forces_z = self.states_dict['rf_contact_forces_z']

        ax.set_xlim(0, self.episode_length)
        ax.set_ylim(-50, 1000)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('contact forces (N)')
        tick_loc = np.arange(0, self.episode_length, 100)
        tick_labels = tick_loc / 100
        ax.xaxis.set_ticks(tick_loc)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.grid(ls='--')

        rf_contact_forces_x_ani, = ax.plot([], [], color='r', label='right foot contact forces x')
        rf_contact_forces_y_ani, = ax.plot([], [], color='g', label='right foot contact forces y')
        rf_contact_forces_z_ani, = ax.plot([], [], color='b', label='right foot contact forces z')
        total_weight_line = ax.axhline(y=self.env.total_weight[0], color='k', linestyle='--', label='total weight')

        ax.legend(loc='upper right')

        # * Define the animation for left foot contact forces
        bx = fig.add_subplot(spec[1])

        def lf_contact_forces_2D_init():
            lf_contact_forces_x_ani.set_data(np.linspace(0, 1, 0), lf_contact_forces_x[0:0])
            lf_contact_forces_y_ani.set_data(np.linspace(0, 1, 0), lf_contact_forces_y[0:0])
            lf_contact_forces_z_ani.set_data(np.linspace(0, 1, 0), lf_contact_forces_z[0:0])
            return [lf_contact_forces_x_ani, lf_contact_forces_y_ani, lf_contact_forces_z_ani, total_weight_line]

        def lf_contact_forces_2D_update(i):
            lf_contact_forces_x_ani.set_data(np.linspace(0, i-1, i), lf_contact_forces_x[0:i])
            lf_contact_forces_y_ani.set_data(np.linspace(0, i-1, i), lf_contact_forces_y[0:i])
            lf_contact_forces_z_ani.set_data(np.linspace(0, i-1, i), lf_contact_forces_z[0:i])
            return [lf_contact_forces_x_ani, lf_contact_forces_y_ani, lf_contact_forces_z_ani, total_weight_line, bx]
        
        lf_contact_forces_x = self.states_dict['lf_contact_forces_x']
        lf_contact_forces_y = self.states_dict['lf_contact_forces_y']
        lf_contact_forces_z = self.states_dict['lf_contact_forces_z']

        bx.set_xlim(0, self.episode_length)
        bx.set_ylim(-50, 1000)
        bx.set_xlabel('time (s)')
        bx.set_ylabel('contact forces (N)')
        tick_loc = np.arange(0, self.episode_length, 100)
        tick_labels = tick_loc / 100
        bx.xaxis.set_ticks(tick_loc)
        bx.xaxis.set_ticklabels(tick_labels)
        bx.grid(ls='--')

        lf_contact_forces_x_ani, = bx.plot([], [], color='r', label='left foot contact forces x')
        lf_contact_forces_y_ani, = bx.plot([], [], color='g', label='left foot contact forces y')
        lf_contact_forces_z_ani, = bx.plot([], [], color='b', label='left foot contact forces z')
        total_weight_line = bx.axhline(y=self.env.total_weight[0], color='k', linestyle='--', label='total weight')
        bx.legend(loc='upper right')

        # * Combine all the animations
        def _init_func():
            artist1 = rf_contact_forces_2D_init()
            artist2 = lf_contact_forces_2D_init()
            return artist1 + artist2
        
        def _update_func(i):
            artist1 = rf_contact_forces_2D_update(i)
            artist2 = lf_contact_forces_2D_update(i)
            return artist1 + artist2
        
        anim = FuncAnimation(fig=fig, init_func=_init_func, func=_update_func, frames=range(self.episode_length), interval=50, blit=False)

        # * Save the animation
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_contact_forces.mp4")
        anim.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])

        print(f"Contact forces animation saved to {filepath}")


    def make_video(self):
        print("Creating video...")
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_gym.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frames[0].shape[1], self.frames[0].shape[0]))

        # Write the frames to the video file
        for frame in self.frames:
            cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

        # Release the video writer and print completion message
        out.release()
        print(f"Video saved to {filepath}")


class SimpleRecorder():

    def __init__(self, env, log_root_path):
        self.env = env
        self.frames = []
        self.fps = int(1/self.env.step_dt)
        self.folderpath = os.path.join(log_root_path)
        os.makedirs(self.folderpath, exist_ok=True)

    def log(self, image):
        # * Log images
        self.frames.append(image)

    def save_video(self):
        self.make_video()

    def make_video(self):
        print("Creating video...")
        filepath = os.path.join(self.folderpath, f"mpc.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frames[0].shape[1], self.frames[0].shape[0]))

        # Write the frames to the video file
        for frame in self.frames:
            cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

        # Release the video writer and print completion message
        out.release()
        print(f"Video saved to {filepath}")

class MPCAnalysisRecorder():

    def __init__(self, env, log_root_path):
        self.env = env
        self.frames = []
        self.states_dict = defaultdict(list)
        self.commands_dict = defaultdict(list)
        self.episode_length = 0
        self.fps = int(1/self.env.step_dt)
        self.folderpath = os.path.join(log_root_path)
        os.makedirs(self.folderpath, exist_ok=True)

    def log(self, image, states_dict, commands_dict):
        # * Log images
        self.frames.append(image)

        # * Log states and commands
        for key, value in states_dict.items():
            if key == 'root_lin_vel_w':
                self.states_dict['COM_vel_x'].append(value[0].item())
                self.states_dict['COM_vel_y'].append(value[1].item())
                # self.states_dict['ipopt_CoM_vel_x'].append(value[0,0].item())
                # self.states_dict['proxqp_CoM_vel_x'].append(value[1,0].item())
                # self.states_dict['customqp_CoM_vel_x'].append(value[2,0].item())
            elif key == 'step_length':
                self.states_dict['step_length'].append(value.item())
            elif key == 'step_width':
                self.states_dict['step_width'].append(value.item())
            elif key == 'p_right_error':
                self.states_dict['p_right_x_error'].append(value[0].item())
                self.states_dict['p_right_y_error'].append(value[1].item())
                self.states_dict['p_right_z_error'].append(value[2].item())
                self.states_dict['p_right_yaw_error'].append(value[3].item())
            elif key == 'v_right_error':
                self.states_dict['v_right_x_error'].append(value[0].item())
                self.states_dict['v_right_y_error'].append(value[1].item())
                self.states_dict['v_right_z_error'].append(value[2].item())
                self.states_dict['v_right_yaw_error'].append(value[3].item())
            elif key == 'p_left_error':
                self.states_dict['p_left_x_error'].append(value[0].item())
                self.states_dict['p_left_y_error'].append(value[1].item())
                self.states_dict['p_left_z_error'].append(value[2].item())
                self.states_dict['p_left_yaw_error'].append(value[3].item())
            elif key == 'v_left_error':
                self.states_dict['v_left_x_error'].append(value[0].item())
                self.states_dict['v_left_y_error'].append(value[1].item())
                self.states_dict['v_left_z_error'].append(value[2].item())
                self.states_dict['v_left_yaw_error'].append(value[3].item())
            elif key == 'contact_forces':
                self.states_dict['F_right_toe_x'].append(value[0].item())
                self.states_dict['F_right_toe_y'].append(value[1].item())
                self.states_dict['F_right_toe_z'].append(value[2].item())
                self.states_dict['F_right_heel_x'].append(value[3].item())
                self.states_dict['F_right_heel_y'].append(value[4].item())
                self.states_dict['F_right_heel_z'].append(value[5].item())
                self.states_dict['F_left_toe_x'].append(value[6].item())
                self.states_dict['F_left_toe_y'].append(value[7].item())
                self.states_dict['F_left_toe_z'].append(value[8].item())
                self.states_dict['F_left_heel_x'].append(value[9].item())
                self.states_dict['F_left_heel_y'].append(value[10].item())
                self.states_dict['F_left_heel_z'].append(value[11].item())
                # self.states_dict['ipopt_F_right_toe_z'].append(value[0,2].item())
                # self.states_dict['proxqp_F_right_toe_z'].append(value[1,2].item())
                # self.states_dict['customqp_F_right_toe_z'].append(value[2,2].item())
            elif key == 'state_error':
                self.states_dict['roll_error'].append(value[0].item())
                self.states_dict['pitch_error'].append(value[1].item())
                self.states_dict['yaw_error'].append(value[2].item())
                self.states_dict['x_error'].append(value[3].item())
                self.states_dict['y_error'].append(value[4].item())
                self.states_dict['z_error'].append(value[5].item())
                self.states_dict['w_x_error'].append(value[6].item())
                self.states_dict['w_y_error'].append(value[7].item())
                self.states_dict['w_z_error'].append(value[8].item())
                self.states_dict['v_x_error'].append(value[9].item())
                self.states_dict['v_y_error'].append(value[10].item())
                self.states_dict['v_z_error'].append(value[11].item())
                
        for key, value in commands_dict.items():
            if key == 'vel_command':
                self.commands_dict['COM_dvel_x'].append(value[0].item())
                self.commands_dict['COM_dvel_y'].append(value[1].item())
            elif key == 'dstep_length':
                self.commands_dict['dstep_length'].append(value.item())
            elif key == 'dstep_width':
                self.commands_dict['dstep_width'].append(value.item())

        self.episode_length += 1

    def save_animation_and_video(self):
        self.make_animation_vel_tracking()
        self.make_plot_foot_tracking()
        self.make_plot_contact_forces()
        # self.make_plot_contact_forces_solver_comparison()
        self.make_plot_state_error()
        self.make_video()

    def make_animation_vel_tracking(self):
        """ Make animation for states_dict and commands_dict using FuncAnimation from matplotlib """
        print("Creating animation...")
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])

        # * Define the animation for COM tracking
        ax = fig.add_subplot(spec[0])

        def COM_vel_2D_init():
            COM_vel_x_ani.set_data(np.linspace(0, 1, 0), COM_vel_x[0:0])
            COM_vel_y_ani.set_data(np.linspace(0, 1, 0), COM_vel_y[0:0])
            return [COM_vel_x_ani, COM_vel_y_ani]

        def COM_vel_2D_update(i):
            COM_vel_x_ani.set_data(np.linspace(0, i-1, i), COM_vel_x[0:i])
            COM_vel_y_ani.set_data(np.linspace(0, i-1, i), COM_vel_y[0:i])
            COM_dvel_x_ani.set_data(np.linspace(0, i-1, i), COM_dvel_x[0:i])
            COM_dvel_y_ani.set_data(np.linspace(0, i-1, i), COM_dvel_y[0:i])
            return [COM_vel_x_ani, COM_vel_y_ani, ax]

        COM_vel_x = self.states_dict['COM_vel_x']
        COM_vel_y = self.states_dict['COM_vel_y']
        COM_dvel_x = self.commands_dict['COM_dvel_x']
        COM_dvel_y = self.commands_dict['COM_dvel_y']
        ax.set_xlim(0, self.episode_length)
        ax.set_ylim(min(min(COM_vel_x), min(COM_vel_y), min(COM_dvel_x), min(COM_dvel_y))-0.1, max(max(COM_vel_x), max(COM_vel_y), max(COM_dvel_x), max(COM_dvel_y))+0.1)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('CoM velocity (m/s)')
        tick_loc = np.arange(0, self.episode_length, self.fps)
        tick_labels = tick_loc / self.fps
        ax.xaxis.set_ticks(tick_loc)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.grid(ls='--')

        COM_vel_x_ani, = ax.plot([], [], color='k', label='CoM velocity x')
        COM_dvel_x_ani, = ax.plot([], [], color='k', linestyle='--', label='desired CoM velocity x')
        COM_vel_y_ani, = ax.plot([], [], color='purple', label='CoM velocity y')
        COM_dvel_y_ani, = ax.plot([], [], color='purple', linestyle='--', label='desired CoM velocity y')
        ax.legend(loc='upper right')

        COM_vel_2D = FuncAnimation(fig=fig, init_func=COM_vel_2D_init, func=COM_vel_2D_update, frames=range(self.episode_length), interval=50, blit=False)
        
        # * Save the animation
        filepath = os.path.join(self.folderpath, "mpc_plot.mp4")
        COM_vel_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])

        print(f"Animation saved to {filepath}")

    def make_plot_foot_tracking(self):
        """ Make plot for the foot tracking error """
        print("Creating foot tracking error plot...")
        fig, axes = plt.subplots(4, 4, figsize=(18, 8))

        axes[0,0].plot(np.arange(self.episode_length), self.states_dict['p_right_x_error'], color='r')
        axes[0,0].axhline(y=0., color='r', linestyle='--')
        axes[0,1].plot(np.arange(self.episode_length), self.states_dict['p_right_y_error'], color='r')
        axes[0,1].axhline(y=0., color='r', linestyle='--')
        axes[0,2].plot(np.arange(self.episode_length), self.states_dict['p_right_z_error'], color='r')
        axes[0,2].axhline(y=0., color='r', linestyle='--')
        axes[0,3].plot(np.arange(self.episode_length), self.states_dict['p_right_yaw_error'], color='r')
        axes[0,3].axhline(y=0., color='r', linestyle='--')

        axes[1,0].plot(np.arange(self.episode_length), self.states_dict['v_right_x_error'], color='r')
        axes[1,0].axhline(y=0., color='r', linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.states_dict['v_right_y_error'], color='r')
        axes[1,1].axhline(y=0., color='r', linestyle='--')
        axes[1,2].plot(np.arange(self.episode_length), self.states_dict['v_right_z_error'], color='r')
        axes[1,2].axhline(y=0., color='r', linestyle='--')
        axes[1,3].plot(np.arange(self.episode_length), self.states_dict['v_right_yaw_error'], color='r')
        axes[1,3].axhline(y=0., color='r', linestyle='--')

        axes[2,0].plot(np.arange(self.episode_length), self.states_dict['p_left_x_error'], color='b')
        axes[2,0].axhline(y=0., color='b', linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.states_dict['p_left_y_error'], color='b')
        axes[2,1].axhline(y=0., color='b', linestyle='--')
        axes[2,2].plot(np.arange(self.episode_length), self.states_dict['p_left_z_error'], color='b')
        axes[2,2].axhline(y=0., color='b', linestyle='--')
        axes[2,3].plot(np.arange(self.episode_length), self.states_dict['p_left_yaw_error'], color='b')
        axes[2,3].axhline(y=0., color='b', linestyle='--')

        axes[3,0].plot(np.arange(self.episode_length), self.states_dict['v_left_x_error'], color='b')
        axes[3,0].axhline(y=0., color='b', linestyle='--')
        axes[3,1].plot(np.arange(self.episode_length), self.states_dict['v_left_y_error'], color='b')
        axes[3,1].axhline(y=0., color='b', linestyle='--')
        axes[3,2].plot(np.arange(self.episode_length), self.states_dict['v_left_z_error'], color='b')
        axes[3,2].axhline(y=0., color='b', linestyle='--')
        axes[3,3].plot(np.arange(self.episode_length), self.states_dict['v_left_yaw_error'], color='b')
        axes[3,3].axhline(y=0., color='b', linestyle='--')

        axes[0,0].set(xlabel='time [s]', ylabel='p_right_x_error [m]')
        axes[0,1].set(xlabel='time [s]', ylabel='p_right_y_error [m]')
        axes[0,2].set(xlabel='time [s]', ylabel='p_right_z_error [m]')
        axes[0,3].set(xlabel='time [s]', ylabel='p_right_yaw_error [rad]')
        axes[1,0].set(xlabel='time [s]', ylabel='v_right_x_error [m/s]')
        axes[1,1].set(xlabel='time [s]', ylabel='v_right_y_error [m/s]')
        axes[1,2].set(xlabel='time [s]', ylabel='v_right_z_error [m/s]')
        axes[1,3].set(xlabel='time [s]', ylabel='v_right_yaw_error [rad/s]')
        axes[2,0].set(xlabel='time [s]', ylabel='p_left_x_error [m]')
        axes[2,1].set(xlabel='time [s]', ylabel='p_left_y_error [m]')
        axes[2,2].set(xlabel='time [s]', ylabel='p_left_z_error [m]')
        axes[2,3].set(xlabel='time [s]', ylabel='p_left_yaw_error [rad]')
        axes[3,0].set(xlabel='time [s]', ylabel='v_left_x_error [m/s]')
        axes[3,1].set(xlabel='time [s]', ylabel='v_left_y_error [m/s]')
        axes[3,2].set(xlabel='time [s]', ylabel='v_left_z_error [m/s]')
        axes[3,3].set(xlabel='time [s]', ylabel='v_left_yaw_error [rad/s]')

        tick_loc = np.arange(0, self.episode_length, 2*self.fps)
        tick_labels = tick_loc / self.fps

        for i in range(4):
            for j in range(4):
                axes[i,j].xaxis.set_ticks(tick_loc)
                axes[i,j].xaxis.set_ticklabels(tick_labels)
                axes[i,j].grid(ls='--')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, "mpc_foot_tracking.pdf")
        plt.savefig(filepath)

        print(f"Foot tracking error plot saved to {filepath}")

    def make_plot_contact_forces(self):
        """ Make plot for the CoM velocity and the contact forces """
        print("Creating CoM velocity and contact forces plot...")
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # * Define the plot for the velocity command
        axes[0].plot(np.arange(self.episode_length), self.commands_dict['COM_dvel_x'], color='k')
        axes[0].plot(np.arange(self.episode_length), self.commands_dict['COM_dvel_y'], color='purple')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['COM_vel_x'], color='k', linestyle='--')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['COM_vel_y'], color='purple', linestyle='--')
        axes[0].set(xlabel='time [s]', ylabel='COM velocity command [m/s]')
        axes[0].legend(['COM_dvel_x', 'COM_dvel_y', 'COM_vel_x', 'COM_vel_y'])
        axes[0].grid(ls='--')

        # * Define the plot for the contact forces
        axes[1].plot(np.arange(self.episode_length), self.states_dict['F_right_toe_x'], color='r', linestyle='--')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['F_right_toe_y'], color='r', linestyle='-.')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['F_right_toe_z'], color='r')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['F_right_heel_x'], color='m', linestyle='--')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['F_right_heel_y'], color='m', linestyle='-.')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['F_right_heel_z'], color='m')
        axes[1].set(xlabel='time [s]', ylabel='contact forces [N]')
        axes[1].legend(['F_right_toe_x', 'F_right_toe_y', 'F_right_toe_z', 'F_right_heel_x', 'F_right_heel_y', 'F_right_heel_z'])
        axes[1].grid(ls='--')

        axes[2].plot(np.arange(self.episode_length), self.states_dict['F_left_toe_x'], color='b', linestyle='--')
        axes[2].plot(np.arange(self.episode_length), self.states_dict['F_left_toe_y'], color='b', linestyle='-.')
        axes[2].plot(np.arange(self.episode_length), self.states_dict['F_left_toe_z'], color='b')
        axes[2].plot(np.arange(self.episode_length), self.states_dict['F_left_heel_x'], color='c', linestyle='--')
        axes[2].plot(np.arange(self.episode_length), self.states_dict['F_left_heel_y'], color='c', linestyle='-.')
        axes[2].plot(np.arange(self.episode_length), self.states_dict['F_left_heel_z'], color='c')
        axes[2].set(xlabel='time [s]', ylabel='contact forces [N]')
        axes[2].legend(['F_left_toe_x', 'F_left_toe_y', 'F_left_toe_z', 'F_left_heel_x', 'F_left_heel_y', 'F_left_heel_z'])
        axes[2].grid(ls='--')

        tick_loc = np.arange(0, self.episode_length, 2*self.fps)
        tick_labels = tick_loc / self.fps

        for i in range(3):
            axes[i].xaxis.set_ticks(tick_loc)
            axes[i].xaxis.set_ticklabels(tick_labels)
            axes[i].grid(ls='--')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, "mpc_contact_forces.pdf")
        plt.savefig(filepath)

        print(f"CoM velocity and contact forces plot saved to {filepath}")

    def make_plot_contact_forces_solver_comparison(self):
        """ Make plot for the CoM velocity and the contact forces for solver comparison (ipopt, proxqp, customqp) """
        print("Creating CoM velocity and contact forces plot for solver comparison...")
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # * Define the plot for the velocity command
        axes[0].plot(np.arange(self.episode_length), self.commands_dict['COM_dvel_x'], color='k', linestyle='--')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['ipopt_CoM_vel_x'], color='r')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['proxqp_CoM_vel_x'], color='g')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['customqp_CoM_vel_x'], color='b')
        axes[0].set(xlabel='time [s]', ylabel='COM velocity command [m/s]')
        axes[0].legend(['COM_dvel_x', 'ipopt_CoM_vel_x', 'proxqp_CoM_vel_x', 'customqp_CoM_vel_x'])
        axes[0].grid(ls='--')

        # * Define the plot for the contact forces
        axes[1].plot(np.arange(self.episode_length), self.states_dict['ipopt_F_right_toe_z'], color='r',)
        axes[1].plot(np.arange(self.episode_length), self.states_dict['proxqp_F_right_toe_z'], color='g',)
        axes[1].plot(np.arange(self.episode_length), self.states_dict['customqp_F_right_toe_z'], color='b')
        axes[1].set(xlabel='time [s]', ylabel='contact forces [N]')
        axes[1].legend(['ipopt_F_right_toe_z', 'proxqp_F_right_toe_z', 'customqp_F_right_toe_z'])
        axes[1].grid(ls='--')

        tick_loc = np.arange(0, self.episode_length, 2*self.fps)
        tick_labels = tick_loc / self.fps

        for i in range(2):
            axes[i].xaxis.set_ticks(tick_loc)
            axes[i].xaxis.set_ticklabels(tick_labels)
            axes[i].grid(ls='--')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, "mpc_contact_forces_solver_comparison.pdf")
        plt.savefig(filepath)

        print(f"CoM velocity and contact forces plot for solver comparison saved to {filepath}")

    def make_plot_state_error(self):
        """ Make plot for the state error """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # * Define the plot for the velocity command
        axes[0].plot(np.arange(self.episode_length), self.commands_dict['COM_dvel_x'], color='k')
        axes[0].plot(np.arange(self.episode_length), self.commands_dict['COM_dvel_y'], color='purple')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['COM_vel_x'], color='k', linestyle='--')
        axes[0].plot(np.arange(self.episode_length), self.states_dict['COM_vel_y'], color='purple', linestyle='--')
        axes[0].set(xlabel='time [s]', ylabel='COM velocity command [m/s]')
        axes[0].legend(['COM_dvel_x', 'COM_dvel_y', 'COM_vel_x', 'COM_vel_y'])
        axes[0].grid(ls='--')

        # * Define the plot for the state error 
        axes[1].plot(np.arange(self.episode_length), self.states_dict['roll_error'], color='r')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['pitch_error'], color='g')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['yaw_error'], color='b')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['x_error'], color='c')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['y_error'], color='m')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['z_error'], color='y')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['w_x_error'], color='k')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['w_y_error'], color='purple')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['w_z_error'], color='orange')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['v_x_error'], color='brown')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['v_y_error'], color='pink')
        axes[1].plot(np.arange(self.episode_length), self.states_dict['v_z_error'], color='gray')
        axes[1].set(xlabel='time [s]', ylabel='state error')
        axes[1].legend(['roll_error', 'pitch_error', 'yaw_error', 'x_error', 'y_error', 'z_error', 'w_x_error', 'w_y_error', 'w_z_error', 'v_x_error', 'v_y_error', 'v_z_error'])
        axes[1].grid(ls='--')

        tick_loc = np.arange(0, self.episode_length, 2*self.fps)
        tick_labels = tick_loc / self.fps

        for i in range(2):
            axes[i].xaxis.set_ticks(tick_loc)
            axes[i].xaxis.set_ticklabels(tick_labels)
            axes[i].grid(ls='--')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, "mpc_state_error.pdf")
        plt.savefig(filepath)

        print(f"State error plot saved to {filepath}")

    def make_video(self):
        print("Creating video...")
        filepath = os.path.join(self.folderpath, "mpc.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frames[0].shape[1], self.frames[0].shape[0]))

        # Write the frames to the video file
        for frame in self.frames:
            cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

        # Release the video writer and print completion message
        out.release()
        print(f"Video saved to {filepath}")