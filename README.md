# LearningHumanoidArmMotion-RAL2025-Code
This repository is an open-sourced code for the following paper presented at the 2025 IEEE Robotics and Automation Letters (RA-L).

#### Title: Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent Reinforcement Learning <br/> 
Paper Link: [https://arxiv.org/pdf/2507.04140](https://arxiv.org/pdf/2507.04140) <br/>
Video Link: [https://youtu.be/BNYML7QZyWQ](https://youtu.be/BNYML7QZyWQ) <br/>
Project Website: [https://hojae-io.github.io/LearningHumanoidArmMotion-RAL2025-Website](https://hojae-io.github.io/LearningHumanoidArmMotion-RAL2025-Website)

<br/>
<div align="center">
  <a href="https://youtu.be/BNYML7QZyWQ">
    <img src="media/intro.gif" width="60%" 
         style="border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.3); object-fit:cover;">
  </a>
</div>
<br/>

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#system-info">System Info</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
  </ol>
</details>

## Getting Started

### Installation

1. Git clone the repository.
2. Set up the git submodules.
   ```bash
   git submodule init
   git submodule update
   ```
   This should set up [IsaacLab](https://github.com/hojae-io/IsaacLab), [cusadi](https://github.com/hojae-io/cusadi), [rsl_rl](https://github.com/hojae-io/rsl_rl) submodules (branch: dev_RAL2025).

3. Go to the [IsaacLab](https://github.com/hojae-io/IsaacLab) directory. <br/>
   Follow the [instruction](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install _IsaacSim_ and _IsaacLab_ anaconda virtual environment. <br/>
   ```bash
   cd IsaacLab
   conda create -n ral2025 python=3.10
   conda activate ral2025
   pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
   ./isaaclab.sh --install none
   ```
   
    > :bulb: **Note** <br/>
    > You _must_ use the customized [IsaacLab](https://github.com/hojae-io/IsaacLab), not the official version. <br/>
    > IsaacSim version: 4.5.0 <br/>
    > Python version: 3.10
    
4. Go to the [rsl_rl](https://github.com/hojae-io/rsl_rl) directory. <br/>
   Check if _rsl_rl_ is already installed during IsaacLab installation by running `pip show rsl_rl`; if so, uninstall it by running `pip uninstall rsl_rl`. <br/>
   Run the following command to install customized [rsl_rl](https://github.com/hojae-io/rsl_rl).
    ```bash
    cd rsl_rl
    pip install -e .
    ```

5. Go to the [cusadi](https://github.com/hojae-io/cusadi) directory. <br/>
    ```bash
    cd cusadi
    pip install -e .
    ```

6. Install required libraries by running `pip install -r requirements.txt` at root directory.
    ```bash
    pip install -r requirements.txt
    ```

> [!IMPORTANT]
> By now, you should be able to run
> ```bash
> pip show isaacsim
> pip show IsaacLab
> pip show rsl_rl
> pip show cusadi
> ```
> And all these packages should be located in `$HOME/anaconda3/envs/ral2025/lib/python3.10/site-packages` and their Editable project located in `LearningHumanoidArmMotion-RAL2025-Code/`

### Code Structure

| Directory      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `extensions/` | Core algorithms (MIT Humanoid env, CAM computation, etc.)                  |
| `IsaacLab/`   | RL wrappers (observation, action, reward, termination managers etc.) |
| `cusadi/`     | GPU parallelization of CasADi functions (pinocchio etc.) |
| `rsl_rl/`     | RL algorithms (PPO etc.) |
| `scripts/`    | Entry points (`play_modular.py`, `train_modular.py` etc.)                       |
| `logs/`       | Saved trained models                                                       |
| `resources/`  | MIT Humanoid URDF    
<br/>

## User Manual

### 0. Generate CusADi Function (Optional)
Necessary CusADi functions for running this code are already generated. So you don't need to run this section.
But if you want to generate custom CusADi functions, follow the instructions below.

1. Install CasADi extension of Pinocchio by following the instruction [here](https://stack-of-tasks.github.io/pinocchio/download.html) or by running `conda install pinocchio -c conda-forge`. <br/>
   You should be able to run `from pinocchio import casadi as cpin` from the script.

2. Run
   ```bash
   python ./extensions/humanoid/dynamics/forward_kinematics.py
   ```
   It will generate casadi functions in `extensions/humanoid/dynamics/casadi_fns` directory.

3. Copy manually these casadi functions to `cusadi/src/casadi_functions`
4. Run
   ```base
   cd cusadi
   python run_codegen.py
   ```
   It will generate `*.cu` files in `cusadi/codegen`. Now we are ready to use _CusADi_ functions!

### 3. Deploy the policy to robot hardware
This repository does not include a code stack for deploying a policy to MIT Humanoid hardware.
Please check the [Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software) for our lab's hardware code stack.

To deploy the trained policy, you would need to set `EXPORT_POLICY=TRUE` in the `scripts/play_modular.py` script.
Then you would get a `*.onnx` file to run on C++ code.

## System Info

Operating System: Ubuntu 22.04 <br/>
GPU: Geforce 3090 / Geforce 4070 Ti


## Troubleshooting

1. If you have any issue with `numpy` version conflict, consider downgrading to version 1.23.1 by `pip install numpy==1.23.1`


## Acknowledgement

We would appreciate it if you would cite it in academic publications:
```
@article{lee2025learning,
  title={Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent Reinforcement Learning},
  author={Lee, Ho Jae and Jeon, Se Hwan and Kim, Sangbae},
  journal={arXiv preprint arXiv:2507.04140},
  year={2025}
}
```
