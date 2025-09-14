# LearningHumanoidArmMotion-RAL2025-Code
This repository is an open-sourced code for the following paper presented at the 2025 IEEE Robotics and Automation Letters (RA-L).

#### Title: Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent Reinforcement Learning <br/> 
Paper Link: [https://arxiv.org/pdf/2507.04140](https://arxiv.org/pdf/2507.04140) <br/>
Video Link: [https://youtu.be/BNYML7QZyWQ](https://youtu.be/BNYML7QZyWQ) <br/>
Project Website: [https://hojae-io.github.io/LearningHumanoidArmMotion-RAL2025-Website](https://hojae-io.github.io/LearningHumanoidArmMotion-RAL2025-Website)


<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#systeminfo">System Info</a></li>
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
   This should set up [IssacLab](https://github.com/hojae-io/IsaacLab), [cusadi](https://github.com/hojae-io/cusadi), [rsl_rl](https://github.com/hojae-io/rsl_rl) submodules (branch: dev_RAL2025).

3. Go to the [IssacLab](https://github.com/hojae-io/IsaacLab) directory. <br/>
   Follow the [instruction](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install _IsaacSim_ and _IsaacLab_ anaconda virtual environment. <br/>
    > :bulb: **Note** <br/>
    > You _must_ use the customized [IssacLab](https://github.com/hojae-io/IsaacLab), not the official version. <br/>
    > IsaacSim version: 4.5.0
    
4. Go to the [rsl_rl](https://github.com/hojae-io/rsl_rl) directory. <br/>
   Check if _rsl_rl_ is already installed during IsaacLab installation by running `pip show rsl_rl`; if so, uninstall it by running `pip uninstall rsl_rl`. <br/>
   Run the following command to install customized [rsl_rl](https://github.com/hojae-io/rsl_rl).
    ```bash
    cd rsl_rl
    pip install -e .
    ```

---
### System Info

Operating System: Ubuntu 22.04 / Ubuntu 24.04 <br/>
GPU: Geforce 3090 / Geforce 4070 Ti

---
### Troubleshooting

1. TODO

---
### Acknowledgement

We would appreciate it if you would cite it in academic publications:
```
@article{lee2025learning,
  title={Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent Reinforcement Learning},
  author={Lee, Ho Jae and Jeon, Se Hwan and Kim, Sangbae},
  journal={arXiv preprint arXiv:2507.04140},
  year={2025}
}
```
