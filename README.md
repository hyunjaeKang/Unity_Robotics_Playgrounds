# Unity_Robot_Playgrounds

### Install ML-Agents package

1. navigating to the menu Window -> Package Manager.
2. In the package manager window click on the + button on the top left of the packages list).
3. Select Add package from disk...
4. Navigate into the ./MLAgents_4.0/com.unity.ml-agents folder.
5. Select the package.json file.

<p align="center"> <img src="https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/images/unity_package_manager_window.png" alt="Unity Package Manager Window" height="150" border="3"> <img src="https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/images/unity_package_json.png" alt="package.json" height="150" border="3"> </p>

- **[NOTE]** This step is required for each Unity Project.

### Setup a conda environment

 ```
 conda create -y -n mlagents python=3.10.12
 conda activate mlagents


 pip install mlagents==1.1.0
 pip install ipykernel ipywidgets
 pip install torchvision torchaudio
 ```

### Test packages
| Package | Version |
| :---: | :---: |
|Unity  | 6000.2.4.f1 |
|ML-Agents (Unity package) | Release 23 [4.0.0] |
| mlagent (python package) | 1.1.0 |

### Learning Environments

| Name| Screenshot | Unity Project | Python API | ML Agents |
| :--- | :---: | :---: | :----  | :--- |
| 3D Ball | <img src=Unity6000_Envs/3DBall.png>| [3DBall](./Unity6000_Projects/3DBall/) | [3DBall.ipynb](/Agent_Scripts/3DBall.ipynb) | [3DBall_ml.ipynb](/Agent_Scripts/3DBall_ml.ipynb)
| GridWorld | <img src=Unity6000_Envs/GridWorld.png>| [GridWorld](./Unity6000_Projects/GridWorld/)| <br>[GridWorld_DQN.ipynb](/Agent_Scripts/GridWorld_DQN.ipynb)</br> <br>[GridWorld_A2C.ipynb](/Agent_Scripts/GridWorld_A2C.ipynb)</br> | [GridWorld_ml.ipynb](/Agent_Scripts/GridWorld_ml.ipynb) |
| Drone | <img src=Unity6000_Envs/Drone.png> | [Drone](./Unity6000_Projects/Drone/) | [Drone_DDPG.ipynb](/Agent_Scripts/Drone_DDPG.ipynb) | [Drone_ml.ipynb](/Agent_Scripts/Drone_ml.ipynb) |
| Kart | <img src=Unity6000_Envs/Kart.png> | [Kart](./Unity6000_Projects/Kart/) | [Kart_BC.ipynb](/Agent_Scripts/Kart_BC.ipynb)| <br>[Kart_ml.ipynb](/Agent_Scripts/Kart_ml.ipynb)</br> <br>[Kart_BC_ml.ipynb](/Agent_Scripts/Kart_BC_ml.ipynb)</br> <br>[Kart_BC_GAIL_ml.ipynb](/Agent_Scripts/Kart_BC_GAIL_ml.ipynb)</br> |
| Dodge | <img src=Unity6000_Envs/Dodge.png> | [Dodge](./Unity6000_Projects/Dodge/) | <br>[Dodge_Random_PPO.ipynb](/Agent_Scripts/Dodge_Random_PPO.ipynb)</br> <br>[Dodge_Curriculum_PPO.ipynb](/Agent_Scripts/Dodge_Curriculum_PPO.ipynb)</br>  | <br>[Dodge_ml.ipynb.ipynb](/Agent_Scripts/Dodge_ml.ipynb.ipynb)</br> |
| Dodge-Attention | <img src=Unity6000_Envs/Dodge_Att.png> | [Dodge_Att](./Unity6000_Projects/Dodge_Attention/) |  [Dodge_Attention_PPO.ipynb](/Agent_Scripts/Dodge_Attention_PPO.ipynb) | [Dodge_Attention_ml.ipynb](/Agent_Scripts/Dodge_Attention_ml.ipynb) |
| Pong | <img src=Unity6000_Envs/Pong.png> | [Pong](./Unity6000_Projects/Pong/) |  [Pong_Adversarial.ipynb](/Agent_Scripts/Pong_Adversarial.ipynb) | [Pong_Adversarial_ml.ipynb](/Agent_Scripts/Pong_Adversarial_ml.ipynb) |
| EscapeRoom | <img src=Unity6000_Envs/EscapeRoom.png> |[EscapeRoom](./Unity6000_Projects/EscapeRoom/) |  [EscapeRoom_MAPOCA.ipynb](/Agent_Scripts/EscapeRoom_MAPOCA.ipynb) | [EscapeRoom_MAPOCA_ml.ipynb](/Agent_Scripts/EscapeRoom_MAPOCA_ml.ipynb) |
| Maze |  <img src=Unity6000_Envs/Maze.png>  |[Maze](./Unity6000_Projects/Maze/) |  [Maze_RND_PPO.ipynb](/Agent_Scripts/Maze_RND_PPO.ipynb) | [Maze_RND_PPO_ml.ipynb](/Agent_Scripts/Maze_RND_PPO_ml.ipynb) |
| TwoMission |  <img src=Unity6000_Envs/TwoMission.png> | [TwoMission](./Unity6000_Projects/TwoMissions/) |  <br>[TwoMission_PPO.ipynb](/Agent_Scripts/TwoMission_PPO.ipynb)</br>  <br>[TwoMission_HyperPPO.ipynb](/Agent_Scripts/TwoMission_HyperPPO.ipynb)</br> | [TwoMission_Hyper_ml.ipynb](/Agent_Scripts/TwoMission_Hyper_ml.ipynb) |
| Crawler |  <img src=Unity6000_Envs/Crawler.png> | [Crawler](./Unity6000_Projects/Crawler/) |   | <br>[Crawer_ml.ipynb](/Agent_Scripts/Crawer_ml.ipynb)</br> <br>[Crawer_BC_ml.ipynb](/Agent_Scripts/Crawer_BC_ml.ipynb)</br> |
| Walker |  <img src=Unity6000_Envs/Walker.png> | [Waler](./Unity6000_Projects/Walker/) |   | [Walker_ml.ipynb](/Agent_Scripts/Walker_ml.ipynb) |
| Worm |  <img src=Unity6000_Envs/Worm.png> | [Worm](./Unity6000_Projects/Worm/) |  | [Worm_ml.ipynb](/Agent_Scripts/Worm_ml.ipynb) |
| SoccerTwos | <img src=Unity6000_Envs/SoccerTwos.png> |[Soccer](./Unity6000_Projects/Soccer/) |   | [SoccerTwos_ml.ipynb](/Agent_Scripts/SoccerTwos_ml.ipynb) |
| StrikersVsGoalie | <img src=Unity6000_Envs/StrikersVsGoalie.png> |[Soccer](./Unity6000_Projects/Soccer/) |   | [StrikersVsGoalie_ml.ipynb](/Agent_Scripts/StrikersVsGoalie_ml.ipynb) |

 ---
### Reference:


- ***Papers***:
    - ....

- ***Blog***:
    - https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html#advanced-local-installation-for-development
    - https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Examples.html
    - https://unity-technologies.github.io/ml-agents/Training-ML-Agents/
    - https://unity-technologies.github.io/ml-agents/Training-Configuration-File/


- ***Github***:
    - https://github.com/Unity-Technologies/ml-agents
    - https://github.com/reinforcement-learning-kr/Unity_ML_Agents_2.0