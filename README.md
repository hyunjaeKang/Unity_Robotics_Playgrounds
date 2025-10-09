# Unity_Robot_Playgrounds

### Install ML-Agents package

1. navigating to the menu Window -> Package Manager.
2. In the package manager window click on the + button on the top left of the packages list).
3. Select Add package from disk...
4. Navigate into the ./MLAgents_4.0/com.unity.ml-agents folder.
5. Select the package.json file.

<p align="center"> <img src="https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/images/unity_package_manager_window.png" alt="Unity Package Manager Window" height="150" border="3"> <img src="https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/images/unity_package_json.png" alt="package.json" height="150" border="3"> </p>

- **[NOTE]** This step is required for each Unity Project.


-----

### Setup a conda environment

 ```
 conda create -y -n mlagents python=3.10.12
 conda activate mlagents


 pip install mlagents==1.1.0
 pip install ipykernel ipywidgets
 pip install torchvision torchaudio
 ```

-----


### Test packages
| Package | Version |
| :---: | :---: |
|Unity  | 6000.2.4.f1 |
|ML-Agents (Unity package) | Release 23 [4.0.0] |
| mlagent (python package) | 1.1.0 |

----

### Playgrounds

<p></p>


<table  border="1">
  <thead>
    <tr>
      <th style="text-align: center;">Playground</th>
      <th style="text-align: center;">Screenshot</th>
      <th style="text-align: center;">Unity Project</th>
      <th style="text-align: center;">Notebook-Python API</th>
      <th style="text-align: center;">Notebook-ML Agents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>3D Ball</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/3DBall.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/3DBall/">3DBall</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/3DBall.ipynb">3DBall.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/3DBall_ml.ipynb">3DBall_ml.ipynb</a></th>
    </tr>
    <tr>center
        <th style="text-align: center; padding: 10px;" rowspan="2"><b>GridWorld</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><img src="./Unity6000_Envs/GridWorld.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Unity6000_Projects/GridWorld/">GridWorld</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/GridWorld_DQN.ipynb">GridWorld_DQN.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Agent_Scripts/GridWorld_ml.ipynb">GridWorld_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/GridWorld_A2C.ipynb">GridWorld_A2C.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>Drone</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/Drone.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Drone/">Drone</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Drone_DDPG.ipynb">Drone_DDPG.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Drone_ml.ipynb">Drone_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="3"><b>Kart</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="3"><img src="./Unity6000_Envs/Kart.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="3"><a href="./Unity6000_Projects/Kart/">Kart</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="3"><a href="./Agent_Scripts/Kart_BC.ipynb">Kart_BC.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Kart_ml.ipynb">Kart_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Kart_BC_ml.ipynb">Kart_BC_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Kart_BC_GAIL_ml.ipynb">Kart_BC_GAIL_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="2"><b>Dodge</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><img src="./Unity6000_Envs/Dodge.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Unity6000_Projects/Dodge/">Dodge</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Dodge_Random_PPO.ipynb">Dodge_Random_PPO.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Agent_Scripts/Dodge_ml.ipynb">Dodge_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Dodge_Curriculum_PPO.ipynb">Dodge_Curriculum_PPO.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>Dodge-Attention</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/Dodge_Att.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Dodge_Attention/">Dodge_Attention</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Dodge_Attention_PPO.ipynb">Dodge_Attention_PPO.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Dodge_Attention_ml.ipynb">Dodge_Attention_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>Pong</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/Pong.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Pong/">Pong</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Pong_Adversarial.ipynb">Pong_Adversarial.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Pong_Adversarial_ml.ipynb">Pong_Adversarial_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>EscapeRoom</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/EscapeRoom.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/EscapeRoom/">EscapeRoom</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/EscapeRoom_MAPOCA.ipynb">EscapeRoom_MAPOCA.ipynb</a></th>
        <th style="text-align: left; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/EscapeRoom_MAPOCA_ml.ipynb">EscapeRoom_MAPOCA_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>Maze</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/Maze.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Maze/">Maze</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Maze_RND_PPO.ipynb">Maze_RND_PPO.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Maze_RND_PPO_ml.ipynb">Maze_RND_PPO_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="2"><b>TwoMission</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><img src="./Unity6000_Envs/TwoMission.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Unity6000_Projects/TwoMission/">TwoMission</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/TwoMission_PPO.ipynb">TwoMission_PPO.ipynb</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Agent_Scripts/TwoMission_Hyper_ml.ipynb">TwoMission_Hyper_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/TwoMission_HyperPPO.ipynb">TwoMission_HyperPPO.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="2"><b>Crawler</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><img src="./Unity6000_Envs/Crawler.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"><a href="./Unity6000_Projects/Crawler/">Crawler</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="2"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Crawer_ml.ipynb">Crawer_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Crawer_BC_ml.ipynb">Crawer_BC_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>Walker</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/Walker.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Walker/">Crawler</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Walker_ml.ipynb">Walker_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>Worm</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/Worm.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Worm/">Worm</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/Worm_ml.ipynb">Worm_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>SoccerTwos</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/SoccerTwos.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Soccer/">Soccer</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/SoccerTwos_ml.ipynb">SoccerTwos_ml.ipynb</a></th>
    </tr>
    <tr>
        <th style="text-align: center; padding: 10px;" rowspan="1"><b>StrikersVsGoalie</b></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><img src="./Unity6000_Envs/StrikersVsGoalie.png?raw=true"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Unity6000_Projects/Soccer/">Soccer</a></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"></th>
        <th style="text-align: center; padding: 10px;" rowspan="1"><a href="./Agent_Scripts/StrikersVsGoalie_ml.ipynb">StrikersVsGoalie_ml.ipynb</a></th>
    </tr>
  </tbody>
</table>

<p></p>


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