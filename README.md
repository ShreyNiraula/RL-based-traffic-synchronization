# TRAFFIC SYNCHRONIZATION IN KATHMANDU USING REINFORCEMENT LEARNING
## This is the project under the quantum hack

# DEMO VIDEO [https://www.youtube.com/watch?v=NHan2zTlRLs]

## Team members:
- Sajil Awale
- Rashik Shrestha
- Shrey Niraula

### Ideas and overall flow
- aim: Traffic signal synchronization across the junctions to reduce the traffic jam in Kathmandu using Reinforcement Learning (RL)

- Overall flow: YOLO vehicle detection ---> Simulation & RL ---> Optimized and Synchronized result ---> Deploy in Kathmandu Roads


## Project Dependencies
### For Vechicle Detection
- YOLO 


### For Simulation
- CityFlow 
- SUMO 


## Detail on Each Dependencies
### For Vechicle Detection
#### YOLO 
##### Further Dependencies
- SORT

### For Simulation
#### CityFlow  (build from source)
##### Further Dependencies
- linux
- CMAKE
- config file path = 'examples/config.json'
- in index.html, NOT `roadnetFile.json`, BUT `replay_roadnet.json`
- `replay_roadnet.json` after engine creatation 


#### SUMO (build from source)
##### Further Dependencies
- linux
- CMAKE



## References:
- [https://traffic-signal-control.github.io]
- [http://kec.edu.np/wp-content/uploads/2018/10/11.pdf]
- [https://arxiv.org/abs/2004.11986]
- [http://ceur-ws.org/Vol-223/53.pdf]



