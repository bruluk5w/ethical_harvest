# Ethical harvest

Based on the Harvest environment from: 
*Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). Multi-agent reinforcement learning in sequential social dilemmas. In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).*

This is the code for the final assignment of the Self-organizing Multiagent Systems course at the University of Barcelona.
## How to run the code:
- install python with pip (tested for Python 3.8.5 )
- install the required libs via ```pip install -r requirements.txt```
- install the dependencies required by tensorflow, see: https://www.tensorflow.org/install/gpu#software_requirements
  
- For the training run ```python Learning.py``` from the desired working directory.
- You can set the name of the training run in the first ```set_config``` call in ```Learning.py```. This is the name of the subfolder in the working directory to which all relevant data for the training run will be saved.
  
- To open the plots in the browser:
    - Either change to ```./impl/vis``` and run ```python vis.py```. This allows to select and view graphs for any of the experiments saved in the working directory.
    - Or edit ```./Learning.py``` and change ```SERVE_VISUALIZATION``` to ```True```. This shows graphs only for the running training session but also updates as new data becomes available.
  
- To evaluate and agent:
  - edit ```Learning.py``` and set the ```EVALUATE_EXPERIMENT``` variable to the name of the folder in the working directory which contains the data from the experiment for which  you want to evaluate an agent.
  - also set the ```EPISODE_NUMBER``` variable to the episode number for which you want the agents to load their respective model weights.
    You can have a look to the folder ```./<experiment_name>/weights/``` in your chosen working directory to see for which previous training episodes model weights have been dumped.
  - To load the trace file from the evaluation run rename the file ```trace_eval.txt``` in the respective subfolder of the working directory to ```trace.txt``` and open the visualization tool as described above. 
    
For more information please read the corresponding report.