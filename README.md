# Ethical harvest

Based on the Harvest environment from: 
*Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). Multi-agent reinforcement learning in sequential social dilemmas. In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).*

This is the code for the final assignment of the Self-organizing Multiagent Systems course at the University of Barcelona.
## How to run the code:
- install python with pip (tested for Python 3.8.5 )
- install the required libs via ```pip install -r requirements.txt```
- install the dependencies required by tensorflow, see: https://www.tensorflow.org/install/gpu#software_requirements
- For the training run ```python Learning.py``` from the desired working directory
- To open the plots in the browser:
    - Either change to ```./impl/vis``` and run ```python vis.py```. This allows to select to show graphs for any of the experiments saved in the working directory.
    - Or edit ```./Learning.py``` and change ```SERVE_VISUALIZATION``` to ```True```. This shows graphs only for the running training session but also updates as new data becomes available.
    
For more information on results and the project structure please read the corresponding report.