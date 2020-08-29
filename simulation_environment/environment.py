import cityflow 
import time

class Simulator():
	def __init__(self, config_path):
	    self.config_path = config_path
	    self.eng = cityflow.Engine(self.config_path, thread_num=1)

	def run_single_step(self):
		self.eng.next_step()

	def run_n_steps(self, n):
		for i in range(n):
			self.run_single_step()






Sim = Simulator('examples/config.json')
Sim.run_n_steps(50)

