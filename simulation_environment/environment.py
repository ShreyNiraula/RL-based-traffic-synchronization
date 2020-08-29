import time
import os, sys
import cityflow 
import pysumo
from tqdm import tqdm
import random
from time import time

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
 	sys.exit("please declare environment variable 'SUMO_HOME'")


class Simulator():
	def __init__(self, config_path):
	    self.config_path = config_path
	    self.eng = cityflow.Engine(self.config_path, thread_num=1)
	    self.state = [0]*28
	    self.actions = ['rGrG','ryry','GrGr','yryr']

	def run_single_step(self):
		self.eng.next_step()

	# run the traning step for n times
	def run_n_steps(self, n):
		for i in range(n):
			self.run_single_step()

	# Get number of waiting vehicles on each lane. 
	# Currently, vehicles with speed less than 0.1m/s is considered as waiting.
	def get_waiting_vehiches(self):
		return self.eng.get_lane_waiting_vehicle_count()

	def get_no_of_vehicles(self):
		lane_vehicles = self.eng.get_lane_vehicles()
		no_of_vehicles = []
		
		for k,v in lane_vehicles.items():
			no_of_veh = len(list(lane_vehicles[k]))
			no_of_vehicles.append(no_of_veh)

		return no_of_vehicles

	def get_queue_length(self):
		queue = self.eng.get_queue_length()
		q = []
		
		for k,v in queue.items():
			queue_length = len(list(lane_vehicles[k]))
			q.append(no_of_veh)

		return q


	# def get_total_queue_length(self):
	# 	queue = self.eng.queue_sequence()
	# 	seq = []
		
	# 	for k,v in queue.items():
	# 		queue_length = len(list(lane_vehicles[k]))
	# 		seq.append(no_of_veh)

	# 	return seq

	def get_state(self):
		wait_vehicles = self.get_waiting_vehiches()

		# print(wait_vehicles)
		
		temp = 0
		count = 0
		for key,value in wait_vehicles.items():
			count += 1
			temp += value

			if count % 3 == 0:
				self.state.append(temp)
				temp = 0

		temp = 0
		count = 0
		vehicles = self.get_no_of_vehicles()
		for v in vehicles:
			count += 1
			temp += value

			if count % 3 == 0:
				self.state.append(temp)
				temp = 0

		print(self.state)

		temp = 0
		count = 0
		qu = self.get_queue_length()
		for q in qu:
			count += 1
			temp += value

			if count % 3 == 0:
				self.state.append(temp)
				temp = 0

		# print(self.state)  # see the states

		return self.state

	def simulate(self):
		self.time_start = time()
		for i in tqdm(range(500)):
			pysumo.simulation_start(cmd)
		#	print 'lanes:', pysumo.tls_getControlledLanes('0');
			print('all lanes', pysumo.lane_list());
			for j in range(1000):
				pysumo.tls_setstate("0",random_action())
				pysumo.simulation_step()
				ids =  pysumo.lane_onLaneVehicles("0_n_0")
				if ids:
					print(ids);
			pysumo.simulation_stop()
		self.time_end = time()

	def random_action(self):
		return random.choice(actions)

	


# test_code
# Sim = Simulator('config.json')
# Sim.run_n_steps(1000)
# Sim.get_state()