import numpy as np 
from numpy.linalg import norm
import collections
import os, sys
import matplotlib.pyplot as plt
import pdb
import scipy 
import sys
sys.path.append("..") 
from utils import smooth

# Smooth function
def sm(data):
  data = data.squeeze()
  length = data.size
  temp1 = data[1:]
  temp0 = data[:length-1]
  diff = temp1 - temp0
  var = np.std(diff)
  return var

class BoundaryAttack(object):
	def __init__(self, model, shape, args, source_step=0.01, spherical_step=0.01,
				step_adaptation=1.5):
		super(BoundaryAttack, self).__init__()
		self.model = model 
		self.source_step = source_step
		self.spherical_step = spherical_step
		self.stats_spherical_adversarial = collections.deque(maxlen=100)
		self.stats_step_adversarial = collections.deque(maxlen=30)

		self.min = np.zeros(shape, dtype=np.float32)
		self.max = np.ones(shape, dtype=np.float32)

		self.type = np.float32

		self.step_adaptation = step_adaptation
		self.log_every_n_steps = 1
		self.args = args

	def prepare_generate_candidates(self, original, perturbed):
		unnormalized_source_direction = original - perturbed
		source_norm = norm(unnormalized_source_direction)
		source_direction = unnormalized_source_direction / source_norm
		return unnormalized_source_direction, source_direction, source_norm
	
	def generate_candidate_default(self, original,
			unnormalized_source_direction, source_direction, source_norm):
		spherical_step = self.spherical_step
		source_step = self.source_step

		perturbation = np.random.randn(*original.shape)
		perturbation = perturbation.astype(np.float32)
		shape = perturbation.shape 
		# apply hanning filter
		if self.args.smooth:
			if self.args.win%2==1: 
				# Verifying Odd Window Size
				perturbation = smooth(perturbation.squeeze(), window_len=self.args.win)
				perturbation = perturbation.reshape(shape)

		# ===========================================================
		# calculate candidate on sphere
		# ===========================================================
		dot = np.vdot(perturbation, source_direction)
		perturbation -= dot * source_direction
		perturbation *= spherical_step * source_norm / norm(perturbation)

		D = 1 / np.sqrt(spherical_step ** 2 + 1)
		direction = perturbation - unnormalized_source_direction
		spherical_candidate = original + D * direction

		# ===========================================================
		# add perturbation in direction of source
		# ===========================================================

		new_source_direction = original - spherical_candidate
		new_source_direction_norm = norm(new_source_direction)

		# length if spherical_candidate would be exactly on the sphere
		length = source_step * source_norm

		# length including correction for deviation from sphere
		deviation = new_source_direction_norm - source_norm
		length += deviation

		# make sure the step size is positive
		length = max(0, length)

		# normalize the length
		length = length / new_source_direction_norm

		candidate = spherical_candidate + length * new_source_direction
		
		
		return (candidate, spherical_candidate)

	def update_step_sizes(self):
		def is_full(deque):
			return len(deque) == deque.maxlen

		if not (
			is_full(self.stats_spherical_adversarial)
			or is_full(self.stats_step_adversarial)
		):
			# updated step size recently, not doing anything now
			return

		def estimate_probability(deque):
			if len(deque) == 0:
				return None
			return np.mean(deque)

		p_spherical = estimate_probability(self.stats_spherical_adversarial)
		p_step = estimate_probability(self.stats_step_adversarial)

		n_spherical = len(self.stats_spherical_adversarial)
		n_step = len(self.stats_step_adversarial)

		def log(message):
			_p_spherical = p_spherical
			if _p_spherical is None:  # pragma: no cover
				_p_spherical = -1.0

			_p_step = p_step
			if _p_step is None:
				_p_step = -1.0

			print(
				"  {} spherical {:.2f} ({:3d}), source {:.2f} ({:2d})".format(
					message, _p_spherical, n_spherical, _p_step, n_step
				)
			)

		if is_full(self.stats_spherical_adversarial):
			# Constrains orthogonal steps based on previous probabilities
			# where the spherical candidate is in the correct target class
			# so it's between .2 and .5 
			if p_spherical > 0.5:
				message = "Boundary too linear, increasing steps:	"
				self.spherical_step *= self.step_adaptation
				self.source_step *= self.step_adaptation
			elif p_spherical < 0.2:
				message = "Boundary too non-linear, decreasing steps:"
				self.spherical_step /= self.step_adaptation
				self.source_step /= self.step_adaptation
			else:
				message = None

			if message is not None:
				self.stats_spherical_adversarial.clear()
				log(message)

		if is_full(self.stats_step_adversarial):
			# Constrains step towards the source after orthogonal step
			# If it's too small, remaining in the targeted class,
			# then we want to be right along the boundary as to lower
			# the rate of sucess of attacks, since they will be more
			# responsive to smaller steps
			if p_step > 0.5:
				message = "Success rate too high, increasing source step:"
				self.source_step *= self.step_adaptation
			elif p_step < 0.2:
				message = "Success rate too low, decreasing source step: "
				self.source_step /= self.step_adaptation
			else:
				message = None

			if message is not None:
				self.stats_step_adversarial.clear()
				log(message)

	def attack(self, target_ecg, orig_ecg, orig, target, max_iterations, max_queries):
		# Added parameter queries to limit query amount, default is infinite
		print("Initial spherical_step = {:.2f}, source_step = {:.2f}".format(
				self.spherical_step, self.source_step
			))
		print("Window size: {}".format(self.args.win))
		m = np.size(orig_ecg)
		n_batches = 25
		perturbed = target_ecg
		original = orig_ecg
		query = 0
		iteration = 1
		query_dist_smooth = np.empty((3,1))
		# Iteration value for row 0
		query_dist_smooth[1][0] = norm(original - perturbed)/m
		# Distance value for row 1
		query_dist_smooth[2][0] = sm(perturbed - original, perturbed.shape[1])
		bool = True
		# Set to false to terminate attack function
		rel_improvement_temp = 1
		# History of adversarial attacks
		hist_adv = np.array(target_ecg).reshape((1,perturbed.shape[1]))
		# Add an iterations variable instead so we can use a while loop
		# Update each time like it normally would at the end of each loop
		# while max_iterations >= iterations && max_queries >= iterations

		while (iteration <= max_iterations) :
						# Only appending every 10th step, not wasting memory
						# on every single step taken, rep. sample
			do_spherical = (iteration % 10 == 0)
			
			unnormalized_source_direction, source_direction, source_norm = self.prepare_generate_candidates(
				original, perturbed)

			distance = source_norm
			for i in range(n_batches):

				# generate candidates
				candidate, spherical_candidate = self.generate_candidate_default(original,
					unnormalized_source_direction, source_direction, source_norm)
				# candidate is the final result after both orthogonal and
				# source steps, while spherical step is just the
				# orthotogonal step wrt to the surface of the sphere
				if do_spherical:
					spherical_is_adversarial = (np.argmax(self.model.predict(spherical_candidate)) == target)
					query += 1
					if (query % 200 == 0):
						temp = np.empty((3,1))
						temp[0][0] = query / 100
						temp[1][0] = norm(original - perturbed)/m
						temp[2][0] = sm(perturbed - original,perturbed.shape[1])
						query_dist_smooth = np.append(query_dist_smooth, temp, axis = 1)
					self.stats_spherical_adversarial.appendleft(
							spherical_is_adversarial)
					is_adversarial = (np.argmax(self.model.predict(candidate)) == target)
					query += 1
					if (query % 200 == 0):
						temp = np.empty((3,1))
						temp[0][0] = query / 100
						temp[1][0] = norm(original - perturbed)/m
						temp[2][0] = sm(perturbed - original,perturbed.shape[1])
						query_dist_smooth = np.append(query_dist_smooth, temp, axis = 1)
					self.stats_step_adversarial.appendleft(is_adversarial)
					# is_adversarial is wrt to the final position
					# and final perturbed image after both steps
					if is_adversarial:
						new_perturbed = candidate
						new_distance = norm(new_perturbed - original)
						break
					# If final step is adversarial, then the for loops breaks
					# Sets the new_perturbed to not None, also updating the distance
				else:
					is_adversarial = (np.argmax(self.model.predict(candidate)) == target)
					query += 1
					if (query % 200 == 0):
						temp = np.empty((3,1))
						temp[0][0] = query / 100
						temp[1][0] = norm(original - perturbed)/186
						temp[2][0] = sm(perturbed - original,perturbed.shape[1])
						query_dist_smooth = np.append(query_dist_smooth, temp, axis = 1)
					if is_adversarial:
						new_perturbed = candidate
						new_distance = norm(new_perturbed - original)
						break
			else:
				new_perturbed = None
				# Determines when to record datapoint for queries vs distance plot 

			message = ""
			# This if statement only runs if out of the X iterations
			# one of them result in an adversarial position wrt to
			# the original image
			# If there is no adversarial perturbed image, then this
			# statement doesn't run
			if new_perturbed is not None:
				abs_improvement = abs(distance - new_distance)
				rel_improvement = abs_improvement / distance
				# Is this relative improvement what we're looking for??
				# How do we decide which percentage is too little change?
				message = "d. reduced by {:.3f}% ({:.4e})".format(
						rel_improvement * 100, abs_improvement
					)
				message += ' queries {}'.format(query)
				# update the variables
				perturbed = new_perturbed
				distance = new_distance

				rel_improvement_temp = rel_improvement

				if iteration % int(max_iterations/100) == 0:
					store_dir = self.args.store_dir
					if not os.path.isdir(store_dir):
						os.makedirs(store_dir, exist_ok=True)
					smooth = sm(perturbed - original,perturbed.shape[1])
					plt.figure()
					plt.ylabel('Normalized Amplitude')
					plt.xlabel('Sample Index')
					plt.title('average L2 distance = {:.4f}, Smoothness = {:.4f} \n Queries {}'.format(distance/m, smooth, query))
					plt.plot(np.arange(len(original.squeeze())), original.squeeze(), 'C1', label='Original ECG')
					plt.plot(np.arange(len(perturbed.squeeze())), perturbed.squeeze(), 'C2', label='Adversarial ECG')
					plt.legend()
					plt.savefig(store_dir+str(iteration)+'.png')
					plt.close()
					hist_adv = np.append(hist_adv, perturbed.reshape((1,perturbed.shape[1])), axis = 0)

				# store adversarial images
				# Potential for iteration to not result in successful adversarial candidate
				# To avoid issues where the final adversarial candidate isn't saved, we allowed for the iterations to be one less
				# than the maximum alloted iterations
				if (iteration == max_iterations) or (iteration == max_iterations - 1) or (query == max_queries):
					print("stored .npy files")
					store_dir = self.args.store_dir
					if not os.path.isdir(store_dir):
						os.makedirs(store_dir, exist_ok=True)
					noise = new_perturbed - original
					np.save(store_dir+"orig.npy", original.astype(np.float16))
					np.save(store_dir+"adv.npy", new_perturbed.astype(np.float16))
					np.save(store_dir+"noise.npy", noise.astype(np.float16))
					np.save(store_dir+"query_dist_smooth.npy", query_dist_smooth.astype(np.float16))
					array = np.array([[iteration],
										[query]])
					np.save(store_dir+"iteration_queries.npy", array.astype(int))
					np.save(store_dir+"hist_adv.npy", hist_adv.astype(np.float16))
					break 

				# Create 1 x 2  array with distance and query number
				# Store each array as vals.npy in store directory

			iteration += 1

				# Add formatting for the different names, also add a different type of save file
				# specifically storing a graph that represents iterations vs distance
				# Do we apply a regression model on the data points, or should we do that after?


			# ===========================================================
			# Update step sizes
			# ===========================================================
			self.update_step_sizes()

			# ===========================================================
			# Log the step
			# ===========================================================
			# self.log_step(iteration, distance, message)

		return perturbed, query 

	# Can be uncommented to print values of step size and source step for each iteration,
	# we removed it as to print less lines while the boundary attack was running

	# def log_step(self, iteration, distance, message="", always=False):
	# 	if not always and iteration % self.log_every_n_steps != 0:
	# 		return
	# 	print(
	# 		"Step {}: {:.5e}, stepsizes = {:.1e}/{:.1e}: {}".format(
	# 			iteration, distance, self.spherical_step, self.source_step, message
	# 		)
	# 	)
