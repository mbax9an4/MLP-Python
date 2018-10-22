import numpy

class UtilityFunctions:

	#method to shuffle two arrays together, so indices are coordinated
	@classmethod
	def shuffle_in_unison(self, a, b):
	    assert len(a) == len(b)
	    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
	    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
	    permutation = numpy.random.permutation(len(a))
	    for old_index, new_index in enumerate(permutation):
	        shuffled_a[new_index] = a[old_index]
	        shuffled_b[new_index] = b[old_index]
	    return shuffled_a, shuffled_b

	#method to scale dataset between -1 and 1
	@classmethod
	def scaleData(self, data):
		max = numpy.amax(data) # B 1-b
		min = numpy.amin(data) # A -1 - a
		return -1 + (data - min) * 2 / (max - min)