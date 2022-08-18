import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def k_cap(input, cap_size):
    output = np.zeros_like(input)
    if np.any(input != output):
        if len(input.shape) == 1:
            idx = np.argsort(input)[-cap_size:]
            output[idx] = 1
        else:
            idx = np.argsort(input, axis=-1)[..., -cap_size:]
            np.put_along_axis(output, idx, 1, axis=-1)
    return output

class BrainArea():
    '''A class representing a single brain area and its input.
    
    
    Attributes
    ----------    
    n_inputs : array of shape (n_input_areas)
        number of inputs (per input area)
        
    n_neurons : int
        number of neurons
        
    cap_size : int
        number of neurons allowed to fire simultaneously
        
    density : float
        probability of random connection between neurons
        
    plasticity : float
        strength of the Hebbian update
        
    norm_init : bool
        whether to initialize weights as normalized
        
    n_input_areas : int
        number of input areas
        
    input_weights : array of shape (n_input_areas, n_inputs, n_neurons)
    	weights between brain area and input areas
    	
    recurrent_weights : array of shape (n_neurons, n_neurons)
    	weights between brain area neurons
    	
    inputs : array of shape (n_inputs)
    	current inputs to the brain area
    	
    activations : array of shape (ref_period, n_neurons)
    	firing patterns in the brain area over the last ref_period rounds, with activations[0] the most recent
        
    Methods
    -------
    set_input(inputs, input_area=0)
        Loads external input to the brain area
        
    step(update=True)
        Updates activations to next time step
        
    read()
        Returns current activations
        
    inhibit()
        Sets current activations to zero
        
    fire(activations)
        Manually fires neurons to match the given pattern
        
        
    normalize()
        Normalizes current incoming weights of each neuron to unit sum
        
    reset()
        Resets weights, activations, and inputs
    '''
    
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity=1e-1, norm_init=False, refraction=False, ref_period=2, ref_penalty=1e2):
        '''
        Parameters
        ----------
        n_inputs : int or array
            number of inputs per area

        n_neurons : int
            number of neurons

        cap_size : int
            number of neurons allowed to fire simultaneously

        density : float
            probability of random connection between neurons

        plasticity : float, optional
            strength of the Hebbian update (default is 0.1)

        norm_init : bool, optional
            whether to initialize weights as normalized (default is False)
            
        refraction : bool, optional
            whether to penalize neurons for firing too many times consecutively (default is False)
            
        ref_period : int, optional
            the number of rounds a neuron may fire consecutively before being penalized (default is 2)
            
        ref_penalty : float, optional
            the refractory penalty (default is 100)
        '''
        if isinstance(n_inputs, int):
            self.n_input_areas = 1
            n_inputs = [n_inputs]
        else:
            self.n_input_areas = len(n_inputs)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.cap_size = cap_size
        self.density = density
        self.plasticity = plasticity
        self.norm_init = norm_init
        self.refraction = refraction
        self.ref_period = ref_period
        self.penalty = ref_penalty
        self.bias = np.zeros(n_neurons)
        self.reset()
        
    def set_input(self, inputs, input_area=0):
        '''Loads input to the brain area.
        
        Parameters
        ----------
        inputs : array of shape (n_inputs[input_area]) or array of arrays
            The input to the brain area on the current time step
            
        input_area : int or array, optional
            The areas to input (default is zero)
        '''
        if isinstance(input_area, int):
            self.inputs[input_area] = inputs
        else:
            for i in input_area:
                self.inputs[i] = inputs[i]
        
    def step(self, update=True):
        '''Updates activations to next time step.
        
        Also resets current inputs to zero after applying it.
        
        Parameters
        ----------
        update : bool, optional
            Whether to update the weights based on the resulting activations (default is True)
        '''
        if self.refraction:
            self.bias -= self.penalty * np.all(self.activations[:-1] > 0, axis=0)

        self.activations[1:] = self.activations[:-1]
        self.activations[0] = k_cap(np.sum([self.weight_factor[i] * self.inputs[i] @ self.input_weights[i] for i in range(self.n_input_areas)], axis=0) + self.activations[1] @ self.recurrent_weights + self.bias, self.cap_size)

        if update:
            for w, inp in zip(self.input_weights, self.inputs):
                w[(inp[:, np.newaxis] > 0) & (self.activations[0][np.newaxis] > 0)] *= 1 + self.plasticity
            self.recurrent_weights[(self.activations[1][:, np.newaxis] > 0) & (self.activations[0][np.newaxis] > 0)] *= 1 + self.plasticity
        self.inputs = [np.zeros(n) for n in self.n_inputs]
        
    def read(self):
        '''Returns current activations.
        '''
        return self.activations[0].copy()
    
    def fire(self, activations, update=True):
        '''Manually fires neurons to match the given pattern.
        
        Parameters
        ----------
        activations : array of shape (n_neurons)
            The firing pattern to match
        
        update : bool, optional
            Whether to update the weights based on the resulting activations (default is True)
        '''
        if update:
            for w, inp in zip(self.input_weights, self.inputs):
                w[(inp[:, np.newaxis] > 0) & (activations[np.newaxis] > 0)] *= 1 + self.plasticity
            self.recurrent_weights[(self.activations[0][:, np.newaxis] > 0) & (activations[np.newaxis] > 0)] *= 1 + self.plasticity
        self.activations[1:] = self.activations[:-1]
        self.activations[0] = activations
        self.inputs = [np.zeros(n) for n in self.n_inputs]
        

    def inhibit(self):
        '''Sets current activations and inputs to zero.
        '''
        self.inputs = [np.zeros(n) for n in self.n_inputs]
        self.activations = np.zeros((self.ref_period+1, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)
    
    def normalize(self):
        '''Normalizes current incoming weights of each neuron to unit sum.
        '''
        norm = np.sum([w.sum(axis=0) for w in self.input_weights], axis=0) + self.recurrent_weights.sum(axis=0)
        for w in self.input_weights:
            w /= norm[np.newaxis, :]
        self.recurrent_weights /= norm[np.newaxis, :]
        
    def reset(self):
        '''Resets weights, activations, and inputs.
        '''
        self.input_weights = [(rng.random((n, self.n_neurons)) < self.density) * 1. for n in self.n_inputs]
        self.recurrent_weights = (rng.random((self.n_neurons, self.n_neurons)) < self.density) * 1.
        self.inhibit()
        self.weight_input(1.)
        if self.norm_init:
            self.normalize()
            
    def project(self, inputs, n_rounds=5, input_area=0, update=True):
        if len(inputs) > 1:
            n_rounds = len(inputs)
        
        for i in range(n_rounds):
            self.set_input(inputs[0 if len(inputs) == 1 else i], input_area=input_area)
            self.step(update=update)
            
    def weight_input(self, factor, input_area=None):
        if input_area is None:
            self.weight_factor = [factor for _ in self.n_inputs]
        else:
            self.weight_factor[input_area] = factor
                        
class ScaffoldNetwork():
    def __init__(self, n_areas, n_inputs, n_neurons, cap_size, density, plasticity=1e-1, norm_init=False):
        self.n_areas = n_areas
        self.areas = [BrainArea(n_inputs + [n_neurons], n_neurons, cap_size, density, plasticity=plasticity, norm_init=norm_init) for _ in range(n_areas)]
        
    def set_input(self, inputs, input_area=0):
        for area in self.areas:
            area.set_input(inputs, input_area=input_area)
                    
    def step(self, update=True):
        for i in range(self.n_areas):
            self.areas[i].set_input(self.areas[i-1].read(), input_area=-1)
        for area in self.areas:
            area.step(update=update)
            
    def fire(self, activations, areas=None, update=True):
        if areas is None:
            areas = [i for i in range(self.n_areas)]
        elif isinstance(areas, int):
            areas = [areas]
            activations = [activations]

        for i in range(self.n_areas):
            self.areas[i].set_input(self.areas[i-1].read(), input_area=-1)
        
        for i, act in zip(areas, activations):
            self.areas[i].fire(act, update=update)    
            
        for i in range(self.n_areas):
            if i not in areas:
                self.areas[i].step(update=update)

    def read(self):
        return [area.read() for area in self.areas]
                    
    def inhibit(self):
        for area in self.areas:
            area.inhibit()
    
    def reset(self):
        for area in self.areas:
            area.reset()       
             
    def normalize(self):
        for area in self.areas:
            area.normalize()
            
    def weight_interarea(self, factor):
        self.weight_input(factor, input_area=-1)

    def weight_input(self, factor, input_area=None):
        for area in self.areas:
            area.weight_input(factor, input_area=input_area)

