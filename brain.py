import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def k_cap(input, cap_size):
    output = np.zeros_like(input)
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
        
    normalize : bool
        whether to initialize weights as normalized
        
    n_input_areas : int
        number of input areas
        
    input_weights : array of shape (n_input_areas, n_inputs, n_neurons)
    	weights between brain area and input areas
    	
    recurrent_weights : array of shape (n_neurons, n_neurons)
    	weights between brain area neurons
    	
    inputs : array of shape (n_inputs)
    	current inputs to the brain area
    	
    neurons : array of shape (n_neurons)
    	currently firing neurons in the brain area
        
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
    
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity=1e-1, normalize=False):
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

        normalize : bool, optional
            whether to initialize weights as normalized (default is False)
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
        if len(inputs) != self.n_inputs[input_area]:
            raise Exception('Mismatched dimensions in input and specified area {} ({} not equal to {})'.format(input_area, len(inputs), self.n_inputs[input_area]))
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
        new_activations = k_cap(np.sum([self.inputs[i] @ self.input_weights[i] for i in range(self.n_input_areas)], axis=0) + self.activations @ self.recurrent_weights, self.cap_size)
        
        if update:
            for w, inp in zip(self.input_weights, self.inputs):
                w[(inp[:, np.newaxis] > 0) & (new_activations[np.newaxis] > 0)] *= 1 + self.plasticity
            self.recurrent_weights[(self.activations[:, np.newaxis] > 0) & (new_activations[np.newaxis] > 0)] *= 1 + self.plasticity

        self.activations = new_activations
        self.inputs = [np.zeros(n) for n in self.n_inputs]
        
    def read(self):
        '''Returns current activations.
        '''
        return self.activations.copy()
    
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
            self.recurrent_weights[(self.activations[:, np.newaxis] > 0) & (activations[np.newaxis] > 0)] *= 1 + self.plasticity
        self.activations = activations
        self.inputs = [np.zeros(n) for n in self.n_inputs]
        

    def inhibit(self):
        '''Sets current activations and inputs to zero.
        '''
        self.inputs = [np.zeros(n) for n in self.n_inputs]
        self.activations = np.zeros(self.n_neurons)
    
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
        if self.normalize:
            self.normalize()
