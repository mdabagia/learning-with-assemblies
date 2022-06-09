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
    n_inputs : int
        number of inputs
        
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
        
    Methods
    -------
    project(input=None, update=True)
        Takes input and updates activations to next time step
        
    step(update=True)
        Updates activations to next time step without input
        
    read()
        Returns current activations
        
    reset()
        Sets current activations to zero
        
    normalize()
        Normalizes current incoming weights of each neuron to unit sum
    '''
    
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity=1e-1, normalize=False):
        '''
        Parameters
        ----------
        n_inputs : int
            number of inputs

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
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.cap_size = cap_size
        self.density = density
        self.plasticity = plasticity
        self.input_weights = (rng.random((n_inputs, n_neurons)) < density) * 1.
        self.recurrent_weights = (rng.random((n_neurons, n_neurons)) < density) * 1.
        if normalize:
            self.normalize()
        self.activations = np.zeros(n_neurons)
        
    def project(self, input=None, update=True):
        '''Takes input and updates activations to next time step.
        
        If no input is passed in, zero input is assumed.
        
        Parameters
        ----------
        input : ndarray of shape (n_neurons), optional
            The input to the brain area on the current time step (default is all zeros)
            
        update : bool, optional
            Whether to update the weights based on the resulting activations (default is True)
        '''
        if input is None:
            input = np.zeros(self.n_inputs)
        new_activations = k_cap(input @ self.input_weights + self.activations @ self.recurrent_weights, self.cap_size)
        
        self.input_weights[(input[:, np.newaxis] > 0) & (new_activations[np.newaxis] > 0)] *= 1 + self.plasticity
        self.recurrent_weights[(self.activations[:, np.newaxis] > 0) & (new_activations[np.newaxis] > 0)] *= 1 + self.plasticity
        self.activations = new_activations
        
    def step(self, update=True):
        '''Updates activations to next time step without input.
        
        Parameters
        ----------
        update : bool, optional
            Whether to update the weights based on the resulting activations (default is True)
        '''
        self.project(input=None, update=update)
        
    def read(self):
        '''Returns current activations.
        '''
        return self.activations
    
    def reset(self):
        '''Sets current activations to zero.
        '''
        self.activations = np.zeros(self.n_neurons)
    
    def normalize(self):
        '''Normalizes current incoming weights of each neuron to unit sum.
        '''
        norm = self.input_weights.sum(axis=0) + self.recurrent_weights.sum(axis=0)
        self.input_weights /= norm[np.newaxis, :]
        self.recurrent_weights /= norm[np.newaxis, :]
