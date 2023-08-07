import numpy as np
rng = np.random.default_rng()

def k_cap(input, cap_size):
    if np.all(input == 0):
        return []
    else:
        return input.argsort(axis=-1)[...,-cap_size:]

def idx_to_vec(idx, shape):
    vec = np.zeros(idx.shape[:-1] + (shape,))
    np.put_along_axis(vec, idx, 1, axis=-1)
    return vec

class FFArea():
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity, norm_init=False):
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
        self.reset()
        
    def reset(self):
        self.input_weights = [(rng.random((n, self.n_neurons)) < self.density) * 1. for n in self.n_inputs]
        self.inhibit()
        if self.norm_init:
            self.normalize()
        
    def inhibit(self):
        self.clear_input()
        self.activations = []
        
    def fire(self, activations, update=True):
        if update:
            self.update(activations)
            
        self.activations = activations
        self.clear_input()
        
    def set_input(self, inputs, input_area=0):
        if isinstance(input_area, int):
            if len(inputs) == self.n_input_areas:
                self.inputs = inputs
            else:
                self.inputs[input_area] = inputs
        else:                    
            for i in input_area:
                self.inputs[i] = inputs[i]
                
    def clear_input(self, input_area=-1):
        if isinstance(input_area, int):
            if input_area < 0:
                self.inputs = [[] for _ in self.n_inputs]
            else:
                self.inputs[input_area] = []
        else:
            for i in input_area:
                self.inputs[i] = []
                
    def get_total_input(self):
        return np.sum([w[inp].sum(axis=0) for w, inp in zip(self.input_weights, self.inputs)], axis=0)
                
    def step(self, update=True):
        new_activations = k_cap(self.get_total_input(), cap_size=self.cap_size)
        if update:
            self.update(new_activations)
            
        self.activations = new_activations
        self.clear_input()
        
    def forward(self, inputs, input_area=0, update=True):
        self.set_input(inputs, input_area=input_area)
        self.step(update=update)
        self.clear_input()
        
                
    def update(self, new_activations):
        for w, inp in zip(self.input_weights, self.inputs):
            w[np.ix_(inp, new_activations)] *= 1 + self.plasticity
    
    def normalize(self):
        for w, inp in zip(self.input_weights, self.inputs):
            w /= w.sum(axis=0, keepdims=True)
        
    def read(self, dense=False):
        if dense:
            return idx_to_vec(self.activations, self.n_neurons)
        else:
            return self.activations
        
class RecurrentArea(FFArea):
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity, norm_init=False):
        super().__init__(n_inputs, n_neurons, cap_size, density, plasticity, norm_init)
        
    def reset(self):
        self.recurrent_weights = (rng.random((self.n_neurons, self.n_neurons)) < self.density) * 1.
        super().reset()                
        
    def update(self, new_activations):
        super().update(new_activations)
        self.recurrent_weights[np.ix_(self.activations, new_activations)] *= 1 + self.plasticity
        
    def get_total_input(self):
        return super().get_total_input() + self.recurrent_weights[self.activations].sum(axis=0)
        
    def normalize(self):
        super().normalize()
        self.recurrent_weights /= self.recurrent_weights.sum(axis=0, keepdims=True)
        
class RefractedArea(FFArea):
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity, norm_init=False):
        super().__init__(n_inputs, n_neurons, cap_size, density, plasticity, norm_init)
        
    def reset(self):
        self.bias = np.zeros(self.n_neurons)
        super().reset()
        
    def update(self, new_activations):
        self.bias[new_activations] += super().get_total_input()[new_activations] * self.plasticity
        super().update(new_activations)
        
    def get_total_input(self):
        return super().get_total_input() - self.bias
        
class RandomChoiceArea(RecurrentArea):
    def __init__(self, n_neurons, cap_size, density, plasticity, norm_init=False):
        super().__init__([], n_neurons, cap_size, density, 3., norm_init)
        
    def train(self, assemblies):
        for assm in assemblies:
            self.inhibit()
            self.fire(assm)
            self.fire(assm)
        self.inhibit()
        
    def flip(self, n_rounds=10):
        self.inhibit()
        self.fire(rng.choice(self.n_neurons, size=self.cap_size, replace=False), update=False)
        for _ in range(n_rounds):
            self.step(update=False)
        choice = self.read()
        self.inhibit()
        return choice
        
class ScaffoldNetwork():
    def __init__(self, n_inputs, n_neurons, cap_size, density, plasticity, norm_init=False):
        self.areas = [RecurrentArea([n_inputs, n_neurons], n_neurons, cap_size, density, plasticity, norm_init=norm_init),
                      RecurrentArea(n_neurons, n_neurons, cap_size, density, plasticity, norm_init=norm_init)]
        
    def reset(self):
        for area in self.areas:
            area.reset()
            
    def inhibit(self):
        for area in self.areas:
            area.inhibit()
            
    def set_input(self, inputs):
        self.areas[0].set_input(inputs)
        
    def step(self, update=True):
        self.areas[0].set_input(self.areas[1].read(), input_area=1)
        self.areas[1].set_input(self.areas[0].read())
        for area in self.areas:
            area.step(update=update)
            
    def forward(self, inputs, update=True):
        self.set_input(inputs)
        self.step(update=update)
    
    def normalize(self):
        for area in self.areas:
            area.normalize()
            
    def read(self, dense=False):
        return self.areas[0].read(dense=dense)
            
class FSMNetwork():
    def __init__(self, n_symbol_neurons, n_state_neurons, n_arc_neurons, cap_size, density, plasticity, norm_init=False):
        self.state_area = FFArea(n_arc_neurons, n_state_neurons, cap_size, density, plasticity, norm_init=norm_init)
        self.arc_area = RefractedArea([n_symbol_neurons, n_state_neurons], n_arc_neurons, cap_size, density, plasticity, norm_init=norm_init)
        
    def reset(self):
        self.state_area.reset()
        self.arc_area.reset()
        
    def inhibit(self):
        self.state_area.inhibit()
        self.arc_area.inhibit()
        
    def forward(self, inputs, update=True):
        self.arc_area.forward([inputs, self.state_area.read()], update=update)
        self.state_area.forward(self.arc_area.read(), update=update)
        
    def read(self, dense=False):
        return self.state_area.read(dense=dense)
        
    def train(self, symbol, state, new_state):
        self.inhibit()
        self.arc_area.forward([symbol, state])
        self.state_area.set_input(self.arc_area.read())
        self.state_area.fire(new_state)
        
class PFANetwork():
    def __init__(self, n_symbol_neurons, n_state_neurons, n_arc_neurons, n_random_neurons, cap_size, density, plasticity, norm_init=False):
        self.symbol_area = FFArea(n_arc_neurons, n_symbol_neurons, cap_size, density, plasticity, norm_init=norm_init)
        self.state_area = FFArea(n_arc_neurons, n_state_neurons, cap_size, density, plasticity, norm_init=norm_init)
        self.arc_area = RefractedArea([n_state_neurons, n_random_neurons], n_arc_neurons, cap_size, density, plasticity, norm_init=norm_init)
        self.random_area = RandomChoiceArea(n_random_neurons, cap_size, density, plasticity)
        self.cap_size = cap_size
        self.random_area.train([np.arange(cap_size), np.arange(cap_size, 2 * cap_size)])
        
    def train(self, state, rand, new_state, symbol):
        self.arc_area.forward([state, np.arange(rand*self.cap_size, (rand+1)*self.cap_size)])
        self.symbol_area.set_input(self.arc_area.read())
        self.symbol_area.fire(symbol)
        self.state_area.set_input(self.arc_area.read(), input_area=0)
        self.state_area.fire(new_state)
        
    def step(self):
        self.arc_area.forward([self.state_area.read(), self.random_area.flip()], update=False)
        self.state_area.forward(self.arc_area.read(), update=False)
        self.symbol_area.forward(self.arc_area.read(), update=False)
        
    def read(self, dense=False):
        return self.symbol_area.read(dense=dense)