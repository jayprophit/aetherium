"""
Aetherium Whole Brain Emulation (WBE) System
Complete digital brain emulation with biological neural mapping
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class BiologicalNeuron:
    """Digital representation of biological neuron"""
    id: str
    neuron_type: str  # pyramidal, interneuron, motor, sensory
    location: Tuple[float, float, float]  # 3D brain coordinates
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    refractory_period: float = 2.0  # ms
    last_spike_time: float = 0.0
    dendrites: List[str] = field(default_factory=list)
    axon_terminals: List[str] = field(default_factory=list)
    neurotransmitters: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class Synapse:
    """Digital synapse with biological properties"""
    id: str
    pre_neuron: str
    post_neuron: str
    weight: float
    delay: float  # ms
    neurotransmitter_type: str  # dopamine, serotonin, acetylcholine, etc.
    plasticity_rule: str = "STDP"  # Spike-timing dependent plasticity
    last_update: float = 0.0

class BrainRegion:
    """Emulated brain region with specific functions"""
    
    def __init__(self, region_name: str, coordinates: Tuple[float, float, float], 
                 size: Tuple[float, float, float]):
        self.region_name = region_name
        self.coordinates = coordinates
        self.size = size
        self.neurons: Dict[str, BiologicalNeuron] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.neural_activity = 0.0
        self.specialized_functions = []
        self.logger = logging.getLogger(__name__)
    
    def add_specialized_function(self, function_name: str, description: str):
        """Add specialized brain function to region"""
        self.specialized_functions.append({
            'name': function_name,
            'description': description,
            'activation_level': 0.0
        })
    
    def create_biological_neuron(self, neuron_type: str, location: Tuple[float, float, float]) -> str:
        """Create biologically accurate neuron"""
        neuron_id = f"{self.region_name}_neuron_{len(self.neurons)}"
        
        # Set neuron-type specific properties
        if neuron_type == "pyramidal":
            threshold = -55.0
            neurotransmitters = {"glutamate": 1.0}
        elif neuron_type == "interneuron":
            threshold = -60.0
            neurotransmitters = {"GABA": 1.0}
        elif neuron_type == "motor":
            threshold = -50.0
            neurotransmitters = {"acetylcholine": 1.0}
        elif neuron_type == "sensory":
            threshold = -65.0
            neurotransmitters = {"glutamate": 0.8, "substance_P": 0.2}
        else:
            threshold = -55.0
            neurotransmitters = {"glutamate": 1.0}
        
        neuron = BiologicalNeuron(
            id=neuron_id,
            neuron_type=neuron_type,
            location=location,
            threshold=threshold,
            neurotransmitters=neurotransmitters
        )
        
        self.neurons[neuron_id] = neuron
        return neuron_id
    
    def create_synapse(self, pre_neuron_id: str, post_neuron_id: str, 
                      weight: float, neurotransmitter: str) -> str:
        """Create biologically accurate synapse"""
        synapse_id = f"syn_{pre_neuron_id}_{post_neuron_id}"
        
        # Calculate delay based on distance
        pre_neuron = self.neurons[pre_neuron_id]
        post_neuron = self.neurons[post_neuron_id]
        distance = np.linalg.norm(np.array(pre_neuron.location) - np.array(post_neuron.location))
        delay = max(0.5, distance * 0.1)  # ms based on conduction velocity
        
        synapse = Synapse(
            id=synapse_id,
            pre_neuron=pre_neuron_id,
            post_neuron=post_neuron_id,
            weight=weight,
            delay=delay,
            neurotransmitter_type=neurotransmitter
        )
        
        self.synapses[synapse_id] = synapse
        self.neurons[pre_neuron_id].axon_terminals.append(synapse_id)
        self.neurons[post_neuron_id].dendrites.append(synapse_id)
        
        return synapse_id
    
    def simulate_neural_activity(self, external_input: Dict[str, float], 
                                current_time: float) -> Dict[str, Any]:
        """Simulate biological neural activity"""
        
        firing_neurons = []
        synaptic_activity = {}
        
        # Process each neuron
        for neuron_id, neuron in self.neurons.items():
            # Apply external input
            input_current = external_input.get(neuron_id, 0.0)
            
            # Apply synaptic inputs
            synaptic_input = 0.0
            for dendrite_id in neuron.dendrites:
                if dendrite_id in self.synapses:
                    synapse = self.synapses[dendrite_id]
                    pre_neuron = self.neurons[synapse.pre_neuron]
                    
                    # Check if pre-synaptic neuron fired recently
                    time_since_spike = current_time - pre_neuron.last_spike_time
                    if time_since_spike <= synapse.delay:
                        # Apply neurotransmitter effect
                        nt_effect = self._get_neurotransmitter_effect(
                            synapse.neurotransmitter_type, synapse.weight
                        )
                        synaptic_input += nt_effect
            
            # Update membrane potential
            total_input = input_current + synaptic_input
            neuron.membrane_potential += total_input * 0.1  # Integration
            
            # Check for spike
            if (neuron.membrane_potential >= neuron.threshold and 
                current_time - neuron.last_spike_time > neuron.refractory_period):
                
                firing_neurons.append(neuron_id)
                neuron.last_spike_time = current_time
                neuron.membrane_potential = -80.0  # Reset potential
                
                # Record synaptic activity
                for terminal_id in neuron.axon_terminals:
                    if terminal_id in self.synapses:
                        synaptic_activity[terminal_id] = current_time
        
        # Update region activity level
        self.neural_activity = len(firing_neurons) / max(1, len(self.neurons))
        
        return {
            'firing_neurons': firing_neurons,
            'synaptic_activity': synaptic_activity,
            'region_activity': self.neural_activity,
            'specialized_functions': self._update_specialized_functions()
        }
    
    def _get_neurotransmitter_effect(self, nt_type: str, weight: float) -> float:
        """Calculate neurotransmitter effect on post-synaptic neuron"""
        
        effects = {
            'glutamate': 1.0,      # Excitatory
            'GABA': -0.8,          # Inhibitory
            'dopamine': 1.2,       # Modulatory excitatory
            'serotonin': 0.6,      # Modulatory
            'acetylcholine': 1.1,  # Excitatory
            'norepinephrine': 0.9, # Modulatory
            'substance_P': 0.7     # Neuropeptide
        }
        
        base_effect = effects.get(nt_type, 1.0)
        return base_effect * weight
    
    def _update_specialized_functions(self) -> List[Dict[str, Any]]:
        """Update specialized brain functions based on activity"""
        
        for function in self.specialized_functions:
            # Calculate activation based on neural activity and function type
            if function['name'] in ['memory_formation', 'learning']:
                function['activation_level'] = min(1.0, self.neural_activity * 1.5)
            elif function['name'] in ['motor_control', 'movement']:
                function['activation_level'] = self.neural_activity
            elif function['name'] in ['sensory_processing', 'perception']:
                function['activation_level'] = min(1.0, self.neural_activity * 2.0)
            else:
                function['activation_level'] = self.neural_activity
        
        return self.specialized_functions

class WholeBrainEmulation:
    """Complete whole brain emulation system"""
    
    def __init__(self, subject_id: str = "default"):
        self.subject_id = subject_id
        self.brain_regions: Dict[str, BrainRegion] = {}
        self.inter_region_connections: Dict[str, Dict] = {}
        self.global_neurotransmitters = {
            'dopamine': 1.0,
            'serotonin': 1.0,
            'norepinephrine': 1.0,
            'acetylcholine': 1.0
        }
        self.consciousness_level = 0.0
        self.cognitive_state = "inactive"
        self.memory_systems = {}
        self.current_time = 0.0
        self.logger = logging.getLogger(__name__)
    
    def initialize_standard_brain(self):
        """Initialize standard human brain structure"""
        
        # Create major brain regions
        regions = {
            'frontal_cortex': {
                'coordinates': (0, 50, 0),
                'size': (80, 40, 30),
                'functions': ['executive_control', 'decision_making', 'working_memory']
            },
            'parietal_cortex': {
                'coordinates': (0, 0, 30),
                'size': (60, 40, 25),
                'functions': ['spatial_processing', 'attention']
            },
            'temporal_cortex': {
                'coordinates': (-40, 0, -10),
                'size': (50, 30, 25),
                'functions': ['auditory_processing', 'language']
            },
            'occipital_cortex': {
                'coordinates': (0, -50, 0),
                'size': (40, 30, 25),
                'functions': ['visual_processing']
            },
            'hippocampus': {
                'coordinates': (-25, -10, -15),
                'size': (15, 10, 8),
                'functions': ['memory_formation', 'learning']
            },
            'amygdala': {
                'coordinates': (-20, 5, -20),
                'size': (8, 6, 5),
                'functions': ['emotional_processing', 'fear_response']
            },
            'thalamus': {
                'coordinates': (0, -5, 5),
                'size': (12, 8, 10),
                'functions': ['sensory_relay', 'consciousness']
            },
            'brainstem': {
                'coordinates': (0, -20, -25),
                'size': (8, 15, 12),
                'functions': ['autonomic_control', 'arousal']
            }
        }
        
        # Create regions
        for region_name, props in regions.items():
            region = BrainRegion(region_name, props['coordinates'], props['size'])
            
            # Add specialized functions
            for function in props['functions']:
                region.add_specialized_function(
                    function, 
                    f"{function.replace('_', ' ').title()} processing in {region_name}"
                )
            
            self.brain_regions[region_name] = region
            
            # Populate with neurons
            self._populate_region_with_neurons(region)
        
        # Create inter-region connections
        self._create_inter_region_connections()
        
        # Initialize memory systems
        self._initialize_memory_systems()
        
        self.logger.info(f"Initialized standard brain for subject: {self.subject_id}")
    
    def _populate_region_with_neurons(self, region: BrainRegion):
        """Populate brain region with appropriate neurons"""
        
        # Calculate neuron density based on region type
        region_volume = np.prod(region.size)
        
        if 'cortex' in region.region_name:
            # Cortical regions: high density, mixed neuron types
            neuron_count = int(region_volume * 0.5)  # Neurons per cubic unit
            neuron_types = ['pyramidal'] * 8 + ['interneuron'] * 2  # 80% pyramidal
        elif region.region_name == 'hippocampus':
            neuron_count = int(region_volume * 0.8)
            neuron_types = ['pyramidal'] * 7 + ['interneuron'] * 3
        elif region.region_name in ['thalamus', 'amygdala']:
            neuron_count = int(region_volume * 0.6)
            neuron_types = ['pyramidal'] * 6 + ['interneuron'] * 4
        else:
            neuron_count = int(region_volume * 0.4)
            neuron_types = ['pyramidal'] * 5 + ['interneuron'] * 3 + ['motor'] * 2
        
        # Create neurons
        for i in range(min(neuron_count, 1000)):  # Limit for performance
            neuron_type = np.random.choice(neuron_types)
            
            # Random location within region
            location = (
                np.random.uniform(0, region.size[0]),
                np.random.uniform(0, region.size[1]),
                np.random.uniform(0, region.size[2])
            )
            
            neuron_id = region.create_biological_neuron(neuron_type, location)
            
            # Create local connections
            if i > 0 and np.random.random() < 0.3:  # 30% connection probability
                other_neurons = list(region.neurons.keys())
                if len(other_neurons) > 1:
                    target = np.random.choice([n for n in other_neurons if n != neuron_id])
                    
                    # Determine neurotransmitter based on neuron type
                    source_neuron = region.neurons[neuron_id]
                    if source_neuron.neuron_type == 'interneuron':
                        nt = 'GABA'
                        weight = -0.5
                    else:
                        nt = 'glutamate'
                        weight = 0.8
                    
                    region.create_synapse(neuron_id, target, weight, nt)
    
    def _create_inter_region_connections(self):
        """Create connections between brain regions"""
        
        # Define major anatomical connections
        connections = [
            ('frontal_cortex', 'parietal_cortex', 0.6),
            ('frontal_cortex', 'temporal_cortex', 0.4),
            ('parietal_cortex', 'occipital_cortex', 0.5),
            ('temporal_cortex', 'hippocampus', 0.8),
            ('hippocampus', 'amygdala', 0.5),
            ('thalamus', 'frontal_cortex', 0.7),
            ('thalamus', 'parietal_cortex', 0.7),
            ('thalamus', 'temporal_cortex', 0.6),
            ('thalamus', 'occipital_cortex', 0.8),
            ('brainstem', 'thalamus', 0.9),
            ('amygdala', 'frontal_cortex', 0.3)
        ]
        
        for source, target, strength in connections:
            if source in self.brain_regions and target in self.brain_regions:
                connection_id = f"{source}_to_{target}"
                self.inter_region_connections[connection_id] = {
                    'source': source,
                    'target': target,
                    'strength': strength,
                    'delay': 5.0,  # ms inter-region delay
                    'plasticity': True
                }
    
    def _initialize_memory_systems(self):
        """Initialize different memory systems"""
        
        self.memory_systems = {
            'working_memory': {
                'capacity': 7,  # Miller's magic number
                'contents': [],
                'decay_rate': 0.1
            },
            'short_term_memory': {
                'capacity': 50,
                'contents': [],
                'consolidation_rate': 0.05
            },
            'long_term_memory': {
                'capacity': float('inf'),
                'contents': {},
                'retrieval_strength': {}
            },
            'episodic_memory': {
                'episodes': [],
                'temporal_context': 0.0
            },
            'procedural_memory': {
                'skills': {},
                'habits': {}
            }
        }
    
    def process_cognitive_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cognitive input through emulated brain"""
        
        self.current_time += 1.0  # Advance simulation time
        
        # Prepare regional inputs
        regional_inputs = self._distribute_input_to_regions(input_data)
        
        # Simulate each region
        regional_outputs = {}
        for region_name, region in self.brain_regions.items():
            region_input = regional_inputs.get(region_name, {})
            output = region.simulate_neural_activity(region_input, self.current_time)
            regional_outputs[region_name] = output
        
        # Process inter-region communication
        integrated_output = self._integrate_regional_outputs(regional_outputs)
        
        # Update memory systems
        self._update_memory_systems(input_data, integrated_output)
        
        # Calculate consciousness level
        self._update_consciousness_level(regional_outputs)
        
        # Generate cognitive response
        cognitive_response = self._generate_cognitive_response(integrated_output)
        
        return {
            'cognitive_response': cognitive_response,
            'consciousness_level': self.consciousness_level,
            'cognitive_state': self.cognitive_state,
            'regional_activity': {name: out['region_activity'] 
                                for name, out in regional_outputs.items()},
            'memory_state': self._get_memory_state(),
            'processing_time': self.current_time
        }
    
    def _distribute_input_to_regions(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Distribute input to appropriate brain regions"""
        
        regional_inputs = {region: {} for region in self.brain_regions.keys()}
        
        # Route inputs based on type
        for input_type, value in input_data.items():
            if isinstance(value, (int, float)):
                signal_strength = float(value)
                
                if input_type in ['visual', 'sight']:
                    regional_inputs['occipital_cortex']['visual_input'] = signal_strength
                elif input_type in ['auditory', 'sound']:
                    regional_inputs['temporal_cortex']['auditory_input'] = signal_strength
                elif input_type in ['motor', 'movement']:
                    regional_inputs['frontal_cortex']['motor_input'] = signal_strength
                elif input_type in ['emotional', 'emotion']:
                    regional_inputs['amygdala']['emotional_input'] = signal_strength
                elif input_type in ['memory', 'recall']:
                    regional_inputs['hippocampus']['memory_input'] = signal_strength
                else:
                    # Default to frontal cortex for complex processing
                    regional_inputs['frontal_cortex'][input_type] = signal_strength
        
        return regional_inputs
    
    def _integrate_regional_outputs(self, regional_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate outputs from all brain regions"""
        
        integrated = {
            'total_activity': 0.0,
            'dominant_regions': [],
            'cognitive_functions': {},
            'emotional_state': 0.0,
            'attention_focus': 'none'
        }
        
        # Calculate total activity
        region_activities = []
        for region_name, output in regional_outputs.items():
            activity = output['region_activity']
            region_activities.append((region_name, activity))
            integrated['total_activity'] += activity
        
        # Find dominant regions
        region_activities.sort(key=lambda x: x[1], reverse=True)
        integrated['dominant_regions'] = [r[0] for r in region_activities[:3]]
        
        # Extract cognitive functions
        for region_name, output in regional_outputs.items():
            for func in output['specialized_functions']:
                if func['activation_level'] > 0.5:
                    integrated['cognitive_functions'][func['name']] = func['activation_level']
        
        # Calculate emotional state
        if 'amygdala' in regional_outputs:
            integrated['emotional_state'] = regional_outputs['amygdala']['region_activity']
        
        return integrated
    
    def _update_memory_systems(self, input_data: Dict[str, Any], output: Dict[str, Any]):
        """Update various memory systems"""
        
        # Working memory
        wm = self.memory_systems['working_memory']
        if len(wm['contents']) >= wm['capacity']:
            wm['contents'].pop(0)  # Remove oldest item
        wm['contents'].append({
            'input': input_data,
            'timestamp': self.current_time,
            'activation': output['total_activity']
        })
        
        # Long-term memory consolidation
        if 'hippocampus' in output.get('dominant_regions', []):
            # Strong hippocampal activity triggers memory consolidation
            ltm = self.memory_systems['long_term_memory']
            memory_key = f"memory_{len(ltm['contents'])}"
            ltm['contents'][memory_key] = {
                'data': input_data,
                'context': output,
                'consolidation_strength': 0.8,
                'timestamp': self.current_time
            }
    
    def _update_consciousness_level(self, regional_outputs: Dict[str, Dict[str, Any]]):
        """Update consciousness level based on brain activity"""
        
        # Consciousness factors
        thalamic_activity = regional_outputs.get('thalamus', {}).get('region_activity', 0.0)
        cortical_integration = np.mean([
            regional_outputs.get('frontal_cortex', {}).get('region_activity', 0.0),
            regional_outputs.get('parietal_cortex', {}).get('region_activity', 0.0),
            regional_outputs.get('temporal_cortex', {}).get('region_activity', 0.0)
        ])
        
        # Integrated Information Theory approximation
        self.consciousness_level = min(1.0, (thalamic_activity + cortical_integration) / 2.0)
        
        # Update cognitive state
        if self.consciousness_level > 0.8:
            self.cognitive_state = "highly_conscious"
        elif self.consciousness_level > 0.5:
            self.cognitive_state = "conscious"
        elif self.consciousness_level > 0.2:
            self.cognitive_state = "drowsy"
        else:
            self.cognitive_state = "unconscious"
    
    def _generate_cognitive_response(self, integrated_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cognitive response based on brain processing"""
        
        response = {
            'decision': 'continue_processing',
            'confidence': integrated_output['total_activity'],
            'emotional_response': integrated_output['emotional_state'],
            'attention_allocation': integrated_output['dominant_regions'],
            'memory_formation': 'hippocampus' in integrated_output['dominant_regions'],
            'executive_control': 'frontal_cortex' in integrated_output['dominant_regions']
        }
        
        return response
    
    def _get_memory_state(self) -> Dict[str, Any]:
        """Get current state of memory systems"""
        
        return {
            'working_memory_load': len(self.memory_systems['working_memory']['contents']),
            'long_term_memories': len(self.memory_systems['long_term_memory']['contents']),
            'recent_activity': self.memory_systems['working_memory']['contents'][-3:] if 
                            self.memory_systems['working_memory']['contents'] else []
        }
    
    def get_brain_status(self) -> Dict[str, Any]:
        """Get comprehensive brain emulation status"""
        
        total_neurons = sum(len(region.neurons) for region in self.brain_regions.values())
        total_synapses = sum(len(region.synapses) for region in self.brain_regions.values())
        
        return {
            'subject_id': self.subject_id,
            'brain_regions': len(self.brain_regions),
            'total_neurons': total_neurons,
            'total_synapses': total_synapses,
            'inter_region_connections': len(self.inter_region_connections),
            'consciousness_level': self.consciousness_level,
            'cognitive_state': self.cognitive_state,
            'simulation_time': self.current_time,
            'memory_systems': {name: len(sys.get('contents', [])) if isinstance(sys.get('contents'), list) 
                             else len(sys.get('contents', {})) if isinstance(sys.get('contents'), dict) 
                             else 0 for name, sys in self.memory_systems.items()},
            'global_neurotransmitters': self.global_neurotransmitters,
            'emulation_status': 'active'
        }

# Example usage and demonstration
async def demo_whole_brain_emulation():
    """Demonstrate whole brain emulation capabilities"""
    
    print("🧠 Whole Brain Emulation Demo")
    
    # Create WBE system
    wbe = WholeBrainEmulation("demo_subject")
    
    # Initialize standard human brain
    wbe.initialize_standard_brain()
    
    # Test cognitive processing
    test_inputs = [
        {
            'visual': 0.8,
            'auditory': 0.3,
            'emotional': 0.6,
            'task_complexity': 0.7
        },
        {
            'memory': 0.9,
            'motor': 0.4,
            'attention': 0.8
        },
        {
            'visual': 0.2,
            'emotional': 0.9,
            'memory': 0.6
        }
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n   Test {i+1}: {test_input}")
        
        result = wbe.process_cognitive_input(test_input)
        
        print(f"   Consciousness: {result['consciousness_level']:.2f}")
        print(f"   State: {result['cognitive_state']}")
        print(f"   Dominant regions: {result['regional_activity']}")
        print(f"   Memory formation: {result['cognitive_response']['memory_formation']}")
    
    # Show final brain status
    status = wbe.get_brain_status()
    print(f"\n   Brain regions: {status['brain_regions']}")
    print(f"   Total neurons: {status['total_neurons']}")
    print(f"   Total synapses: {status['total_synapses']}")
    print(f"   Final consciousness: {status['consciousness_level']:.2f}")
    
    print("✅ Whole Brain Emulation system operational")

if __name__ == "__main__":
    asyncio.run(demo_whole_brain_emulation())