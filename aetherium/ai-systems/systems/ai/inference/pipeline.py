"""
Inference Pipeline

This module provides a flexible pipeline for running inference with preprocessing and postprocessing.
"""
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class PipelineStep:
    """A single step in the inference pipeline."""
    name: str
    function: Callable
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None

class InferencePipeline:
    """A flexible pipeline for running inference with preprocessing and postprocessing."""
    
    def __init__(self, steps: Optional[List[PipelineStep]] = None):
        """Initialize the pipeline.
        
        Args:
            steps: List of pipeline steps to execute in order
        """
        self.steps = steps or []
        self.state = {}
    
    def add_step(
        self,
        name: str,
        function: Callable,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        **kwargs
    ) -> 'InferencePipeline':
        """Add a step to the pipeline.
        
        Args:
            name: Name of the step (must be unique)
            function: Function to execute for this step
            input_key: Key of the input in the state dictionary (default: previous output)
            output_key: Key to store the output under in the state dictionary
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            self (for method chaining)
        """
        step = PipelineStep(
            name=name,
            function=function,
            input_key=input_key,
            output_key=output_key or name,
            kwargs=kwargs or {}
        )
        self.steps.append(step)
        return self
    
    def run(self, initial_input: Any = None, **initial_state) -> Dict[str, Any]:
        """Run the pipeline.
        
        Args:
            initial_input: Initial input to the pipeline
            **initial_state: Additional initial state values
            
        Returns:
            Dictionary containing the final state after all steps
        """
        # Initialize state
        self.state = {
            'input': initial_input,
            'output': None,
            **initial_state
        }
        
        # Execute each step
        for step in tqdm(self.steps, desc="Running pipeline"):
            try:
                # Get input for this step
                if step.input_key is not None:
                    inputs = self.state[step.input_key]
                else:
                    # Default to previous output or initial input
                    inputs = self.state.get('output', initial_input)
                
                # Call the step function
                if isinstance(inputs, (list, tuple)):
                    result = step.function(*inputs, **step.kwargs)
                elif isinstance(inputs, dict):
                    result = step.function(**inputs, **step.kwargs)
                else:
                    result = step.function(inputs, **step.kwargs)
                
                # Store the result
                self.state[step.output_key] = result
                self.state['output'] = result
                
            except Exception as e:
                raise RuntimeError(f"Error in pipeline step '{step.name}': {str(e)}") from e
        
        return self.state
    
    def __call__(self, *args, **kwargs) -> Any:
        """Run the pipeline (same as run method)."""
        return self.run(*args, **kwargs)
    
    def clear(self) -> None:
        """Clear the pipeline state."""
        self.state = {}
    
    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name.
        
        Args:
            name: Name of the step to get
            
        Returns:
            The PipelineStep with the given name, or None if not found
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def remove_step(self, name: str) -> bool:
        """Remove a step by name.
        
        Args:
            name: Name of the step to remove
            
        Returns:
            True if the step was found and removed, False otherwise
        """
        for i, step in enumerate(self.steps):
            if step.name == name:
                self.steps.pop(i)
                return True
        return False
