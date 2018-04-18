import torch

def apply_advanced(parent_module, function):
    """Applies ``function`` recursively to every submodule (as returned by ``.children()``).
       Similar to module.apply() but also returns the parent object of every module being
       traversed.
       
       Parameters
       ----------
       parent_module : nn.Module
           Module object representing the root of the computation graph.
           
       function : function closure
           Function with signature (child_module (nn.Module), child_name (str), parent_module (nn.Module) 

    """
        
    for child_name, child_module in parent_module.named_children():
    
        function(child_module, child_name, parent_module)    
        
        apply_advanced(child_module, function)