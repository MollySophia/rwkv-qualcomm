import os,sys
import torch
from transformers.utils.fx import symbolic_trace
from typing import Callable
# from nectar.model_wrappers import HFModelWrapper
from aimet_torch.model_preparer import _prepare_traced_model
from aimet_torch.model_validator.model_validator import ModelValidator
# from utils.model_utils import input_keys_for_causal_llm, get_input_output_names

# class Select(torch.nn.Module):
#     """ Custom module for a functional bmm"""

#     @staticmethod
#     def forward(*args, **kwargs) -> torch.Tensor:
#         """Forward-pass routine for torch.bmm op"""
#         return torch.select(*args, **kwargs)

# class Bmm(torch.nn.Module):
#     """ Custom module for a functional bmm"""

#     @staticmethod
#     def forward(*args, **kwargs) -> torch.Tensor:
#         """Forward-pass routine for torch.bmm op"""
#         return torch.bmm(*args, **kwargs)

# class Baddbmm(torch.nn.Module):
#     """Custom module for a functional baddbmm"""
#     @staticmethod
#     def forward(*args, **kwargs) -> torch.Tensor:
#         """Forward-pass routine for torch.baddbmm"""
#         tensor, batch1, batch2, beta, alpha = args

#         return tensor.baddbmm(batch1, batch2, beta=beta, alpha=alpha)

class ModelPreparer:
    @staticmethod
    def prepare_hf_model(orig_model, dummy_input, validator: Callable[[], bool]):
        model = orig_model
        #######################################################################################
        #These can go away once this lands: https://jira-dc.qualcomm.com/jira/browse/AIMET-2528
        # from aimet_torch import model_preparer
        # model_preparer.functional_with_stateless_api.update({'bmm': Bmm})
        # model_preparer.functional_with_stateless_api.update({'baddbmm': Baddbmm})
        # model_preparer.functional_with_stateless_api.update({'select': Select})
        #######################################################################################
        #Ensure that the keys in dummy input are in the list of inputs we are tracing the model with
        # for key in dummy_input.keys():
        #     assert key in input_keys_for_causal_llm

        # Trace the model
        traced = symbolic_trace(model, list(dummy_input.keys()))
        print("Traced model:")
        traced.print_readable()

        # Prepare the model
        _prepare_traced_model(traced)

        print("Prepared model:")
        traced.print_readable()

        #print("Tabular form of the prepared graph")
        traced.graph.print_tabular()

        model_inputs_tuple = tuple([dummy_input[input] for input in dummy_input.keys()])

        ModelValidator.validate_model(traced, model_inputs_tuple)

        if validator:
            ModelPreparer.__compare_model_outputs(model, traced, model_inputs_tuple, dummy_input, validator)

        print('Model validation complete')

        return traced
    
    @staticmethod
    def set_model_preparer_dependencies(sdk_version_args):
        sdk_version = os.environ.get('LATEST_QNN_SDK', sdk_version_args)
        sdk_root =  os.environ.get('QNN_SDK_ROOT',None)
        if sdk_root is None:
            raise Exception("Please set up QNN_SDK_ROOT environment variable to where you have QNN SDKs as in the README.md")
        sdk_path = os.path.join(sdk_root, sdk_version)
        if os.path.exists(sdk_path):
            sys.path.append(os.path.join(sdk_path, 'target/x86_64-linux-clang/python/'))
        else:
            raise FileNotFoundError('QNN SDK path was not found for the versions mentioned.')

    # @staticmethod
    # def prepare_hf_model_pro(model, dummy_input, sdk_version, output_dir, export_mode):
    #     # Python path must be updated before importing 
    #     ModelPreparer.set_model_preparer_dependencies(sdk_version)
    #     from aimet_torch.pro.model_preparer import prepare_model as prepare_model_pro
    #     input_names, output_names = get_input_output_names(export_mode, model.config)
    #     model_inputs_tuple = tuple([dummy_input[input] for input in dummy_input.keys()])
    #     prepared_model = prepare_model_pro(model, model_inputs_tuple, path=output_dir, input_names=input_names, output_names=output_names)
    #     prepared_model.device = getattr(model, 'device',  next(model.parameters()).device)
    #     return prepared_model

    @staticmethod
    def __compare_model_outputs(orig_model, prepared_model, dummy_inputs_tuple, dummy_input_dict, validator: Callable[[], bool]):
        with torch.no_grad():
            # Run the dummy input through the model and get the output from the FP model
            fp_dummy_out = orig_model(**dummy_input_dict)

            # Run the dummy input through the model and get the output from the prepared model
            prepared_dummy_out = prepared_model(*dummy_inputs_tuple)

        assert validator(fp_dummy_out, prepared_dummy_out)
