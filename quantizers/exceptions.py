import re
import json
from collections import defaultdict

import torch

from aimet_torch.qc_quantize_op import LearnedGridQuantWrapper, QcQuantizeWrapper
try:
    from aimet_torch.nn.modules.custom import Cast, CustomGather, MatMul, Permute, Reshape, Split
except:
    from aimet_torch.elementwise_ops import Cast, CustomGather, MatMul, Permute, Reshape, Split
import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantizationDataType

nop_exception = {
    "param_exceptions": "None",
    "input_exceptions": "None",
    "output_exceptions": "None"
}

modulename_2_module_map = {
    "LayerNorm": torch.nn.LayerNorm,
    "Embedding": torch.nn.Embedding,
    "MatMul": MatMul,
    "Softmax": torch.nn.Softmax,
    "Sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "CustomGather": CustomGather,
    "Reshape": Reshape,
    "Permute": Permute,
    "Cast": Cast,
    "Split": Split,
}

enc_override_reserved_keywords = ['same_as_param']

class ExceptionConfigurator:
    def __init__(self, args):
        self.exception_types = ("module", "name")
        self.fp16 = args.fp16
        # disable all activation quantizers if do_train==True
        self.disable_act_quantizers = (args.do_train and args.disable_act_quant_on_train) or args.disable_act_quantizers
        self.pre_calib_exceptions = {k: defaultdict(lambda: nop_exception) for k in self.exception_types}
        self.post_calib_exceptions = {k: defaultdict(lambda: nop_exception) for k in self.exception_types}
        if args.exceptions_file is not None:
            with open(args.exceptions_file) as f:
                exception_config = json.load(f)
                
                #Populate op_list here
                for etype in self.exception_types:
                    for item in exception_config[f'{etype}_list']:
                        print(item)
                        if item['exception_stage'] == "pre-calibration":
                            self.pre_calib_exceptions[etype].update({item['module_name']: item['exceptions']})
                        elif item['exception_stage'] == "post-calibration":
                            self.post_calib_exceptions[etype].update({item['module_name']: item['exceptions']})

    def get_pre_calib_exception(self, op_name, etype='module'):
        return self.pre_calib_exceptions[etype][op_name]

    def get_post_calib_exception(self, op_name, etype='module'):
        return self.post_calib_exceptions[etype][op_name]

    def extract_encoding_override(self, encoding_override):
        return int(encoding_override["offset"]), float(encoding_override["min"]), \
            float(encoding_override["max"]), int(encoding_override["bitwidth"]), \
            float(encoding_override["scale"]) if "scale" in encoding_override else None

    def apply_exceptions_to_module(self, exception, name, module):
        print(f'Applying exception to {name} of type: {module}')
        self.apply_param_exception_to_module(exception["param_exceptions"], module)
        self.apply_input_exception_to_module(exception["input_exceptions"], module)
        self.apply_output_exception_to_module(exception["output_exceptions"], module)

    def apply_param_exception_to_module(self, param_exceptions, module):
        #Apply Input exceptions
        if param_exceptions != "None":
            is_enable = (int(param_exceptions["bitwidth"]) < 16) if "bitwidth" in param_exceptions and self.fp16 else True
            if not is_enable:
                param_exceptions["enabled"] = is_enable

            for key in param_exceptions.keys():
                if key=="enabled":
                    module.param_quantizers["weight"].enabled = param_exceptions[key]
                if key=="asymmetric":
                    if param_exceptions[key] == "True":
                        module.param_quantizers["weight"].use_symmetric_encodings = False
                if key=="bitwidth":
                        module.param_quantizers["weight"].bitwidth = int(param_exceptions[key])

    def apply_input_exception_to_module(self, input_exceptions, module):
        if input_exceptions != "None" and not self.disable_act_quantizers:
            assert len(module.input_quantizers) >= len(input_exceptions)
            for index in range(len(input_exceptions)):
                is_enable = True
                if "enabled" in input_exceptions[index]:
                    is_enable = module.input_quantizers[index].enabled = input_exceptions[index]["enabled"]
                    
                for key in input_exceptions[index].keys():
                    if key=="input_index":
                        input_index = int(input_exceptions[index][key])
                        assert index == input_index
                        module.input_quantizers[index].enabled = True and is_enable
                    if key=="asymmetric":
                        if input_exceptions[index][key] == "False":
                            module.input_quantizers[index].use_symmetric_encodings = True
                    if key=="bitwidth":
                        module.input_quantizers[index].bitwidth = int(input_exceptions[index][key])

    def apply_output_exception_to_module(self, output_exceptions, module):
        if output_exceptions != "None" and not self.disable_act_quantizers:
            assert len(module.output_quantizers) >= len(output_exceptions)
            for index in range(len(output_exceptions)):
                for key in output_exceptions[index].keys():
                    if key=="enabled":
                        module.output_quantizers[index].enabled = output_exceptions[index][key]
                    if key=="output_index":
                        output_index = int(output_exceptions[index][key])
                        assert index == output_index
                    if key=="bitwidth":
                        module.output_quantizers[index].bitwidth = int(output_exceptions[index][key])
                    if key=="encoding_overrides":
                        if(isinstance(output_exceptions[index][key], dict)):
                            offset, min, max, bw, scale = self.extract_encoding_override(output_exceptions[index][key])
                            new_enc = libpymo.TfEncoding()
                            new_enc.delta = scale if scale else 1 / (2 ** bw - 1)
                            new_enc.bw = bw
                            new_enc.offset = offset
                            new_enc.min = min
                            new_enc.max = max
                            module.output_quantizers[index].encoding = new_enc
                        elif output_exceptions[index][key] in enc_override_reserved_keywords:
                            if isinstance(module, LearnedGridQuantWrapper):
                                if output_exceptions[index][key] == "same_as_param":
                                    encoding_min = module.weight_encoding_min
                                    encoding_max = module.weight_encoding_max
                                    assert encoding_min is not None
                                    assert encoding_max is not None
                                    setattr(module, f'output{index}_encoding_min', encoding_min)
                                    setattr(module, f'output{index}_encoding_max', encoding_max)
                            else:
                                if output_exceptions[index][key] == "same_as_param":
                                    enc = module.param_quantizers["weight"].encoding
                                    assert enc
                                    module.output_quantizers[index].encoding = enc
                        else:
                            raise ValueError("no valid encodings provided")
                        module.output_quantizers[index]._is_encoding_frozen = True
                    if key=="asymmetric":
                        if output_exceptions[index][key] == "False":
                            module.output_quantizers[index].use_symmetric_encodings = True


    def apply_pre_calibration_exceptions(self, quant_sim):

        for etype in self.exception_types:
            exception_modules = tuple([
                (modulename_2_module_map[key] if etype == 'module' else key)
                for key in self.pre_calib_exceptions[etype].keys()])

            for name, module in quant_sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    exception = None
                    if etype == 'module' and isinstance(module._module_to_wrap, exception_modules):
                        exception = self.get_pre_calib_exception(module._module_to_wrap._get_name(), etype=etype)
                    elif etype == 'name':
                        for key in exception_modules:
                            if name.endswith(key):
                                exception = self.get_pre_calib_exception(key, etype=etype)
                            if key.startswith('reg:') and re.search(key[4:],name):
                                exception = self.get_pre_calib_exception(key, etype=etype)
                    if exception is not None:
                        self.apply_exceptions_to_module(exception, name, module)

    def apply_post_calibration_exceptions(self, quant_sim):
        for etype in self.exception_types:
            exception_modules = tuple([
                (modulename_2_module_map[key] if etype == 'module' else key)
                for key in self.post_calib_exceptions[etype].keys()])

            for name, module in quant_sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    exception = None
                    if etype == 'module' and isinstance(module._module_to_wrap, exception_modules):
                        exception = self.get_post_calib_exception(module._module_to_wrap._get_name(), etype=etype)
                    elif etype == 'name':
                        for key in exception_modules:
                            if name.endswith(key):
                                exception = self.get_post_calib_exception(key, etype=etype)
                    
                    if exception is not None:
                        self.apply_exceptions_to_module(exception, name, module)
        
        if self.disable_act_quantizers:
            print(" ----- Disable all activation quantizers")
            for name, module in quant_sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    for index, input_quantizer in enumerate(module.input_quantizers):
                        # Reasons of disabling all activation quantizers
                        #   1. Do QAT with FP16 as a proxy of QT16 to save memory usage
                        #      - lm_head also requires 16-bit output quantizer, because it is Linear module
                        #   2. 8-bit Embedding layer doesn't have to have additional output quantizer
                        if True: # input_quantizer.bitwidth >= 16:
                            input_quantizer.enabled = False
                        else:
                            print(f"\tInput quantizer:{index} of {module} has bitwidth:{input_quantizer.bitwidth}, keep enabled={input_quantizer.enabled}")
                    for index, output_quantizer in enumerate(module.output_quantizers):
                        if True: # output_quantizer.bitwidth >= 16:
                            output_quantizer.enabled = False
                        else:
                            print(f"\tOutput quantizer:{index} of {module} has bitwidth:{output_quantizer.bitwidth}, keep enabled={output_quantizer.enabled}")    

