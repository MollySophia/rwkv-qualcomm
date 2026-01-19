from utils.htp_devices_config import htp_devices, dump_htp_config, dump_htp_link_config
import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Make context cache from model libs')
    parser.add_argument('model_dlc', type=Path, help='Path to RWKV pth files')
    parser.add_argument('output_path', type=Path, help='Path to output folder')
    parser.add_argument('platform', type=str, choices=htp_devices.keys(), help='Platform name')
    parser.add_argument('--use_optrace', action='store_true', help='Use optrace profiling')
    parser.add_argument('--wkv_customop', action='store_true', help='Use wkv custom op')
    parser.add_argument('--output_name', type=str, default=None, help='Output name for the binary file')
    args = parser.parse_args()
    qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
    if not qnn_sdk_root:
        print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
        exit(1)

    model_paths = [str(args.model_dlc)] if ',' not in str(args.model_dlc) else str(args.model_dlc).split(',')
    model_names = [str(path).split('/')[-1].replace('.dlc', '') for path in model_paths]
    print(f"Processing model {model_names}")
    dump_htp_config(args.platform, model_names, model_paths[0].replace('.dlc', f'_{args.platform}_htp_config.json'))
    dump_htp_link_config(model_paths[0].replace('.dlc', f'_{args.platform}_htp_link.json'), qnn_sdk_root)
    convert_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-context-binary-generator"
    convert_cmd += f" --model {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnModelDlc.so"
    convert_cmd += f" --backend {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnHtp.so"
    convert_cmd += f" --dlc_path {','.join(model_paths)}"
    convert_cmd += f" --output_dir {args.output_path}"
    output_name = model_names[0].replace('lib', '') if args.output_name is None else args.output_name
    convert_cmd += f" --binary_file {output_name}"
    convert_cmd += f" --config_file {model_paths[0].replace('.dlc', f'_{args.platform}_htp_link.json')}"
    if args.use_optrace:
        convert_cmd += " --profiling_level detailed --profiling_option optrace"

    if args.wkv_customop:
        convert_cmd += " --op_packages hexagon/HTP/RwkvWkvOpPackage/build/x86_64-linux-clang/libQnnRwkvWkvOpPackage.so:RwkvWkvOpPackageInterfaceProvider"

    convert_cmd += " --input_output_tensor_mem_type memhandle"
    # convert_cmd += " --enable_intermediate_outputs"
    result = os.system(convert_cmd)
    exit(result)

if __name__ == '__main__':
    main()
