from utils.htp_devices_config import htp_devices, dump_htp_config, dump_htp_link_config
import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Make context cache from model libs')
    parser.add_argument('model_lib', type=Path, help='Path to RWKV pth file')
    parser.add_argument('output_path', type=Path, help='Path to output folder')
    parser.add_argument('platform', type=str, choices=htp_devices.keys(), help='Platform name')
    args = parser.parse_args()
    qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
    if not qnn_sdk_root:
        print("Please set QNN_SDK_ROOT environment variable to the root of the Qualcomm Neural Processing SDK")
        exit(1)

    model_name = str(args.model_lib).split('/')[-1].replace('.so', '')
    dump_htp_config(args.platform, [model_name], str(args.model_lib).replace('.so', '_htp_config.json'))
    dump_htp_link_config(str(args.model_lib).replace('.so', '_htp_link.json'), qnn_sdk_root)
    convert_cmd = f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-context-binary-generator"
    convert_cmd += f" --backend {qnn_sdk_root}/lib/x86_64-linux-clang/libQnnHtp.so"
    convert_cmd += f" --model {args.model_lib}"
    convert_cmd += f" --output_dir {args.output_path}"
    convert_cmd += f" --binary_file {model_name.replace('lib', '')}"
    convert_cmd += f" --config_file {str(args.model_lib).replace('.so', '_htp_link.json')}"
    os.system(convert_cmd)

if __name__ == '__main__':
    main()
