"""
@Author  : ChenXin
@Data    : 2021/10/23 1:26
@Brief   : tf的模型转为pytorch模型
"""
import argparse
import logging
import torch
from transformers import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert

logging.basicConfig(level=logging.INFO)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, albert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = AlbertConfig.from_json_file(albert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = AlbertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_albert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default='./resources/albert_base_zh/model.ckpt-best', type=str, required=False, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--albert_config_file",
        default='./resources/albert_base_zh/albert_config.json',
        type=str,
        required=False,
        help="The config json file corresponding to the pre-trained ALBERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default='./resources/albert_base_zh/pytorch_model.bin', type=str, required=False, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.albert_config_file, args.pytorch_dump_path)


