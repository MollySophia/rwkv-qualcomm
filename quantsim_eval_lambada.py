#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

import torch
from tqdm import tqdm

from rwkv_src.rwkv_model import RWKV_RNN
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from rwkv_src.rwkv_v7_modules_conv import L2Norm, Wkv7Op, Wkv7OutputState, Wkv7OutputX
from utils.model_utils import get_dummy_input_for_rwkv_causal_llm, get_dummy_state_kvcache, register_customop_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RWKV AIMET QuantSim on LAMBADA")
    parser.add_argument("model", type=Path, help="Path to RWKV .pth file")
    parser.add_argument("encodings", type=Path, help="Path to AIMET torch encodings")
    parser.add_argument("--text_path", type=Path, default=Path("assets/lambada_test.txt"))
    parser.add_argument("--vocab", type=Path, default=Path("assets/rwkv_vocab_v20230424.txt"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--heads_per_split", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("lambada_results.txt"))
    parser.add_argument("--sample_log", type=Path, default=None)
    parser.add_argument(
        "--sequential_prefill",
        action="store_true",
        help="Feed the prompt one token at a time through the decode graph.",
    )
    return parser.parse_args()


def resolve_use_cuda(device: str) -> bool:
    if device == "cpu":
        return False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return True
    return torch.cuda.is_available()


def make_model_args(args: argparse.Namespace) -> types.SimpleNamespace:
    use_cuda = resolve_use_cuda(args.device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bf16" if use_cuda else "fp32"
    return types.SimpleNamespace(
        USE_CUDA=use_cuda,
        fp16=dtype == "fp16",
        bf16=dtype == "bf16",
        USE_EMBEDDING=True,
        RESCALE_LAYER=0,
        wkv_customop=True,
        output_last=False,
        EXTERNAL_HEAD=False,
        heads_per_split=args.heads_per_split,
        MODEL_NAME=str(args.model),
    )


def install_quantized_custom_modules():
    from aimet_torch.v2.nn import QuantizationMixin

    @QuantizationMixin.implements(Wkv7Op)
    class QuantizedWkv7Op(QuantizationMixin, Wkv7Op):
        def __quant_init__(self):
            super().__quant_init__()
            self.input_quantizers = torch.nn.ModuleList([None, None, None, None, None, None, None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, r, w, k, v, a, b, state2):
            with self._patch_quantized_parameters():
                return super().forward(r, w, k, v, a, b, state2)

    @QuantizationMixin.implements(Wkv7OutputX)
    class QuantizedWkv7OutputX(QuantizationMixin, Wkv7OutputX):
        def __quant_init__(self):
            super().__quant_init__()
            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, input):
            with self._patch_quantized_parameters():
                ret = super().forward(input)
            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)
            return ret

    @QuantizationMixin.implements(Wkv7OutputState)
    class QuantizedWkv7OutputState(QuantizationMixin, Wkv7OutputState):
        def __quant_init__(self):
            super().__quant_init__()
            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, input):
            with self._patch_quantized_parameters():
                return super().forward(input)

    @QuantizationMixin.implements(L2Norm)
    class QuantizedL2Norm(QuantizationMixin, L2Norm):
        def __quant_init__(self):
            super().__quant_init__()
            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, x):
            if self.input_quantizers[0]:
                x = self.input_quantizers[0](x)
            with self._patch_quantized_parameters():
                ret = super().forward(x)
            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)
            return ret


def build_sim(model: RWKV_RNN, device: torch.device):
    from aimet_common.defs import QuantScheme
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.v2 import quantization as Q

    dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, device, model.args)
    dummy_input = (dummy_input["in0"], dummy_input["state"])

    sim = QuantizationSimModel(
        model.eval(),
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=8,
        default_output_bw=16,
        config_file="quantizers/configs/htp_quantsim_config_v75.json",
        in_place=True,
    )

    quant16 = lambda: Q.affine.Quantize((), bitwidth=16, symmetric=False).to(device)

    for block in sim.model.blocks:
        block.att.pre_permute_r.output_quantizers[0] = quant16()
        block.att.pre_permute_w.output_quantizers[0] = quant16()
        block.att.pre_permute_k.output_quantizers[0] = quant16()
        block.att.pre_permute_v.output_quantizers[0] = quant16()

        for head in block.att.heads:
            head.post_permute_a.output_quantizers[0] = quant16()
            head.post_permute_g.output_quantizers[0] = quant16()
            head.post_permute_r.output_quantizers[0] = quant16()
            head.post_permute_w.output_quantizers[0] = quant16()
            head.post_permute_k.output_quantizers[0] = quant16()
            head.post_permute_v.output_quantizers[0] = quant16()
            head.post_permute_v1.output_quantizers[0] = quant16()

            head.mul_ln_x.input_quantizers[1] = quant16()
            head.add_ln_x.input_quantizers[1] = quant16()
            head.mul_gate.input_quantizers[1] = quant16()
            head.scale_w.input_quantizers[1] = quant16()
            head.mix_kk.input_quantizers[1] = quant16()
            head.mix_ka_add.input_quantizers[0] = quant16()
            head.mix_ka_sub.input_quantizers[1] = quant16()
            head.mix_ka_mul_a.input_quantizers[1] = quant16()
            head.mul_r_k.input_quantizers[1] = quant16()

            head.wkv7.wkv.output_quantizers[0] = None
            for i in range(7):
                head.wkv7.wkv.input_quantizers[i] = None
            head.wkv7.wkv_output_x.input_quantizers[0] = None
            head.wkv7.wkv_output_x.output_quantizers[0] = quant16()
            head.wkv7.wkv_output_state.input_quantizers[0] = None
            head.wkv7.wkv_output_state.output_quantizers[0] = None

        block.ffn.pre_conv_transpose.output_quantizers[0] = quant16()
        block.ffn.post_conv_transpose.output_quantizers[0] = quant16()
        block.ffn.pre_conv_transpose2.output_quantizers[0] = quant16()
        block.ffn.post_conv_transpose2.output_quantizers[0] = quant16()
        block.ffn.mul_x_k.input_quantizers[1] = quant16()

        block.att.lerp_mul_r.input_quantizers[1] = quant16()
        block.att.lerp_mul_w.input_quantizers[1] = quant16()
        block.att.lerp_mul_k.input_quantizers[1] = quant16()
        block.att.lerp_mul_v.input_quantizers[1] = quant16()
        block.att.lerp_mul_a.input_quantizers[1] = quant16()
        block.att.lerp_mul_g.input_quantizers[1] = quant16()

    sim.model.head_pre_permute.output_quantizers[0] = quant16()
    sim.model.head_post_permute.output_quantizers[0] = quant16()
    sim.model.head_pre_reshape.output_quantizers[0] = quant16()
    sim.model.head_post_reshape.output_quantizers[0] = quant16()
    return sim


def load_lambada(path: Path) -> list[str]:
    return [text for text in path.read_text().split("|") if text]


def safe_decode(tokenizer: RWKV_TOKENIZER, token_id: int) -> str:
    token = tokenizer.idx2token.get(int(token_id), b"")
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return repr(token)


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    register_customop_symbols()
    install_quantized_custom_modules()

    model_args = make_model_args(args)
    print(
        f"device={'cuda' if model_args.USE_CUDA else 'cpu'} "
        f"dtype={'fp16' if model_args.fp16 else 'bf16' if model_args.bf16 else 'fp32'}",
        flush=True,
    )
    model = RWKV_RNN(model_args).eval()
    device = model.device
    sim = build_sim(model, device)
    print(f"loading encodings: {args.encodings}", flush=True)
    sim.load_encodings(args.encodings, allow_overwrite=False)
    sim.model.eval()

    tokenizer = RWKV_TOKENIZER(str(args.vocab))
    texts = load_lambada(args.text_path)
    end = len(texts) if args.max_samples <= 0 else min(len(texts), args.start_sample + args.max_samples)
    selected = texts[args.start_sample:end]
    print(f"evaluating samples [{args.start_sample}, {end}) / {len(texts)}", flush=True)

    sample_log = args.sample_log.open("w") if args.sample_log else None
    xsum = 0.0
    xacc = 0
    xcnt = 0
    try:
        with torch.inference_mode():
            for sample_index, text in enumerate(tqdm(selected), start=args.start_sample):
                src_text = text.rsplit(" ", 1)[0]
                target_text = " " + text.rsplit(" ", 1)[1]
                targets = tokenizer.encode(target_text)

                state = get_dummy_state_kvcache(1, sim.model.args, sim.model.device)
                prompt_tokens = [0] + tokenizer.encode(src_text)
                if args.sequential_prefill:
                    logits = None
                    for prompt_token in prompt_tokens:
                        input_data = torch.tensor([[prompt_token]], dtype=torch.long, device=sim.model.device)
                        logits, state = sim.model(input_data, state)
                    if logits is None:
                        raise RuntimeError("empty prompt")
                else:
                    input_data = torch.tensor([prompt_tokens], dtype=torch.long, device=sim.model.device)
                    logits, state = sim.model(input_data, state)
                    logits = logits[:, -1, :]

                logits_list = []
                for token in targets:
                    logits = logits.reshape(1, -1, logits.shape[-1])
                    logits_list.append(logits)
                    next_token = torch.tensor([[token]], dtype=torch.long, device=sim.model.device)
                    logits, state = sim.model(next_token, state)

                probs = torch.nn.functional.softmax(torch.cat(logits_list, dim=1), dim=-1)
                pred = torch.argmax(probs, dim=-1).squeeze().detach().cpu().numpy().tolist()
                if isinstance(pred, int):
                    pred = [pred]

                correct = pred == targets
                if correct:
                    xacc += 1
                for i, target in enumerate(targets):
                    xsum += probs[0, i, target].log().item()
                xcnt += 1

                if sample_log:
                    sample_log.write(
                        json.dumps(
                            {
                                "sample": sample_index,
                                "correct": correct,
                                "target_ids": targets,
                                "pred_ids": pred,
                                "target_text": target_text,
                                "pred_text": "".join(safe_decode(tokenizer, token_id) for token_id in pred),
                                "accuracy": xacc / xcnt,
                                "perplexity": math.exp(-xsum / xcnt),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    sample_log.flush()

                if args.log_every > 0 and (xcnt == 1 or xcnt % args.log_every == 0):
                    print(
                        f"sample={sample_index} count={xcnt} "
                        f"accuracy={xacc}/{xcnt}={xacc / xcnt:.6f} "
                        f"perplexity={math.exp(-xsum / xcnt):.6f}",
                        flush=True,
                    )
    finally:
        if sample_log:
            sample_log.close()

    result = {
        "samples": xcnt,
        "accuracy_count": xacc,
        "accuracy": xacc / xcnt if xcnt else 0.0,
        "perplexity": math.exp(-xsum / xcnt) if xcnt else float("nan"),
    }
    print(json.dumps(result, indent=2), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
