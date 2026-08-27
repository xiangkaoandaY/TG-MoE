from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import json
from diffusers import AutoencoderKL
import kornia as K
import io
import torchvision.transforms as T
import torch.nn.functional as F


def collate(examples):
    # SD3: 这里的 input_ids 是一个包含三个 tensor 的列表的列表
    # example["instance_prompt_ids"] = [tensor(77), tensor(77), tensor(512)]

    # 1. 整理 Pixel Values
    pixel_values = [example["instance_image"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # 2. 整理 Prompt IDs (无 Trigger)
    # 我们需要把 tokenizer_1 的放在一起，tokenizer_2 的放在一起...
    input_ids_list = []
    num_tokenizers = len(examples[0]["instance_prompt_ids"])  # 应该是 3

    for i in range(num_tokenizers):
        # 收集当前 batch 中所有样本的第 i 个 tokenizer 的输出
        ids = [example["instance_prompt_ids"][i] for example in examples]
        input_ids_list.append(torch.cat(ids, dim=0))

    # 3. 整理 Prompt IDs (有 Trigger)
    input_ids_trigger_list = []
    for i in range(num_tokenizers):
        ids = [example["instance_prompt_ids_with_trigger"][i] for example in examples]
        input_ids_trigger_list.append(torch.cat(ids, dim=0))

    batch = {
        "input_ids_list": input_ids_list,  # [Batch, 77], [Batch, 77], [Batch, 512]
        "input_ids_trigger_list": input_ids_trigger_list,
        "pixel_values": pixel_values
    }

    return batch


def encode_prompt(text_encoders, input_ids_list):
    """
    SD3 专用的 encode_prompt。
    参数:
        text_encoders: [text_encoder_1, text_encoder_2, text_encoder_3]
        input_ids_list: [input_ids_1, input_ids_2, input_ids_3]
    返回:
        prompt_embeds, pooled_prompt_embeds
    """
    # 1. 分别获取 CLIP 和 T5 的编码器
    clip_encoder_1 = text_encoders[0]
    clip_encoder_2 = text_encoders[1]
    t5_encoder = text_encoders[2]  # 可能为 None，如果显存不够没加载

    clip_input_ids_1 = input_ids_list[0]
    clip_input_ids_2 = input_ids_list[1]
    t5_input_ids = input_ids_list[2]

    # 2. 处理 CLIP Encoder 1
    prompt_embeds_1 = clip_encoder_1(clip_input_ids_1, output_hidden_states=True)
    pooled_prompt_embeds_1 = prompt_embeds_1[0]
    prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]  # 取倒数第二层

    # 3. 处理 CLIP Encoder 2
    prompt_embeds_2 = clip_encoder_2(clip_input_ids_2, output_hidden_states=True)
    pooled_prompt_embeds_2 = prompt_embeds_2[0]
    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    # 4. 处理 T5 Encoder (如果存在)
    if t5_encoder is not None:
        prompt_embeds_3 = t5_encoder(t5_input_ids, output_hidden_states=True)
        # T5 没有 pooled output，只取最后一层 hidden states
        prompt_embeds_3 = prompt_embeds_3[0]
    else:
        # 如果为了省显存没加载 T5，创建一个全零的占位符 (Batch, Seq, Dim)
        # SD3 T5 dim is 4096
        batch_size = clip_input_ids_1.shape[0]
        # 假设 T5 max length 是 256 或 512，这里取 input_ids 的长度
        seq_len = t5_input_ids.shape[1]
        prompt_embeds_3 = torch.zeros(
            (batch_size, seq_len, 4096),
            device=clip_encoder_1.device,
            dtype=clip_encoder_1.dtype
        )

    # 5. 拼接 Pooled Embeddings (用于输入 Transformer 的 pooled_projections)
    # CLIP 1 (768) + CLIP 2 (1280) = 2048
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

    # 6. 处理 CLIP Embeddings 的拼接和 Padding
    # SD3 逻辑：CLIP outputs 需要 pad 到和 T5 维度一致或者拼接
    # 通常 CLIP dim 是 768 和 1280，T5 是 4096。
    # SD3 Transformer 期望输入维度是 4096。

    # 填充 CLIP 1 (768 -> 4096)
    prompt_embeds_1 = torch.nn.functional.pad(
        prompt_embeds_1, (0, 4096 - prompt_embeds_1.shape[-1])
    )
    # 填充 CLIP 2 (1280 -> 4096)
    prompt_embeds_2 = torch.nn.functional.pad(
        prompt_embeds_2, (0, 4096 - prompt_embeds_2.shape[-1])
    )

    # 7. 最终拼接 Hidden States
    # 顺序：[CLIP_1, CLIP_2, T5]
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2, prompt_embeds_3], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class DreamBoothDataset_modified(Dataset):
    """
    SD3 Version: 支持多个 tokenizer
    """

    def __init__(
            self,
            instance_data_root,
            tokenizers,  # 接收一个列表 [tok1, tok2, tok3]
            size=512,
            center_crop=False,
            tokenizer_max_length=None,  # 可以是一个整数(统一)或列表 [77, 77, 512]
            prompt_trigger='',
            use_null_prompt=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizers = tokenizers  # List of tokenizers

        # 处理 max_length
        if tokenizer_max_length is None:
            # 默认 SD3 设置: CLIPs=77, T5=256 or 512
            self.tokenizer_max_lengths = [77, 77, 512]
        elif isinstance(tokenizer_max_length, int):
            self.tokenizer_max_lengths = [tokenizer_max_length] * len(tokenizers)
        else:
            self.tokenizer_max_lengths = tokenizer_max_length

        with open(f'{instance_data_root}/metadata.jsonl') as f:
            metadata = [json.loads(line) for line in f]
        file_names = [os.path.join(instance_data_root, item['file_name']) for item in metadata]
        prompts = [item['text'] for item in metadata]
        self.instance_images_path = file_names
        self.instance_prompts = prompts
        self._length = len(self.instance_images_path)
        self.prompt_trigger = prompt_trigger
        self.use_null_prompt = use_null_prompt

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # 1. 处理图像 (保持不变)
        instance_image = Image.open(self.instance_images_path[index % self._length])
        instance_image = exif_transpose(instance_image)

        if self.use_null_prompt:
            instance_prompt = ""
        else:
            instance_prompt = self.instance_prompts[index % self._length]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        transforms_pipeline = transforms.Compose(
            [
                transforms.CenterCrop(min(instance_image.size)) if self.center_crop else transforms.RandomCrop(
                    min(instance_image.size)),
                transforms.Resize((self.size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        example["instance_image"] = transforms_pipeline(instance_image)

        # 2. 处理 Tokenization (循环处理 3 个 tokenizer)
        # ===============================================
        input_ids_list = []
        input_ids_trigger_list = []

        instance_prompt_with_trigger = self.prompt_trigger + instance_prompt

        for i, tokenizer in enumerate(self.tokenizers):
            # 获取当前 tokenizer 的最大长度
            curr_max_len = self.tokenizer_max_lengths[i] if i < len(self.tokenizer_max_lengths) else 77

            # Tokenize 无 Trigger
            text_inputs = tokenize_prompt(
                tokenizer, instance_prompt, tokenizer_max_length=curr_max_len
            )
            input_ids_list.append(text_inputs.input_ids)

            # Tokenize 有 Trigger
            text_inputs_trigger = tokenize_prompt(
                tokenizer, instance_prompt_with_trigger, tokenizer_max_length=curr_max_len
            )
            input_ids_trigger_list.append(text_inputs_trigger.input_ids)

        example["instance_prompt_ids"] = input_ids_list
        example["instance_prompt_ids_with_trigger"] = input_ids_trigger_list

        return example


# ====================================================
# 以下辅助函数通常不需要针对 SD3 修改，保持原样即可
# 除非 img_to_DMlatents 中的 vae scaling factor 在 SD3 中有变化 (SD3 vae 通常也是 0.18215 或 1.5305，取决于具体实现)
# SD3 VAE scaling factor is usually different from SD1.5.
# SD1.5 is 0.18215. SD3 is approximately 1.5305.
# 建议在主程序中通过 vae.config.scaling_factor 动态获取，不要写死。
# 下面的函数已经使用了 vae.config.scaling_factor，所以是兼容的。

def coefficient_wm(t, t_threshold, max_weight, steepness):
    sigmoid_weight = max_weight * torch.sigmoid(-(t - t_threshold) / steepness)
    return sigmoid_weight


def coefficient_preserve(t, t_threshold, steepness):
    sigmoid_weight = torch.sigmoid((t - t_threshold) / steepness)
    return sigmoid_weight


def img_to_DMlatents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist.sample()
    latents = posterior * vae.config.scaling_factor
    return latents


def DMlatent2img(latents: torch.Tensor, vae: AutoencoderKL):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents)['sample']
    image_tensor = image / 2.0 + 0.5
    return image_tensor


def distorsion_unit(encoded_images, type):
    # 这部分代码是通用的图像处理，不需要修改
    if type == 'identity':
        distorted_images = encoded_images
    elif type == 'brightness':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),
            contrast=(1.0, 1.0),
            saturation=(1.0, 1.0),
            hue=(0.0, 0.0),
            p=1
        )(encoded_images)
    elif type == 'contrast':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),
            contrast=(0.8, 1.2),
            saturation=(1.0, 1.0),
            hue=(0.0, 0.0),
            p=1
        )(encoded_images)
    elif type == 'saturation':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),
            contrast=(1.0, 1.0),
            saturation=(0.8, 1.2),
            hue=(0.0, 0.0),
            p=1
        )(encoded_images)
    elif type == 'blur':
        distorted_images = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_images)
    elif type == 'noise':
        distorted_images = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_images)
    elif type == 'jpeg_compress':
        B = encoded_images.shape[0]
        distorted_images = []
        for i in range(B):
            buffer = io.BytesIO()
            pil_image = T.ToPILImage()(encoded_images[i].squeeze(0))
            pil_image.save(buffer, format='JPEG', quality=50)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            distorted_images.append(T.ToTensor()(pil_image).to(encoded_images.device).unsqueeze(0))
        distorted_images = torch.cat(distorted_images, dim=0)
    elif type == 'resize':
        distorted_images = F.interpolate(
            encoded_images,
            scale_factor=(0.5, 0.5),
            mode='bilinear')
    elif type == 'sharpness':
        distorted_images = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_images)

    else:
        raise ValueError(f'Wrong distorsion type in add_distorsion().')

    distorted_images = torch.clamp(distorted_images, 0, 1)
    return distorted_images