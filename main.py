import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import io
import cv2
import base64
import numpy as np
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Qwen2TokenizerFast,
)
from qwen_vl_utils import process_vision_info
from preprocess import image_preprocess


def run_inference(image1_path, image2_path, model_path):
    print(f"正在从 {model_path} 加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
    print("模型加载完成。")

    pil_image1 = Image.open(image1_path).convert("RGB") if image1_path else None
    pil_image2 = Image.open(image2_path).convert("RGB") if image2_path else None

    if not pil_image1 and not pil_image2:
        print("错误：请至少提供一张图片路径。")
        return


    processed_image_np = image_preprocess(pil_image1, pil_image2)
    
    temp_save_path = "temp_inference_input.png"
    cv2.imwrite(temp_save_path, processed_image_np)
    print(f"预处理完成，中间图像已保存至: {temp_save_path}")

    prompt = "<image>这是一个人脸图像，叠加了微表情的可视化光流图。在此图像中，每个面部区域的颜色表示运动的方向，亮度表示运动的幅度。颜色到方向的映射如下：\n向上：紫色\n右上：品红色，粉红色\n向右：红色\n右下：橙黄色\n向下：黄绿色\n左下：绿色\n向左：青色\n左上：蓝色\n你需要按以下步骤进行分析：\n视觉检查：仔细观察颜色块在不同面部区域的分布。排除整体头部运动和亮度较低（运动较弱）的区域。仅关注明显的局部运动单元。识别每个关键区域的颜色、相应的运动方向和运动强度。并基于特定面部区域的运动方向和幅度，推断图像中涉及的面部动作单元 (AUs)。\n表情推理：结合上下文信息，并根据识别出的面部动作单元 (AUs) 及其组合，推理出人脸所表达的微表情。\n最终结论：请总结分析结果，明确列出推断出的面部动作单元 (AUs) 以及最终推断的微表情。"

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": temp_save_path}],
        }
    ]


    text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")}<|im_end|>\n<|im_start|>assistant\n'
    

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")


    generate_ids = model.generate(
        **inputs, 
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True
    )
    
    output_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    generated_text = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(generated_text)


if __name__ == "__main__":
    # 在这里配置你的图片路径和模型路径
    IMG1 = "path/to/your/onset.jpg" 
    IMG2 = "path/to/your/apex.jpg"
    MODEL_DIR = "model_path"

    run_inference(IMG1, IMG2, MODEL_DIR)
