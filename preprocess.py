import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
import face_alignment
import argparse
from tqdm import tqdm
from PIL import Image


def compute_adaptive_flow_mask(flow, percent=95):
    """
    使用分位数法生成自适应阈值掩码，用于保留光流强度较高的区域。
    参数:
        flow (np.ndarray): 光流向量场，形状为 (H, W, 2)。
        percent (int): 百分位数阈值，表示保留光流强度大于此分位数的像素。默认95。
    返回:
        np.ndarray: 二值掩码，形状为 (H, W)，数据类型为 np.uint8，前景为255，背景为0。
    """
    # 计算光流向量的幅度
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 计算指定分位数作为阈值（保留最大的 (100 - percent)% 的光流）
    threshold = np.percentile(magnitude, percent)
    # 根据阈值创建二值掩码
    return (magnitude > threshold).astype(np.uint8) * 255


def compute_tvl1_optical_flow(prev_img, next_img):
    """
    使用TVL1算法计算两张图片之间的光流。
    参数:
        prev_img (np.ndarray): 前一帧图像。
        next_img (np.ndarray): 后一帧图像。
    返回:
        np.ndarray: 计算得到的光流向量场，形状为 (H, W, 2)。
    """
    # 确保输入图像是灰度图，如果不是则进行转换
    prev_gray = (
        cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        if len(prev_img.shape) == 3
        else prev_img
    )
    next_gray = (
        cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        if len(next_img.shape) == 3
        else next_img
    )
    # 创建DualTVL1OpticalFlow_create对象
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    # 计算光流
    flow = optical_flow.calc(prev_gray, next_gray, None)
    return flow


def quantize_angle(angle_deg, step=30):
    """
    将角度量化到指定步长。
    参数:
        angle_deg (np.ndarray或float): 输入角度（度）。
        step (int): 量化步长。默认30。
    返回:
        np.ndarray或float: 量化后的角度。
    """
    # 将角度限制在0-360度之间
    angle_deg = angle_deg % 360
    # 进行量化
    quantized = np.round(angle_deg / step) * step
    # 确保量化后的角度仍在0-360度之间
    return quantized % 360


def visualize_flow(flow):
    """
    将光流可视化为BGR图像（通常用于光流场的颜色编码）。
    参数:
        flow (np.ndarray): 光流向量场，形状为 (H, W, 2)。
    返回:
        np.ndarray: 可视化后的BGR图像，形状为 (H, W, 3)。
    """
    # 将光流从笛卡尔坐标转换为极坐标（幅度magnitude和角度angle）
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 将角度从弧度转换为度
    angle_deg = angle * 180 / np.pi
    # 量化角度
    quantized_angle_deg = quantize_angle(angle_deg, step=30)
    # 计算HSV色相（Hue）分量，范围为0-180
    hsv_hue = quantized_angle_deg / 2
    # 创建HSV图像，初始化为全黑
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    # 设置HSV图像的色相、饱和度和亮度
    hsv[..., 0] = hsv_hue.astype(np.uint8)  # 色相
    hsv[..., 1] = 255  # 饱和度设为最大
    hsv[..., 2] = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    )  # 亮度根据幅度归一化
    # 将HSV图像转换为BGR格式
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def image_preprocess(onset_image, apex_image):
    """
    对起始帧和峰值帧图像进行预处理，包括人脸对齐、光流计算、头部运动校正和光流可视化叠加。
    参数:
        onset_image (PIL.Image.Image或np.ndarray): 起始帧图像。
        apex_image (PIL.Image.Image或np.ndarray): 峰值帧图像。
    返回:
        np.ndarray: 叠加了校正后光流可视化结果的灰度图像。
    """
    # 将PIL图像转换为OpenCV格式（BGR）
    if isinstance(onset_image, Image.Image):
        onset_image = np.array(onset_image)
        onset_image = cv2.cvtColor(onset_image, cv2.COLOR_RGB2BGR)
    if isinstance(apex_image, Image.Image):
        apex_image = np.array(apex_image)
        apex_image = cv2.cvtColor(apex_image, cv2.COLOR_RGB2BGR)
    # 初始化人脸对齐模型
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device="cuda",
    )
    # 获取起始帧图像中的人脸关键点
    preds = fa.get_landmarks(onset_image)
    if not preds:
        # 如果没有检测到人脸，则返回原始图像的灰度版本
        print("未在起始图像中检测到人脸，返回原始灰度图像。")
        return cv2.cvtColor(onset_image, cv2.COLOR_BGR2GRAY)
    preds = preds[0]  # shape: (68, 2)
    # 计算起始帧和峰值帧之间的光流
    flow = compute_tvl1_optical_flow(onset_image, apex_image)
    # 取鼻尖关键点（索引30）作为头部运动的参考点
    nose_point = preds[30]  # (x, y)
    nose_x, nose_y = int(nose_point[0]), int(nose_point[1])
    # 设置鼻尖局部区域大小（可调），用于估计头部运动
    region_size = 5
    x1 = max(nose_x - region_size, 0)
    x2 = min(nose_x + region_size + 1, flow.shape[1])
    y1 = max(nose_y - region_size, 0)
    y2 = min(nose_y + region_size + 1, flow.shape[0])
    # 计算鼻尖局部区域的平均光流（作为头部运动的估计）
    if x2 <= x1 or y2 <= y1:  # 避免空区域
        print("鼻尖局部区域计算异常，可能导致头部运动校正不准确。")
        mean_flow = np.array([0.0, 0.0])  # 如果区域无效，则假设没有头部运动
    else:
        nose_region_flow = flow[y1:y2, x1:x2]
        # 确保nose_region_flow不是空的
        if nose_region_flow.size > 0:
            mean_flow = np.mean(nose_region_flow.reshape(-1, 2), axis=0)
        else:
            print("鼻尖区域光流为空，无法计算平均光流。")
            mean_flow = np.array([0.0, 0.0])
    # 用鼻尖光流校正整体光流，消除头部运动
    flow_corrected = flow - mean_flow
    # 将原始图像转换为三通道灰度图像，以便叠加光流可视化结果
    if len(onset_image.shape) == 3:  # 若原图为彩色
        gray_img = cv2.cvtColor(onset_image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # 转三通道
    else:  # 若原图已经是单通道灰度
        gray_img = cv2.cvtColor(onset_image, cv2.COLOR_GRAY2BGR)
    # 可视化校正后的光流
    bgr_flow = visualize_flow(flow_corrected)
    # 计算自适应光流掩码，可以根据需要调整百分位数
    mask = compute_adaptive_flow_mask(
        flow_corrected, percent=0
    )  # percent=0表示不过滤，保留所有光流
    # 将光流可视化结果叠加到灰度图像上
    overlay = cv2.bitwise_and(bgr_flow, bgr_flow, mask=mask)
    result = cv2.add(gray_img, overlay)
    return result


