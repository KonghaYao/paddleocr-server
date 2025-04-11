# -*- coding: utf-8 -*-
import logging
import mimetypes
import os
import tempfile
import uuid
from urllib.parse import urlparse

import cv2
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置参数
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
DOWNLOAD_TIMEOUT = 30  # 下载超时时间(秒)


def allowed_file(filename):
    """检查文件类型是否允许"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def download_file(url, save_path=None):
    """
    从URL下载文件，支持HTTP重定向

    Args:
        url: 文件URL
        save_path: 保存路径，如果为None则生成临时文件

    Returns:
        保存的文件路径
    """
    try:
        # 使用requests自动处理重定向
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()

        # 检查文件大小
        content_length = int(response.headers.get("content-length", 0))
        if content_length > MAX_FILE_SIZE:
            raise ValueError(
                f"文件大小 ({content_length} 字节) 超过最大限制 ({MAX_FILE_SIZE} 字节)"
            )

        # 检查文件类型
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"不支持的文件类型: {content_type}")

        # 如果未提供保存路径，创建临时文件
        if save_path is None:
            # 获取合适的文件扩展名
            ext = mimetypes.guess_extension(content_type) or ".jpg"
            # 使用tempfile模块自动创建临时文件
            temp_fd, save_path = tempfile.mkstemp(suffix=ext)
            os.close(temp_fd)  # 关闭文件描述符，让后续操作可以使用这个文件

        # 下载并保存文件内容
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"文件已下载到: {save_path}")
        return save_path

    except requests.RequestException as e:
        logger.error(f"下载文件失败: {str(e)}")
        raise ValueError(f"下载文件失败: {str(e)}")
    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}")
        raise ValueError(f"处理文件失败: {str(e)}")



def extract_text_with_layout(image_path):
    """
    从图像中提取文本并保持布局

    Args:
        image_path: 图像文件路径

    Returns:
        按照原始布局格式化的文本
    """
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

        # 识别图像中的文本
    result = ocr.ocr(img, cls=True)

    # 提取文本块及其位置信息
    text_blocks = []
    for line in result[0]:
        box = line[0]  # 文本框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]  # 识别的文本
        confidence = line[1][1]  # 置信度

        # 计算中心点的y坐标作为行标识
        center_y = (box[0][1] + box[2][1]) / 2
        # 计算左边界的x坐标作为列标识
        left_x = box[0][0]

        text_blocks.append(
            {
                "text": text,
                "box": box,
                "center_y": center_y,
                "left_x": left_x,
                "confidence": confidence,
            }
        )

        # 根据y坐标对文本块进行分组，将接近的y坐标视为同一行

    def group_by_rows(blocks, y_threshold=40):
        """将文本块按行分组"""
        if not blocks:
            return []

            # 按center_y排序
        sorted_blocks = sorted(blocks, key=lambda b: b["center_y"])

        rows = []
        current_row = [sorted_blocks[0]]

        for block in sorted_blocks[1:]:
            if abs(block["center_y"] - current_row[0]["center_y"]) <= y_threshold:
                # 如果y坐标接近，则归为同一行
                current_row.append(block)
            else:
                # 否则开始新的一行
                rows.append(current_row)
                current_row = [block]

        rows.append(current_row)
        return rows

        # 将文本块分组为行

    rows = group_by_rows(text_blocks)

    # 在每行内按x坐标排序文本块
    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda b: b["left_x"])

        # 计算各列之间的间距，用于决定制表符数量

    def calculate_tabs(row):
        """计算行内各文本块之间应插入的制表符数量"""
        if len(row) <= 1:
            return []

            # 计算行内各文本块的间距
        gaps = []
        for i in range(len(row) - 1):
            current_right = max(p[0] for p in row[i]["box"])
            next_left = min(p[0] for p in row[i + 1]["box"])
            gap = next_left - current_right
            gaps.append(gap)

            # 如果有间距，归一化间距为制表符数量
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            # 将间距转换为制表符数量(根据间距大小调整)
            tab_counts = [max(1, int(gap / (avg_gap * 0.7))) for gap in gaps]
            return tab_counts
        return []

        # 构建格式化的文本输出

    formatted_text = ""
    for i, row in enumerate(rows):
        if i > 0:
            formatted_text += "\n"  # 行间换行

        tab_counts = calculate_tabs(row)

        for j, block in enumerate(row):
            formatted_text += block["text"]
            if j < len(row) - 1:
                # 添加制表符
                if j < len(tab_counts):
                    formatted_text += "\t" * tab_counts[j]
                else:
                    formatted_text += "\t"

    return formatted_text


@app.route("/ocr", methods=["POST"])
def ocr_api():
    """
    OCR API接口

    请求参数:
    - image_url: 图像文件URL

    返回:
    - JSON格式的OCR识别结果
    """
    # 获取请求数据
    data = request.json
    if not data or "image_url" not in data:
        return jsonify({"success": False, "error": "缺少image_url参数"}), 400

    image_url = data["image_url"]
    logger.info(f"处理图像URL: {image_url}")

    try:
        # 下载文件
        temp_image_path = download_file(image_url)

        # OCR处理
        logger.info(f"开始OCR处理: {temp_image_path}")
        formatted_text = extract_text_with_layout(temp_image_path)

        # 删除临时文件
        try:
            os.remove(temp_image_path)
            logger.info(f"临时文件已删除: {temp_image_path}")
        except Exception as e:
            logger.warning(f"删除临时文件失败: {str(e)}")

            # 返回结果
        return jsonify({"success": True, "text": formatted_text})

    except Exception as e:
        logger.error(f"OCR处理失败: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    # 健康检查接口


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "paddle-ocr-api"})


# API文档
@app.route("/", methods=["GET"])
def api_docs():
    return """  
    <html>  
        <head><title>PaddleOCR API</title></head>  
        <body>  
            <h1>PaddleOCR API 文档</h1>  
            <h2>接口：/ocr</h2>  
            <p>方法：POST</p>  
            <p>参数：JSON格式，包含image_url字段</p>  
            <p>示例请求：</p>  
            <pre>  
            {  
                "image_url": "https://example.com/image.jpg"  
            }  
            </pre>  
            <p>示例响应：</p>  
            <pre>  
            {  
                "success": true,  
                "text": "识别出的文本内容..."  
            }  
            </pre>  
        </body>  
    </html>  
    """


if __name__ == "__main__":
    # 在生产环境中应使用gunicorn或uwsgi
    app.run(host="0.0.0.0", port=5000, debug=False)
