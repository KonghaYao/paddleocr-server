#!/bin/bash

# 向OCR API发送请求
echo "正在测试OCR API..."
curl -X POST \
    http://localhost:5000/ocr \
    -H "Content-Type: application/json" \
    -d "{\"image_url\": \"https://remarksoftware.com/wp-content/uploads/2015/07/High-School-Test-blank.jpg\"}"

# 如果没有安装jq，可以去掉"| jq ."部分
