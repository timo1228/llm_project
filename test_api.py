"""
测试后端API端点
"""
import requests
import io
from PIL import Image

# 创建一个简单的测试图片
img = Image.new('RGB', (100, 100), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# 测试 /generate 端点
print("测试 /generate 端点...")
url = "http://localhost:8000/generate"
files = {'image': ('test.png', img_bytes, 'image/png')}

try:
    response = requests.post(url, files=files)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\n返回结果:")
        print(f"文件名: {result.get('filename')}")
        print(f"大小: {result.get('size')} 字节")
        print(f"内容类型: {result.get('content_type')}")
        print(f"\n生成的文本:\n{result.get('text')}")
        print("\n✅ API测试成功！")
    else:
        print(f"❌ 错误: {response.text}")
except Exception as e:
    print(f"❌ 请求失败: {e}")



