# 营养报告生成后端服务

基于 Qwen2.5-VL-7B-Instruct 模型和 LoRA 适配器的营养报告生成 API。

## 环境要求

- Python 3.10+
- CUDA (可选，用于 GPU 加速)
- 足够的磁盘空间存储模型文件（约 15GB）

## 安装依赖

```bash
conda activate image-upload
cd backend
pip install -r requirements.txt
```

## 配置

### 环境变量

可以通过环境变量配置模型路径：

- `HF_CACHE_DIR`: Hugging Face 模型缓存目录（默认: `~/.cache/huggingface`）
- `ADAPTER_PATH`: LoRA 适配器路径（默认: `./output/Qwen2.5-VL-7B-nutrition/checkpoint-215`）

示例（Windows）:
```bash
set HF_CACHE_DIR=D:/cache/huggingface
set ADAPTER_PATH=./output/Qwen2.5-VL-7B-nutrition/checkpoint-215
```

示例（Linux/Mac）:
```bash
export HF_CACHE_DIR=~/.cache/huggingface
export ADAPTER_PATH=./output/Qwen2.5-VL-7B-nutrition/checkpoint-215
```

### 模型文件

确保以下文件/目录存在：

1. **基础模型**: 将从 Hugging Face 自动下载到 `HF_CACHE_DIR/models/Qwen/Qwen2.5-VL-7B-Instruct/`
2. **LoRA 适配器**: 应位于 `ADAPTER_PATH` 指定的路径

## 启动服务

```bash
conda activate image-upload
cd backend
python main.py
```

服务将在 `http://localhost:8000` 启动。

## API 端点

### GET /

健康检查端点，返回服务状态。

**响应示例**:
```json
{
  "message": "Nutrition Report Generation API is running",
  "endpoint": "/generate",
  "model_status": "loaded",
  "device": "cuda"
}
```

### POST /generate

接收图片文件，返回生成的营养报告。

**请求**:
- Content-Type: `multipart/form-data`
- Body: `file` (图片文件)

**响应**:
```json
{
  "text": "生成的营养报告文本...",
  "report": "生成的营养报告文本...",
  "filename": "image.jpg",
  "status": "success"
}
```

**错误响应**:
- `400`: 文件不是图片
- `500`: 处理图片时出错
- `503`: 模型未加载或推理错误

## 注意事项

1. **首次启动**: 首次运行时会从 Hugging Face 下载模型文件，可能需要较长时间
2. **内存要求**: 模型需要较大的内存/显存，建议至少 16GB RAM 或 8GB VRAM
3. **GPU 加速**: 如果系统有 CUDA，会自动使用 GPU 加速
4. **模型加载**: 模型在服务启动时加载，可能需要几分钟时间

## 故障排除

### 模型加载失败

- 检查 `HF_CACHE_DIR` 路径是否正确
- 确保有足够的磁盘空间
- 检查网络连接（需要下载模型）

### 适配器加载失败

- 检查 `ADAPTER_PATH` 路径是否正确
- 确保适配器文件完整

### CUDA 错误

- 如果使用 CPU，确保 `device` 设置为 `"cpu"`
- 检查 PyTorch CUDA 版本是否与系统 CUDA 版本匹配

### 内存不足

- 减少 `max_new_tokens` 参数
- 使用 CPU 模式（虽然会较慢）
- 考虑使用量化模型



