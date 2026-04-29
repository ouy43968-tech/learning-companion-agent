# AI Learning Companion

一个基于 OpenAI + FastAPI + Knowledge Graph 的 AI 个性化学习陪练系统。

## 功能

- 个性化学习路径
- AI 出题
- AI 批改
- 错误类型分析
- 自适应难度
- FastAPI API
- Docker 部署

## 安装

```bash
pip install -r requirements.txt
```

## 启动

```bash
python app.py
```

或：

```bash
uvicorn app:app --reload
```

## Docker

```bash
docker build -t ai-learning-agent .
docker run -p 8000:8000 ai-learning-agent
```
