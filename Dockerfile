FROM python:3.8-slim

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install Flask gunicorn
RUN pip install torch numpy onnx onnxruntime
RUN pip install opencv-python

# 启动 Web 服务
# 这里我们使用了 gunicorn 作为 Server，1 个 worker 和 8 个线程
# 如果您的容器实例拥有多个 CPU 核心，我们推荐您把线程数设置为与 CPU 核心数一致
CMD exec gunicorn --bind :8080 --workers 1 --threads 1 --timeout 0 main:app
