FROM elasticsearch:8.12.0

USER root
# 复制本地文件安装
COPY elasticsearch-analysis-ik-8.12.0.zip /tmp/ik.zip
RUN ./bin/elasticsearch-plugin install --batch file:///tmp/ik.zip && rm /tmp/ik.zip
USER elasticsearch