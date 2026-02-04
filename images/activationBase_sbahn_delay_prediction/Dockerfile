# Dockerfile für activationBase_sbahn_delay_prediction
# Basierend auf busybox (gemäß Subgoal 3 PDF-Anforderungen)
FROM busybox:latest

RUN mkdir -p /data

COPY activation_data.csv /data/activation_data.csv
COPY activation_data_ols.csv /data/activation_data_ols.csv
COPY README.md /Readme.md

CMD ["sh","-c","mkdir -p /tmp/activationBase && cp /data/activation_data*.csv /tmp/activationBase/ && ls -la /tmp/activationBase && echo OK"]

