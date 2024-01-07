FROM python:3.10.11

WORKDIR /DURF
ADD requirements.txt /DURF

RUN pip install -r requirements.txt
CMD ["tail", "-f", "/dev/null"]