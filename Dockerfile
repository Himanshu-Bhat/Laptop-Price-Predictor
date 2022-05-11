FROM python:3.10-alpine
WORKDIR app
COPY requirements.txt ./requirements.txt
COPY . /app
RUN pip3 install -r requirements.txt
RUN pip3 install --force-reinstall numpy==1.22.3
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ['webpage.py']

