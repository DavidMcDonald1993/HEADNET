FROM continuumio/miniconda3

# python dependencies
COPY ./environment.yml /environment.yml
RUN conda env update --name base --file /environment.yml

RUN conda clean --all


RUN mkdir /headnet
COPY ./headnet /headnet/headnet 
COPY ./utils /headnet/utils 
COPY ./main.py /headnet/main.py 

RUN mkdir /headnet/input

WORKDIR /headnet

ENTRYPOINT ["python", "main.py", "--graph", "input/graph.npz", "--embedding", "input/embeddings"]