from transformers import pipeline 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") #model to summarize summary from hugging face

text = '''Docker is a platform for developing, shipping and running applications using containerization. A container is a lightweight, standalone, executable package that include everything which is required to run a piece of software — code, runtime, libraries and system tools.
Terminologies:
Images: These are something like classes which are used to create instances of a container, a image can have multiple container associate with it each with its own space and env. You can pull an already existing image from Docker-Hub or create your own image and push it for others to use.Containers: These are running instance of Images which are mainly used for isolated and lightweight env. without installing those env. locally on your own machine. You can start, stop and kill a container. These containers can have volume associate with them to sustain data in case they are restarted after being stopped.
DockerFile: It is a file which consist script to build a Docker image. It have onion like structure it support caching of layers in case not all content of image is changed.
4. Volumes: These are used persist data outside container i.e. when it is started after being stopped. It is useful databases, logs etc.
5. Docker Compose: It is tool used in big projects to run multi-container applications . In it docker-compose.yaml file is used to setup these containers.
Some useful Docker commands:
Image & Container management:
docker build -t <my-image> . //build a images with my-image name
docker images //show all images in your m/c
docker run -d -p 8000:80 <my-image> //run container in detach mode with port mapping of 8000 in pc to 80 in env. m/c.
docker ps //show all running container
docker stop <container-name> //used to stop a container
docker rm <container-id> //used to remove a container with specified id
docher rmi <image-name> //used to remove image from your m/
docker-compose up //used to start services specified in .yaml file
docker-compose down //used to stop service specified in .yaml file"'''

summary = summarizer(text, max_length=230, min_length=130, do_sample=False) #give parmeters to limit summary within given token limits
print(summary)