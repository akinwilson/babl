# Babl webserver application 

## Overview 
The code accompanying this folder is a django webserver application which allows endusers to interact with a selection of large language models; [T5](https://en.wikipedia.org/wiki/T5_(language_model)), [LLaMA](https://en.wikipedia.org/wiki/Llama_language_model) and 
[Bloom](https://en.wikipedia.org/wiki/BLOOM_(language_model))


## Usage 

**NOTE**
<br>
It is best suited to run the webserver in tadem with at least the `../serve` application since requests from the frontend will be made to the serving service API. This can be achieved by editing the `docker-compose` file in the directory above. Having said that, hotreloading changes to the webserver source code is currently not working. 
I need to follow [this blog](https://docs.appseed.us/technologies/django/docker-auto-reload/). I believe simply mounting the source directory inside the container should allow for the hotreloading, but I have not read the blog

To run the server in development mode
```
python manage.py runserver 127.0.0.1:8080 
```


TO DO 21 Jan 2025:
- [ ] Frontend development: checkout [this video](https://www.youtube.com/watch?v=_sxoqRIbW0c) and walk through it step by step integrating it into the frontend part of this application 

