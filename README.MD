


# Realtime Semantic Segmentation in the browser

 To a more detailed description about this project, [check my article](https://lnkd.in/dFm95MJ) at Towards Data Science.

This project uses TensorFlow.js to perform realtime instance segmentation in the browser using [RefineNet](https://arxiv.org/pdf/1611.06612.pdf). Many thanks to  [Bernardo de Oliveira](https://github.com/bernieOllie), that help me to fix some bugs in the code, and [Vladimir Nekrasov](https://drsleep.github.io/) that provided the *.js weights* of the [Light-Weight RefineNet](http://bmvc2018.org/contents/papers/0494.pdf) - a more compact implementation of RefineNet, suitable for tasks requiring real-time performance.

  
  ![enter image description here](./git_media/objects_demo.gif)
  
  

### Getting started
Tensorflow.js models have to be served through an URL to be loaded in javascript. The  Refine Net model is based on a main JSON file and shared weights files in a binary format. The idea is to create an HTTP server that will make the model available in a URL allowing requests and be thread as REST API. When loading the model, the TensorFlow.js will do the following requests:

    GET /model.json
    GET /group1-shard1of4
    GET /group1-shard2of4
    GET /group1-shard3of4
    GET /group1-shard4of4

The [http-server](https://www.npmjs.com/package/http-server) can be installed using the following command:

`npm install http-server -g`

Once you have the program installed, go to the weights folder, run the command below and the model you be available at http://127.0.0.1:8080.

`http-server -c1 --cors .`

To install the required packages, it can be used `npm` or `yarn`. 

If you are using  `yarn`:

	yarn
	yarn start

If you are using  `npm`:

	npm install
	npm start

To test the project just go to  http://localhost:3000/ and the application will be running there.

### References
[1] Romera-Paredes, Bernardino, and Philip Hilaire Sean Torr. “Recurrent instance segmentation.” _European conference on computer vision_. Springer, Cham, 2016.

[2] Lin, Guosheng, et al. “Refinenet: Multi-path refinement networks for high-resolution semantic segmentation.” _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2017.

[3] Nekrasov, Vladimir, Chunhua Shen, and Ian Reid. “Light-weight refinenet for real-time semantic segmentation.” _arXiv preprint arXiv:1810.03272_ (2018).

[4] Medium. 2020. _Introducing Tensorflow.Js: Machine Learning In Javascript_. [online] Available at: <https://medium.com/tensorflow/introducing-tensorflow-js-machine-learning-in-javascript-bf3eab376db> [Accessed 3 June 2020].

[5] [https://www.tensorflow.org/js/tutorials/conversion/import_keras](https://www.tensorflow.org/js/tutorials/conversion/import_keras)

[6] [https://github.com/DrSleep/light-weight-refinenet](https://github.com/DrSleep/light-weight-refinenet)

[7] [https://www.tensorflow.org/js/guide/conversion](https://www.tensorflow.org/js/guide/conversion)
