import React from "react";
import ReactDOM from "react-dom";
import * as tf from '@tensorflow/tfjs';
import "./styles.css";
tf.setBackend('webgl');
let backend = new tf.webgl.MathBackendWebGL()

const pascalvoc = [[ 0,0,0 ],[ 128,0,0 ],[ 0,128,0 ],
                    [ 128,128,0 ],[ 0,0,128 ],[ 128,0,128 ],
                    [ 0,128,128 ],[ 128,128,128 ],[ 64,0,0 ],
                    [ 192,0,0 ],[ 64,128,0 ],[ 192,128,0 ],
                    [ 64,0,128 ],[ 192,0,128 ],[ 64,128,128 ],
                    [ 192,128,128 ],[ 0,64,0 ],[ 128,64,0 ],
                    [ 0,192,0 ],[ 128,192,0 ],[ 0,64,128 ],
                    [ 128,64,128 ],[ 0,192,128 ],[ 128,192,128 ],
                    [ 64,64,0 ],[ 192,64,0 ],[ 64,192,0 ],
                    [ 192,192,0 ],[ 64,64,128 ],[ 192,64,128 ],
                    [ 64,192,128 ],[ 192,192,128 ],[ 0,0,64 ],
                    [ 128,0,64 ],[ 0,128,64 ],[ 128,128,64 ],
                    [ 0,0,192 ],[ 128,0,192 ],[ 0,128,192 ],
                    [ 128,128,192 ],[ 64,0,64 ]];


async function load_model() {
  const model = await tf.loadLayersModel("http://127.0.0.1:8080/model.json");
  return model;
}

const modelPromise = load_model();

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();

  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });
      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model) => {
      const predictions = model.predict(this.process_input(video));
      this.renderPredictions(predictions);
      predictions.dispose();
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
  };

  process_input(video_frame){
    const img = tf.browser.fromPixels(video_frame).toFloat();
    const scale = tf.scalar(255.);
    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normalised = img.div(scale).sub(mean).div(std);
    const batched = normalised.transpose([2,0,1]).expandDims();
    img.dispose();
    scale.dispose();
    mean.dispose();
    std.dispose();
    normalised.dispose();
    batched.dispose();
    return batched;
  };
    renderPredictions = async (predictions) => {
    const img_shape = [480, 480]
    const offset = 0;
    const segmPred = tf.image.resizeBilinear(predictions.transpose([0,2,3,1]),
                                              img_shape);
    const segmMask = segmPred.argMax(3).reshape(img_shape);
    const width = segmMask.shape.slice(0, 1);
    const height = segmMask.shape.slice(1, 2);
    const data = await segmMask.data();
    const bytes = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; ++i) {
      const partId = data[i];
      const j = i * 4;
      if (partId === -1) {
          bytes[j + 0] = 255;
          bytes[j + 1] = 255;
          bytes[j + 2] = 255;
          bytes[j + 3] = 255;
      } else {
          const color = pascalvoc[partId + offset];

          if (!color) {
              throw new Error(`No color could be found for part id ${partId}`);
          }
          bytes[j + 0] = color[0];
          bytes[j + 1] = color[1];
          bytes[j + 2] = color[2];
          bytes[j + 3] = 255;
      }
    }
    const out = new ImageData(bytes, width, height);
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.scale(1.5, 1.5);
    ctx.putImageData(out, 520, 60);
    console.log(tf.memory())
    //console.log('  Tensors:', tf.memory().numTensors);
    backend.dispose()
    backend = new tf.webgl.MathBackendWebGL()
  };

  render() {
    return (
      <div>
        <h1>Real-Time Semantic Segmentation</h1>
        <h3>Refine Net</h3>
        <video
          style={{height: '600px', width: "480px"}}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width= "480"
          height= "480"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="960"
          height="480"
        />
      </div>
    );
  }

}
const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);

