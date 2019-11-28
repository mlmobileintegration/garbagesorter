from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import base64, sys, numpy as np

#import os
import tensorflow as tf
import sys
import cv2
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print("Tensorflow Version : ", tf.__version__)

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

path = Path(__file__).parent
#model_file_url = 'YOUR MODEL.h5 DIRECT / RAW DOWNLOAD URL HERE!'
#model_file_name = 'model'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

IMG_FILE_SRC = '/tmp/saved_image.png'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/app/models/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/app/models/labelmap.pbtxt'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    # await download_file(model_file_url, MODEL_PATH)
    # model = load_model(MODEL_PATH) # Load your Custom trained model
    # model._make_predict_function()
    #model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
#loop = asyncio.get_event_loop()
#tasks = [asyncio.ensure_future(setup_model())]
#model = loop.run_until_complete(asyncio.gather(*tasks))[0]
#loop.close()

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = data["img"]
    bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
    return predict(IMG_FILE_SRC)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)      

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:   
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def run_inference_for_single_image(image):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    
    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def predict(image_path):

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.                      

    output_dict = run_inference_for_single_image(image_np_expanded)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4)

    # Save the file into the predictedimages folder for later use
    cv2.imwrite("/app/static/images/predicted.png", image_np)
    predicted = path/'static'/'predict.html'
    return HTMLResponse(predicted.open().read())



if __name__ == "__main__":

    ##################################################
    # Tensorflow part
    ##################################################    
    print("Frozen Path", PATH_TO_FROZEN_GRAPH)
    print("Label Path", PATH_TO_LABELS)
    print('Loading the model')
    detection_graph = load_graph(PATH_TO_FROZEN_GRAPH)

    print('Loading the label map')    
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print(category_index)

    # Get handles to input and output tensors
    tensor_dict = {}
    with detection_graph.as_default():
        with tf.compat.v1.Session() as findtensor:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
    
    print(tensor_dict)

    print('Starting Session')
    sess = tf.compat.v1.Session(graph=detection_graph)

    ##################################################
    # END Tensorflow part
    ##################################################

    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)

#docker build . -t oapi -f Dockerfile
#docker run -p 8080:8080 -t oapi