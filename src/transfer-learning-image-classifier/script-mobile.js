import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as knnClassifier from '@tensorflow-models/knn-classifier'

let net
const webcamElement = document.getElementById('webcam')
const classifier = knnClassifier.create()

async function app () {
  console.log('loading mobilenet...')

  // load the model
  net = await mobilenet.load()
  console.log('Successfully loaded model')

  const webcam = await tf.data.webcam(webcamElement)

  const addExample = async classId => {
    const img = await webcam.capture()

    const activation = net.infer(img, 'conv_preds')
    activation.print(true)

    classifier.addExample(activation, classId)

    img.dispose()
  }

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0))
  document.getElementById('class-b').addEventListener('click', () => addExample(1))
  document.getElementById('class-c').addEventListener('click', () => addExample(2))

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture()

      const activation = net.infer(img, 'conv_preds')

      const result = await classifier.predictClass(activation)

      const classes = ['A', 'B', 'C']

      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `

      // Dispose the tensor to release the memory.
      img.dispose()
    }

    await tf.nextFrame()
  }
}

app()
