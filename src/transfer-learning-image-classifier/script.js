import * as mobilenet from '@tensorflow-models/mobilenet'

let net

async function app () {
  console.log('loading mobilent..')

  net = await mobilenet.load()
  console.log('Successfully loaded moedl')

  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('img')
  const result = await net.classify(imgEl)
  console.log(result)
}

app()
