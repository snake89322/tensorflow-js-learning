import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData () {
  const carsDataReq = await window.fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
  const carsData = await carsDataReq.json()
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower
  }))
    .filter(car => (car.mpg != null && car.horsepower != null))

  return cleaned
}

async function run () {
  const data = await getData()
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }))

  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )

  const model = createModel()
  tfvis.show.modelSummary({
    name: 'Model Summary'
  }, model)

  const tensorData = convertToTensor(data)
  const { inputs, labels } = tensorData

  // Train the model
  await trainModel(model, inputs, labels)

  testModel(model, data, tensorData)
}

document.addEventListener('DOMContentLoaded', run)

function createModel () {
  // Create a squential model
  const model = tf.sequential()

  // Add a signle hidden layer
  model.add(tf.layers.dense({
    inputShape: [1],
    units: 1,
    useBias: true
  }))

  // try to add a sigmoid
  // 原文使用的 sigmoid 不是例子的最终效果，主要取决于激活函数
  // 使用 relu 函数可以达到预期效果
  model.add(tf.layers.dense({
    units: 50,
    activation: 'relu'
  }))

  // Add an output layer
  model.add(tf.layers.dense({
    units: 1,
    useBias: true
  }))

  return model
}

function convertToTensor (data) {
  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data)

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)

    // Here we have inputs.
    // length examples and each example has 1 input feature (the horsepower).
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    // https://developers.google.com/machine-learning/data-prep/transform/normalization
    // 归一化数据至 0～1 有时候是 -1～1
    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }
  })
}

function trainModel (model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    // https://developers.google.com/machine-learning/glossary/#MSE
    metrics: ['mse']
  })

  const batchSize = 32
  const epochs = 50

  return model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  })
}

function testModel (model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = model.predict(xs.reshape([100, 1]))

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin)

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin)

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()]
  })

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {
      x: val,
      y: preds[i]
    }
  })

  const originalPoints = inputData.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }))

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted']
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
}
