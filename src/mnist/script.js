import { MnistData } from './data.js'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

async function showExamples (data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' })

  // Get the examples
  const examples = data.nextTestBatch(20)
  const numExamples = examples.xs.shape[0]

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1])
    })

    const canvas = document.createElement('canvas')
    canvas.width = 28
    canvas.height = 28
    canvas.style = 'margin: 4px;'
    await tf.browser.toPixels(imageTensor, canvas)
    surface.drawArea.appendChild(canvas)

    imageTensor.dispose()
  }
}

async function run () {
  const data = new MnistData()
  await data.load()
  await showExamples(data)

  const model = getModel()
  tfvis.show.modelSummary({ name: 'Model Architecture' }, model)

  await train(model, data)

  await showAccuracy(model, data)
  await showConfusion(model, data)
}

document.addEventListener('DOMContentLoaded', run)

// 正式开始 tf
function getModel () {
  const model = tf.sequential()

  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
  const IMAGE_CHANNELS = 1

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    // inputShape ((null | number)[]) If defined,
    // will be used to create an input layer to insert before this layer.
    // If both inputShape and batchInputShape are defined,
    // batchInputShape will be used.
    // This argument is only applicable to input layers (the first layer of a model).
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    // kernelSize (number|number[]) The dimensions of the convolution window.
    // If kernelSize is a number, the convolutional window will be square.
    // NOTE 这玩意儿也得调
    kernelSize: 5,
    // filters (number) The dimensionality of the output space
    // (i.e. the number of filters in the convolution).
    // The number of filter windows of size kernelSize to apply to the input data.
    // Here, we will apply 8 filters to the data.
    // 卷积和的次数, 卷积窗的算子是随机的，所以每次都能得到一个新的值，这里的次数就是这个意思
    filters: 8,
    // strides (number|number[]) The strides of the convolution in each dimension.
    // If strides is a number, strides in both dimensions are equal.
    // The "step size" of the sliding window—i.e.,
    // how many pixels the filter will shift each time it moves over the image.
    // Here, we specify strides of 1,
    // which means that the filter will slide over the image in steps of 1 pixel.
    strides: 1,
    // 激活函数 卷积完成之后得到一个值，用函数输出值
    // 函数是控制输入大小，函数是可导
    // 典型 sigmoid，relu，主要看应用场景 还有 tanh，某些场景应用会更好
    // relu 更适用于图像
    activation: 'relu',
    // TODO 方差归一化，具体啥作用还不清楚
    kernelInitializer: 'varianceScaling'
  }))

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  // 池化 averagePooling2d，maxPooling2d
  // 取平均值，取最大值
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }))
  model.add(tf.layers.averagePooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))

  // TODO 对输入的标准图片特征进行相似性度量，看一下训练效果

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten())
  // 这里的输出是一个标准的全联接神经网络，展开后的每一个值是一个神经元
  // 最后输出10个神经元，每一个神经元链接到分类的权制

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  // 分类上 90% 以上都用 softmax，概率函数
  const NUM_OUTPUT_CLASSES = 10
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }))

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  // 准确率 和 代价函数，训练目的
  // loss function，误差函数，训练过程误差越来越小，就是 ok 的，这个差值注意⚠️，是所有训练样本与对应标签差值的均值（有时候是方差均值，信息论中墒的评价，反正就是各种数学度量）
  // 当 loss function 最小时，把 全联接权制记录下来，卷积窗的核的值记录下来
  // accuracy 是在 predict 用的
  const optimizer = tf.train.adam()
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}

async function train (model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc']
  const container = {
    name: 'Model Training',
    styles: {
      height: '1000px'
    }
  }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 5500
  const TEST_DATA_SIZE = 1000

  // Using this method helps avoid memory leaks.
  // In general, wrap calls to operations in tf.tidy() for automatic memory cleanup.
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    // 迭代次数，以loss function 为衡量指标，这个值取1 对整体没有什么意义
    // 现在无法解决，迭代多少次能达到最小，所以这个值是人调的
    // NOTE 调就这个玩意儿
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  })
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

function doPrediction (model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
  const testData = data.nextTestBatch(testDataSize)
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
  // Notably the argmax function is what
  // gives us the index of the highest probability class.
  // Remember that the model outputs a probability for each class.
  // Here we find out the highest probability and assign use that as the prediction.
  const labels = testData.labels.argMax([-1])
  const preds = model.predict(testxs).argMax([-1])

  testxs.dispose()
  return [preds, labels]
}

async function showAccuracy (model, data) {
  const [preds, labels] = doPrediction(model, data)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
  const container = { name: 'Accuracy', tab: 'Evaluation' }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)

  labels.dispose()
}

async function showConfusion (model, data) {
  const [preds, labels] = doPrediction(model, data)
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
  const container = { name: 'Confusion Matrix', tab: 'Evaluation' }
  tfvis.render.confusionMatrix(
    container, { values: confusionMatrix }, classNames)

  labels.dispose()
}
