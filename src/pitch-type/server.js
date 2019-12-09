require('@tensorflow/tfjs-node')
// require('@tensorflow/tfjs-node-gpu')

const http = require('http')
const socketio = require('socket.io')
const pitchType = require('./pitchType')

const TIMEOUT_BETWEEN_EPOCHS_MS = 500
const PORT = 8001

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function run () {
  const port = process.env.PORT || PORT
  const server = http.createServer()
  const io = socketio(server)

  server.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`)
  })

  io.on('connection', (socket) => {
    socket.on('predictSample', async (sample) => {
      io.emit('predictResult', await pitchType.predictSample(sample))
    })
  })

  const numTrainingIterations = 10
  for (var i = 0; i < numTrainingIterations; i++) {
    console.log(`Training iteration : ${i + 1} / ${numTrainingIterations}`)
    await pitchType.model.fitDataset(pitchType.trainingData, { epochs: 1 })
    console.log('accuracyPerClass', await pitchType.evaluate(true))
    await sleep(TIMEOUT_BETWEEN_EPOCHS_MS)
  }

  io.emit('trainingComplete', true)
}

run()
