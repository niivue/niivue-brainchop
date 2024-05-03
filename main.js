import { Niivue } from "@niivue/niivue"
import {runInference } from './brainchop-mainthread.js'
import { inferenceModelsList, brainChopOpts } from "./brainchop-parameters.js"
import { isChrome, localSystemDetails } from "./brainchop-telemetry.js"
import MyWorker from "./brainchop-webworker.js?worker"

async function main() {
  smoothCheck.onchange = function () {
    nv1.setInterpolation(!smoothCheck.checked)
  }
  aboutBtn.onclick = function () {
    window.alert("BrainChop models https://github.com/neuroneural/brainchop")
  }
  diagnosticsBtn.onclick = function () {
    if (diagnosticsString.length < 1) {
      window.alert('No diagnostic string generated: run a model to create diagnostics')
      return
    }
    navigator.clipboard.writeText(diagnosticsString)
    window.alert('Diagnostics copied to clipboard\n' + diagnosticsString)
  }
  opacitySlider0.oninput = function () {
    nv1.setOpacity(0, opacitySlider0.value / 255)
    nv1.updateGLVolume()
  }
  opacitySlider1.oninput = function () {
    nv1.setOpacity(1, opacitySlider1.value / 255)
  }
  async function ensureConformed() {
    let nii = nv1.volumes[0]
    let isConformed = ((nii.dims[1] === 256) && (nii.dims[2] === 256) && (nii.dims[3] === 256))
    if ((nii.permRAS[0] !== -1) || (nii.permRAS[1] !== 3) || (nii.permRAS[2] !== -2))
      isConformed = false
    if (isConformed)
      return
    let nii2 = await nv1.conform(nii, false)
    nv1.removeVolume(nv1.volumes[0])
    nv1.addVolume(nii2)
  }
  async function closeAllOverlays() {
    while (nv1.volumes.length > 1) {
      await nv1.removeVolume(nv1.volumes[1])
    }
  }
  modelSelect.onchange = async function () {
    if (this.selectedIndex < 0)
      modelSelect.selectedIndex = 11
    await closeAllOverlays()
    await ensureConformed()
    let model = inferenceModelsList[this.selectedIndex]
    let opts = brainChopOpts
    opts.rootURL = location.href
    const isLocalhost = Boolean(
      window.location.hostname === 'localhost' ||
      // [::1] is the IPv6 localhost address.
      window.location.hostname === '[::1]' ||
      // 127.0.0.1/8 is considered localhost for IPv4.
      window.location.hostname.match(
          /^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/
      )
    )
    if (isLocalhost)
      opts.rootURL = location.protocol + '//' + location.host
    if (workerCheck.checked) {
      if(typeof(chopWorker) !== "undefined") {
          console.log('Unable to start new segmentation: previous call has not completed')
          return
      }
      chopWorker = await new MyWorker({ type: "module" })
      let hdr = {datatypeCode: nv1.volumes[0].hdr.datatypeCode, dims: nv1.volumes[0].hdr.dims}
      let msg = {opts:opts, modelEntry: model, niftiHeader: hdr, niftiImage: nv1.volumes[0].img}
      chopWorker.postMessage(msg)
      chopWorker.onmessage = function(event) {
        let cmd = event.data.cmd
        if (cmd === 'ui') {
            if (event.data.modalMessage !== "") {
              chopWorker.terminate()
              chopWorker = undefined
            }
            callbackUI(event.data.message, event.data.progressFrac, event.data.modalMessage, event.data.statData)
        }
        if (cmd === 'img') {
            chopWorker.terminate()
            chopWorker = undefined
            callbackImg(event.data.img, event.data.opts, event.data.modelEntry)
        }
      }
    } else {
      runInference(opts, model, nv1.volumes[0].hdr, nv1.volumes[0].img, callbackImg, callbackUI)
    }
  }
  saveBtn.onclick = function () {
    nv1.volumes[1].saveToDisk("Custom.nii")
  }
  sceneBtn.onclick = function () {
    nv1.saveDocument("niivue.brainchop.nvd")
  }
  workerCheck.onchange = function () {
    modelSelect.onchange()
  }
  clipCheck.onchange = function () {
    if (clipCheck.checked) {
      nv1.setClipPlane([0, 0, 90])
    } else {
      nv1.setClipPlane([2, 0, 90])
    }
  }
  function doLoadImage() {
    opacitySlider0.oninput()
  }
  async function fetchJSON(fnm) {
    const response = await fetch(fnm)
    const js = await response.json()
    return js
  }
  async function callbackImg(img, opts, modelEntry) {
    closeAllOverlays()
    let overlayVolume = await nv1.volumes[0].clone()
    overlayVolume.zeroImage()
    overlayVolume.hdr.scl_inter = 0
    overlayVolume.hdr.scl_slope = 1
    overlayVolume.img = new Uint8Array(img)
    if (modelEntry.colormapPath) {
      let cmap = await fetchJSON(modelEntry.colormapPath)
      overlayVolume.setColormapLabel(cmap)
      // n.b. most models create indexed labels, but those without colormap mask scalar input
      overlayVolume.hdr.intent_code = 1002 // NIFTI_INTENT_LABEL
    } else {
      let colormap = opts.atlasSelectedColorTable.toLowerCase()
      const cmaps = nv1.colormaps()
      if (!cmaps.includes(colormap)) {
        colormap = 'actc'
      }
      overlayVolume.colormap = colormap
    }
    overlayVolume.opacity = opacitySlider1.value / 255
    await nv1.addVolume(overlayVolume)
  }
  async function reportTelemetry(statData) {
    if (typeof statData === 'string' || statData instanceof String) {
      function strToArray(str) {
        const list = JSON.parse(str)
        const array = []
        for (const key in list) {
            array[key] = list[key]
        }
        return array
      }
      statData = strToArray(statData)
    }
    statData = await localSystemDetails(statData, nv1.gl)
    diagnosticsString = ':: Diagnostics can help resolve issues https://github.com/neuroneural/brainchop/issues ::\n'
    for (var key in statData){
      diagnosticsString +=  key + ': ' + statData[key]+'\n'
    }
  }
  function callbackUI(message = "", progressFrac = -1, modalMessage = "", statData = []) {
    if (message !== "") {
      console.log(message)
      document.getElementById("location").innerHTML = message
    }
    if (isNaN(progressFrac)) { //memory issue
      memstatus.style.color = "red"
      memstatus.innerHTML = "Memory Issue"
    } else if (progressFrac >= 0) {
      modelProgress.value = progressFrac * modelProgress.max
    }
    if (modalMessage !== "") {
      window.alert(modalMessage)
    }
    if (Object.keys(statData).length > 0) {
      reportTelemetry(statData)
    }
  }
  function handleLocationChange(data) {
    document.getElementById("location").innerHTML = "&nbsp;&nbsp;" + data.string
  }
  let defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    show3Dcrosshair: true,
    onLocationChange: handleLocationChange,
  }
  var diagnosticsString = ''
  var chopWorker
  let nv1 = new Niivue(defaults)
  nv1.attachToCanvas(gl1)
  nv1.opts.dragMode = nv1.dragModes.pan
  nv1.opts.multiplanarForceRender = true
  nv1.opts.yoke3Dto2DZoom = true
  nv1.opts.crosshairGap = 11
  smoothCheck.onchange()
  await nv1.loadVolumes([{ url: "./t1_crop.nii.gz" }])
  for (let i = 0; i < inferenceModelsList.length; i++) {
    var option = document.createElement("option")
    option.text = inferenceModelsList[i].modelName
    option.value = inferenceModelsList[i].id.toString()
    modelSelect.appendChild(option)
  }
  nv1.onImageLoaded = doLoadImage
  modelSelect.selectedIndex = -1
  workerCheck.checked = await isChrome() //TODO: Safari does not yet support WebGL TFJS webworkers, test FireFox
  // uncomment next two lines to automatically run segmentation when web page is loaded
  //   modelSelect.selectedIndex = 11
  //   modelSelect.onchange()

}

main()