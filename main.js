import { Niivue } from "@niivue/niivue"
import { runInference, inferenceModelsList, brainChopOpts } from "./brainchop.js"

async function main() {
  let defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    show3Dcrosshair: true,
    onLocationChange: handleLocationChange,
  }
  let nv1 = new Niivue(defaults)
  nv1.attachToCanvas(gl1)
  nv1.opts.dragMode = nv1.dragModes.pan
  nv1.opts.multiplanarForceRender = true
  nv1.opts.yoke3Dto2DZoom = true
  await nv1.loadVolumes([{ url: "./t1_crop.nii.gz" }])

  aboutBtn.onclick = function () {
    window.alert("BrainChop models https://github.com/neuroneural/brainchop")
  }
  opacitySlider.oninput = function () {
    nv1.setOpacity(1, opacitySlider.value / 255)
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

  modelSelect.onchange = async function () {
    await ensureConformed()
    let model = inferenceModelsList[this.selectedIndex]
    let opts = brainChopOpts
    runInference(opts, model, nv1.volumes[0].hdr, nv1.volumes[0].img, callbackImg, callbackUI)
  }

  saveBtn.onclick = function () {
    nv1.volumes[1].saveToDisk("Custom.nii")
  }

  async function callbackImg(img, opts, modelEntry) {

    while (nv1.volumes.length > 1) {
      nv1.removeVolume(nv1.volumes[1])
    }

    let overlayVolume = await nv1.volumes[0].clone()
    overlayVolume.zeroImage()
    overlayVolume.hdr.scl_inter = 0
    overlayVolume.hdr.scl_slope = 1
    overlayVolume.img = new Uint8Array(img)
    let colormap = opts.atlasSelectedColorTable.toLowerCase()
    const cmaps = nv1.colormaps()
    if (!cmaps.includes(colormap))
      colormap = 'actc'
    overlayVolume.colormap = colormap
    overlayVolume.opacity = opacitySlider.value / 255
    nv1.addVolume(overlayVolume)
  }
  function callbackUI(message = "", progressFrac = -1, modalMessage = "") {
    console.log(message)
    document.getElementById("location").innerHTML = message
    if (isNaN(progressFrac)) { //memory issue
      memstatus.style.color = "red"
      memstatus.innerHTML = "Memory Issue"
    } else if (progressFrac >= 0) {
      modelProgress.value = progressFrac * modelProgress.max
    }
    if (modalMessage !== "") {
      window.alert(modalMessage)
    }
  }
  function handleLocationChange(data) {
    document.getElementById("location").innerHTML = "&nbsp;&nbsp;" + data.string
  }
  for (let i = 0; i < inferenceModelsList.length; i++) {
    var option = document.createElement("option");
    option.text = inferenceModelsList[i].modelName
    option.value = inferenceModelsList[i].id.toString()
    modelSelect.appendChild(option);
  }
  modelSelect.selectedIndex = -1;
}

main()