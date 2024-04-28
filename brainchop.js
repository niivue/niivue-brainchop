import * as tf from '@tensorflow/tfjs'
import { BWLabeler } from './bwlabels.js'
export { runInference, inferenceModelsList, brainChopOpts }

const brainChopOpts = {
  // General settings for input shape [batchSize, batch_D, batch_H, batch_W, numOfChan]
  batchSize: 1, // How many batches are used during each inference iteration
  numOfChan: 1, // num of channel of the input shape
  isColorEnable: true, // If false, grey scale will enabled
  isAutoColors: true, // If false, manualColorsRange will be in use
  bgLabelValue: 0, // Semenatic Segmentation background label value
  drawBoundingVolume: false, // plot bounding volume used to crop the brain
  isBrainCropMaskBased: true, // Check if brain masking will be used for cropping & optional show or brain tissue will be used
  showPhase1Output: false, // This will load to papaya the output of phase-1 (ie. brain mask or brain tissue)
  isPostProcessEnable: true, // If true 3D Connected Components filter will apply
  isContoursViewEnable: false, // If true 3D contours of the labeled regions will apply
  browserArrayBufferMaxZDim: 30, // This value depends on Memory available
  telemetryFlag: false, // Ethical and transparent collection of browser usage while adhering to security and privacy standards
  chartXaxisStepPercent: 10, // percent from total labels on Xaxis
  uiSampleName: 'BC_UI_Sample', // Sample name used by interface
  atlasSelectedColorTable: 'Fire' // Select from ["Hot-and-Cold", "Fire", "Grayscale", "Gold", "Spectrum"]
}

// Inference Models, the ids must start from 1 in sequence
const inferenceModelsList = [
  {
    id: 1,
    type: 'Segmentation',
    path: './models/model5_gw_ae/model.json',
    modelName: '\u26A1 Tissue GWM (light)',
    labelsPath: './models/model5_gw_ae/labels.json',
    colorsPath: './models/model5_gw_ae/colorLUT.json',
    colormapPath: './models/model5_gw_ae/colormap3.json',
    preModelId: null, // Model run first e.g.  crop the brain   { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 18, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning: null, // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Gray and white matter segmentation model. Operates on full T1 image in a single pass, but uses only 5 filters per layer. Can work on integrated graphics cards but is barely large enough to provide good accuracy. Still more accurate than the subvolume model.'
  },
  {
    id: 2,
    type: 'Segmentation',
    path: './models/model20chan3cls/model.json',
    modelName: '\u{1F52A} Tissue GWM (High Acc)',
    labelsPath: './models/model20chan3cls/labels.json',
    colorsPath: './models/model20chan3cls/colorLUT.json',
    colormapPath: './models/model20chan3cls/colormap.json',
    preModelId: null, // Model run first e.g.  crop the brain   { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0.2, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: true, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.",
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Gray and white matter segmentation model. Operates on full T1 image in a single pass but needs a dedicated graphics card to operate. Provides the best accuracy with hard cropping for better speed'
  },
  {
    id: 3,
    type: 'Segmentation',
    path: './models/model20chan3cls/model.json',
    modelName: '\u{1F52A} Tissue GWM (High Acc, Low Mem)',
    labelsPath: './models/model20chan3cls/labels.json',
    colorsPath: './models/model20chan3cls/colorLUT.json',
    colormapPath: './models/model20chan3cls/colormap.json',
    preModelId: null, // Model run first e.g.  crop the brain   { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0.2, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: true, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.",
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Gray and white matter segmentation model. Operates on full T1 image in a single pass but needs a dedicated graphics card to operate. Provides high accuracy and fit low memory available but slower'
  },
  {
    id: 4,
    type: 'Atlas',
    path: './models/model30chan18cls/model.json',
    modelName: '\u{1FA93} Subcortical + GWM (High Mem, Fast)',
    labelsPath: './models/model30chan18cls/labels.json',
    colorsPath: './models/model30chan18cls/colorLUT.json',
    colormapPath: './models/model30chan18cls/colormap.json',
    preModelId: null, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0.2, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Parcellation of the brain into 17 regions: gray and white matter plus subcortical areas. This is a robust model able to handle range of data quality, including varying saturation, and even clinical scans. It may work on infant brains, but your mileage may vary.'
  },
  {
    id: 5,
    type: 'Atlas',
    path: './models/model30chan18cls/model.json',
    modelName: '\u{1FA93} Subcortical + GWM (Low Mem, Slow)',
    labelsPath: './models/model30chan18cls/labels.json',
    colorsPath: './models/model30chan18cls/colorLUT.json',
    colormapPath: './models/model30chan18cls/colormap.json',
    preModelId: null, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0.2, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Parcellation of the brain into 17 regions: gray and white matter plus subcortical areas. This is a robust model able to handle range of data quality, including varying saturation, and even clinical scans. It may work on infant brains, but your mileage may vary.'
  },
  {
    id: 6,
    type: 'Atlas',
    path: './models/model18cls/model.json',
    modelName: '\u{1FA93} Subcortical + GWM (Low Mem, Faster)',
    labelsPath: './models/model18cls/labels.json',
    colorsPath: './models/model18cls/colorLUT.json',
    colormapPath: './models/model18cls/colormap.json',
    preModelId: null, // model run first e.g.  Brain_Extraction  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0.2, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Parcellation of the brain into 17 regions: gray and white matter plus subcortical areas. This is a robust model able to handle range of data quality, including varying saturation, and even clinical scans. It may work on infant brains, but your mileage may vary.'
  },
  {
    id: 7,
    type: 'Atlas',
    path: './models/model30chan18cls/model.json',
    modelName: '\u{1F52A}\u{1FA93} Subcortical + GWM (Failsafe, Less Acc)',
    labelsPath: './models/model30chan18cls/labels.json',
    colorsPath: './models/model30chan18cls/colorLUT.json',
    colormapPath: './models/model30chan18cls/colormap.json',
    preModelId: 1, // model run first e.g.  Brain_Extraction  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Parcellation of the brain into 17 regions: gray and white matter plus subcortical areas. This is not a robust model, it may work on low data quality, including varying saturation, and even clinical scans. It may work also on infant brains, but your mileage may vary.'
  },
  {
    id: 8,
    type: 'Atlas',
    path: './models/model30chan50cls/model.json',
    modelName: '\u{1F52A} Aparc+Aseg 50 (High Mem, Fast)',
    labelsPath: './models/model30chan50cls/labels.json',
    colorsPath: './models/model30chan50cls/colorLUT.json',
    colormapPath: './models/model30chan50cls/colormap.json',
    preModelId: 1, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: true, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'This is a 50-class model, that segments the brain into the Aparc+Aseg Freesurfer Atlas but one where cortical homologues are merged into a single class.'
  },
  {
    id: 9,
    type: 'Atlas',
    path: './models/model30chan50cls/model.json',
    modelName: '\u{1F52A} Aparc+Aseg 50 (Low Mem, Slow)',
    labelsPath: './models/model30chan50cls/labels.json',
    colorsPath: './models/model30chan50cls/colorLUT.json',
    colormapPath: './models/model30chan50cls/colormap.json',
    preModelId: 1, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: true, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last laye
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'This is a 50-class model, that segments the brain into the Aparc+Aseg Freesurfer Atlas but one where cortical homologues are merged into a single class. The model use sequential convolution for inference to overcome browser memory limitations but leads to longer computation time.'
  },
  // './models/model5_gw_ae/colorLUT.json',
  {
    id: 10,
    type: 'Brain_Extraction',
    path: './models/model5_gw_ae/model.json',
    modelName: '\u26A1 Extract the Brain (FAST)',
    labelsPath: null,
    colorsPath: null,
    preModelId: null, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 18, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning: null, // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Extract the brain fast model operates on full T1 image in a single pass, but uses only 5 filters per layer. Can work on integrated graphics cards but is barely large enough to provide good accuracy. Still more accurate than the failsafe version.'
  },
  {
    id: 11,
    type: 'Brain_Extraction',
    path: './models/model11_gw_ae/model.json',
    modelName: '\u{1F52A} Extract the Brain (High Acc, Slow)',
    labelsPath: null,
    colorsPath: null,
    preModelId: null, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.",
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'Extract the brain high accuracy model operates on full T1 image in a single pass, but uses only 11 filters per layer. Can work on dedicated graphics cards. Still more accurate than the fast version.'
  },
  {
    id: 12,
    type: 'Brain_Masking',
    path: './models/model5_gw_ae/model.json',
    modelName: '\u26A1 Brain Mask (FAST)',
    labelsPath: null,
    colorsPath: null,
    colormapPath: './models/model5_gw_ae/colormap.json',
    preModelId: null, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 17, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning: null, // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'This fast masking model operates on full T1 image in a single pass, but uses only 5 filters per layer. Can work on integrated graphics cards but is barely large enough to provide good accuracy. Still more accurate than failsafe version.'
  },
  {
    id: 13,
    type: 'Brain_Masking',
    path: './models/model11_gw_ae/model.json',
    modelName: '\u{1F52A} Brain Mask (High Acc, Low Mem)',
    labelsPath: null,
    colorsPath: null,
    preModelId: null, // Model run first e.g.  crop the brain  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 0, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: true, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.",
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'This masking model operates on full T1 image in a single pass, but uses 11 filters per layer. Can work on dedicated graphics cards. Still more accurate than fast version.'
  },
  {
    id: 14,
    type: 'Atlas',
    path: './models/model21_104class/model.json',
    modelName: '\u{1F52A} Aparc+Aseg 104 (High Mem, Fast)',
    labelsPath: './models/model21_104class/labels.json',
    colorsPath: './models/model21_104class/colorLUT.json',
    colormapPath: './models/model21_104class/colormap.json',
    preModelId: 1, // model run first e.g.  Brain_Extraction  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: false, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'FreeSurfer aparc+aseg atlas 104 parcellate brain areas into 104 regions. It contains a combination of the Desikan-Killiany atlas for cortical area and also segmentation of subcortical regions.'
  },
  {
    id: 15,
    type: 'Atlas',
    path: './models/model21_104class/model.json',
    modelName: '\u{1F52A} Aparc+Aseg 104 (Low Mem, Slow)',
    labelsPath: './models/model21_104class/labels.json',
    colorsPath: './models/model21_104class/colorLUT.json',
    colormapPath: './models/model21_104class/colormap.json',
    preModelId: 1, // model run first e.g.  Brain_Extraction  { null, 1, 2, ..  }
    preModelPostProcess: false, // If true, perform postprocessing to remove noisy regions after preModel inference generate output.
    isBatchOverlapEnable: false, // create extra overlap batches for inference
    numOverlapBatches: 200, // Number of extra overlap batches for inference
    enableTranspose: true, // Keras and tfjs input orientation may need a tranposing step to be matched
    enableCrop: true, // For speed-up inference, crop brain from background before feeding to inference model to lower memory use.
    cropPadding: 0, // Padding size add to cropped brain
    autoThreshold: 0, // Threshold between 0 and 1, given no preModel and tensor is normalized either min-max or by quantiles. Will remove noisy voxels around brain
    enableQuantileNorm: false, // Some models needs Quantile Normaliztion.
    filterOutWithPreMask: false, // Can be used to multiply final output with premodel output mask to crean noisy areas
    enableSeqConv: true, // For low memory system and low configuration, enable sequential convolution instead of last layer
    textureSize: 0, // Requested Texture size for the model, if unknown can be 0.
    warning:
      "This model may need dedicated graphics card.  For more info please check with Browser Resources <i class='fa fa-cogs'></i>.", // Warning message to show when select the model.
    inferenceDelay: 100, // Delay in ms time while looping layers applying.
    description:
      'FreeSurfer aparc+aseg atlas 104 parcellate brain areas into 104 regions. It contains a combination of the Desikan-Killiany atlas for cortical area and also segmentation of subcortical regions. The model use sequential convolution for inference to overcome browser memory limitations but leads to longer computation time. '
  }
] // inferenceModelsList

async function checkZero(timeValue) {
  return timeValue < 10 ? timeValue : '0' + timeValue
}

async function detectBrowser() {
  if (navigator.userAgent.indexOf('OPR/') > -1) {
    return 'Opera'
  } else if (navigator.userAgent.indexOf('Edg/') > -1) {
    return 'Edge'
  } else if (navigator.userAgent.indexOf('Falkon/') > -1) {
    return 'Falkon'
  } else if (navigator.userAgent.indexOf('Chrome/') > -1) {
    return 'Chrome'
  } else if (navigator.userAgent.indexOf('Firefox/') > -1) {
    return 'Firefox'
  } else if (navigator.userAgent.indexOf('Safari/') > -1) {
    return 'Safari'
  } else if (navigator.userAgent.indexOf('MSIE/') > -1 || navigator.userAgent.indexOf('rv:') > -1) {
    return 'IExplorer'
  } else {
    return 'Unknown'
  }
}

async function detectBrowserVersion() {
  if (navigator.userAgent.indexOf('OPR/') > -1) {
    return parseInt(navigator.userAgent.split('OPR/')[1])
  } else if (navigator.userAgent.indexOf('Edg/') > -1) {
    return parseInt(navigator.userAgent.split('Edg/')[1])
  } else if (navigator.userAgent.indexOf('Falkon/') > -1) {
    return parseInt(navigator.userAgent.split('Falkon/')[1])
  } else if (navigator.userAgent.indexOf('Chrome/') > -1) {
    return parseInt(navigator.userAgent.split('Chrome/')[1])
  } else if (navigator.userAgent.indexOf('Firefox/') > -1) {
    return parseInt(navigator.userAgent.split('Firefox/')[1])
  } else if (navigator.userAgent.indexOf('Safari/') > -1) {
    return parseInt(navigator.userAgent.split('Safari/')[1])
  } else if (navigator.userAgent.indexOf('MSIE/') > -1 || navigator.userAgent.indexOf('rv:') > -1) {
    return parseInt(navigator.userAgent.split('MSIE/')[1])
  } else {
    return Infinity
  }
}

async function detectOperatingSys() {
  if (navigator.userAgent.indexOf('Win') > -1) {
    return 'Windows'
  } else if (navigator.userAgent.indexOf('Mac') > -1) {
    return 'MacOS'
  } else if (navigator.userAgent.indexOf('Linux') > -1) {
    return 'Linux'
  } else if (navigator.userAgent.indexOf('UNIX') > -1) {
    return 'UNIX'
  } else {
    return 'Unknown'
  }
}

async function checkWebGl2(callbackUI) {
  const gl = document.createElement('canvas').getContext('webgl2')
  if (!gl) {
    if (typeof WebGL2RenderingContext !== 'undefined') {
      const msg = 'WebGL2 may be disabled. Please try updating video card drivers'
      callbackUI(msg, -1, msg)
    } else {
      console.log('WebGL2 is not supported')
    }
    return false
  } else {
    console.log('WebGl2 is enabled')
    return true
  }
}

async function detectGPUVendor() {
  const gl = document.createElement('canvas').getContext('webgl')
  let debugInfo
  if (gl) {
    debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
    if (debugInfo) {
      const result = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
      // --e.g. : NVIDIA Corporation
      if (result.indexOf('(') > -1 && result.indexOf(')') > -1) {
        return result.substring(result.indexOf('(') + 1, result.indexOf(')'))
      }
      return result
    }
  }
  return null
}

async function detectGPUVendor_v0() {
  const gl = document.createElement('canvas').getContext('webgl')

  if (gl) {
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
    return debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : null
  } else {
    return null
  }
}

async function detectGPUCardType_v0() {
  const gl = document.createElement('canvas').getContext('webgl')
  if (gl) {
    if (detectBrowser() === 'Firefox') {
      // -- return e.g: "GeForce GTX 980/PCIe/SSE2"
      return gl.getParameter(gl.RENDERER)
    }

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
    return debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : null
  } else {
    return null
  }
}

async function detectGPUCardType() {
  const gl = document.createElement('canvas').getContext('webgl')
  let debugInfo

  if (gl) {
    if (detectBrowser() === 'Firefox') {
      // -- return e.g: "GeForce GTX 980/PCIe/SSE2"
      return gl.getParameter(gl.RENDERER)
    }

    debugInfo = gl.getExtension('WEBGL_debug_renderer_info')

    if (debugInfo) {
      let result = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
      // --e.g. : ANGLE (NVIDIA Corporation, GeForce GTX 1050 Ti/PCIe/SSE2, OpenGL 4.5.0 NVIDIA 390.144) as with Chrome
      // Or:  GeForce GTX 1050 Ti/PCIe/SSE2    as with fireFox

      if (result.indexOf('(') > -1 && result.indexOf(')') > -1 && result.indexOf('(R)') === -1) {
        result = result.substring(result.indexOf('(') + 1, result.indexOf(')'))

        if (result.split(',').length === 3) {
          return result.split(',')[1].trim()
        }
      }

      return result
    }
  }
  return null
}

async function getCPUNumCores() {
  return navigator.hardwareConcurrency
}

async function isChrome() {
  return /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor)
}

async function submitTiming2GoogleSheet(dataObj, callbackUI) {
  if (navigator.onLine) {
    const msg = 'Telemetry not yet supported'
    callbackUI(msg, -1, msg)
    console.log(dataObj)

    /*
    // -- Fill form with data to submit
    Object.keys(dataObj).forEach(dataKey =>{
         document.getElementById(dataKey).value = dataObj[dataKey];
    })
    //-- Settings of submission
    const scriptURL = 'https://script.google.com/macros/s/AKfycbwn-Ix6IVGOwUSU1VBU8hFcABT9PqwCwN90UxfK_fXp5CEfxvIoQHZXs2XQRZQo_N8I/exec'
    const form = document.forms['google-sheet']
    //-- Add event handler to the form.
    form.addEventListener('submit', e => {
          e.preventDefault()
          fetch(scriptURL, { method: 'POST', body: new FormData(form)})
            .then(response => console.log("------Done------"))
            .catch(error => console.error('Error!', error.message))
    })
    //-- Submit the form
    document.getElementById("SubmitStatisticalData").click();
    */
  } else {
    console.log(' Offline Mode ')
  }
}

async function getModelNumParameters(modelObj) {
  let numParameters = 0
  for (let layerIdx = 0; layerIdx < modelObj.layers.length; layerIdx++) {
    numParameters += modelObj.layers[layerIdx].countParams()
  }
  return numParameters
}

async function getModelNumLayers(modelObj) {
  return modelObj.layers.length
}

async function isModelChnlLast(modelObj) {
  for (let layerIdx = 0; layerIdx < modelObj.layers.length; layerIdx++) {
    if (modelObj.layersByDepth[layerIdx][0].dataFormat) {
      return modelObj.layersByDepth[layerIdx][0].dataFormat === 'channelsLast'
    }
  }
}

async function load_model(modelUrl) {
  return await tf.loadLayersModel(modelUrl)
}

async function getAllSlices2D(allSlices, slice_height, slice_width) {
  const allSlices_2D = []
  for (let sliceIdx = 0; sliceIdx < allSlices.length; sliceIdx++) {
    allSlices_2D.push(tf.tensor(allSlices[sliceIdx], [slice_height, slice_width]))
  }
  return allSlices_2D
}

async function getSlices3D(allSlices_2D) {
  return tf.stack(allSlices_2D)
}

async function getAllSlicesData1D(num_of_slices, niftiHeader, niftiImage) {
  // Get nifti dimensions
  const cols = niftiHeader.dims[1] // Slice width
  const rows = niftiHeader.dims[2] // Slice height
  let typedData
  if (niftiHeader.datatypeCode === 2) {
    // enum from nvimage/utils DT_UINT8 = 2
    typedData = new Uint8Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 4) {
    // DT_INT16 = 4
    typedData = new Int16Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 8) {
    // DT_INT32 = 8
    typedData = new Int32Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 16) {
    // DT_FLOAT32 = 16
    typedData = new Float32Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 64) {
    // DT_FLOAT64 = 64
    typedData = new Float64Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 256) {
    // DT_INT8 = 256
    typedData = new Int8Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 512) {
    // DT_UINT16 = 512
    typedData = new Uint16Array(niftiImage)
  } else if (niftiHeader.datatypeCode === 768) {
    // DT_UINT32 = 768
    typedData = new Uint32Array(niftiImage)
  } else {
    return
  }
  const allSlices = []
  let offset3D = 0
  // Draw pixels
  for (let slice = 0; slice < num_of_slices; slice++) {
    const slice = new Array(rows * cols)
    let offset2D = 0
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const value = typedData[offset3D++]
        // Create 1Dim Array of pixel value, this 1 dim represents one channel
        slice[offset2D++] = value & 0xff
      }
    }
    allSlices.push(slice)
  }
  return allSlices
}

async function calculateQuantiles(tensor, lowerQuantile = 0.01, upperQuantile = 0.99) {
  // Flatten the tensor
  const flatTensor = tensor.flatten()

  // Convert the flattened tensor to an array to sort it
  const flatArray = await flatTensor.array()
  flatArray.sort((a, b) => a - b) // Sort the array in ascending order

  // Convert the sorted array back to a tensor
  const sortedTensor = tf.tensor1d(flatArray)

  // Calculate the indices for the quantiles
  const numElements = sortedTensor.shape[0]
  const lowIndex = Math.floor(numElements * lowerQuantile)
  const highIndex = Math.ceil(numElements * upperQuantile) - 1 // Subtract 1 because indices are 0-based

  // Slice the sorted tensor to get qmin and qmax
  const qmin = sortedTensor.slice(lowIndex, 1) // Get the value at the low index
  const qmax = sortedTensor.slice(highIndex, 1) // Get the value at the high index

  // Get the actual values from the tensors
  const qminValue = (await qmin.array())[0]
  const qmaxValue = (await qmax.array())[0]

  // Clean up tensors to free memory
  flatTensor.dispose()
  sortedTensor.dispose()
  qmin.dispose()
  qmax.dispose()

  return { qmin: qminValue, qmax: qmaxValue }
}

async function quantileNormalizeVolumeData(tensor, lowerQuantile = 0.05, upperQuantile = 0.95) {
  // Call calculateQuantiles and wait for the result
  const { qmin, qmax } = await calculateQuantiles(tensor, lowerQuantile, upperQuantile)

  // Convert qmin and qmax back to scalars
  const qminScalar = tf.scalar(qmin)
  const qmaxScalar = tf.scalar(qmax)

  // Perform the operation: (tensor - qmin) / (qmax - qmin)
  const resultTensor = tensor.sub(qminScalar).div(qmaxScalar.sub(qminScalar))

  // Dispose of the created scalars to free memory
  qminScalar.dispose()
  qmaxScalar.dispose()

  // Return the resulting tensor
  return resultTensor
}

async function minMaxNormalizeVolumeData(volumeData) {
  // Normalize the data to the range 0 - 1 using min-max scaling
  const volumeData_Max = volumeData.max()
  const volumeData_Min = volumeData.min()
  const normalizedSlices_3d = await volumeData.sub(volumeData_Min).div(volumeData_Max.sub(volumeData_Min))
  return normalizedSlices_3d
}

async function inferenceFullVolumeSeqCovLayer(
  model,
  slices_3d,
  input_shape,
  isChannelLast,
  num_of_slices,
  slice_height,
  slice_width
) {
  window.alert('inferenceFullVolumeSeqCovLayer() is not dead code?')
}

async function inferenceFullVolume(
  model,
  slices_3d,
  input_shape,
  isChannelLast,
  num_of_slices,
  slice_height,
  slice_width
) {
  window.alert('inferenceFullVolume() is not dead code?')
}

async function inferenceSubVolumes(model, slices_3d, num_of_slices, slice_height, slice_width, pipeline1_out = null) {
  window.alert('inferenceSubVolumes() is not dead code?')
}

async function tensor2LightBuffer(tensor, dtype) {
  window.alert('tensor2LightBuffer() is not dead code?')
  // return new Buffer(tensor.shape, dtype, Array.from(tensor.dataSync()) );
}

async function draw3dObjBoundingVolume(unstackOutVolumeTensor) {
  window.alert('draw3dObjBoundingVolume() is not dead code?')
  /*
  console.log("Plot cropped  volume shape ... ");
  // Convert all slices into 1 Dim array to download
  
  let allOutputSlices3DCC = [];
  let allOutputSlices3DContours = [];
  
  
  // dataSync() using to flatten array. Takes around 1.5 s
  for(let sliceTensorIdx = 0; sliceTensorIdx < unstackOutVolumeTensor.length; sliceTensorIdx++ ) {
    allOutputSlices3DCC[sliceTensorIdx] = Array.from(unstackOutVolumeTensor[sliceTensorIdx].dataSync());
  }
  
  // if(false) { // Enable contour for overlay option
  //     // Remove noisy regions using 3d CC
  //     let sliceWidth = niftiHeader.dims[1];
  //     let sliceHeight = niftiHeader.dims[2];
  //     allOutputSlices3DCC = findVolumeContours(allOutputSlices3DCC, sliceHeight, sliceWidth, 2 );
  // }
  
  let allOutputSlices3DCC1DimArray = [];
  // Use this conversion to download output slices as nii file. Takes around 0.5 s
  for(let sliceIdx = 0; sliceIdx < allOutputSlices3DCC.length; sliceIdx++ ) {
    allOutputSlices3DCC1DimArray.push.apply(allOutputSlices3DCC1DimArray, allOutputSlices3DCC[sliceIdx]);
  }
  
  console.log("Done with allOutputSlices3DCC1DimArray ")
  
  let  brainOut = [];
  
  
  let brainMaskTensor1d =  binarizeVolumeDataTensor(tf.tensor1d(allOutputSlices3DCC1DimArray));
  brainOut = Array.from(brainMaskTensor1d.dataSync());
  
  
  // labelArrayBuffer = createNiftiOutArrayBuffer(rawNiftiData, brainExtractionData1DimArr);
  let labelArrayBuffer = createNiftiOutArrayBuffer(rawNiftiData, brainOut);
  
  TODO Papaa specific code? 
  if(true) { // flag to not draw for now
    if(opts.isColorEnable) {
        let blob = new Blob([labelArrayBuffer], {type: "application/octet-binary;charset=utf-8"});
        let file = new File([blob], "temp.nii");
        params_label["files"] = [file];
        params_label[file["name"]] = {lut: "Grayscale", interpolation: false};

    } else {
        params_label["binaryImages"] = [labelArrayBuffer];
    }

    // Set the view of container-2 as container-1
    params_label["mainView"] = papayaContainers[0].viewer.mainImage.sliceDirection == 1? "axial" :
                               papayaContainers[0].viewer.mainImage.sliceDirection == 2? "coronal" : "sagittal";


    papaya.Container.resetViewer(1, params_label);
    papayaContainers[1].viewer.screenVolumes[0].alpha = 0.2;  // 0 to 1 screenVolumes[0] is first image loaded in Labels viewer
    papayaContainers[1].viewer.drawViewer(true, false);


    // To sync swap view button
    document.getElementById(PAPAYA_CONTROL_MAIN_SWAP_BUTTON_CSS + papayaContainers[0].containerIndex).addEventListener("click", function(){
          papayaContainers[1].viewer.rotateViews()

    })

    document.getElementById(PAPAYA_CONTROL_MAIN_SWAP_BUTTON_CSS + papayaContainers[1].containerIndex).addEventListener("click", function(){
          papayaContainers[0].viewer.rotateViews()

      })
    
  }
  */
}

async function argMaxLarge(outVolumeBuffer, num_of_slices, slice_height, slice_width, numOfClasses, dtype = 'float32') {
  window.alert('argMaxLarge() is not dead code?')
  /*
  if( findMinNumOfArrBufs(num_of_slices, slice_height, slice_width, numOfClasses, dtype) == 1) {

     // console.log("Convert output tensor to buffer");
    // reshape modelOutTensor.shape  : [ 1, 256, 256, 256, 3 ] to [ 256, 256, 256, 3 ]
    //-- let outVolumeBuffer = tensor2Buffer(modelOutTensor.relu().reshape([num_of_slices, slice_height, slice_width, numOfClasses]));

    //-- let  outVolumeBuffer = tensor2Buffer(modelOutTensor.reshape([num_of_slices, slice_height, slice_width, numOfClasses]));
    //-- let  outVolumeBuffer = tensor2LightBuffer(modelOutTensor.reshape([num_of_slices, slice_height, slice_width, numOfClasses]), dtype);

    console.log("Start argMaxLarge for  buffer  with last axis -1")

    let outBuffer = tf.buffer([num_of_slices, slice_height, slice_width ], dtype=tf.float32);

    for(let depthIdx = 0; depthIdx < num_of_slices; depthIdx += 1) {
        for(let rowIdx = 0; rowIdx < slice_height; rowIdx += 1) {
            for(let colIdx = 0; colIdx < slice_width; colIdx += 1) {
                // index of buffer with max Freq or max number so the index of that buffer is the right concensus label
                let indexOfMaxVotedBuffer = -1;
                // let maxVoxelValue = -Infinity;
                let maxVoxelValue = -1000000;

                for(let bufferIdx = 0; bufferIdx < numOfClasses; bufferIdx += 1) {
                    //Requested out of range element at 1,0,0,0.   Buffer shape=1,256,256,256,3
                    let voxelValue = outVolumeBuffer.get(depthIdx, rowIdx, colIdx, bufferIdx );

                    if(maxVoxelValue <= voxelValue) {
                          maxVoxelValue = voxelValue;
                          indexOfMaxVotedBuffer = bufferIdx;
                    }
                }

                outBuffer.set(indexOfMaxVotedBuffer, depthIdx, rowIdx, colIdx);

            }
        }
    }

    console.log("argMaxLarge for buffer ..Done");

    return outBuffer.toTensor();

  } else {
     console.log(" Terminated due to browser memory limitation (TODO: callbackUI)");
     console.log("argMaxLarge needs buffer division .. ");
     return 0;
  }
  */
}

async function addZeroPaddingTo3dTensor(tensor3d, rowPadArr = [1, 1], colPadArr = [1, 1], depthPadArr = [1, 1]) {
  if (tensor3d.rank !== 3) {
    throw new Error('Tensor must be 3D')
  }
  return tensor3d.pad([rowPadArr, colPadArr, depthPadArr])
}

async function removeZeroPaddingFrom3dTensor(tensor3d, rowPad = 1, colPad = 1, depthPad = 1) {
  if (tensor3d.rank !== 3) {
    throw new Error('Tensor must be 3D')
  }
  let h, w, d
  ;[h, w, d] = tensor3d.shape
  return tensor3d.slice([rowPad, colPad, depthPad], [h - 2 * rowPad, w - 2 * colPad, d - 2 * depthPad])
}

async function resizeWithZeroPadding(croppedTensor3d, newDepth, newHeight, newWidth, refVoxel, boundVolSizeArr) {
  const row_pad_befor = refVoxel[0]
  const col_pad_befor = refVoxel[1]
  const depth_pad_befor = refVoxel[2]
  // last and lower volume voxel
  const row_max = row_pad_befor + boundVolSizeArr[0] - 1 // size [2, 2, 2] means 2 voxels total in each dim
  const col_max = col_pad_befor + boundVolSizeArr[1] - 1
  const depth_max = depth_pad_befor + boundVolSizeArr[2] - 1

  const row_pad_after = newHeight - row_max - 1 > 0 ? newHeight - row_max - 1 : 0
  const col_pad_after = newWidth - col_max - 1 > 0 ? newWidth - col_max - 1 : 0
  const depth_pad_after = newDepth - depth_max - 1 > 0 ? newDepth - depth_max - 1 : 0

  return croppedTensor3d.pad([
    [row_pad_befor, row_pad_after],
    [col_pad_befor, col_pad_after],
    [depth_pad_befor, depth_pad_after]
  ])
}

async function applyMriThreshold(tensor, percentage) {
  // Perform asynchronous operations outside of tf.tidy
  const maxTensor = tensor.max()
  const thresholdTensor = maxTensor.mul(percentage)
  const threshold = await thresholdTensor.data() // Extracts the threshold value

  // Dispose tensors not needed anymore
  maxTensor.dispose()
  thresholdTensor.dispose()

  // Use tf.tidy for synchronous operations
  return tf.tidy(() => {
    const dataForProcessing = tensor.clone()

    // Thresholding (assuming background has very low values compared to the head)
    const mask = dataForProcessing.greater(threshold[0])
    // -- const denoisedMriData = dataForProcessing.mul(mask);

    // No need to  manually dispose dataForProcessing and mask, as tf.tidy() will dispose them auto.
    return mask
  })

  // -- return denoisedMriData;
}

async function binarizeVolumeDataTensor(volumeDataTensor) {
  const alpha = 0
  // element-wise: (x > 0 ? 1 : alpha * x );  e.g. Tenosr [0, 0.9, 0.8, -3] => Tensor [0, 1, 1, 0]
  return volumeDataTensor.step(alpha)
}
async function generateBrainMask(
  unstackOutVolumeTensor,
  num_of_slices,
  slice_height,
  slice_width,
  modelEntry,
  opts,
  callbackUI,
  callbackImg,
  isFinalImage = true
) {
  console.log('Generate Brain Masking ... ')
  // Convert all slices into 1 Dim array to download

  let allOutputSlices3DCC = []
  // const allOutputSlices3DContours = []

  // dataSync() using to flatten array. Takes around 1.5 s
  for (let sliceTensorIdx = 0; sliceTensorIdx < unstackOutVolumeTensor.length; sliceTensorIdx++) {
    allOutputSlices3DCC[sliceTensorIdx] = Array.from(unstackOutVolumeTensor[sliceTensorIdx].dataSync())
  }
  const isPreModelPostProcessEnable = modelEntry.preModelPostProcess
  // let isPreModelPostProcessEnable = inferenceModelsList[$$("selectModel").getValue() - 1]["preModelPostProcess"];

  if (isPreModelPostProcessEnable) {
    console.log('Phase-1 Post processing enabled ... ')
    allOutputSlices3DCC = tf.tidy(() => {
      // Remove noisy regions using 3d CC
      // const sliceWidth = niftiHeader.dims[1]
      // const sliceHeight = niftiHeader.dims[2]
      // return postProcessSlices3D(allOutputSlices3DCC, slice_height, slice_width)
      const errTxt = 'postProcessSlices3D() should be upgraded to BWLabeler'
      callbackUI(errTxt, -1, errTxt)
    })
    console.log('Post processing done ')
  } else {
    console.log('Phase-1 Post processing disabled ... ')
  }
  // Use this conversion to download output slices as nii file. Takes around 30 ms
  // does not use `push` to avoid stack overflows. In future: consider .set() with typed arrays
  const allOutputSlices3DCC1DimArray = new Array(allOutputSlices3DCC[0].length * allOutputSlices3DCC.length)
  let index = 0;
  for (let sliceIdx = 0; sliceIdx < allOutputSlices3DCC.length; sliceIdx++) {
      for (let i = 0; i < allOutputSlices3DCC[sliceIdx].length; i++) {
          allOutputSlices3DCC1DimArray[index++] = allOutputSlices3DCC[sliceIdx][i];
      }
  }
  let brainOut = []

  if (opts.isBrainCropMaskBased) {
    //  Mask-based

    const brainMaskTensor1d = await binarizeVolumeDataTensor(tf.tensor1d(allOutputSlices3DCC1DimArray))
    brainOut = Array.from(brainMaskTensor1d.dataSync())
  } else {
    //  Brain tissue
    window.alert('getAllSlicesData1D() is not dead code? niftiHeader and niftiImage required by getAllSlicesData1D')
    /*const allSlices = getAllSlicesData1D(num_of_slices, niftiHeader, niftiImage)
    for (let sliceIdx = 0; sliceIdx < allOutputSlices3DCC.length; sliceIdx++) {
      for (let pixelIdx = 0; pixelIdx < slice_height * slice_width; pixelIdx++) {
        // Filter smaller regions original MRI data
        if (allOutputSlices3DCC[sliceIdx][pixelIdx] === 0) {
          allSlices[sliceIdx][pixelIdx] = 0
        }
      }

      brainOut.push.apply(brainOut, allSlices[sliceIdx])
    }*/
  }
  if (isFinalImage || opts.showPhase1Output) {//all done
    callbackImg(brainOut, opts, modelEntry)
    callbackUI('Segmentation finished', 0)
  }
  return tf.tensor(brainOut, [num_of_slices, slice_height, slice_width])
}

async function convByOutputChannelAndInputSlicing(input, filter, biases, stride, pad, dilationRate, sliceSize) {
  // const batchSize = input.shape[0]
  // const depth = input.shape[1]
  // const height = input.shape[2]
  // const width = input.shape[3]
  const inChannels = input.shape[4]
  const outChannels = filter.shape[4]

  // Create an empty array to hold the output channels
  let outputChannels = null

  // Slice the input tensor and process one output channel at a time
  for (let channel = 0; channel < outChannels; channel++) {
    const numSlices = Math.ceil(inChannels / sliceSize)
    const biasesSlice = biases.slice([channel], [1])
    let outputChannel = null

    for (let i = 0; i < numSlices; i++) {
      const startChannel = i * sliceSize
      const endChannel = Math.min((i + 1) * sliceSize, inChannels)

      // Only proceed if there are channels to process
      if (startChannel < inChannels) {
        const resultSlice = tf.tidy(() => {
          const inputSlice = input.slice([0, 0, 0, 0, startChannel], [-1, -1, -1, -1, endChannel - startChannel])
          const filterSlice = filter.slice([0, 0, 0, startChannel, channel], [-1, -1, -1, endChannel - startChannel, 1])
          // Perform the convolution for the current slice and output channel
          return tf.conv3d(inputSlice, filterSlice, stride, pad, 'NDHWC', dilationRate)
        })

        if (outputChannel === null) {
          outputChannel = resultSlice
        } else {
          const updatedOutputChannel = outputChannel.add(resultSlice)
          outputChannel.dispose()
          resultSlice.dispose()
          outputChannel = updatedOutputChannel
        }
      }
    }

    // Add the biases to the accumulated convolutions for this channel
    const biasedOutputChannel = outputChannel.add(biasesSlice)
    outputChannel.dispose()
    biasesSlice.dispose()

    // Accumulate the channel to the output array
    if (outputChannels == null) {
      outputChannels = biasedOutputChannel
    } else {
      const updatedOutputChannels = await tf.concat([outputChannels, biasedOutputChannel], 4)
      biasedOutputChannel.dispose()
      outputChannels.dispose()
      outputChannels = updatedOutputChannels
    }
  }

  return outputChannels
}

function processTensorInChunks(inputTensor, filterWeights, chunkSize) {
  // Assuming inputTensor's shape: [batch, depth, height, width, inChannels]
  // and filterWeights's shape: [filterDepth, filterHeight, filterWidth, inChannels, outChannels]
  const stride = 1
  const pad = 0
  const dilationRate = 1
  const inChannels = inputTensor.shape[4]
  const numSlices = Math.ceil(inChannels / chunkSize)

  let accumulatedResult = null

  for (let i = 0; i < numSlices; i++) {
    const startChannel = i * chunkSize
    const endChannel = Math.min((i + 1) * chunkSize, inChannels)
    const channels = endChannel - startChannel

    const inputSlice = tf.tidy(() => {
      // Slice the input tensor to get the current chunk
      return inputTensor.slice([0, 0, 0, 0, startChannel], [-1, -1, -1, -1, channels])
    })

    const filterSlice = tf.tidy(() => {
      // Slice the filter weights to match the input tensor's current chunk
      return filterWeights.slice([0, 0, 0, startChannel, 0], [-1, -1, -1, channels, -1])
    })

    const resultSlice = tf.conv3d(inputSlice, filterSlice, stride, pad, 'NDHWC', dilationRate)
    // Clean up the slices to free memory
    inputSlice.dispose()
    filterSlice.dispose()

    // Squeeze the result slice to remove dimensions of size 1
    const squeezedResultSlice = tf.squeeze(resultSlice)
    resultSlice.dispose() // Dispose of the original resultSlice after squeezing

    if (accumulatedResult === null) {
      accumulatedResult = squeezedResultSlice
    } else {
      // Accumulate the result by adding the new result slice to it
      const newAccumulatedResult = accumulatedResult.add(squeezedResultSlice)

      // Dispose of the previous accumulatedResult and squeezedResultSlice
      accumulatedResult.dispose()
      // Dispose of squeezedResultSlice only if it wasn't assigned to accumulatedResult
      if (accumulatedResult !== squeezedResultSlice) {
        squeezedResultSlice.dispose()
      }
      // Update accumulatedResult with the new result
      accumulatedResult = newAccumulatedResult
    }

    tf.tidy(() => {
      tf.matMul(tf.zeros([1, 1]), tf.zeros([1, 1]))
    })
  }

  return accumulatedResult
}

class SequentialConvLayer {
  constructor(model, chunkSize, isChannelLast, callbackUI) {
    this.model = model
    this.outChannels = model.outputLayers[0].kernel.shape[4]
    this.chunkSize = chunkSize
    this.isChannelLast = isChannelLast
    this.callbackUI = callbackUI // fork
  }

  /**
    * Apply sequential convolution layer
    * @since 3.0.0
    * @member SequentialConvLayer
    * @param {tf.Tensor}  inputTensor  e.g.  [ 1, 256, 256, 256, 5 ]
    * @return {promise}
    *
    * convLayer.rank -> 3
    * typeof(convLayer) -> "object"
    * convLayer:  Object { dataFormat: "channelsLast", dilationRate: Array(3) [ 1, 1, 1 ], inputSpec: Array [ {…} ],
    *                      name: "output", padding: "same", strides: Array(3) [ 1, 1, 1 ], ...}
    *
    * weights.shape ->  Array(5) [ 1, 1, 1, 5, 3 ]
    * weights.print()
    * //=>    Tensor
    *           [[[[[0.146999 , -1.4474995, -2.8961499],
    *               [1.1067894, 0.6897876 , -0.7573005],
    *               [-0.38512 , -0.2812168, -0.8637539],
    *               [0.9341159, -0.0344299, -2.3668685],
    *               [0.1052373, 1.266812  , 0.6542516 ]]]]]
    *
    * biases.shape ->  Array [ 3 ]
    * biases.print()
    * //=>      Tensor
    *             [-0.7850812, -2.3238883, 2.1639345]
    *
    * for idx = 0 -> filterWeights.shape  -> Array(5) [ 1, 1, 1, 5, 1 ]
    * filterWeights.print()
    * //=>  Tensor
    *         [[[[[0.146999 ],
    *             [1.1067894],
    *             [-0.38512 ],
    *             [0.9341159],
    *             [0.1052373]]]]]
    *
    * for idx = 0 -> filterBiases.shape  -> Array [1]
    * filterBiases.print()
    * //=>   Tensor
    *          [-0.7850812]

    */

  async apply(inputTensor) {
    const oldDeleteTextureThreshold = tf.ENV.get('WEBGL_DELETE_TEXTURE_THRESHOLD')
    tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0)

    const self = this
    // Important to avoid "undefined" class var members inside the timer.
    // "this" has another meaning inside the timer.

    // document.getElementById("progressBarChild").parentElement.style.visibility = "visible";

    return new Promise((resolve) => {
      const startTime = performance.now()

      const convLayer = self.model.layers[self.model.layers.length - 1]
      const weights = convLayer.getWeights()[0] //
      const biases = convLayer.getWeights()[1]
      const outputShape = self.isChannelLast ? inputTensor.shape.slice(1, -1) : inputTensor.shape.slice(2)
      // -- e.g.  outputShape : [256,256,256] or cropped Dim
      // -- if inputTensor [ 1, D, H, W, 50 ], channelLast true ->   outputShape : outputShape [D, H, W]
      // -- if inputTensor [ 1, 50, D, H, W ], channelLast false ->   outputShape : outputShape [D, H, W]

      let outB = tf.mul(tf.ones(outputShape), -10000)
      // -- e.g. outB.shape  [256,256,256]
      let outC = tf.zeros(outputShape)
      // -- e.g. outC.shape  [256,256,256]
      let chIdx = 0

      // console.log("---------------------------------------------------------");
      console.log(' channel loop')

      const seqTimer = window.setInterval(async function () {
        tf.engine().startScope() // Start TensorFlow.js scope
        console.log('=======================')
        const memoryInfo0 = tf.memory()
        console.log(`| Number of Tensors: ${memoryInfo0.numTensors}`)
        console.log(`| Number of Data Buffers: ${memoryInfo0.numDataBuffers}`)
        console.log('Channel : ', chIdx)

        const result = await tf.tidy(() => {
          const filterWeights = weights.slice([0, 0, 0, 0, chIdx], [-1, -1, -1, -1, 1])
          // -- e.g. filterWeights.shape [ 1, 1, 1, 5, 1 ]
          const filterBiases = biases.slice([chIdx], [1])
          // -- e.g. filterBiases.shape [1] -> Tensor  [-0.7850812]
          const outA = processTensorInChunks(
            inputTensor,
            filterWeights,
            Math.min(self.chunkSize, self.outChannels)
          ).add(filterBiases)
          const greater = tf.greater(outA, outB)
          const newoutB = tf.where(greater, outA, outB)
          const newoutC = tf.where(greater, tf.fill(outC.shape, chIdx), outC)
          // Dispose the old tensors before reassigning
          tf.dispose([outB, outC, filterWeights, filterBiases, outA, greater])
          // Dummy operation to trigger cleanup
          tf.tidy(() => tf.matMul(tf.ones([1, 1]), tf.ones([1, 1])))
          return [newoutC, newoutB]
        })

        // -- await showMemStatus(chIdx, self.outChannels);

        const memoryInfo1 = tf.memory()
        console.log(`| Number of Tensors: ${memoryInfo1.numTensors}`)
        console.log(`| Number of Data Buffers: ${memoryInfo1.numDataBuffers}`)
        console.log('=======================')

        // Log memory usage

        const memoryInfo = tf.memory()
        self.callbackUI(`Iteration ${chIdx}`, chIdx / self.outChannels)
        console.log(`Number of Tensors: ${memoryInfo.numTensors}`)
        console.log(`Number of Data Buffers: ${memoryInfo.numDataBuffers}`)
        console.log(`Bytes In Use: ${memoryInfo.numBytes}`)
        console.log(`Megabytes In Use: ${(memoryInfo.numBytes / 1048576).toFixed(3)} MB`)
        if (memoryInfo.unreliable) {
          console.log(`Unreliable: ${memoryInfo.unreliable}`)
        }
        // Dispose of previous values before assigning new tensors to outC and outB
        if (typeof outC !== 'undefined') {
          outC.dispose()
        }
        if (typeof outB !== 'undefined') {
          outB.dispose()
        }
        // Assign the new values to outC and outB
        outC = tf.keep(result[0])
        outB = tf.keep(result[1])
        // // Assign the new values to outC and outB
        // outC = result[0];
        // outB = result[1];
        tf.engine().endScope()

        if (chIdx === self.outChannels - 1) {
          window.clearInterval(seqTimer)
          // document.getElementById("progressBarChild").style.width = 0 + "%";
          tf.dispose(outB)
          const endTime = performance.now()
          const executionTime = endTime - startTime
          console.log(`Execution time for output layer: ${executionTime} milliseconds`)
          tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', oldDeleteTextureThreshold)
          resolve(outC)
        } else {
          chIdx++

          // the seemingly strange sequence of operations
          // below prevents tfjs from uncontrolably
          // grabbing buffers, even when all tensors have
          // already been disposed

          const outCShape = outC.shape
          const outCdata = outC.dataSync()
          const outBShape = outC.shape
          const outBdata = outB.dataSync()
          outC.dispose()
          outB.dispose()
          // tf.disposeVariables()
          outC = tf.tensor(outCdata, outCShape)
          outB = tf.tensor(outBdata, outBShape)

          // document.getElementById("progressBarChild").style.width = (chIdx + 1) * 100 / self.outChannels + "%";
        }

        // Artificially introduce a pause to allow for garbage collection to catch up
        await new Promise((resolve) => setTimeout(resolve, 300))
      }, 0)
    })
  }
} // <<<< End of class

async function generateOutputSlicesV2(
  img,
  OutVolumeTensorShape,
  OutVolumeTensorType,
  num_of_slices,
  numSegClasses,
  slice_height,
  slice_width,
  modelEntry,
  opts,
  niftiImage
) {
  // Convert all slices into 1 Dim array
  // const allOutputSlices3DContours = []
  if (opts.isPostProcessEnable) {
    const BWInstance = new BWLabeler()
    const dim = new Uint32Array(OutVolumeTensorShape)
    const conn = 26 // Example connectivity
    const binarize = true
    const onlyLargestClusterPerClass = true
    const [labelCount, labeledImage] = BWInstance.bwlabel(img, dim, conn, binarize, onlyLargestClusterPerClass)
    for (let i = 0; i < img.length; i++) {
      img[i] *= labeledImage[i]
    }
  } // if isPostProcessEnable
  const typedArrayConstructor = {
    float32: Float32Array,
    int32: Int32Array
    // Add other cases as needed for different dtypes
  }[OutVolumeTensorType]
  // Create a new TypedArray from img with the same type as outLabelVolume
  const allOutputSlices3DCC1DimArray = new Uint8Array(img)

  // ? const maskBrainExtraction = false

  // let labelArrayBuffer
  const modelType = modelEntry.type

  // return img
  switch (modelType) {
    case 'Brain_Masking': {
      const brainMask = new Uint8Array(allOutputSlices3DCC1DimArray.length)
      for (let i = 0; i < allOutputSlices3DCC1DimArray.length; i++) {
        brainMask[i] = allOutputSlices3DCC1DimArray[i] !== 0 ? 1 : 0
      }
      // labelArrayBuffer = createNiftiOutArrayBuffer(rawNiftiData, brainMask);
      // allOutputSlices3DCC1DimArray = brainMask;
      // --labelsHistogramMap = null;
      // maskBrainExtraction = true;
      return brainMask
      // break;
    }
    case 'Brain_Extraction': {
      const maskedData = new Uint8Array(allOutputSlices3DCC1DimArray.length)
      // const brainData = nifti2data(rawNiftiData);

      for (let i = 0; i < allOutputSlices3DCC1DimArray.length; i++) {
        // Create the mask - 1 where the value is non-zero, 0 where it is zero.
        const maskValue = allOutputSlices3DCC1DimArray[i] !== 0 ? 1 : 0
        // Apply the mask to the data - multiply by the mask value.
        maskedData[i] = niftiImage[i] * maskValue
      }
      // labelArrayBuffer = createNiftiOutArrayBuffer(rawNiftiData, maskedData);

      // Update `allOutputSlices3DCC1DimArray` if needed.
      // allOutputSlices3DCC1DimArray = maskedData;

      // Other operations...
      // maskBrainExtraction = true;
      return maskedData
      // break;
    }
  }

  return img
}

async function inferenceFullVolumeSeqCovLayerPhase2(
  opts,
  modelEntry,
  model,
  slices_3d,
  num_of_slices,
  slice_height,
  slice_width,
  pipeline1_out,
  callbackUI,
  callbackImg,
  statData,
  niftiImage
) {
  // --Phase-2, After remove the skull try to allocate brain volume and make inferece

  console.log(' ---- Start FullVolume Inference with Sequential Conv Layer for phase-II ---- ')
  // console.log("BOB", callbackUI); console.log("UNCLE",callbackImg); return
  const quantileNorm = modelEntry.enableQuantileNorm
  if (quantileNorm) {
    // Quantile normalize function needs specific models to be used
    console.log('preModel Quantile normalization enabled')
    slices_3d = await quantileNormalizeVolumeData(slices_3d)
  } else {
    // Min Max Nomalize MRI data to be from 0 to 1
    console.log('preModel Min Max normalization enabled')
    slices_3d = await minMaxNormalizeVolumeData(slices_3d)
  }

  let mask_3d

  if (pipeline1_out == null) {
    // preModel is null

    // Check if thresholding the MRI to remove noisy voxels for better cropping is needed.
    const autoThresholdValue = modelEntry.autoThreshold

    if (autoThresholdValue > 0 && autoThresholdValue <= 1) {
      // Filtered MRI from noisy voxel below  autoThresholdValue
      mask_3d = await applyMriThreshold(slices_3d, autoThresholdValue)
    } else {
      console.log('No valid crop threshold value')
      // binarize original image
      mask_3d = await slices_3d.greater([0]).asType('bool')
    }
  } else {
    mask_3d = await pipeline1_out.greater([0]).asType('bool')
    // -- pipeline1_out.dispose();
  }

  console.log(' mask_3d shape :  ', mask_3d.shape)

  const coords = await tf.whereAsync(mask_3d)
  // -- Get each voxel coords (x, y, z)

  mask_3d.dispose()

  const coordsArr = coords.arraySync()

  let row_min = slice_height
  let row_max = 0
  let col_min = slice_width
  let col_max = 0
  let depth_min = num_of_slices
  let depth_max = 0

  for (let i = 0; i < coordsArr.length; i++) {
    if (row_min > coordsArr[i][0]) {
      row_min = coordsArr[i][0]
    } else if (row_max < coordsArr[i][0]) {
      row_max = coordsArr[i][0]
    }

    if (col_min > coordsArr[i][1]) {
      col_min = coordsArr[i][1]
    } else if (col_max < coordsArr[i][1]) {
      col_max = coordsArr[i][1]
    }

    if (depth_min > coordsArr[i][2]) {
      depth_min = coordsArr[i][2]
    } else if (depth_max < coordsArr[i][2]) {
      depth_max = coordsArr[i][2]
    }
  }

  console.log('row min and max  :', row_min, row_max)
  console.log('col min and max  :', col_min, col_max)
  console.log('depth min and max  :', depth_min, depth_max)

  // -- Reference voxel that cropped volume started slice with it
  const refVoxel = [row_min, col_min, depth_min]
  // -- Starting form refVoxel, size of bounding volume
  const boundVolSizeArr = [row_max - row_min + 1, col_max - col_min + 1, depth_max - depth_min + 1]

  coords.dispose()

  // -- Extract 3d object (e.g. brain)
  const cropped_slices_3d = await slices_3d.slice(
    [row_min, col_min, depth_min],
    [row_max - row_min + 1, col_max - col_min + 1, depth_max - depth_min + 1]
  )
  slices_3d.dispose()

  // -- Padding size add to cropped brain
  const pad = modelEntry.cropPadding

  // Create margin around the bounding volume
  let cropped_slices_3d_w_pad = await addZeroPaddingTo3dTensor(cropped_slices_3d, [pad, pad], [pad, pad], [pad, pad])
  console.log(' cropped slices_3d with padding shape:  ', cropped_slices_3d_w_pad.shape)

  cropped_slices_3d.dispose()

  if (opts.drawBoundingVolume) {
    let testVol = await removeZeroPaddingFrom3dTensor(cropped_slices_3d_w_pad, pad, pad, pad)
    console.log(' outLabelVolume without padding shape :  ', testVol.shape)

    testVol = await resizeWithZeroPadding(testVol, num_of_slices, slice_height, slice_width, refVoxel, boundVolSizeArr)
    console.log(' outLabelVolume final shape after resizing :  ', testVol.shape)

    draw3dObjBoundingVolume(tf.unstack(testVol))
    testVol.dispose()

    return 0
  }

  statData.Brainchop_Ver = 'FullVolume'
  // model.then(function (res) {
  // console.log("--->>>>", opts.drawBoundingVolume); return
  const res = await model
  try {
    let startTime = performance.now()
    const inferenceStartTime = performance.now()
    // maxLabelPredicted in whole volume of the brain
    let maxLabelPredicted = 0
    const transpose = modelEntry.enableTranspose
    const delay = modelEntry.inferenceDelay
    console.log('Inference delay :', delay)

    if (transpose) {
      cropped_slices_3d_w_pad = await cropped_slices_3d_w_pad.transpose()
      console.log('Input transposed for pre-model')
    } else {
      console.log('Transpose not enabled for pre-model')
    }

    let i = 1
    const layersLength = res.layers.length
    console.log('res.layers.length ', layersLength)

    const isChannelLast = isModelChnlLast(res)
    const batchSize = opts.batchSize
    const numOfChan = opts.numOfChan
    let adjusted_input_shape
    // -- Adjust model input shape
    if (isChannelLast) {
      res.layers[0].batchInputShape[1] = cropped_slices_3d_w_pad.shape[0]
      res.layers[0].batchInputShape[2] = cropped_slices_3d_w_pad.shape[1]
      res.layers[0].batchInputShape[3] = cropped_slices_3d_w_pad.shape[2]

      adjusted_input_shape = [
        batchSize,
        res.layers[0].batchInputShape[1],
        res.layers[0].batchInputShape[2],
        res.layers[0].batchInputShape[3],
        numOfChan
      ]
    } else {
      res.layers[0].batchInputShape[2] = cropped_slices_3d_w_pad.shape[0]
      res.layers[0].batchInputShape[3] = cropped_slices_3d_w_pad.shape[1]
      res.layers[0].batchInputShape[4] = cropped_slices_3d_w_pad.shape[2]

      adjusted_input_shape = [
        batchSize,
        numOfChan,
        res.layers[0].batchInputShape[2],
        res.layers[0].batchInputShape[3],
        res.layers[0].batchInputShape[4]
      ]
    }

    console.log(' Model batch input shape : ', res.layers[0].batchInputShape)
    // -- batchInputShape {Array} input_shape - e.g. [?, D, H, W, Ch] or [?, Ch, D, H, W]

    statData.Input_Shape = JSON.stringify(res.layers[0].batchInputShape)
    statData.Output_Shape = JSON.stringify(res.output.shape)
    statData.Channel_Last = isChannelLast
    statData.Model_Param = await getModelNumParameters(res)
    statData.Model_Layers = await getModelNumLayers(res)
    statData.Model = modelEntry.modelName
    statData.Extra_Info = null

    // Determine the number of output channels in the last layer of the model
    //  e.g. 3, 50, 104
    const outputLayer = res.layers[res.layers.length - 1]
    console.log('Output Layer : ', outputLayer)

    const expected_Num_labels = isChannelLast
      ? outputLayer.outputShape[outputLayer.outputShape.length - 1]
      : outputLayer.outputShape[1]
    console.log('Num of output channels : ', expected_Num_labels)

    const curTensor = []
    curTensor[0] = await cropped_slices_3d_w_pad.reshape(adjusted_input_shape)
    // console.log("curTensor[0] :", curTensor[0].dataSync());

    // let curProgBar = parseInt(document.getElementById("progressBar").style.width);

    const timer = window.setInterval(async function () {
      try {
        if (res.layers[i].activation.getClassName() !== 'linear') {
          curTensor[i] = await res.layers[i].apply(curTensor[i - 1])
        } else {
          curTensor[i] = await convByOutputChannelAndInputSlicing(
            curTensor[i - 1],
            res.layers[i].getWeights()[0],
            res.layers[i].getWeights()[1],
            res.layers[i].strides,
            res.layers[i].padding,
            res.layers[i].dilationRate,
            3
          ) // important for memory use
        }

        // // Log memory usage
        // const memoryInfo = tf.memory();
        // console.log(`Iteration ${i}:`);
        // console.log(`Number of Tensors: ${memoryInfo.numTensors}`);
        // console.log(`Number of Data Buffers: ${memoryInfo.numDataBuffers}`);
        // console.log(`Bytes In Use: ${memoryInfo.numBytes}`);
        // console.log(`Megabytes In Use: ${(memoryInfo.numBytes / 1048576).toFixed(3)} MB`);
        // console.log(`Unreliable: ${memoryInfo.unreliable}`);

        tf.dispose(curTensor[i - 1])
      } catch (err) {
        // ? original code provided special dialog for shaders if( err.message === "Failed to compile fragment shader.") {
        callbackUI(err.message, -1, err.message)

        window.clearInterval(timer)
        tf.engine().endScope()
        tf.engine().disposeVariables()

        statData.Inference_t = Infinity
        statData.Postprocess_t = Infinity
        statData.Status = 'Fail'
        statData.Error_Type = err.message
        statData.Extra_Err_Info = 'Failed while model layer ' + i + ' apply'

        if (opts.telemetryFlag) {
          await submitTiming2GoogleSheet(statData, callbackUI)
        }

        return 0
      }

      console.log('layer output Tensor shape : ', curTensor[i].shape)
      console.log('layer count params ', res.layers[i].countParams())

      res.layers[i].dispose()
      curTensor[i - 1].dispose()

      // bork
      callbackUI('Layer ' + i.toString(), (i + 1) / layersLength)
      if (tf.memory().unreliable) {
        const unreliableReasons = 'unreliable reasons :' + tf.memory().reasons
        callbackUI(unreliableReasons, NaN, unreliableReasons)
      }
      if (i === layersLength - 2) {
        // Stop before the last layer or classification layer.

        window.clearInterval(timer)

        // // Create an instance of SequentialConvLayer
        // The second parameter is important for memory,
        // the larger it is, the more memory it uses
        // it was 8, but I set it to 3, got a different error
        // let seqConvLayer = new SequentialConvLayer(res, 10, isChannelLast);
        const seqConvLayer = await new SequentialConvLayer(res, 10, isChannelLast, callbackUI)

        // Apply the last output tensor to the seq. instance
        let outputTensor = null
        const profileInfo = await tf.profile(async () => {
          // Your tensor operations here
          outputTensor = await seqConvLayer.apply(curTensor[i])
        })
        console.log('profileInfo : ', profileInfo)

        // -- document.getElementById("progressBarChild").style.width = 0 + "%";;

        // Dispose the previous layer input tensor
        tf.dispose(curTensor[i])
        // delete the used class
        // ? delete seqConvLayer;

        // You can now use 'outputTensor' as needed
        console.log(' Output tensor', outputTensor)
        console.log(' Output tensor shape : ', outputTensor.shape)
        // Array(3) [ 256, 256, 256 ]

        if (outputTensor.shape.length !== 3) {
          const msg = 'Output tensor shape should be 3 dims but it is ' + outputTensor.shape.length
          callbackUI(msg, -1, msg)
        }

        const Inference_t = ((performance.now() - startTime) / 1000).toFixed(4)

        console.log(' find array max ')
        const curBatchMaxLabel = await outputTensor.max().dataSync()[0]
        if (maxLabelPredicted < curBatchMaxLabel) {
          maxLabelPredicted = curBatchMaxLabel
        }

        const numSegClasses = maxLabelPredicted + 1
        console.log('Predicted num of segmentation classes', numSegClasses)
        statData.Actual_Labels = numSegClasses
        statData.Expect_Labels = expected_Num_labels
        statData.NumLabels_Match = numSegClasses === expected_Num_labels
        if (numSegClasses !== expected_Num_labels) {
          const msg = 'expected ' + expected_Num_labels + ' labels, but the predicted are ' + numSegClasses
          callbackUI(msg, -1, msg)
        }

        // -- Transpose back to fit Papaya display settings
        let outLabelVolume = outputTensor.reshape([
          cropped_slices_3d_w_pad.shape[0],
          cropped_slices_3d_w_pad.shape[1],
          cropped_slices_3d_w_pad.shape[2]
        ])
        tf.dispose(outputTensor)

        // Transpose MRI data to be match pytorch/keras input output
        if (transpose) {
          console.log('outLabelVolume transposed')
          outLabelVolume = outLabelVolume.transpose()
        }

        outLabelVolume = await removeZeroPaddingFrom3dTensor(outLabelVolume, pad, pad, pad)
        console.log(' outLabelVolume without padding shape :  ', outLabelVolume.shape)
        outLabelVolume = await resizeWithZeroPadding(
          outLabelVolume,
          num_of_slices,
          slice_height,
          slice_width,
          refVoxel,
          boundVolSizeArr
        )
        console.log(' outLabelVolume final shape after resizing :  ', outLabelVolume.shape)

        // let filterOutWithPreMask =  inferenceModelsList[$$("selectModel").getValue() - 1]["filterOutWithPreMask"];
        const filterOutWithPreMask = modelEntry.filterOutWithPreMask
        // To clean the skull area wrongly segmented inphase-2.
        if (pipeline1_out != null && opts.isBrainCropMaskBased && filterOutWithPreMask) {
          const bin = await binarizeVolumeDataTensor(pipeline1_out)
          outLabelVolume = await outLabelVolume.mul(bin)
        }

        startTime = performance.now()
        // Generate output volume or slices
        console.log('Generating correct output')
        let outimg
        try {
          const img = await new Uint32Array(outLabelVolume.dataSync())
          const Vshape = outLabelVolume.shape
          const Vtype = outLabelVolume.dtype
          outimg = await generateOutputSlicesV2(
            img,
            Vshape,
            Vtype,
            num_of_slices,
            numSegClasses,
            slice_height,
            slice_width,
            modelEntry,
            opts,
            niftiImage
          )
          console.log(' Phase-2 num of tensors after generateOutputSlicesV2: ', tf.memory().numTensors)

          tf.dispose(outLabelVolume)
          tf.engine().endScope()
          tf.engine().disposeVariables()
        } catch (error) {
          // -- Timing data to collect
          tf.engine().endScope()
          tf.engine().disposeVariables()
          console.log('Error while generating output: ', error)
          const msg = 'Failed while generating output due to limited browser memory available'
          callbackUI(msg, -1, msg)

          statData.Inference_t = Inference_t
          statData.Postprocess_t = Infinity
          statData.Status = 'Fail'
          statData.Error_Type = error.message
          statData.Extra_Err_Info = 'Failed while generating output'

          if (opts.telemetryFlag) {
            await submitTiming2GoogleSheet(statData, callbackUI)
          }

          return 0
        }
        const Postprocess_t = ((performance.now() - startTime) / 1000).toFixed(4)

        console.log(
          'Processing the whole brain volume in tfjs for multi-class output mask took : ',
          ((performance.now() - inferenceStartTime) / 1000).toFixed(4) + '  Seconds'
        )

        // -- Timing data to collect
        statData.Inference_t = Inference_t
        statData.Postprocess_t = Postprocess_t
        statData.Status = 'OK'

        if (opts.telemetryFlag) {
          await submitTiming2GoogleSheet(statData, callbackUI)
        }
        callbackUI('Segmentation finished', 0)
        callbackImg(outimg, opts, modelEntry)
        return 0
      } else {
        i++
      }
    }, delay)
  } catch (err) {
    callbackUI(err.message, -1, err.message)
    console.log(
      'If webgl context is lost, try to restore webgl context by visit the link ' +
        '<a href="https://support.biodigital.com/hc/en-us/articles/218322977-How-to-turn-on-WebGL-in-my-browser">here</a>'
    )
    if (tf.memory().unreliable) {
      const unreliableReasons = 'unreliable reasons :' + tf.memory().reasons
      callbackUI(unreliableReasons, NaN, unreliableReasons)
    }
  }
  // });
}

async function inferenceFullVolumePhase2(
  model,
  slices_3d,
  num_of_slices,
  slice_height,
  slice_width,
  pipeline1_out,
  modelEntry,
  statData,
  opts,
  callbackImg,
  callbackUI,
  niftiImage
) {
  let outimg = []
  // --Phase-2, After remove the skull try to allocate brain volume and make inferece
  console.log(' ---- Start FullVolume inference phase-II ---- ')
  const quantileNorm = modelEntry.enableQuantileNorm
  if (quantileNorm) {
    // Quantile normalize function needs specific models to be used
    console.log('preModel Quantile normalization enabled')
    slices_3d = await quantileNormalizeVolumeData(slices_3d)
  } else {
    // Min Max Nomalize MRI data to be from 0 to 1
    console.log('preModel Min Max normalization enabled')
    slices_3d = await minMaxNormalizeVolumeData(slices_3d)
  }
  let mask_3d
  if (pipeline1_out == null) {
    // preModel is null

    // Check if thresholding the MRI to remove noisy voxels for better cropping is needed.
    const autoThresholdValue = modelEntry.autoThreshold

    if (autoThresholdValue > 0 && autoThresholdValue <= 1) {
      // Filtered MRI from noisy voxel below  autoThresholdValue
      mask_3d = await applyMriThreshold(slices_3d, autoThresholdValue)
    } else {
      console.log('No valid crop threshold value')
      // binarize original image
      mask_3d = await slices_3d.greater([0]).asType('bool')
    }
  } else {
    mask_3d = pipeline1_out.greater([0]).asType('bool')
    // -- pipeline1_out.dispose()
  }
  console.log(' mask_3d shape :  ', mask_3d.shape)
  const coords = await tf.whereAsync(mask_3d)
  // -- Get each voxel coords (x, y, z)
  mask_3d.dispose()
  const coordsArr = coords.arraySync()

  let row_min = slice_height
  let row_max = 0
  let col_min = slice_width
  let col_max = 0
  let depth_min = num_of_slices
  let depth_max = 0

  for (let i = 0; i < coordsArr.length; i++) {
    if (row_min > coordsArr[i][0]) {
      row_min = coordsArr[i][0]
    } else if (row_max < coordsArr[i][0]) {
      row_max = coordsArr[i][0]
    }

    if (col_min > coordsArr[i][1]) {
      col_min = coordsArr[i][1]
    } else if (col_max < coordsArr[i][1]) {
      col_max = coordsArr[i][1]
    }

    if (depth_min > coordsArr[i][2]) {
      depth_min = coordsArr[i][2]
    } else if (depth_max < coordsArr[i][2]) {
      depth_max = coordsArr[i][2]
    }
  }

  console.log('row min and max  :', row_min, row_max)
  console.log('col min and max  :', col_min, col_max)
  console.log('depth min and max  :', depth_min, depth_max)

  // -- Reference voxel that cropped volume started slice with it
  const refVoxel = [row_min, col_min, depth_min]
  console.log('refVoxel :', refVoxel)

  // -- Starting form refVoxel, size of bounding volume
  const boundVolSizeArr = [row_max - row_min + 1, col_max - col_min + 1, depth_max - depth_min + 1]

  console.log('boundVolSizeArr :', boundVolSizeArr)

  coords.dispose()

  // -- Extract 3d object (e.g. brain)
  const cropped_slices_3d = slices_3d.slice(
    [row_min, col_min, depth_min],
    [row_max - row_min + 1, col_max - col_min + 1, depth_max - depth_min + 1]
  )

  slices_3d.dispose()

  // -- Padding size add to cropped brain
  const pad = modelEntry.cropPadding

  // Create margin around the bounding volume
  let cropped_slices_3d_w_pad = await addZeroPaddingTo3dTensor(cropped_slices_3d, [pad, pad], [pad, pad], [pad, pad])
  console.log(' cropped slices_3d with padding shape:  ', cropped_slices_3d_w_pad.shape)

  cropped_slices_3d.dispose()

  // -- Test dim after padding ..
  // for (let i = 0; i < cropped_slices_3d_w_pad.rank; i++) {
  //     if(cropped_slices_3d_w_pad.shape[i] > 256) {
  //          console.log(" cropped_slices_3d_w_pad > 256 ")
  //     }

  // }

  if (opts.drawBoundingVolume) {
    let testVol = await removeZeroPaddingFrom3dTensor(cropped_slices_3d_w_pad, pad, pad, pad)
    console.log(' outLabelVolume without padding shape :  ', testVol.shape)

    testVol = await resizeWithZeroPadding(testVol, num_of_slices, slice_height, slice_width, refVoxel, boundVolSizeArr)
    console.log(' outLabelVolume final shape after resizing :  ', testVol.shape)
    // todo draw3dObjBoundingVolume()
    draw3dObjBoundingVolume(tf.unstack(testVol))
    testVol.dispose()

    return 0
  }

  statData.Brainchop_Ver = 'FullVolume'
  let startTime = performance.now()
  let adjusted_input_shape = []
  const res = await model
  // ?
  // model.then(function (res) {
  try {
    startTime = performance.now()
    const inferenceStartTime = performance.now()
    // maxLabelPredicted in whole volume of the brain
    let maxLabelPredicted = 0
    const transpose = modelEntry.enableTranspose
    const delay = modelEntry.inferenceDelay
    console.log('Inference delay :', delay)

    if (transpose) {
      cropped_slices_3d_w_pad = cropped_slices_3d_w_pad.transpose()
      console.log('Input transposed for pre-model')
    } else {
      console.log('Transpose not enabled for pre-model')
    }

    let i = 1
    const layersLength = res.layers.length
    console.log('res.layers.length ', layersLength)

    const isChannelLast = isModelChnlLast(res)
    const batchSize = opts.batchSize
    const numOfChan = opts.numOfChan

    // -- Adjust model input shape
    if (isChannelLast) {
      res.layers[0].batchInputShape[1] = cropped_slices_3d_w_pad.shape[0]
      res.layers[0].batchInputShape[2] = cropped_slices_3d_w_pad.shape[1]
      res.layers[0].batchInputShape[3] = cropped_slices_3d_w_pad.shape[2]

      adjusted_input_shape = [
        batchSize,
        res.layers[0].batchInputShape[1],
        res.layers[0].batchInputShape[2],
        res.layers[0].batchInputShape[3],
        numOfChan
      ]
    } else {
      res.layers[0].batchInputShape[2] = cropped_slices_3d_w_pad.shape[0]
      res.layers[0].batchInputShape[3] = cropped_slices_3d_w_pad.shape[1]
      res.layers[0].batchInputShape[4] = cropped_slices_3d_w_pad.shape[2]

      adjusted_input_shape = [
        batchSize,
        numOfChan,
        res.layers[0].batchInputShape[2],
        res.layers[0].batchInputShape[3],
        res.layers[0].batchInputShape[4]
      ]
    }

    console.log(' Model batch input shape : ', res.layers[0].batchInputShape)
    // -- batchInputShape {Array} input_shape - e.g. [?, D, H, W, Ch] or [?, Ch, D, H, W]

    statData.Input_Shape = JSON.stringify(res.layers[0].batchInputShape)
    statData.Output_Shape = JSON.stringify(res.output.shape)
    statData.Channel_Last = isChannelLast
    statData.Model_Param = await getModelNumParameters(res)
    statData.Model_Layers = await getModelNumLayers(res)
    statData.Model = modelEntry.modelName
    statData.Extra_Info = null

    const curTensor = []
    curTensor[0] = cropped_slices_3d_w_pad.reshape(adjusted_input_shape)
    // console.log("curTensor[0] :", curTensor[0].dataSync())

    // ? let curProgBar = parseInt(document.getElementById("progressBar").style.width)
    // let timer = window.setInterval(function() {
    // ???? subsequent await are required
    const timer = window.setInterval(async function () {
      try {
        // -- curTensor[i] = res.layers[i].apply( curTensor[i-1])
        curTensor[i] = res.layers[i].apply(curTensor[i - 1])
      } catch (err) {
        callbackUI(err.message, -1, err.message)
        window.clearInterval(timer)
        tf.engine().endScope()
        tf.engine().disposeVariables()

        statData.Inference_t = Infinity
        statData.Postprocess_t = Infinity
        statData.Status = 'Fail'
        statData.Error_Type = err.message
        statData.Extra_Err_Info = 'Failed while model layer ' + i + ' apply'

        if (opts.telemetryFlag) {
          submitTiming2GoogleSheet(statData, callbackUI)
        }

        return 0
      }
      callbackUI('Layer ' + i.toString(), (i + 1) / layersLength)
      console.log('layer output Tensor shape : ', curTensor[i].shape)
      console.log('layer count params ', res.layers[i].countParams())
      res.layers[i].dispose()
      curTensor[i - 1].dispose()
      if (tf.memory().unreliable) {
        const unreliableReasons = 'unreliable reasons :' + tf.memory().reasons
        callbackUI(unreliableReasons, NaN, unreliableReasons)
      }
      // ? document.getElementById("memoryStatus").style.backgroundColor =  memStatus

      if (i === layersLength - 1) {
        window.clearInterval(timer)

        // prediction = res.layers[res.layers.length-1].apply(curTensor[i])
        // curTensor[i].print()
        // outputDataBeforArgmx = Array.from(curTensor[i].dataSync())

        const axis = isChannelLast ? -1 : 1
        console.log(' find argmax ')
        console.log('last Tensor shape : ', curTensor[i].shape)
        // -- curTensor[i].shape  e.g. [ 1, 256, 256, 256, 3 ]
        const expected_Num_labels = isChannelLast ? curTensor[i].shape[4] : curTensor[i].shape[1]
        let prediction_argmax

        // Try for argMax with model output tensor.

        try {
          const argMaxTime = performance.now()
          console.log(' Try tf.argMax for fullVolume ..')
          prediction_argmax = tf.argMax(curTensor[i], axis)
          console.log('tf.argMax for fullVolume takes : ', ((performance.now() - argMaxTime) / 1000).toFixed(4))
        } catch (err1) {
          // if channel last
          if (axis === -1) {
            try {
              const argMaxLargeTime = performance.now()
              console.log(' tf.argMax failed .. try argMaxLarge ..')
              // todo tensor2LightBuffer()
              const modelOutBuffer = tensor2LightBuffer(
                curTensor[i].reshape([
                  cropped_slices_3d_w_pad.shape[0],
                  cropped_slices_3d_w_pad.shape[1],
                  cropped_slices_3d_w_pad.shape[2],
                  expected_Num_labels
                ]),
                'float16'
              )
              // todo argMaxLarge()
              prediction_argmax = argMaxLarge(
                modelOutBuffer,
                cropped_slices_3d_w_pad.shape[0],
                cropped_slices_3d_w_pad.shape[1],
                cropped_slices_3d_w_pad.shape[2],
                expected_Num_labels,
                'float16'
              )
              console.log(
                'argMaxLarge for fullVolume takes : ',
                ((performance.now() - argMaxLargeTime) / 1000).toFixed(4)
              )
            } catch (err2) {
              const errTxt = "argMax buffer couldn't be created due to limited memory resources."
              callbackUI(errTxt, -1, errTxt)

              window.clearInterval(timer)
              tf.engine().endScope()
              tf.engine().disposeVariables()

              statData.Inference_t = Infinity
              statData.Postprocess_t = Infinity
              statData.Status = 'Fail'
              statData.Error_Type = err2.message
              statData.Extra_Err_Info = 'prediction_argmax from argMaxLarge failed'

              if (opts.telemetryFlag) {
                submitTiming2GoogleSheet(statData, callbackUI)
              }

              return 0
            }
          } else {
            // if channel first ..
            const errTxt = "argMax buffer couldn't be created due to limited memory resources."
            callbackUI(errTxt, -1, errTxt)

            prediction_argmax.dispose()

            window.clearInterval(timer)
            tf.engine().endScope()
            tf.engine().disposeVariables()

            statData.Inference_t = Infinity
            statData.Postprocess_t = Infinity
            statData.Status = 'Fail'
            statData.Error_Type = err1.message
            statData.Extra_Err_Info = 'prediction_argmax from argMaxLarge not support yet channel first'

            if (opts.telemetryFlag) {
              submitTiming2GoogleSheet(statData, callbackUI)
            }

            return 0
          }
        }

        console.log(' prediction_argmax shape : ', prediction_argmax.shape)
        // -- prediction_argmax.shape  : [ 1, 256, 256, 256]

        const Inference_t = ((performance.now() - startTime) / 1000).toFixed(4)

        // outputDataBeforArgmx = Array.from(prediction_argmax.dataSync())
        tf.dispose(curTensor[i])
        // allPredictions.push({"id": allBatches[j].id, "coordinates": allBatches[j].coordinates, "data": Array.from(prediction_argmax.dataSync()) })
        const curBatchMaxLabel = await prediction_argmax.max().dataSync()[0]
        if (maxLabelPredicted < curBatchMaxLabel) {
          maxLabelPredicted = curBatchMaxLabel
        }

        const numSegClasses = maxLabelPredicted + 1
        console.log('numSegClasses', numSegClasses)
        statData.Actual_Labels = numSegClasses
        statData.Expect_Labels = expected_Num_labels
        statData.NumLabels_Match = numSegClasses === expected_Num_labels

        if (numSegClasses !== expected_Num_labels) {
          // errTxt = "expected " + expected_Num_labels + " labels, but the predicted are " + numSegClasses + ". For possible solutions please refer to <a href='https://github.com/neuroneural/brainchop/wiki/FAQ#Q3' target='_blank'><b> FAQ </b></a>.", "alert-error"
          const errTxt = 'expected ' + expected_Num_labels + ' labels, but the predicted are ' + numSegClasses
          callbackUI(errTxt, -1, errTxt)
        }

        // -- Transpose back to fit Papaya display settings
        let outLabelVolume = prediction_argmax.reshape([
          cropped_slices_3d_w_pad.shape[0],
          cropped_slices_3d_w_pad.shape[1],
          cropped_slices_3d_w_pad.shape[2]
        ])
        tf.dispose(prediction_argmax)

        // Transpose MRI data to be match pytorch/keras input output
        if (transpose) {
          console.log('outLabelVolume transposed')
          outLabelVolume = outLabelVolume.transpose()
        }
        // ???? await
        outLabelVolume = await removeZeroPaddingFrom3dTensor(outLabelVolume, pad, pad, pad)
        console.log(' outLabelVolume without padding shape :  ', outLabelVolume.shape)
        // ???? await
        outLabelVolume = await resizeWithZeroPadding(
          outLabelVolume,
          num_of_slices,
          slice_height,
          slice_width,
          refVoxel,
          boundVolSizeArr
        )
        console.log(' outLabelVolume final shape after resizing :  ', outLabelVolume.shape)

        const filterOutWithPreMask = modelEntry.filterOutWithPreMask
        // To clean the skull area wrongly segmented in phase-2.
        if (pipeline1_out != null && opts.isBrainCropMaskBased && filterOutWithPreMask) {
          const bin = binarizeVolumeDataTensor(pipeline1_out)
          outLabelVolume = outLabelVolume.mul(bin)
        }

        startTime = performance.now()
        // Generate output volume or slices
        console.log('Generating correct output')

        try {
          const img = new Uint32Array(outLabelVolume.dataSync())
          const Vshape = outLabelVolume.shape
          const Vtype = outLabelVolume.dtype
          tf.dispose(outLabelVolume)
          tf.engine().endScope()
          tf.engine().disposeVariables()
          // ???? await
          // BINGO
          outimg = await generateOutputSlicesV2(
            img,
            Vshape,
            Vtype,
            num_of_slices,
            numSegClasses,
            slice_height,
            slice_width,
            modelEntry,
            opts,
            niftiImage
          )
          console.log(' Phase-2 num of tensors after generateOutputSlicesV2: ', tf.memory().numTensors)
        } catch (error) {
          // -- Timing data to collect
          tf.engine().endScope()
          tf.engine().disposeVariables()

          const errTxt = 'Failed while generating output due to limited browser memory available'
          callbackUI(errTxt, -1, errTxt)
          statData.Inference_t = Inference_t
          statData.Postprocess_t = Infinity
          statData.Status = 'Fail'
          statData.Error_Type = error.message
          statData.Extra_Err_Info = 'Failed while generating output'

          if (opts.telemetryFlag) {
            submitTiming2GoogleSheet(statData, callbackUI)
          }

          return 0
        }

        const Postprocess_t = ((performance.now() - startTime) / 1000).toFixed(4)

        // ? $$("downloadBtn").enable()
        // ? $$("segmentBtn").enable()
        //    $$("imageUploader").enable()
        // tf.engine().endScope()
        tf.engine().disposeVariables()

        console.log(
          'Processing the whole brain volume in tfjs for multi-class output mask took : ',
          ((performance.now() - inferenceStartTime) / 1000).toFixed(4) + '  Seconds'
        )

        // -- Timing data to collect
        statData.Inference_t = Inference_t
        statData.Postprocess_t = Postprocess_t
        statData.Status = 'OK'

        if (opts.telemetryFlag) {
          submitTiming2GoogleSheet(statData, callbackUI)
        }
        clearInterval(timer)
        callbackUI('Segmentation finished', 0)
        callbackImg(outimg, opts, modelEntry)
        return 0
      }
      i++
    }, delay)
  } catch (err) {
    callbackUI(err.message, -1, err.message)
    console.log(
      'If webgl context is lost, try to restore webgl context by visit the link ' +
        '<a href="https://support.biodigital.com/hc/en-us/articles/218322977-How-to-turn-on-WebGL-in-my-browser">here</a>'
    )
    // ? document.getElementById("webGl2Status").style.backgroundColor =  isWebGL2ContextLost() ? "Red" : "Green"

    // ? document.getElementById("memoryStatus").style.backgroundColor =  tf.memory().unreliable ? "Red" : "Green"
  }
  //            })
}

async function inferenceFullVolumePhase1(
  model,
  slices_3d,
  num_of_slices,
  slice_height,
  slice_width,
  isModelFullVol,
  modelEntry,
  statData,
  opts,
  callbackImg,
  callbackUI,
  niftiImage
) {
  statData.No_SubVolumes = 1
  // load pre-model for inference first, can be null if no pre-model such as GWM models
  if (modelEntry.preModelId) {
    const preModel = await load_model(inferenceModelsList[modelEntry.preModelId - 1].path)
    const transpose = inferenceModelsList[modelEntry.preModelId - 1].enableTranspose
    const quantileNorm = inferenceModelsList[modelEntry.preModelId - 1].enableQuantileNorm
    let preModel_slices_3d = null

    // -- If pre-model is not null then slices_3d mask will be generated..
    // -- The mask is needed to remove the skull and set noise in background to 0, and get the brain bounding volume properly
    const slices_3d_mask = null

    if (quantileNorm) {
      // Quantile normalize function needs specific models to be used
      console.log('preModel Quantile normalization enabled')
      preModel_slices_3d = await quantileNormalizeVolumeData(slices_3d)
    } else {
      // Min Max Nomalize MRI data to be from 0 to 1
      console.log('preModel Min Max normalization enabled')
      preModel_slices_3d = await minMaxNormalizeVolumeData(slices_3d)
    }

    // -- Transpose MRI data to be match pytorch/keras input output
    // -- Check if pre-model needs transpose..
    if (transpose) {
      preModel_slices_3d = await preModel_slices_3d.transpose()
      console.log('Input transposed for pre-model')
    } else {
      console.log('Transpose not enabled for pre-model')
    }

    statData.Brainchop_Ver = 'PreModel_FV' // e.g. "PreModel_FV"

    // preModel.then(function (res) {
    const res = await preModel

    try {
      const inferenceStartTime = performance.now()
      const preModelObject = res

      // read input shape from model.json object
      const preModelBatchInputShape = preModelObject.layers[0].batchInputShape
      console.log(' Pre-Model batch input shape : ', preModelBatchInputShape)

      // -- Verify input shape
      if (preModelBatchInputShape.length !== 5) {
        const errTxt = 'The pre-model input shape must be 5D '
        callbackUI(errTxt, -1, errTxt)
        return 0
      }

      const isPreModelChannelLast = isModelChnlLast(preModelObject)
      const batchSize = opts.batchSize
      const numOfChan = opts.numOfChan
      let batch_D, batch_H, batch_W
      let preModel_input_shape
      if (isPreModelChannelLast) {
        console.log('Pre-Model Channel Last')
        if (isNaN(preModelBatchInputShape[4]) || preModelBatchInputShape[4] !== 1) {
          const errTxt = 'The number of channels for pre-model input shape must be 1'
          callbackUI(errTxt, -1, errTxt)
          return 0
        }

        batch_D = preModelBatchInputShape[1]
        batch_H = preModelBatchInputShape[2]
        batch_W = preModelBatchInputShape[3]

        preModel_input_shape = [batchSize, batch_D, batch_H, batch_W, numOfChan]
      } else {
        console.log('Pre-Model Channel First')
        if (isNaN(preModelBatchInputShape[1]) || preModelBatchInputShape[1] !== 1) {
          const errTxt = 'The number of channels for pre-model input shape must be 1'
          callbackUI(errTxt, -1, errTxt)
          return 0
        }

        batch_D = preModelBatchInputShape[2]
        batch_H = preModelBatchInputShape[3]
        batch_W = preModelBatchInputShape[4]

        preModel_input_shape = [batchSize, numOfChan, batch_D, batch_H, batch_W]
      }

      statData.Input_Shape = JSON.stringify(preModel_input_shape)
      statData.Output_Shape = JSON.stringify(preModelObject.output.shape)
      statData.Channel_Last = isPreModelChannelLast
      statData.Model_Param = await getModelNumParameters(preModelObject)
      statData.Model_Layers = await getModelNumLayers(preModelObject)
      // ? statData["Model"] = inferenceModelsList[ modelEntry["preModelId"] - 1]["modelName"]
      // ? statData["Extra_Info"] = inferenceModelsList[$$("selectModel").getValue() - 1]["modelName"]

      // maxLabelPredicted in whole volume of the brain
      let maxLabelPredicted = 0
      const delay = inferenceModelsList[modelEntry.preModelId - 1].inferenceDelay

      let i = 1
      const layersLength = res.layers.length

      const curTensor = []
      // -- reshape MRI to model input shape
      curTensor[0] = preModel_slices_3d.reshape(preModel_input_shape)

      // Dispose the volume
      tf.dispose(preModel_slices_3d)

      const timer = window.setInterval(async function () {
        try {
          curTensor[i] = res.layers[i].apply(curTensor[i - 1])
        } catch (err) {
          // ? original code provided special dialog for fragment shader if( err.message === "Failed to compile fragment shader.")
          callbackUI(err.message, -1, err.message)

          window.clearInterval(timer)
          tf.engine().endScope()
          tf.engine().disposeVariables()

          statData.Inference_t = Infinity
          statData.Postprocess_t = Infinity
          statData.Status = 'Fail'
          statData.Error_Type = err.message
          statData.Extra_Err_Info = 'PreModel Failed while model layer ' + i + ' apply'

          if (opts.telemetryFlag) {
            await submitTiming2GoogleSheet(statData, callbackUI)
          }

          return 0
        }

        res.layers[i].dispose()
        curTensor[i - 1].dispose()

        callbackUI('Layer ' + i.toString(), (i + 1) / layersLength)
        if (tf.memory().unreliable) {
          const unreliableReasons = 'unreliable reasons :' + tf.memory().reasons
          callbackUI(unreliableReasons, NaN, unreliableReasons)
        }

        if (i === layersLength - 1) {
          window.clearInterval(timer)

          // -- prediction = res.layers[res.layers.length-1].apply(curTensor[i])
          // -- curTensor[i].print()
          // -- outputDataBeforArgmx = Array.from(curTensor[i].dataSync())

          const axis = isPreModelChannelLast ? -1 : 1
          console.log(' find argmax ')
          console.log('last Tensor shape : ', curTensor[i].shape)
          // -- curTensor[i].shape  : [ 1, 256, 256, 256, 3 ]
          const expected_Num_labels = isPreModelChannelLast ? curTensor[i].shape[4] : curTensor[i].shape[1]
          let prediction_argmax

          // Try for argMax with model output tensor.

          try {
            console.log(' Try tf.argMax for fullVolume ..')
            prediction_argmax = await tf.argMax(curTensor[i], axis)
          } catch (err1) {
            // if channel last
            if (axis === -1) {
              try {
                const argMaxLargeTime = performance.now()
                console.log(' tf.argMax failed .. try argMaxLarge ..')
                const modelOutBuffer = tensor2LightBuffer(
                  curTensor[i].reshape([num_of_slices, slice_height, slice_width, expected_Num_labels]),
                  'float16'
                )
                prediction_argmax = argMaxLarge(
                  modelOutBuffer,
                  num_of_slices,
                  slice_height,
                  slice_width,
                  expected_Num_labels,
                  'float16'
                )
                console.log(
                  'argMaxLarge for fullVolume takes : ',
                  ((performance.now() - argMaxLargeTime) / 1000).toFixed(4)
                )
              } catch (err2) {
                const errTxt = "argMax buffer couldn't be created due to limited memory resources."
                callbackUI(errTxt, -1, errTxt)

                prediction_argmax.dispose()

                window.clearInterval(timer)
                tf.engine().endScope()
                tf.engine().disposeVariables()

                statData.Inference_t = Infinity
                statData.Postprocess_t = Infinity
                statData.Status = 'Fail'
                statData.Error_Type = err2.message
                statData.Extra_Err_Info = 'preModel prediction_argmax from argMaxLarge failed'

                if (opts.telemetryFlag) {
                  await submitTiming2GoogleSheet(statData, callbackUI)
                }

                return 0
              }
            } else {
              // if channel first ..
              const errTxt = "argMax buffer couldn't be created due to limited memory resources."
              callbackUI(errTxt, -1, errTxt)

              prediction_argmax.dispose()

              window.clearInterval(timer)
              tf.engine().endScope()
              tf.engine().disposeVariables()

              statData.Inference_t = Infinity
              statData.Postprocess_t = Infinity
              statData.Status = 'Fail'
              statData.Error_Type = err1.message
              statData.Extra_Err_Info = 'preModel prediction_argmax from argMaxLarge not support yet channel first'

              if (opts.telemetryFlag) {
                await submitTiming2GoogleSheet(statData, callbackUI)
              }

              return 0
            }
          }

          console.log(' Pre-model prediction_argmax shape : ', prediction_argmax.shape)
          // -- prediction_argmax.shape  : [ 1, 256, 256, 256]

          const Inference_t = ((performance.now() - inferenceStartTime) / 1000).toFixed(4)

          tf.dispose(curTensor[i])

          console.log(' Pre-model find array max ')
          const curBatchMaxLabel = await prediction_argmax.max().dataSync()[0]
          if (maxLabelPredicted < curBatchMaxLabel) {
            maxLabelPredicted = curBatchMaxLabel
          }

          const numSegClasses = maxLabelPredicted + 1
          console.log('Pre-model numSegClasses', numSegClasses)

          statData.Actual_Labels = numSegClasses
          statData.Expect_Labels = expected_Num_labels
          statData.NumLabels_Match = numSegClasses === expected_Num_labels

          // -- Transpose back to fit Papaya display settings
          let outLabelVolume = await prediction_argmax.reshape([num_of_slices, slice_height, slice_width])
          tf.dispose(prediction_argmax)
          // Transpose MRI data to be match pytorch/keras input output
          if (transpose) {
            console.log('Pre-model outLabelVolume transposed')
            outLabelVolume = outLabelVolume.transpose()
          }
          const startTime = performance.now()
          // Generate output volume or slices
          console.log('Generating pre-model output')
          let slices_3d_mask
          try {
            const unstackOutVolumeTensor = await tf.unstack(outLabelVolume)
            slices_3d_mask = await generateBrainMask(
              unstackOutVolumeTensor,
              num_of_slices,
              slice_height,
              slice_width,
              modelEntry,
              opts,
              callbackUI,
              callbackImg,
              false
            )
            await tf.dispose(outLabelVolume)
            console.log(' Phase-1 num of tensors after generateBrainMask: ', tf.memory().numTensors)
          } catch (error) {
            // -- Timing data to collect
            tf.engine().endScope()
            tf.engine().disposeVariables()

            const errTxt = 'Failed while generating pre-model output due to limited browser memory available'
            callbackUI(errTxt, -1, errTxt)

            statData.Inference_t = Inference_t
            statData.Postprocess_t = Infinity
            statData.Status = 'Fail'
            statData.Error_Type = error.message
            statData.Extra_Err_Info = 'Pre-model failed while generating output'

            if (opts.telemetryFlag) {
              await submitTiming2GoogleSheet(statData, callbackUI)
            }

            return 0
          }
          const Postprocess_t = ((performance.now() - startTime) / 1000).toFixed(4)
          console.log(
            'Pre-model processing the whole brain volume in tfjs tooks for multi-class output mask : ',
            ((performance.now() - inferenceStartTime) / 1000).toFixed(4) + '  Seconds'
          )

          // -- Timing data to collect
          statData.Inference_t = Inference_t
          statData.Postprocess_t = Postprocess_t
          statData.Status = 'OK'

          if (opts.telemetryFlag) {
            await submitTiming2GoogleSheet(statData, callbackUI)
          }

          if (slices_3d_mask == null) {
            const msg = 'slice_3d_mask failed ...'
            callbackUI(msg, -1, msg)
            return 0
          } else {
            // --Phase-2, After remove the skull try to allocate brain volume and make inferece
            console.log('--- pre-model done ---')
            // --mask_3d = slices_3d_mask.greater([0]).asType('bool')
            // --slices_3d_mask.dispose()

            if (isModelFullVol) {
              if (modelEntry.enableSeqConv) {
                // Mask cropping & seq conv
                // Non-Atlas model (e.g. GWM) needs sequential convolution layer.
                // Sequential convolution layer to be used after cropping - slow but reliable on most machines
                console.log('------ Mask Cropping & Seq Convoluton ------')
                await inferenceFullVolumeSeqCovLayerPhase2(
                  opts,
                  modelEntry,
                  model,
                  slices_3d,
                  num_of_slices,
                  slice_height,
                  slice_width,
                  slices_3d_mask,
                  callbackUI,
                  callbackImg,
                  statData,
                  niftiImage
                )
                return 0
                // inferenceFullVolumeSeqCovLayerPhase2(model, slices_3d.transpose(), num_of_slices, slice_height, slice_width, slices_3d_mask)
              } else {
                // Mask cropping BUT no seq conv
                console.log('------ Mask Cropping  -  NO Seq Convoluton ------')
                // ? await
                // todo output not used outimg = await
                await inferenceFullVolumePhase2(
                  model,
                  slices_3d,
                  num_of_slices,
                  slice_height,
                  slice_width,
                  slices_3d_mask,
                  modelEntry,
                  statData,
                  opts,
                  callbackImg,
                  callbackUI,
                  niftiImage
                )
                // inferenceFullVolumePhase2(model, slices_3d.transpose(), num_of_slices, slice_height, slice_width, slices_3d_mask)
              }
            } else {
              // -- In version 3.0.0 this function not used
              await inferenceSubVolumes(model, slices_3d, num_of_slices, slice_height, slice_width, slices_3d_mask)
              // inferenceSubVolumes(model, slices_3d.transpose(), num_of_slices, slice_height, slice_width, slices_3d_mask)
            }
          }
        }
        i++
      }, delay)
    } catch (err) {
      callbackUI(err.message, -1, err.message)
      console.log(
        'If webgl context is lost, try to restore webgl context by visit the link ' +
          '<a href="https://support.biodigital.com/hc/en-us/articles/218322977-How-to-turn-on-WebGL-in-my-browser">here</a>'
      )

      // document.getElementById("webGl2Status").style.backgroundColor =  isWebGL2ContextLost() ? "Red" : "Green"

      // document.getElementById("memoryStatus").style.backgroundColor =  tf.memory().unreliable ? "Red" : "Green"
    }
    // })

    // -- if(...)  end
  } else {
    // No preModel

    // --Phase-2, After remove the skull try to allocate brain volume and make inferece
    console.log('--- No pre-model is selected ---')
    console.log('------ Run voxel cropping ------')
    // -- mask_3d = slices_3d.greater([0]).asType('bool')

    if (isModelFullVol) {
      if (modelEntry.enableSeqConv) {
        // Voxel cropping & seq conv
        // Non-Atlas model (e.g. GWM) needs sequential convolution layer.
        // Sequential convolution layer to be used after cropping - slow but reliable on most machines
        console.log('------ Seq Convoluton ------')
        await inferenceFullVolumeSeqCovLayerPhase2(
          opts,
          modelEntry,
          model,
          slices_3d,
          num_of_slices,
          slice_height,
          slice_width,
          null,
          callbackUI,
          callbackImg,
          statData,
          niftiImage
        )
      } else {
        // Voxel cropping BUT no seq conv
        // todo: we do not use result const outimg = await
        inferenceFullVolumePhase2(
          model,
          slices_3d,
          num_of_slices,
          slice_height,
          slice_width,
          null,
          modelEntry,
          statData,
          opts,
          callbackImg,
          callbackUI,
          niftiImage
        )
      }
    } else {
      // -- In version 3.0.0 this function not used
      inferenceSubVolumes(model, slices_3d, num_of_slices, slice_height, slice_width, null)
    }
  }
}

async function enableProductionMode(textureF16Flag = true) {
  // -- tf.setBackend('cpu')
  // -- tf.removeBackend('cpu')
  // -- Calling enableProdMode() method
  await tf.enableProdMode()
  // -- Setting debug mode of the  environment
  tf.env().set('DEBUG', false)
  tf.env().set('WEBGL_FORCE_F16_TEXTURES', textureF16Flag)
  // -- set this flag so that textures are deleted when tensors are disposed.
  tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0)
  // -- tf.env().set('WEBGL_PACK', false)
  // -- Put ready after sets above
  await tf.ready()
  // -- Printing output
  console.log('tf env() flags :', tf.env().flags)
  console.log('tf env() features :', tf.env().features)
  console.log('tf env total features: ', Object.keys(tf.env().features).length)
  console.log(tf.getBackend())
}

async function runInference(opts, modelEntry, niftiHeader, niftiImage, callbackImg, callbackUI) {
  callbackUI('Segmentation started', 0)
  const startTime = performance.now()
  const batchSize = opts.batchSize
  const numOfChan = opts.numOfChan
  if (isNaN(batchSize) || batchSize !== 1) {
    const errTxt = 'The batch Size for input shape must be 1'
    callbackUI(errTxt, -1, errTxt)
    return 0
  }
  if (isNaN(numOfChan) || numOfChan !== 1) {
    const errTxt = 'The number of channels for input shape must be 1'
    callbackUI(errTxt, -1, errTxt)
    return 0
  }
  tf.engine().startScope()
  console.log('Batch size: ', batchSize)
  console.log('Num of Channels: ', numOfChan)
  const model = await load_model(modelEntry.path)
  await enableProductionMode(true)
  const modelObject = model
  let batchInputShape = []
  // free global variable of 16777216 voxel
  // allOutputSlices3DCC1DimArray = []
  // outputSceneRendered = false
  // read input shape from model.json object
  batchInputShape = modelObject.layers[0].batchInputShape
  console.log(' Model batch input shape : ', batchInputShape)
  // -- Verify input shape
  if (batchInputShape.length !== 5) {
    const errTxt = 'The model input shape must be 5D'
    callbackUI(errTxt, -1, errTxt)
    return 0
  }
  let batch_D, batch_H, batch_W
  let input_shape
  const slice_width = niftiHeader.dims[1]
  const slice_height = niftiHeader.dims[2]
  const num_of_slices = niftiHeader.dims[3]
  const isChannelLast = await isModelChnlLast(modelObject)
  if (isChannelLast) {
    console.log('Model Channel Last')
    if (isNaN(batchInputShape[4]) || batchInputShape[4] !== 1) {
      const errTxt = 'The number of channels for input shape must be 1'
      callbackUI(errTxt, -1, errTxt)
      return 0
    }
    batch_D = batchInputShape[1]
    batch_H = batchInputShape[2]
    batch_W = batchInputShape[3]

    input_shape = [batchSize, batch_D, batch_H, batch_W, numOfChan]
  } else {
    console.log('Model Channel First')
    if (isNaN(batchInputShape[1]) || batchInputShape[1] !== 1) {
      const errTxt = 'The number of channels for input shape must be 1'
      callbackUI(errTxt, -1, errTxt)
      return 0
    }
    batch_D = batchInputShape[2]
    batch_H = batchInputShape[3]
    batch_W = batchInputShape[4]
    input_shape = [batchSize, numOfChan, batch_D, batch_H, batch_W]
  }
  //  //-- Atlas version check
  // if ( (batch_D > 30) && (batch_H == 256) && (batch_W == 256) ) {
  //    const errTxt = "The subvolume dimension in z-axis shouldn't exceed 30 number of slices for browser limitation"
  //    callbackUI(errTxt, -1, errTxt)
  //    return 0
  // }
  // --Check whether the model will make inference at once as FullVolumeModel
  let isModelFullVol
  if (batch_D === 256 && batch_H === 256 && batch_W === 256) {
    isModelFullVol = true
  } else {
    isModelFullVol = false
  }
  // ? const modelNumLayers = modelObject.layers.length
  // Model output number of segmentations
  // ? const outLabels = modelObject.layers[modelNumLayers - 1].bias.shape[0]
  let allSlices = await getAllSlicesData1D(num_of_slices, niftiHeader, niftiImage)
  const allSlices_2D = await getAllSlices2D(allSlices, slice_height, slice_width)
  // free array from mem
  allSlices = null
  // Get slices_3d tensor
  let slices_3d = await getSlices3D(allSlices_2D)
  // free tensor from mem
  tf.dispose(allSlices_2D)
  const statData = []
  if (opts.telemetryFlag) {
    const Preprocess_t = ((performance.now() - startTime) / 1000).toFixed(4)
    // -- Timing data to collect
    const today = new Date()
    if (isModelFullVol) {
      statData.Brainchop_Ver = 'FullVolume'
    } else {
      statData.Brainchop_Ver = 'SubVolumes'
    }

    /* let geoData = getBrowserLocationInfo()
    if(geoData) {
          statData["Country"] = geoData["Country"]
          statData["State"] = geoData["Region"]
          statData["City"] = geoData["City"]
    } else {
          statData["Country"] = ""
          statData["State"] = ""
          statData["City"] = ""
    } */

    statData.Date = parseInt(today.getMonth() + 1) + '/' + today.getDate() + '/' + today.getFullYear()
    statData.Time =
      (await checkZero(today.getHours())) + ':' + checkZero(today.getMinutes()) + ':' + checkZero(today.getSeconds())
    // ? statData["File_Name"] = refFileName == "" ? opts.uiSampleName: refFileName
    statData.Input_Shape = JSON.stringify(batchInputShape)
    statData.Output_Shape = JSON.stringify(modelObject.output.shape)
    statData.Channel_Last = isChannelLast
    statData.Model_Param = await getModelNumParameters(modelObject)
    statData.Model_Layers = await getModelNumLayers(modelObject)

    statData.Preprocess_t = Preprocess_t
    statData.Model = modelEntry.modelName
    statData.Browser = await detectBrowser()
    statData.Browser_Ver = await detectBrowserVersion()
    statData.OS = await detectOperatingSys()
    // ? NiiVue requires WebGL2, all contemporary browsers support it statData["WebGL1"] = checkWebGl1()
    statData.WebGL2 = await checkWebGl2(callbackUI)
    statData.GPU_Vendor = await detectGPUVendor()
    statData.GPU_Card = await detectGPUCardType()
    statData.GPU_Vendor_Full = await detectGPUVendor_v0()
    statData.GPU_Card_Full = await detectGPUCardType_v0()
    statData.CPU_Cores = await getCPUNumCores()
    statData.TF_Backend = tf.getBackend()

    statData.Which_Brainchop = 'latest'
    // ? statData["Seq_Conv"] =  inferenceModelsList[$$("selectModel").getValue() - 1]["enableSeqConv"]
    statData.Seq_Conv = modelEntry.enableSeqConv

    // -- Init
    statData.Actual_Labels = Infinity
    statData.Expect_Labels = Infinity
    statData.NumLabels_Match = null
    statData.Inference_t = Infinity
    statData.Merge_t = Infinity
    statData.Postprocess_t = Infinity
    statData.Status = null
    statData.Error_Type = null
    statData.Extra_Err_Info = null
    statData.Extra_Info = null

    if (isChrome()) {
      statData.Heap_Size_MB = window.performance.memory.totalJSHeapSize / (1024 * 1024).toFixed(2)
      statData.Used_Heap_MB = window.performance.memory.usedJSHeapSize / (1024 * 1024).toFixed(2)
      statData.Heap_Limit_MB = window.performance.memory.jsHeapSizeLimit / (1024 * 1024).toFixed(2)
    }
    const gl = checkWebGl2() ? document.createElement('canvas').getContext('webgl2') : null

    console.log('MAX_TEXTURE_SIZE :', gl.getParameter(gl.MAX_TEXTURE_SIZE))
    console.log('MAX_RENDERBUFFER_SIZE :', gl.getParameter(gl.MAX_RENDERBUFFER_SIZE))

    // -- check to see   if  machine has two graphics card: one is the builtin e.g. Intel Iris Pro, the other is NVIDIA GeForce GT 750M.
    // -- check browser use which one, if debugInfo is null then installed  GPU is not used
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
    console.log('VENDOR WEBGL:', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL))

    if (gl) {
      statData.Texture_Size = gl.getParameter(gl.MAX_TEXTURE_SIZE) // --returns the maximum dimension the GPU can address
    } else {
      statData.Texture_Size = null
    }
  } // if telemetryFlag
  const transpose = modelEntry.enableTranspose
  const enableCrop = modelEntry.enableCrop
  if (isModelFullVol) {
    if (enableCrop) {
      // FullVolume with Crop option before inference ..
      // pre-model to mask the volume, can also be null and the cropping will be on the MRI.
      await inferenceFullVolumePhase1(
        model,
        slices_3d,
        num_of_slices,
        slice_height,
        slice_width,
        isModelFullVol,
        modelEntry,
        statData,
        opts,
        callbackImg,
        callbackUI,
        niftiImage
      )
    } else {
      // Transpose MRI data to be match pytorch/keras input output
      console.log('Cropping Disabled')

      if (transpose) {
        slices_3d = slices_3d.transpose()
        console.log('Input transposed')
      } else {
        console.log('Transpose NOT Enabled')
      }

      const enableSeqConv = modelEntry.enableSeqConv

      if (enableSeqConv) {
        console.log('Seq Convoluton Enabled')
        await inferenceFullVolumeSeqCovLayer(
          model,
          slices_3d,
          input_shape,
          isChannelLast,
          num_of_slices,
          slice_height,
          slice_width
        )
      } else {
        console.log('Seq Convoluton Disabled')
        await inferenceFullVolume(
          model,
          slices_3d,
          input_shape,
          isChannelLast,
          num_of_slices,
          slice_height,
          slice_width
        )
      }
    }
  }
}
