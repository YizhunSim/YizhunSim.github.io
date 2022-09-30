const canvasParent = document.getElementById('canvas-parent');
const weirdFilter = document.getElementById('weird-filter');
const edgeDetectionFilter = document.getElementById('edge-detection-filter');
const gpuEnabled = document.getElementById('gpu-enabled');
const fpsNumber = document.getElementById('fps-number');
const computationLogMsg = document.getElementById('computation-log-message');
const image = document.createElement('img');
image.src = '../stingray.jpeg';
const displayMode = document.getElementById('display-mode');
// ---------------------------------------------
//  Event Handlers setup to handle Javascript checkbox
//  selection from the user
// ---------------------------------------------
const radioButtons_weirdFilters = document.querySelectorAll(
  'input[name="weird-filterss"]'
);
for (const radioButton of radioButtons_weirdFilters) {
  radioButton.addEventListener('change', showSelectedWeirdFilter);
  radioButton.disabled = true;
}

const radioButtons_edgeDetectionFilters = document.querySelectorAll(
  'input[name="edge-detection-filters"]'
);
for (const radioButton of radioButtons_edgeDetectionFilters) {
  radioButton.addEventListener('change', showSelectedEdgeDetectionFilter);
  radioButton.disabled = true;
}

gpuEnabled.addEventListener('change', (e) => {
  if (e.target.checked) {
    displayMode.innerText = 'GPU Mode';
  } else {
    displayMode.innerText = 'CPU Mode';
  }
});

weirdFilter.addEventListener('change', (e) => {
  if (e.target.checked) {
    document
      .getElementsByName('weird-filterss')
      .forEach((x) => (x.disabled = false));
    // document.getElementById('background-image').checked = true;
  } else {
    document
      .getElementsByName('weird-filterss')
      .forEach((x) => (x.checked = false));
    document
      .getElementsByName('weird-filterss')
      .forEach((x) => (x.disabled = true));
  }
});

edgeDetectionFilter.addEventListener('change', (e) => {
  if (e.target.checked) {
    document
      .getElementsByName('edge-detection-filters')
      .forEach((x) => (x.disabled = false));
    // document.getElementById('identity-matrix').checked = true;
  } else {
    document
      .getElementsByName('edge-detection-filters')
      .forEach((x) => (x.checked = false));
    document
      .getElementsByName('edge-detection-filters')
      .forEach((x) => (x.disabled = true));
  }
});

let MAX_NUM_TIMINGS;

let sliderInput = document.getElementById('input-slider'),
  sliderOutput = document.querySelector('output');

sliderOutput.innerHTML = sliderInput.value;

// use 'change' instead to see the difference in response
sliderInput.addEventListener(
  'change',
  function () {
    sliderOutput.innerHTML = sliderInput.value;
    MAX_NUM_TIMINGS = sliderInput.value;
  },
  false
);

// Utility to fetch a kernel, while timing the runtime of each invocation.
const getKernelTimed = (function () {
  const timings = {};
  MAX_NUM_TIMINGS = sliderOutput.innerHTML;
  console.log('MAX_NUM_TIMINGS: ' + MAX_NUM_TIMINGS);

  return function (mode, kernelName, kernel, filters) {
    // Create timing array for the kernel if doesn't exist.
    if (!timings[kernelName]) timings[kernelName] = [];
    const kernelTimings = timings[kernelName];

    return function () {
      const { time, returnValue } = timeThis(() => {
        kernel.apply(null, arguments);
      });
      kernelTimings.push(time);

      if (kernelTimings.length >= MAX_NUM_TIMINGS) {
        const avg = kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
        let computationLogMessage = `Average runtime of ${kernelName} (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on ${mode} mode`;
        console.log(computationLogMessage);
        computationLogMessage += `\n Filters in pipeline: `;

        console.log(`Filters in pipeline: `);
        for (let filter of filters) {
          computationLogMessage += filter.name + '-> ';
          console.log(filter);
        }
        console.log(
          'computationLogMessage: ' +
            computationLogMessage.substring(0, computationLogMessage.length - 3)
        );
        computationLogMsg.innerText = computationLogMessage.substring(
          0,
          computationLogMessage.length - 3
        );

        timings[kernelName] = [];
      }
      return returnValue;
    };
  };
})();

let lastCalledTime = Date.now();
let fps;
let delta;
let dispose = setup();
gpuEnabled.onchange = () => {
  if (dispose) dispose();
  dispose = setup();
};

function setup() {
  let disposed = false;
  const gpu = new GPU({ mode: gpuEnabled.checked ? 'gpu' : 'cpu' });

  // ------------------------
  // START KERNAL DEFINITIONS
  // ------------------------

  const gpuSettings = {
    output: [1024, 768],
    graphical: true,
    tactic: 'precision',
    pipeline: true,
  };

  const gpuMatrixSettings = {
    output: [1024, 768],
    graphical: true,
    tactic: 'precision',
    pipeline: true,
    immutable: true,
  };

  //Edge Detection Convolution Matrixes
  const prewittMatrix = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
  const sobelMatrix = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const laplacianMatrix = [0, 1, 0, 1, -4, 1, 0, 1, 0];
  const sharpenMatrix = [0, -1, 0, -1, 5, -1, 0, -1, 0];
  const gaussianBlurMatrix = [1, 2, 1, 2, 4, 2, 1, 2, 1].map((x) => x * 0.0625);
  const embossMatrix = [-2, -1, 0, -1, 1, 1, 0, 1, 2];

  // Kernel: Renders a 3-D array into a 2-D graphic array via a Canvas
  const kernelRenderGraphical = gpu.createKernel(
    function (combineTextures) {
      const pixel = combineTextures[this.thread.y][this.thread.x];

      this.color(pixel[0], pixel[1], pixel[2], pixel[3]);
    },
    {
      output: [1024, 768],
      graphical: true,
      tactic: 'precision',
    }
  );

  // Kernel: Renders a 3-D array texture into a 2-D graphic array (texture)
  const kernelWithVideoFeed = gpu.createKernel(function (frame) {
    const pixel = frame[this.thread.y][this.thread.x];
    this.color(pixel.r, pixel.g, pixel.b, pixel.a);
  }, gpuSettings);

  const kernelWithBackgroundImage = gpu.createKernel(
    function (frame, bg) {
      const pixel = frame[this.thread.y][this.thread.x];
      const bgpixel = bg[this.thread.y][this.thread.x];
      if (pixel.r > 0.5 && pixel.g > 0.5 && pixel.b > 0.5) {
        this.color(bgpixel.r, bgpixel.g, bgpixel.b, 1);
      } else {
        this.color(pixel.r, pixel.g, pixel.b, pixel.a);
      }
    },

    gpuSettings
  );

  // Kernel: Grey Filter
  const kernelWithGreyFilter = gpu.createKernel(
    function (frame) {
      const pixel = frame[this.thread.y][this.thread.x];
      let r = pixel[0],
        g = pixel[1],
        b = pixel[2],
        a = pixel[3];
      let c = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      this.color(c, c, c, 1);
    },

    gpuSettings
  );

  // Kernel: Invert Filter
  const kernelWithInvertFilter = gpu.createKernel(
    function (frame) {
      const pixel = frame[this.thread.y][this.thread.x];
      // Invert
      let r = 1.0 - pixel[0];
      let g = 1.0 - pixel[1];
      let b = 1.0 - pixel[2];
      this.color(r, g, b, 1);
    },

    gpuSettings
  );

  // Kernel: Red Filter
  const kernelWithRedFilter = gpu.createKernel(
    function (frame) {
      const pixel = frame[this.thread.y][this.thread.x];
      this.color(1 - pixel.r, pixel.g, pixel.b, pixel.a);
    },

    gpuSettings
  );

  // Kernel: Green Filter
  const kernelWithGreenFilter = gpu.createKernel(
    function (frame) {
      const pixel = frame[this.thread.y][this.thread.x];
      this.color(pixel.r, 1 - pixel.g, pixel.b, pixel.a);
    },

    gpuSettings
  );

  const kernelWithBlueFilter = gpu.createKernel(
    function (frame) {
      const pixel = frame[this.thread.y][this.thread.x];
      this.color(pixel.r, pixel.g, 1 - pixel.b, pixel.a);
    },

    gpuSettings
  );

  // Kernel: Mirror Filter
  const kernelWithMirrorFilter = gpu.createKernel(
    function (frame) {
      // Mirror
      const pixelMirror = frame[this.thread.y][1024 - this.thread.x];
      this.color(pixelMirror[0], pixelMirror[1], pixelMirror[2], 1);
    },

    gpuSettings
  );

  // Kernel: Flip Filter
  const kernelWithFlipFilter = gpu.createKernel(
    function (frame) {
      const pixelFlip = frame[768 - this.thread.y][this.thread.x];
      this.color(pixelFlip[0], pixelFlip[1], pixelFlip[2], 1);
    },

    gpuSettings
  );

  // Kernel: Sepia Filter
  const kernelWithSepiaFilter = gpu.createKernel(
    function (frame) {
      const pixelSepia = frame[this.thread.y][this.thread.x];
      let r = pixelSepia[0],
        g = pixelSepia[1],
        b = pixelSepia[2];
      let s = 0.3 * r + 0.59 * g + 0.11 * b;
      this.color(s + 40.0 / 255.0, s + 20.0 / 255.0, s - 20.0 / 255.0, 1);
    },

    gpuSettings
  );

  // Kernel: Saturation Filter
  const kernelWithSaturationFilter = gpu.createKernel(
    function (frame) {
      const pixelSaturation = frame[this.thread.y][this.thread.x];
      let rw = 0.3086,
        rg = 0.6084,
        rb = 0.082;
      let rw0 = (1 - 2.9) * rw + 2.9;
      let rw1 = (1 - 2.9) * rw;
      let rw2 = (1 - 2.9) * rw;
      let rg0 = (1 - 2.9) * rg;
      let rg1 = (1 - 2.9) * rg + 2.9;
      let rg2 = (1 - 2.9) * rg;
      let rb0 = (1 - 2.9) * rb;
      let rb1 = (1 - 2.9) * rb;
      let rb2 = (1 - 2.9) * rb + 2.9;

      let r =
        rw0 * pixelSaturation[0] +
        rg0 * pixelSaturation[1] +
        rb0 * pixelSaturation[2];
      let g =
        rw1 * pixelSaturation[0] +
        rg1 * pixelSaturation[1] +
        rb1 * pixelSaturation[2];
      let b =
        rw2 * pixelSaturation[0] +
        rg2 * pixelSaturation[1] +
        rb2 * pixelSaturation[2];
      this.color(r, g, b, 1.0);
    },

    gpuSettings
  );

  // Kernel: Edge Detection dynamic filter
  const kernelWithMatrix = gpu.createKernel(
    function (frame, matrix) {
      const pixel = frame[this.thread.y][this.thread.x];
      var col = [0, 0, 0];
      if (
        this.thread.y > 0 &&
        this.thread.y < 768 - 2 &&
        this.thread.x < 1024 - 2 &&
        this.thread.x > 0
      ) {
        const a0 = frame[this.thread.y + 1][this.thread.x - 1];
        const a1 = frame[this.thread.y + 1][this.thread.x];
        const a2 = frame[this.thread.y + 1][this.thread.x + 1];
        const a3 = frame[this.thread.y][this.thread.x - 1];
        const a4 = frame[this.thread.y][this.thread.x];
        const a5 = frame[this.thread.y][this.thread.x + 1];
        const a6 = frame[this.thread.y - 1][this.thread.x - 1];
        const a7 = frame[this.thread.y - 1][this.thread.x];
        const a8 = frame[this.thread.y - 1][this.thread.x + 1];
        for (var i = 0; i < 3; i++) {
          // Compute the convolution for each of red [0], green [1] and blue [2]
          col[i] =
            a0[i] * matrix[0] +
            a1[i] * matrix[1] +
            a2[i] * matrix[2] +
            a3[i] * matrix[3] +
            a4[i] * matrix[4] +
            a5[i] * matrix[5] +
            a6[i] * matrix[6] +
            a7[i] * matrix[7] +
            a8[i] * matrix[8];
        }
        this.color(col[0], col[1], col[2], 1);
      } else {
        this.color(pixel.r, pixel.g, pixel.b, pixel.a);
      }
    },

    gpuMatrixSettings
  );

  canvasParent.appendChild(kernelRenderGraphical.canvas);
  const videoElement = document.querySelector('video');

  const backgroundImageCheckBox = document.getElementById('background-image');
  const grayscaleFilterCheckBox = document.getElementById('grayscale-filter');
  const invertFilterCheckBox = document.getElementById('invert-filter');
  const redFilterCheckBox = document.getElementById('red-filter');
  const greenFilterCheckBox = document.getElementById('green-filter');
  const blueFilterCheckBox = document.getElementById('blue-filter');
  const mirrorFilterCheckBox = document.getElementById('mirror-filter');
  const flipFilterCheckBox = document.getElementById('flip-filter');
  const sepiaFilterCheckBox = document.getElementById('sepia-filter');
  const saturationFilterCheckBox = document.getElementById('saturation-filter');
  const prewittMatrixCheckBox = document.getElementById('prewit-matrix');
  const sobelMatrixCheckBox = document.getElementById('sobel-matrix');
  const laplacianMatrixCheckBox = document.getElementById('laplacian-matrix');
  const sharpenMatrixCheckBox = document.getElementById('sharpen-matrix');
  const gaussianMatrixCheckBox = document.getElementById(
    'gaussian-blur-matrix'
  );
  const embossMatrixCheckBox = document.getElementById('emboss-matrix');

  function render() {
    if (disposed) {
      return;
    }

    let filters = [];
    let result;
    if (gpuEnabled.checked) {
      let filter = {
        mode: 'gpu',
        name: 'kernelWithVideoFeed',
        kernelFunction: kernelWithVideoFeed,
        frame: videoElement,
        frameArg: null,
      };
      filters.push(filter);
      result = kernelWithVideoFeed(videoElement);

      if (backgroundImageCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithBackgroundImage',
          kernelFunction: kernelWithBackgroundImage,
          frame: result,
          frameArg: image,
        };
        filters.push(filter);
        result = kernelWithBackgroundImage(result, image);
      }

      if (grayscaleFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithGreyFilter',
          kernelFunction: kernelWithGreyFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithGreyFilter(result);
      }

      if (redFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithRedFilter',
          kernelFunction: kernelWithRedFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithRedFilter(result);
      }

      if (greenFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithGreenFilter',
          kernelFunction: kernelWithGreenFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithGreenFilter(result);
      }

      if (blueFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithBlueFilter',
          kernelFunction: kernelWithBlueFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithBlueFilter(result);
      }

      if (invertFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithInvertFilter',
          kernelFunction: kernelWithInvertFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithInvertFilter(result);
      }

      if (mirrorFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithMirrorFilter',
          kernelFunction: kernelWithMirrorFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithMirrorFilter(result);
      }

      if (flipFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithFlipFilter',
          kernelFunction: kernelWithFlipFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithFlipFilter(result);
      }

      if (sepiaFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithSepiaFilter',
          kernelFunction: kernelWithSepiaFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithSepiaFilter(result);
      }

      if (saturationFilterCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithSaturationFilter',
          kernelFunction: kernelWithSaturationFilter,
          frame: result,
          frameArg: null,
        };
        filters.push(filter);
        result = kernelWithSaturationFilter(result);
      }

      if (prewittMatrixCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithPrewittMatrix',
          kernelFunction: kernelWithMatrix,
          frame: result,
          frameArg: prewittMatrix,
        };
        filters.push(filter);
        result = kernelWithMatrix(result, prewittMatrix);
      }

      if (sobelMatrixCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithSobelMatrix',
          kernelFunction: kernelWithMatrix,
          frame: result,
          frameArg: sobelMatrix,
        };
        filters.push(filter);
        result = kernelWithMatrix(result, sobelMatrix);
      }

      if (laplacianMatrixCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithLaplacianMatrix',
          kernelFunction: kernelWithMatrix,
          frame: result,
          frameArg: laplacianMatrix,
        };
        filters.push(filter);
        result = kernelWithMatrix(result, laplacianMatrix);
      }

      if (sharpenMatrixCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithSharpenMatrix',
          kernelFunction: kernelWithMatrix,
          frame: result,
          frameArg: sharpenMatrix,
        };
        filters.push(filter);
        result = kernelWithMatrix(result, sharpenMatrix);
      }

      if (gaussianMatrixCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithGaussianMatrix',
          kernelFunction: kernelWithMatrix,
          frame: result,
          frameArg: gaussianBlurMatrix,
        };
        filters.push(filter);
        result = kernelWithMatrix(result, gaussianBlurMatrix);
      }

      if (embossMatrixCheckBox.checked) {
        let filter = {
          mode: 'gpu',
          name: 'kernelWithEmbossMatrix',
          kernelFunction: kernelWithMatrix,
          frame: result,
          frameArg: embossMatrix,
        };
        filters.push(filter);
        result = kernelWithMatrix(result, embossMatrix);
      }
      console.log('----- Kernel Computation on GPU -----');

      getKernelTimed(
        'gpu',
        'kernelRenderGraphical',
        kernelRenderGraphical,
        filters
      )(result);
    } else {
      backgroundImageCheckBox.disabled = true;
      mirrorFilterCheckBox.disabled = true;
      flipFilterCheckBox.disabled = true;
      sepiaFilterCheckBox.disabled = true;
      saturationFilterCheckBox.disabled = true;

      console.log('----- Kernel Computation on CPU -----');
      kernelWithVideoFeed(videoElement);

      const timings = {};
      const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

      if (!timings['kernelWithGreyFilter'])
        timings['kernelWithGreyFilter'] = [];
      const kernelTimings = timings['kernelWithGreyFilter'];

      const { time, returnValue } = timeThis(() => {
        kernelWithVideoFeed(videoElement);
      });
      kernelTimings.push(time);

      if (kernelTimings.length >= MAX_NUM_TIMINGS) {
        const avg = kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
        let computationLogMessage = `Average runtime of kernelWithVideoFeed (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
        console.log(computationLogMessage);
        computationLogMsg.innerText = computationLogMessage;
        timings['kernelWithVideoFeed'] = [];
      }

      /* Pipelining not possible with this kernel, due to 2 threads running in parallel
      if (backgroundImageCheckBox.checked) {
        kernelWithBackgroundImage(videoElement, image);
      } */
      if (grayscaleFilterCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithGreyFilter'])
          timings['kernelWithGreyFilter'] = [];
        const kernelTimings = timings['kernelWithGreyFilter'];

        const { time, returnValue } = timeThis(() => {
          kernelWithGreyFilter(videoElement);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithGreyFilter (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithGreyFilter'] = [];
        }
        // kernelWithGreyFilter(videoElement);
      } else if (redFilterCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithRedFilter'])
          timings['kernelWithRedFilter'] = [];
        const kernelTimings = timings['kernelWithRedFilter'];

        const { time, returnValue } = timeThis(() => {
          kernelWithRedFilter(videoElement);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithRedFilter (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithRedFilter'] = [];
        }
        // kernelWithRedFilter(videoElement);
      } else if (greenFilterCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithGreenFilter'])
          timings['kernelWithGreenFilter'] = [];
        const kernelTimings = timings['kernelWithGreenFilter'];

        const { time, returnValue } = timeThis(() => {
          kernelWithGreenFilter(videoElement);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithGreenFilter (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithGreenFilter'] = [];
        }
        // kernelWithGreenFilter(videoElement);
      } else if (blueFilterCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithBlueFilter'])
          timings['kernelWithBlueFilter'] = [];
        const kernelTimings = timings['kernelWithBlueFilter'];

        const { time, returnValue } = timeThis(() => {
          kernelWithBlueFilter(videoElement);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithBlueFilter (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithBlueFilter'] = [];
        }
        // kernelWithBlueFilter(videoElement);
      } else if (invertFilterCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithInvertFilter'])
          timings['kernelWithInvertFilter'] = [];
        const kernelTimings = timings['kernelWithInvertFilter'];

        const { time, returnValue } = timeThis(() => {
          kernelWithInvertFilter(videoElement);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithInvertFilter (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithInvertFilter'] = [];
        }
        // kernelWithInvertFilter(videoElement);
      } else if (prewittMatrixCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithPrewittMatrix'])
          timings['kernelWithPrewittMatrix'] = [];
        const kernelTimings = timings['kernelWithPrewittMatrix'];

        const { time, returnValue } = timeThis(() => {
          kernelWithMatrix(videoElement, prewittMatrix);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithPrewittMatrix (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithPrewittMatrix'] = [];
        }
        // kernelWithMatrix(videoElement, prewittMatrix);
      } else if (sobelMatrixCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithSobelMatrix'])
          timings['kernelWithSobelMatrix'] = [];
        const kernelTimings = timings['kernelWithSobelMatrix'];

        const { time, returnValue } = timeThis(() => {
          kernelWithMatrix(videoElement, sobelMatrix);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithSobelMatrix (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithSobelMatrix'] = [];
        }
        // kernelWithMatrix(videoElement, sobelMatrix);
      } else if (laplacianMatrixCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithLaplacianMatrix'])
          timings['kernelWithLaplacianMatrix'] = [];
        const kernelTimings = timings['kernelWithLaplacianMatrix'];

        const { time, returnValue } = timeThis(() => {
          kernelWithMatrix(videoElement, laplacianMatrix);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithLaplacianMatrix (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithLaplacianMatrix'] = [];
        }
        // kernelWithMatrix(videoElement, laplacianMatrix);
      } else if (sharpenMatrixCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithSharpenMatrix'])
          timings['kernelWithSharpenMatrix'] = [];
        const kernelTimings = timings['kernelWithSharpenMatrix'];

        const { time, returnValue } = timeThis(() => {
          kernelWithMatrix(videoElement, sharpenMatrix);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithSharpenMatrix (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithSharpenMatrix'] = [];
        }
        // kernelWithMatrix(videoElement, sharpenMatrix);
      } else if (gaussianMatrixCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithGaussianMatrix'])
          timings['kernelWithGaussianMatrix'] = [];
        const kernelTimings = timings['kernelWithGaussianMatrix'];

        const { time, returnValue } = timeThis(() => {
          kernelWithMatrix(videoElement, gaussianBlurMatrix);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithGaussianMatrix (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithGaussianMatrix'] = [];
        }
        // kernelWithMatrix(videoElement, gaussianBlurMatrix);
      } else if (embossMatrixCheckBox.checked) {
        const timings = {};
        const MAX_NUM_TIMINGS = sliderOutput.innerHTML;

        if (!timings['kernelWithEmbossMatrix'])
          timings['kernelWithEmbossMatrix'] = [];
        const kernelTimings = timings['kernelWithEmbossMatrix'];

        const { time, returnValue } = timeThis(() => {
          kernelWithMatrix(videoElement, embossMatrix);
        });
        kernelTimings.push(time);

        if (kernelTimings.length >= MAX_NUM_TIMINGS) {
          const avg =
            kernelTimings.reduce((sum, t) => sum + t) / MAX_NUM_TIMINGS;
          let computationLogMessage = `Average runtime of kernelWithEmbossMatrix (avg of ${MAX_NUM_TIMINGS} runs) is ${avg} ms. Running on CPU mode`;
          console.log(computationLogMessage);
          computationLogMsg.innerText = computationLogMessage;
          timings['kernelWithEmbossMatrix'] = [];
        }
        // kernelWithMatrix(videoElement, embossMatrix);
      }
      /* Pipelining not possible with kernel Mirror, due to calculation unable to be made by CPU
      else if (mirrorFilterCheckBox.checked) {
        kernelWithMirrorFilter(videoElement);
      }*/
      /* Pipelining not possible with kernel Mirror, due to calculation unable to be made by CPU
      else if (flipFilterCheckBox.checked) {
        kernelWithFlipFilter(videoElement);
      }*/
      /* Pipelining not possible with kernel Mirror, due to calculation unable to be made by CPU
      else if (sepiaFilterCheckBox.checked) {
        kernelWithSepiaFilter(videoElement);
      }*/
      /* Pipelining not possible with kernel Mirror, due to calculation unable to be made by CPU
      else if (saturationFilterCheckBox.checked) {
        kernelWithSaturationFilter(videoElement);
      }*/
    }

    window.requestAnimationFrame(render);
    calcFPS();
  }

  render();
  return () => {
    canvasParent.removeChild(kernelRenderGraphical.canvas);
    gpu.destroy();
    disposed = true;
  };
}

function streamHandler(stream) {
  try {
    video.srcObject = stream;
  } catch (error) {
    video.src = URL.createObjectURL(stream);
  }
  video.play();
  console.log('In startStream');
  requestAnimationFrame(render);
}

addEventListener('DOMContentLoaded', initialize);

function calcFPS() {
  delta = (Date.now() - lastCalledTime) / 1000;
  lastCalledTime = Date.now();
  fps = 1 / delta;
  fpsNumber.innerHTML = fps.toFixed(0);
}

function showSelectedWeirdFilter(e) {
  if (this.checked) {
    console.log(`You selected ${this.value}`);
  }
}

function showSelectedEdgeDetectionFilter(e) {
  if (this.checked) {
    console.log(`You selected ${this.value}`);
  }
}

// Utility to time the invocation of a lambda function.
function timeThis(lambda) {
  const start = performance.now();
  const returnValue = lambda();
  const end = performance.now();
  return { time: end - start, returnValue };
}
